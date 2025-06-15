# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def reached_target(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the rover is within a certain threshold distance from the target.
    If the target is reached, a scaled reward is returned based on the remaining time steps.
    """
    # Accessing the target's position w.r.t. the robot frame
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)
    time_steps_to_goal = env.max_episode_length - env.episode_length_buf
    reward_scale = time_steps_to_goal / env.max_episode_length

    # Return the reward, scaled depending on the remaining time steps
    return torch.where(distance < threshold, 1.0 * reward_scale, 0)


def action_penalty_near_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize large actions when close to obstacles.

    Returns negative reward proportional to action magnitude and inverse distance to nearest obstacle.
    Only active when within critical_dist.
    """
    sensor: RayCasterCfg = env.scene.sensors[sensor_cfg.name]

    differences = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1).expand(-1, sensor.data.ray_hits_w.size(1), -1)
    norm_differences = torch.norm(differences, p=2, dim=2)

    clipped_distances = torch.where(
        torch.isinf(norm_differences),
        sensor.cfg.max_distance,
        torch.clamp(norm_differences, max=sensor.cfg.max_distance),
    )

    # normalization coefficients
    critical_dist = 1.5
    sigmoid_coeff = 5.0

    sigmoid_distances = 1 / (1 + torch.exp(-sigmoid_coeff * (clipped_distances - critical_dist)))
    min_sigmoid_distance = sigmoid_distances.min(dim=1).values

    action = env.action_manager.action

    # penalty coefficients
    critical_sigmoid_dist_for_action = 0.08
    max_action_penalty = 0.5
    power = 2.0

    reward = torch.where(
        min_sigmoid_distance < critical_sigmoid_dist_for_action,
        max_action_penalty
        * (torch.norm(action, p=2, dim=1) ** power)
        * (1 - min_sigmoid_distance / critical_sigmoid_dist_for_action),
        0.0,
    )

    return reward


def penalty_for_sideways_movement(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Penalize sideways movement (large absolute Y-axis actions).

    Returns negative reward proportional to the squared Y-component of the action,
    encouraging the dog to move forward rather than sideways.
    """
    action = env.action_manager.action

    y_action = action[:, 1].abs()

    max_penalty = 0.5  # Maximum penalty when y_action is at threshold
    y_threshold = 0.3  # Threshold above which penalty starts applying
    power = 2.0  # How sharply penalty increases with y_action

    reward = torch.where(
        y_action > y_threshold, max_penalty * ((y_action - y_threshold) / (1 - y_threshold)) ** power, 0.0
    )

    return reward
