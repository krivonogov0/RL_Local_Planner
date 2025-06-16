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
    """Computes a reward signal when the agent reaches within threshold distance of the target.

    The reward can optionally include a time-based scaling factor that increases the reward
    when the target is reached earlier in the episode.

    Args:
        env (ManagerBasedRLEnv): The RL training environment instance.
        command_name (str): Name of the command term containing target position.
        threshold (float): Maximum distance (in meters) to consider target as reached.
        max_reward (float, optional): Base reward value when target is reached. Defaults to 1.0.
        time_bonus (bool, optional): Whether to scale reward by remaining timesteps. Defaults to True.

    Returns:
        torch.Tensor: A tensor of rewards with shape (num_envs,) containing:
            - max_reward * time_scale if target reached (distance < threshold)
            - 0 otherwise

    Notes:
        - Target position is assumed to be in robot frame coordinates (relative position)
        - Time scaling factor: (remaining_steps / max_episode_length)
        - For multiple environments, computes rewards independently per environment
        - Typical usage: Terminal reward component in sparse reward scenarios

    Example:
        >>> reward = reached_target(env, "target_position", 0.5)
        >>> # If reached at 75% of episode duration, reward = 1.0 * 0.25 = 0.25
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


def action_penalty_near_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, critical_dist: float = 1.5, sigmoid_coeff: float = 5.0, critical_sigmoid_dist_for_action: float = 0.08, max_action_penalty: float = 0.5, power: float = 2.0) -> torch.Tensor:
    """Computes penalty for large actions when the agent is near obstacles.
    
    The penalty increases when:
    1. The agent is close to obstacles (within critical_dist)
    2. The agent takes large actions (higher magnitude actions are penalized more)
    
    The penalty is computed as a product of three factors:
    - Base penalty (max_action_penalty)
    - Action magnitude (L2 norm raised to given power)
    - Proximity factor (1 - normalized distance when close to obstacles)

    Args:
        env (ManagerBasedRLEnv): The RL training environment instance.
        sensor_cfg (SceneEntityCfg): Configuration for the obstacle detection sensor.
        critical_dist (float, optional): Distance threshold (in meters) where penalty becomes active. 
            Beyond this distance, penalty is zero. Defaults to 1.5.
        sigmoid_coeff (float, optional): Coefficient controlling smoothness of distance transition.
            Higher values make the transition sharper. Defaults to 5.0.
        critical_sigmoid_dist_for_action (float, optional): Normalized distance threshold [0-1] 
            where action penalty reaches maximum. Defaults to 0.08.
        max_action_penalty (float, optional): Maximum penalty value when conditions are worst.
            Defaults to 0.5.
        power (float, optional): Exponent for action magnitude scaling. Higher values penalize
            large actions more aggressively. Defaults to 2.0 (quadratic).

    Returns:
        torch.Tensor: Negative reward (penalty) tensor with shape (num_envs,), where:
            - 0 when not near obstacles (distance > critical_dist)
            - max_penalty * (action_norm^power) * proximity_factor when near obstacles
            
    Notes:
        - Uses sigmoid function to smoothly transition penalty activation near critical_dist
        - The penalty grows as the agent gets closer to obstacles (lower distance)
        - The penalty scales with the magnitude of the action (higher actions = higher penalty)
        - For multiple environments, computes penalties independently per environment
        
    Example:
        >>> penalty = action_penalty_near_obstacles(env, "lidar", critical_dist=1.0)
        >>> # If very close to obstacle with large action, penalty might be 0.5
        >>> # If far from obstacles, penalty will be 0
    """
    sensor: RayCasterCfg = env.scene.sensors[sensor_cfg.name]

    differences = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1).expand(-1, sensor.data.ray_hits_w.size(1), -1)
    norm_differences = torch.norm(differences, p=2, dim=2)

    clipped_distances = torch.where(
        torch.isinf(norm_differences),
        sensor.cfg.max_distance,
        torch.clamp(norm_differences, max=sensor.cfg.max_distance),
    )

    sigmoid_distances = 1 / (1 + torch.exp(-sigmoid_coeff * (clipped_distances - critical_dist)))
    min_sigmoid_distance = sigmoid_distances.min(dim=1).values

    action = env.action_manager.action

    reward = torch.where(
        min_sigmoid_distance < critical_sigmoid_dist_for_action,
        max_action_penalty
        * (torch.norm(action, p=2, dim=1) ** power)
        * (1 - min_sigmoid_distance / critical_sigmoid_dist_for_action),
        0.0,
    )

    return reward


def penalty_for_sideways_movement(env: ManagerBasedRLEnv, max_penalty: float = 0.5, y_threshold: float = 0.3, power: float = 2.0) -> torch.Tensor:
    """Computes a penalty for sideways movement (Y-axis actions) to encourage forward motion.

    The penalty is zero when Y-axis action magnitude is below threshold, and increases
    smoothly when exceeding the threshold. The penalty follows a power-law curve.

    Args:
        env (ManagerBasedRLEnv): The RL training environment instance.
        max_penalty (float, optional): Maximum penalty value when Y-action is 1.0. 
            Defaults to 0.5.
        y_threshold (float, optional): Action threshold [0-1] below which no penalty is applied.
            Defaults to 0.3.
        power (float, optional): Exponent controlling penalty curve shape. Higher values make
            the penalty increase more sharply. Defaults to 2.0 (quadratic).
        clip_action (bool, optional): Whether to clip input actions to [0,1] range before
            processing. Defaults to True.

    Returns:
        torch.Tensor: Negative reward (penalty) tensor with shape (num_envs,), where:
            - 0 when |Y-action| ≤ y_threshold
            - max_penalty * normalized_excess^power when |Y-action| > y_threshold

    Notes:
        - Normalized excess = (|Y-action| - y_threshold)/(1 - y_threshold)
        - Designed to encourage forward/backward movement (X-axis) over sideways (Y-axis)
        - For multiple environments, computes penalties independently per environment
        - Typical use case: Quadruped locomotion where sideways movement is inefficient

    Example:
        >>> penalty = penalty_for_sideways_movement(env)
        >>> # With y_threshold=0.3 and Y-action=0.5:
        >>> # penalty = 0.5 * ((0.5-0.3)/0.7)^2 ≈ 0.0408
    """
    action = env.action_manager.action

    y_action = action[:, 1].abs()

    reward = torch.where(
        y_action > y_threshold, max_penalty * ((y_action - y_threshold) / (1 - y_threshold)) ** power, 0.0
    )

    return reward
