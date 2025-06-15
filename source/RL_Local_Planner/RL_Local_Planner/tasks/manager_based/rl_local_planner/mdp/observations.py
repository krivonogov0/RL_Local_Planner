import RL_Local_Planner.tasks.manager_based.rl_local_planner.visualizers as rr_visualizers
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg


def circle_scanner_observation(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, use_rerun: bool = False
) -> torch.Tensor:
    """Computes distance observations from a circular scanning sensor (ray caster).

    This function processes raycast hit data from a circular scanner sensor to compute
    the distances to detected objects. The distances are clamped to the sensor's maximum
    range, and infinite values (indicating no hit) are replaced with the maximum range.
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

    result = 1 / (1 + torch.exp(-sigmoid_coeff * (clipped_distances - critical_dist)))

    if use_rerun:
        rr_visualizers.circle_scanner_visualizer(distances=result)

    return result


def generated_commands_normalized(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The normalized generated command from command term in the command manager with the given name."""
    # normalization coefficients
    max_distance = 5.0
    max_angle = torch.pi

    command = env.command_manager.get_command(command_name)  # [x, y, z, angle]

    xy_angle_command = torch.cat([command[:, :2], command[:, 3:4]], dim=1)
    xy_angle_command[:, :2] = xy_angle_command[:, :2] / max_distance
    xy_angle_command[:, 2] = xy_angle_command[:, 2] / max_angle

    return xy_angle_command
