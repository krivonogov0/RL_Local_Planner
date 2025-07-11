import RL_Local_Planner.tasks.manager_based.rl_local_planner.visualizers as rr_visualizers
import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, TiledCamera


def circle_scanner_observation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    use_rerun: bool = False,
    critical_dist: float = 1.5,
    sigmoid_coeff: float = 5.0,
) -> torch.Tensor:
    """Computes distance observations from a circular scanning sensor (ray caster) and applies
    a sigmoid transformation to the distances.

    The function processes raycast hit data from a circular scanner sensor to compute:
    1. Euclidean distances to detected objects
    2. Clamps distances to the sensor's maximum range (replaces infinity with max_distance)
    3. Applies a sigmoid transformation to make the output smoother and bounded

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance.
        sensor_cfg (SceneEntityCfg): Configuration of the sensor containing its name and parameters.
        use_rerun (bool, optional): If True, visualizes the sensor data using Rerun. Defaults to False.
        critical_dist (float, optional): The critical distance threshold (in meters) where the sigmoid
            output should be 0.5. Distances below this value will produce outputs > 0.5.
            Represents the "safety boundary" for obstacle avoidance. Defaults to 1.5.
        sigmoid_coeff (float, optional): Coefficient controlling the steepness of the sigmoid transition.
            Higher values make the transition sharper (more binary), lower values make it smoother.
            Typical range: 3-10. Defaults to 5.0.

    Returns:
        torch.Tensor: A tensor containing transformed distance observations with shape (N_rays,).
            Values range between 0 (very close) and 1 (far or no hit).

    Notes:
        - The sigmoid transformation formula: 1 / (1 + exp(-sigmoid_coeff * (distance - critical_dist)))
        - Output interpretation:
            * 0.5 means exactly at critical distance
            * >0.5 means closer than critical distance (potential danger)
            * <0.5 means farther than critical distance (safe zone)
        - Infinite distances (no hits) are converted to sensor.cfg.max_distance before transformation
    """
    sensor: RayCasterCfg = env.scene.sensors[sensor_cfg.name]

    differences = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1).expand(-1, sensor.data.ray_hits_w.size(1), -1)
    norm_differences = torch.norm(differences, p=2, dim=2)

    clipped_distances = torch.where(
        torch.isinf(norm_differences),
        sensor.cfg.max_distance,
        torch.clamp(norm_differences, max=sensor.cfg.max_distance),
    )

    result = 1 / (1 + torch.exp(-sigmoid_coeff * (clipped_distances - critical_dist)))

    if use_rerun:
        rr_visualizers.circle_scanner_visualizer(distances=result)

    return result


def generated_commands_normalized(
    env: ManagerBasedRLEnv, command_name: str, max_distance: float = 5.0, max_angle: float = 3.14
) -> torch.Tensor:
    """Normalizes generated commands from the command manager to a standardized range.

    Processes 3D position + angle commands (x,y,z,angle) by:
    1. Extracting the command vector from the command manager
    2. Normalizing xy-position components by max_distance
    3. Normalizing angle component by max_angle
    4. Dropping the z-coordinate (not normalized)

    Args:
        env (ManagerBasedRLEnv): The reinforcement learning environment instance containing
            the command manager.
        command_name (str): Name of the command term to retrieve from the command manager.
        max_distance (float, optional): Maximum expected distance (in meters) for normalization.
            Used to scale xy-position components to [0, 1] range. Defaults to 5.0.
        max_angle (float, optional): Maximum expected angle (in radians) for normalization.
            Used to scale angle component to [0, 1] range. Defaults to π (≈3.14) for [-π,π] range.

    Returns:
        torch.Tensor: Normalized command vector with shape (N, 3) containing:
            - [:, 0]: x-position normalized by max_distance (range ~[-1, 1])
            - [:, 1]: y-position normalized by max_distance (range ~[-1, 1])
            - [:, 2]: angle normalized by max_angle (range ~[-1, 1])

    Notes:
        - Input command format: [x, y, z, angle] where angle is in radians
        - Z-coordinate is intentionally dropped from output
        - Normalization is linear scaling: normalized_value = raw_value / max_value
        - Output ranges may exceed [-1, 1] if input values exceed max_distance/max_angle
        - Typical usage: Normalizing commands for neural network input or reward calculation
    """
    command = env.command_manager.get_command(command_name)  # [x, y, z, angle]

    xy_angle_command = torch.cat([command[:, :2], command[:, 3:4]], dim=1)  # [x, y, angle]
    xy_angle_command[:, :2] = xy_angle_command[:, :2] / max_distance
    xy_angle_command[:, 2] = xy_angle_command[:, 2] / max_angle

    return xy_angle_command


def top_view_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Retrieves and processes the depth map from a top-down camera sensor.

    This function:
    1. Accesses the depth output of a top-down camera configured in `sensor_cfg`.
    2. Adjusts the camera's height to a fixed z-offset (ignoring robot height variations).
    3. Applies a sigmoid scaling to normalize depth values to the range [0.1, 1.0].

    Primarily used for privileged information in simulation-based RL training.

    Args:
        env (ManagerBasedRLEnv): The RL environment containing the scene and robot.
        sensor_cfg (SceneEntityCfg): Configuration for the camera sensor. Must include:
            - `name`: Sensor name in `env.scene.sensors`.
            - `data_types`: Must contain `"depth"` (configured in `TiledCameraCfg`).

    Returns:
        torch.Tensor: Processed depth values as a tensor of shape
            `(batch_size, height, width, channels)`. Values scaled to [0.1, 1.0].
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]  # type: ignore

    robot_positions = env.scene.articulations["robot"].data.root_pos_w
    fixed_height = sensor.cfg.offset.pos[2]

    robot_positions_fixed_z = robot_positions.clone()
    robot_positions_fixed_z[:, 2] = fixed_height

    sensor.set_world_poses(robot_positions_fixed_z)

    depth = sensor.data.output["depth"]  # make sure to set `data_types=["depth"]` in TiledCameraCfg!
    scaled_sigmoid = 0.1 + 0.9 * torch.sigmoid(depth)

    return scaled_sigmoid
