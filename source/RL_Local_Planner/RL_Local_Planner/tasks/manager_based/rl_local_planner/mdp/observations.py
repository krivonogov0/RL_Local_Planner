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
        rr_visualizers.circle_scanner_visualizer(frame_name="clipped", distances=clipped_distances)
        rr_visualizers.circle_scanner_visualizer(frame_name="sigmoid", distances=result)

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
    use_rerun: bool = False,
) -> torch.Tensor:
    """Generates a binary obstacle mask from a top-down depth camera.

    This function:
    1. Accesses the depth output of a top-down camera configured in `sensor_cfg`.
    2. Adjusts the camera's height to a fixed z-offset (ignoring robot height variations).
    3. Creates a binary mask where:
       - 1.0 indicates obstacles (depth < threshold - 0.05)
       - 0.0 indicates free space (depth >= threshold - 0.05)

    Primarily used for privileged information in simulation-based RL training.

    Args:
        env (ManagerBasedRLEnv): The RL environment containing the scene and robot.
        sensor_cfg (SceneEntityCfg): Configuration for the camera sensor. Must include:
            - `name`: Sensor name in `env.scene.sensors`.
            - `data_types`: Must contain `"depth"` (configured in `TiledCameraCfg`).
        use_rerun (bool, optional): If True, visualizes both raw depth and binary mask
                                  using Rerun. Defaults to False.

    Returns:
        torch.Tensor: Binary obstacle mask tensor of shape
                    `(batch_size, height, width, channels)` where:
                    - 1.0 = obstacle (too close to camera)
                    - 0.0 = free space
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]  # type: ignore

    robot_positions = env.scene.articulations["robot"].data.root_pos_w
    fixed_height = sensor.cfg.offset.pos[2]

    robot_positions_fixed_z = robot_positions.clone()
    robot_positions_fixed_z[:, 2] = fixed_height

    sensor.set_world_poses(robot_positions_fixed_z)

    depth = sensor.data.output["depth"]  # make sure to set `data_types=["depth"]` in TiledCameraCfg!

    threshold = sensor.cfg.offset.pos[2]
    binary_mask = (depth < threshold - 0.05).float()

    if use_rerun:
        rr_visualizers.depth_top_view_visualizer(frame_name="raw", depth_image=depth)
        rr_visualizers.depth_top_view_visualizer(frame_name="binary_mask", depth_image=binary_mask)

    return binary_mask


def top_view_obstacle(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    use_rerun: bool = False,
) -> torch.Tensor:
    """Generates a top-view obstacle binary mask from semantic segmentation data.

    The camera is positioned at a fixed height above each robot. The mask identifies
    non-black pixels (where black represents free space in semantic segmentation).

    Args:
        env: RL environment manager.
        sensor_cfg: Configuration for the tiled camera sensor.
        use_rerun: If True, visualizes the mask using Rerun. Defaults to False.

    Returns:
        torch.Tensor: Semantic segmentation data tensor (N, H, W, 1) containing depth values.
    """
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]  # type: ignore

    robot_positions = env.scene.articulations["robot"].data.root_pos_w
    fixed_height = sensor.cfg.offset.pos[2]

    robot_positions_fixed_z = robot_positions.clone()
    robot_positions_fixed_z[:, 2] = fixed_height

    sensor.set_world_poses(robot_positions_fixed_z)

    sematic_data = sensor.data.output[
        "semantic_segmentation"
    ]  # make sure to set `data_types=["semantic_segmentation"]` in TiledCameraCfg!
    r, g, b, a = sematic_data.unbind(dim=-1)

    is_black = (r == 0) & (g == 0) & (b == 0) & (a == 255)

    binary_mask = ~is_black
    binary_mask = binary_mask.float().unsqueeze(-1)

    if use_rerun:
        rr_visualizers.depth_top_view_visualizer(frame_name="obstacle_mask", depth_image=binary_mask)

    return binary_mask


def top_view_robot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    use_rerun: bool = False,
) -> torch.Tensor:
    """
    Generates a binary top-view semantic mask indicating robot positions and orientations.

    This optimized implementation creates rectangular markers for each robot, rotated according
    to their orientation quaternions. The rectangles are centered in the image with their
    longer side indicating the robot's forward direction.

    Args:
        use_rerun (bool, optional): Whether to visualize using Rerun. Defaults to False.

    Returns:
        torch.Tensor: Binary mask tensor of shape (N, width, height, 1) where:
                     - N: Number of robots
                     - width: Sensor image width in pixels
                     - height: Sensor image height in pixels
                     - Last dimension contains binary values (1=robot, 0=background)

    Features:
        - Automatic size adaptation based on sensor resolution
        - GPU-accelerated computations
        - Correct handling of robot orientation via quaternion conversion
        - Optimized vectorized operations for performance

    Note:
        The rectangle dimensions are automatically scaled for Unitree B1 as:
        - Width: 1/14 of image width
        - Height: 1/7 of image width
        This maintains consistent proportions across different resolutions.
    """

    robot_quats = env.scene.articulations["robot"].data.root_quat_w
    num_robots = robot_quats.shape[0]

    width = env.scene.sensors[sensor_cfg.name].cfg.width
    height = env.scene.sensors[sensor_cfg.name].cfg.height

    binary_mask = torch.zeros((num_robots, width, height, 1), device="cuda:0", dtype=torch.uint8)

    half_w, half_h = width / 14, width / 7
    center = width / 2

    y, x = torch.meshgrid(torch.arange(width, device="cuda:0"), torch.arange(height, device="cuda:0"), indexing="ij")
    xy_rel = torch.stack([x - center, y - center], dim=-1).float()

    for i in range(num_robots):
        q = robot_quats[i]
        yaw = 2 * torch.atan2(q[3], q[0])

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        rot_matrix = torch.tensor([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], device="cuda:0")

        xy_rot = torch.einsum("ij,hwj->hwi", rot_matrix, xy_rel)

        mask = (torch.abs(xy_rot[..., 0]) <= half_w) & (torch.abs(xy_rot[..., 1]) <= half_h)

        binary_mask[i][mask] = 1

    if use_rerun:
        rr_visualizers.depth_top_view_visualizer(
            frame_name="robot_mask",
            depth_image=binary_mask,
        )

    return binary_mask


def top_view_semantic(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    use_rerun: bool = False,
) -> torch.Tensor:
    """
    Merge two masks.
    """
    binary_mask_robot = top_view_robot(env, sensor_cfg, use_rerun)
    binary_mask_obstacle = top_view_obstacle(env, sensor_cfg, use_rerun)

    binary_mask = torch.clamp(binary_mask_robot + binary_mask_obstacle, 0, 1)

    if use_rerun:
        rr_visualizers.depth_top_view_visualizer(
            frame_name="obstacle_robot_mask",
            depth_image=binary_mask,
        )

    return binary_mask
