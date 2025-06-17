import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def is_success(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Determines whether the agent has successfully reached the target position.

    Computes the Euclidean distance to the target and returns a boolean tensor indicating
    whether the agent is within the specified threshold distance.

    Args:
        env (ManagerBasedRLEnv): The RL training environment instance.
        command_name (str): Name of the command term containing target position.
        threshold (float): Success threshold distance in meters.
        use_3d_distance (bool, optional): If True, computes 3D distance (x,y,z).
            If False, computes 2D distance (x,y). Defaults to False.

    Returns:
        torch.Tensor: Boolean tensor of shape (num_envs,) where:
            - True indicates success (distance < threshold)
            - False indicates failure (distance ≥ threshold)

    Notes:
        - Target position is assumed to be in the robot's local frame (relative position)
        - For multiple environments, evaluates success independently per environment
        - Typical usage: Terminal condition or success metric in navigation tasks

    Example:
        >>> success = is_success(env, "target_position", 0.5)
        >>> # Returns tensor([True, False]) if first env reached target, second didn't
    """
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance < threshold, True, False)
