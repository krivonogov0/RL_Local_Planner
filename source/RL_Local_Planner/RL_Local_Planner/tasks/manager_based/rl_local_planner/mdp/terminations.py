import torch
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


def is_success(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """
    Determine whether the target has been reached.

    This function checks if the robot is within a certain threshold distance from the target.
    """
    target = env.command_manager.get_command(command_name)
    target_position = target[:, :2]

    # Calculating the distance and determining if the target is reached
    distance = torch.norm(target_position, p=2, dim=-1)

    return torch.where(distance < threshold, True, False)
