from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

PATH_TO_PARENT_FOLDER = Path(__file__).parents[6]

JETBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{PATH_TO_PARENT_FOLDER}/assets/robots/jetbot/jetbot.usd",
        activate_contact_sensors=True,
    ),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)


class TwoWheeledRobotActionTerm(ActionTerm):
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: TwoWheeledRobotActionTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._vel_omega_command = torch.zeros(self.num_envs, 2, device=self.device)
        self._vel_left_right = torch.zeros(self.num_envs, 2, device=self.device)

        self.base_line = cfg.base_line
        self.radius_wheel = cfg.radius_wheel
        self.scale = cfg.scale
        self.joint_ids = cfg.joint_ids

    @property
    def action_dim(self) -> int:
        return self._vel_omega_command.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._vel_omega_command

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._vel_left_right

    def process_actions(self, actions: torch.Tensor):
        self._vel_omega_command[:] = actions

        v, w = actions[:, :1], actions[:, 1:]

        v_left = (v - 0.5 * w * self.base_line) / self.radius_wheel
        v_right = (v + 0.5 * w * self.base_line) / self.radius_wheel

        self._vel_left_right = torch.cat([v_left, v_right], dim=1) * self.scale

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._vel_left_right, joint_ids=self.joint_ids)
        self._asset.write_data_to_sim()


@configclass
class TwoWheeledRobotActionTermCfg(ActionTermCfg):
    """Configuration for the two wheeled robots action term."""

    class_type: type = TwoWheeledRobotActionTerm  # type: ignore
    """The class corresponding to the action term."""

    base_line: float = MISSING  # type: ignore
    """Distance between left and right wheels"""

    radius_wheel: float = MISSING  # type: ignore
    """Radius of wheel"""

    scale: float = 1.0  # type: ignore
    """Scale to v_left, v_right"""

    joint_ids: list[int] = MISSING  # type: ignore
    """ids of wheel joints"""


@configclass
class JetbotActionTermCfg(TwoWheeledRobotActionTermCfg):
    base_line: float = 0.9  # type: ignore
    """Distance between left and right wheels"""

    radius_wheel: float = 0.3  # type: ignore
    """Radius of wheel"""

    joint_ids: list[int] = [0, 1]  # type: ignore
    """ids of wheel joints"""
