import math

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.navigation.mdp as mdp
import RL_Local_Planner.tasks.manager_based.rl_local_planner.mdp as custom_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c.flat_env_cfg import (
    AnymalCFlatEnvCfg,
)
from RL_Local_Planner.tasks.manager_based.rl_local_planner.terrain.config.indoor_nadigation.indoor_nadigation_cfg import (
    INDOOR_NAVIGATION_CFG,
    INDOOR_NAVIGATION_PLAY_CFG,
)

USE_RERUN = True

SUCCESS_DISTANCE = 0.5

LOW_LEVEL_ENV_CFG = AnymalCFlatEnvCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        pose_command = ObsTerm(func=custom_mdp.generated_commands_normalized, params={"command_name": "pose_command"})
        circle_scanner = ObsTerm(
            func=custom_mdp.circle_scanner_observation,
            params={"sensor_cfg": SceneEntityCfg("circle_scanner"), "use_rerun": USE_RERUN, "critical_dist": 1.5 , "sigmoid_coeff": 5.0},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reached_target = RewTerm(  # type: ignore
        func=custom_mdp.reached_target,
        weight=50.0,
        params={"command_name": "pose_command", "threshold": SUCCESS_DISTANCE},
    )
    action_near_obstacles_penalty = RewTerm(  # type: ignore
        func=custom_mdp.action_penalty_near_obstacles,
        weight=-0.05,
        params={"sensor_cfg": SceneEntityCfg("circle_scanner")},
    )
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    y_action_penalty = RewTerm(
        func=custom_mdp.penalty_for_sideways_movement,
        weight=-0.2,
        params={},
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.TerrainBasedPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.TerrainBasedPose2dCommandCfg.Ranges(heading=(-math.pi, math.pi)),
        goal_pose_visualizer_cfg=CUBOID_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_goal"),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    is_success = DoneTerm(
        func=custom_mdp.is_success,
        params={"command_name": "pose_command", "threshold": SUCCESS_DISTANCE},
    )


@configclass
class RlLocalPlannerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # environment settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # mdp settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        self.scene.num_envs = 5
        self.scene.env_spacing = 5

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=INDOOR_NAVIGATION_CFG,
            max_init_terrain_level=INDOOR_NAVIGATION_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        self.scene.circle_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
            attach_yaw_only=False,
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1, vertical_fov_range=(-1.0, 1.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
            ),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
            max_distance=10.0,
        )

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = (
                self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
            )
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RlLocalPlannerEnvPLAYCfg(RlLocalPlannerEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=INDOOR_NAVIGATION_PLAY_CFG,
            max_init_terrain_level=INDOOR_NAVIGATION_PLAY_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        self.events.benchmark = EventTerm(func=custom_mdp.benchmark, mode="reset")
