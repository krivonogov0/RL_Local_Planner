import isaaclab.sim as sim_utils
import RL_Local_Planner.tasks.manager_based.rl_local_planner.mdp as custom_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from RL_Local_Planner.tasks.manager_based.rl_local_planner.rl_local_planner_env_cfg import (
    RlLocalPlannerEnvCfg,
)
from RL_Local_Planner.tasks.manager_based.rl_local_planner.terrain.config.indoor_nadigation.indoor_nadigation_cfg import (
    INDOOR_NAVIGATION_PLAY_CFG,
)

USE_RERUN = False


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
            params={
                "sensor_cfg": SceneEntityCfg("circle_scanner"),
                "use_rerun": USE_RERUN,
                "critical_dist": 1.5,
            },
        )
        top_view = ObsTerm(
            func=custom_mdp.top_view_depth,
            params={
                "sensor_cfg": SceneEntityCfg(name="tiled_camera"),
                "use_rerun": USE_RERUN,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RlLocalPlannerPrivilegedInfoEnvCfg(RlLocalPlannerEnvCfg):
    """Configuration for the navigation environment."""

    observations: ObservationsCfg = ObservationsCfg()  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.tiled_camera = TiledCameraCfg(  # make sure to add the --enable_cameras argument!
            prim_path="{ENV_REGEX_NS}/Robot/tiled_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 5.0), rot=(0.707, 0.0, 0.707, 0.0), convention="world"),
            data_types=["depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            ),
            width=64,
            height=64,
        )


@configclass
class RlLocalPlannerPrivilegedInfoEnvPLAYCfg(RlLocalPlannerPrivilegedInfoEnvCfg):
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

        self.events.benchmark = EventTerm(func=custom_mdp.benchmark, mode="reset")  # type: ignore
