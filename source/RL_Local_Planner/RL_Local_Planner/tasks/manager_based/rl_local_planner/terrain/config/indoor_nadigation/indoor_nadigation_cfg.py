import isaaclab.terrains as terrain_gen
import isaaclab.terrains.height_field as hf_gen
from isaaclab.terrains import FlatPatchSamplingCfg

INDOOR_NAVIGATION_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "medium_density": hf_gen.HfDiscreteObstaclesTerrainCfg(
            size=(4.0, 4.0),
            horizontal_scale=0.05,
            vertical_scale=0.1,
            num_obstacles=15,
            obstacle_width_range=(0.7, 1.5),
            obstacle_height_range=(1.0, 1.0),
            platform_width=2.0,
            border_width=0.5,
            obstacle_height_mode="fixed",
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=3,
                    x_range=(-3, 3),
                    y_range=(-3, 3),
                    z_range=(-0.01, 0.1),
                    patch_radius=1.0,
                    max_height_diff=0.01,
                ),
            },
        ),
        "less_density": hf_gen.HfDiscreteObstaclesTerrainCfg(
            size=(4.0, 4.0),
            horizontal_scale=0.05,
            vertical_scale=0.1,
            num_obstacles=3,
            obstacle_width_range=(1.5, 2.5),
            obstacle_height_range=(1.0, 1.0),
            platform_width=2.0,
            border_width=0.5,
            obstacle_height_mode="fixed",
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=3,
                    x_range=(-3, 3),
                    y_range=(-3, 3),
                    z_range=(-0.01, 0.1),
                    patch_radius=1.0,
                    max_height_diff=0.01,
                ),
            },
        ),
        # "more_density": hf_gen.HfDiscreteObstaclesTerrainCfg(
        #     size=(2.0, 2.0),
        #     horizontal_scale=0.05,
        #     vertical_scale=0.1,
        #     num_obstacles=50,
        #     obstacle_width_range=(0.5, 0.8),
        #     obstacle_height_range=(1.0, 1.0),
        #     platform_width=2.0,
        #     border_width=0.3,
        #     obstacle_height_mode="fixed",
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=5,
        #             x_range=(-2, 2),
        #             y_range=(-2, 2),
        #             z_range=(-0.1, 0.1),
        #             patch_radius=0.5,
        #             max_height_diff=0.05,
        #         ),
        #     },
        # ),
    },
)


INDOOR_NAVIGATION_PLAY_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=14,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "medium_density": hf_gen.HfDiscreteObstaclesTerrainCfg(
            size=(4.0, 4.0),
            horizontal_scale=0.05,
            vertical_scale=0.1,
            num_obstacles=10,
            obstacle_width_range=(0.8, 1.5),
            obstacle_height_range=(1.0, 1.0),
            platform_width=2.0,
            border_width=0.5,
            obstacle_height_mode="fixed",
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=3,
                    x_range=(-6, 6),
                    y_range=(-6, 6),
                    z_range=(-0.01, 0.1),
                    patch_radius=0.5,
                    max_height_diff=0.01,
                ),
            },
        ),
    },
)
