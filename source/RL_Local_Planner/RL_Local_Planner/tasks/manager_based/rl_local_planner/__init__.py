# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Rl-Local-Planner-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_local_planner_env_cfg:RlLocalPlannerEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Rl-Local-Planner-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_local_planner_env_cfg:RlLocalPlannerEnvPLAYCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Rl-Local-Planner-Jetbot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_rl_local_planner_env_cfg:RlLocalPlannerJetbotEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:jetbot_skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Rl-Local-Planner-Jetbot-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jetbot_rl_local_planner_env_cfg:RlLocalPlannerJetbotEnvPLAYCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:jetbot_skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Rl-Local-Planner-Privileged-Info-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.privileged_info_rl_local_planner_env_cfg:RlLocalPlannerPrivilegedInfoEnvCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:privileged_info_skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-Rl-Local-Planner-Privileged-Info-PLAY-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.privileged_info_rl_local_planner_env_cfg:RlLocalPlannerPrivilegedInfoEnvPLAYCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:privileged_info_skrl_ppo_cfg.yaml",
    },
)
