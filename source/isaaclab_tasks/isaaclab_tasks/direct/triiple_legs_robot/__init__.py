# Copyright (c) 2023-2025, 三足机器人项目开发者.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
三足机器人TLR6的强化学习环境。
"""

import gymnasium as gym
import os

from . import agents

##
# 注册Gym环境
##

gym.register(
    id="Isaac-Velocity-Flat-TLR6-Direct-v0",
    entry_point=f"{__name__}.tripple_legs_robot_env:TLR6Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripple_legs_robot_env_cfg:TLR6FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TLR6FlatPPORunnerCfg",
        # 关节可视化配置 - 仅保留实时显示功能
        "enable_joint_visualization": True,
        "max_datapoints": 1000
    },
)

# 注册跳跃环境
gym.register(
    id="Isaac-Jump-Flat-TLR6-Direct-v0",
    entry_point=f"{__name__}.tripple_legs_robot_env:TLR6JumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tripple_legs_robot_env_cfg:TLR6JumpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TLR6JumpPPORunnerCfg",
        # 关节可视化配置 - 仅保留实时显示功能
        "enable_joint_visualization": True,
        "max_datapoints": 1000
    },
)

