# Copyright (c) 2023, TLR6 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TLR6三足机器人配置文件

可用配置:
* TLR6_CFG: 基本的TLR6三足机器人配置
* TLR6_JUMP_CFG: 为跳跃行为优化的TLR6配置
"""

# Note: These imports are from Isaac Sim/Lab and should be available when Isaac Sim is installed
# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg
import os

# 获取当前Python文件的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录
current_dir = os.path.dirname(current_file)
# 获取项目根目录（假设当前文件在source/isaaclab_assets/isaaclab_assets/robots/下）
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
# 构建USD文件的绝对路径（但基于项目根目录，所以具有可移植性）
TLR6_USD_PATH = os.path.join(project_root, "TLR7_0629new/urdf/TLR7_0629/TLR7_0629.usd")
# TLR6_USD_PATH = "D:/biyesheji/TRL7/urdf/TRL7/TRL7.usd"

# 基本配置
TLR6_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TLR6_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            "L1_thigh_joint": -0.52,
            "L2_thigh_joint": -0.52,
            "L3_thigh_joint": -0.52,
            "L1_shin_joint": 1.75,
            "L2_shin_joint": 1.75,
            "L3_shin_joint": 1.75,
        },
    ),
    actuators={
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=["L[1-3]_thigh_joint"],
            effort_limit=3.0,
            velocity_limit=15.0,
            stiffness=60.0,
            damping=3.0,
        ),
        "shins": ImplicitActuatorCfg(
            joint_names_expr=["L[1-3]_shin_joint"],
            effort_limit=3.49,
            velocity_limit=15.0,
            stiffness=60.0,
            damping=3.0,
        ),
    },
)
"""TLR6三足机器人的基本配置。
使用适中的刚度和阻尼值，适合一般的运动控制。
三条腿分别配置，便于单独控制每条腿。
"""

# 跳跃配置
TLR6_JUMP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TLR6_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8,  # 增加求解器迭代次数，提高跳跃精度
            solver_velocity_iteration_count=2   # 添加速度迭代，改善动态性能
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            "L1_thigh_joint": 0.5,
            "L2_thigh_joint": 0.5,
            "L3_thigh_joint": 0.5,
            "L1_shin_joint": 2.08,
            "L2_shin_joint": 2.08,
            "L3_shin_joint": 2.08,
        },
    ),
    actuators={
        "thighs": ImplicitActuatorCfg(
            joint_names_expr=["L[1-3]_thigh_joint"],
            effort_limit=3.0,
            velocity_limit=15.0,
            stiffness=100.0,
            damping=5.0,
        ),
        "shins": ImplicitActuatorCfg(
            joint_names_expr=["L[1-3]_shin_joint"],
            effort_limit=3.49,
            velocity_limit=15.0,
            stiffness=100.0,
            damping=5.0,
        ),
    },
)
"""为跳跃研究优化的TLR6配置。
提高了力矩和速度限制，使用更高的刚度和适当的阻尼。
增加了求解器迭代次数，改善动态性能，特别是在跳跃过程中。
"""