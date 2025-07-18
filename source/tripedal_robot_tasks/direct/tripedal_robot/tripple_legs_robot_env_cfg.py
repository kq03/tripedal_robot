# Copyright (c) 2023-2025, 三足机器人项目开发者.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# 预定义配置
##
from tripedal_robot_assets.tripedal_robot_assets.tlr7_config import TLR6_CFG, TLR6_JUMP_CFG  # isort: skip

@configclass
class EventCfg:
    """随机化配置"""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base_Link"),
            "mass_distribution_params": (-0.5, 0.5),  # 比ANYmal更小的质量变化
            "operation": "add",
        },
    )


@configclass
class TLR6FlatEnvCfg(DirectRLEnvCfg):
    # 环境
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 6  # 3条腿，每条腿2个关节
    observation_space = 45  # 修正观测空间大小：4+3+3+6+6+1+1+6+6+6+3=45
    state_space = 0

    # 仿真
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 场景
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)

    # 事件
    events: EventCfg = EventCfg()

    # 机器人
    robot: ArticulationCfg = TLR6_CFG
    robot.prim_path = "/World/envs/env_.*/Robot"
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # 奖励比例 - 统一规范
    # 主要行为奖励
    lin_vel_reward_scale = 10.0        # 线速度跟踪奖励
    yaw_rate_reward_scale = 1.0        # 偏航角速度跟踪奖励
    
    # 稳定性相关惩罚
    z_vel_reward_scale = -5.0          # z方向速度惩罚
    ang_vel_reward_scale = -0.05       # 角速度惩罚(x,y)
    flat_orientation_reward_scale = -10.0 # 平坦姿态奖励（保持身体水平）
    
    # 效率相关惩罚
    joint_torque_reward_scale = -2.5e-5 # 关节扭矩惩罚
    joint_accel_reward_scale = -2.5e-7 # 关节加速度惩罚
    action_rate_reward_scale = -0.01   # 动作变化率惩罚
    
    # 步态相关奖励
    feet_air_time_reward_scale = 5.0   # 足部空气时间奖励
    undesired_contact_reward_scale = -15.0 # 不期望的接触惩罚（大腿与地面接触）
    
    # 对称性和协调性奖励（步态协调）
    symmetry_reward_scale = 0.5        # 对称性奖励系数（新增）
    phase_sync_reward_scale = 0.5      # 相位同步奖励系数（新增）


@configclass
class TLR6RoughEnvCfg(TLR6FlatEnvCfg):
    # 环境
    observation_space = 175  # 粗糙地形需要更多观察空间（减少6个default_joint_pos的维度）

    # 地形配置
    # 注意：这里需要定义ROUGH_TERRAINS_CFG，或者使用其他地形生成器
    # 由于我们没有完整的地形生成器定义，这里先注释掉相关代码
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # 暂时使用平面地形
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # 为感知运动添加高度扫描器
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 10.0)),  # 降低高度
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.8)),  # 调整适合三足机器人大小
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # 奖励比例（从平地配置覆盖）
    flat_orientation_reward_scale = 0.0


@configclass
class TLR6JumpEnvCfg(TLR6FlatEnvCfg):
    """三足机器人TLR6跳跃环境配置"""
    
    # 环境
    episode_length_s = 8.0  # 缩短回合时间，专注于跳跃动作
    action_scale = 0.5  # 动作幅度
    decimation = 4  # 明确定义decimation参数
    
    # 仿真
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,  # 更高的模拟频率，提供更精确的物理模拟
        render_interval=4,  # 直接使用数值而不是引用decimation
        # disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,  # 增加静摩擦力
            dynamic_friction=1.0,
            restitution=0.1,  # 增加一些反弹性
        ),
    )
    
    # 场景 - 减少环境数量以提高训练稳定性
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)
    
    # 机器人 - 使用为跳跃优化的配置
    robot: ArticulationCfg = TLR6_JUMP_CFG
    robot.prim_path = "/World/envs/env_.*/Robot"
    
    # 简化后的奖励系数 - 只保留必要的奖励项
    ang_vel_reward_scale = -1.5   # 角速度惩罚（防止倾倒）
    joint_torque_reward_scale = -5.0e-5  # 关节扭矩惩罚
    joint_accel_reward_scale = -5.0e-7   # 关节加速度惩罚
    flat_orientation_reward_scale = -5.0  # 平坦姿态奖励（防止倾倒）
    
    # 重要的跳跃奖励
    jump_height_reward_scale = 20.0  # 增加高度奖励
    height_change_reward_scale = 12.0  # 增加快速向上高度变化奖励（代替z方向速度奖励）
    
    # 对称性奖励
    symmetry_reward_scale = 2.0   # 对称性奖励系数
    phase_sync_reward_scale = 2.0  # 相位同步奖励系数
    
    # 新增：静止状态惩罚
    static_penalty_scale = 6.0    # 静止状态惩罚系数（惩罚长时间静止不动）
    
    # 新增：不期望接触惩罚 - 防止大腿与地面接触
    undesired_contact_reward_scale = -10.0  # 较大的惩罚系数，防止大腿电机触地
    
    # 以下奖励已不再使用，但保留配置以避免代码错误
    lin_vel_reward_scale = 0      # 不再奖励线速度
    yaw_rate_reward_scale = 0.0   # 不再奖励偏航角速度
    action_rate_reward_scale = 0  # 不再惩罚动作变化率
    feet_air_time_reward_scale = 0.0  # 不再使用足部空气时间奖励
    jump_landing_reward_scale = 0  # 不再使用着陆奖励
    air_time_reward_scale = 0     # 不再使用空中时间奖励
    z_accel_reward_scale = 0      # 不再使用z方向加速度奖励
    preparation_reward_scale = 0
    burst_reward_scale = 0
    air_pose_reward_scale = 0
    landing_buffer_reward_scale = 0
    continuity_reward_scale = 0
  