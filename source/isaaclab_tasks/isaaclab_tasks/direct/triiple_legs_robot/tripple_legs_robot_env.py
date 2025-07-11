# Copyright (c) 2023-2025, 三足机器人项目开发者.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import numpy as np
from typing import Dict, List, Optional
import os

# Note: These imports are from Isaac Sim/Lab and should be available when Isaac Sim is installed
# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sensors import ContactSensor, RayCaster
# from isaaclab.utils import math as math_utils
from .tripple_legs_robot_env_cfg import TLR6FlatEnvCfg, TLR6JumpEnvCfg, TLR6RoughEnvCfg


class TLR6Env(DirectRLEnv):
    cfg: TLR6FlatEnvCfg | TLR6RoughEnvCfg

    def __init__(self, cfg: TLR6FlatEnvCfg | TLR6RoughEnvCfg, render_mode: str | None = None, **kwargs):
        # 首先调用父类的初始化方法，确保scene等基本属性已设置
        super().__init__(cfg, render_mode, **kwargs)
        
        # 修改关节数据可视化设置，使用BFS顺序：所有大腿关节在前，所有小腿关节在后
        self._joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
        self._enable_joint_visualization = kwargs.get("enable_joint_visualization", False)
        self._max_datapoints = kwargs.get("max_datapoints", 1000)
        
        # 回合计数
        self._episode_counter = 0
        
        # 关节位置命令（偏离默认关节位置）
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y线性速度和偏航角速度命令
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 添加关节速度历史记录，用于计算加速度估计
        self._prev_joint_vel = torch.zeros(self.num_envs, self._robot.data.joint_vel.shape[1], device=self.device)
        
        # 存储默认关节位置，用于动作缩放 - 在实际机器人部署时应该由标定过程确定
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()
        
        # 模拟测距传感器 - 只提供高度值，不依赖于全局坐标
        # 在实际机器人中，这会被真实传感器的读数替代
        self._height_sensor_value = torch.zeros(self.num_envs, device=self.device)
        self._prev_height_sensor_value = torch.zeros(self.num_envs, device=self.device)
        self._estimated_z_vel = torch.zeros(self.num_envs, device=self.device)

        # 日志记录 - 确保所有奖励项都有对应的日志记录
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # 基本行为和稳定性奖励
                "lin_vel_reward",
                "yaw_rate_reward",
                "z_vel_reward",
                "ang_vel_reward",
                "flat_orientation_reward",
                
                # 效率相关惩罚
                "joint_torque_reward",
                "joint_accel_reward",
                "action_rate_reward",
                
                # 步态相关奖励
                "feet_air_time_reward",
                "undesired_contact_reward",
            ]
        }
        
        # 添加数据收集缓冲区，用于JointMonitorRunner
        self._joint_data_buffer = {
            "joint_pos": [],
            "joint_vel": [], 
            "joint_torque": []
        }
        # 设置最大数据点数量
        self._max_datapoints = 1000
        
        # 新增：线速度与命令速度数据缓冲区
        self._vel_data_buffer = {
            "root_lin_vel": [],  # 机器人基座线速度 (3,)
            "command_vel": []    # 命令线速度 (2,)
        }
        
        # 添加接触历史记录
        self._contact_history = []
        
        # 调用关节顺序调试函数
        self._debug_joint_order()

    def _debug_joint_order(self):
        """调试函数：验证关节排序的实际情况"""
        print("\n===== 关节顺序调试信息 =====")
        
        # 打印关节名称（如果可用）
        if hasattr(self._robot.data, "joint_names"):
            print("物理引擎解析的关节名称:", self._robot.data.joint_names)
        else:
            print("未找到joint_names属性")
            
        # 打印链接名称（如果可用）
        if hasattr(self._robot.data, "body_names"):
            print("物理引擎解析的链接名称:", self._robot.data.body_names)
        else:
            print("未找到body_names属性")
        
        # 尝试查找其他可能包含关节信息的属性
        for attr in dir(self._robot.data):
            if "joint" in attr.lower() or "dof" in attr.lower():
                try:
                    value = getattr(self._robot.data, attr)
                    if not callable(value):  # 排除方法
                        print(f"找到可能相关的属性: {attr}")
                except:
                    pass
        
        # 打印默认关节位置，用于比较
        print("默认关节位置:", self._robot.data.default_joint_pos[0])
        
        # 打印配置中的关节初始位置
        if hasattr(self.cfg.robot, "init_state") and hasattr(self.cfg.robot.init_state, "joint_pos"):
            print("配置中的关节初始位置:")
            for joint, pos in self.cfg.robot.init_state.joint_pos.items():
                print(f"  {joint}: {pos}")
        
        print("===== 调试信息结束 =====\n")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # 如果是粗糙地形，添加高度扫描器
        if isinstance(self.cfg, TLR6RoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
            
        # 设置地形
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # 克隆和复制环境
        self.scene.clone_environments(copy_from_source=False)
        
        # 添加灯光
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

      

    def _update_joint_visualization(self):
        """更新关节可视化数据 - 现在只收集数据，不进行实时可视化"""
        # 即使不启用实时可视化，也需要收集关节数据供JointMonitorRunner使用
        try:
            # 获取当前步骤
            current_step = self.episode_length_buf[0].item() if hasattr(self, "episode_length_buf") else 0
            # 只在关键点输出调试信息
            should_debug = current_step == 0 or current_step % 100 == 0
            
            # 从第一个环境中获取数据
            joint_pos = self._robot.data.joint_pos[0]
            joint_vel = self._robot.data.joint_vel[0]
            applied_torque = self._robot.data.applied_torque[0] if hasattr(self._robot.data, "applied_torque") else torch.zeros_like(joint_vel)
            
            # 存储数据到缓冲区
            if len(self._joint_data_buffer["joint_pos"]) >= self._max_datapoints:
                for key in self._joint_data_buffer:
                    self._joint_data_buffer[key].pop(0)
            
            self._joint_data_buffer["joint_pos"].append(joint_pos.detach().clone())
            self._joint_data_buffer["joint_vel"].append(joint_vel.detach().clone())
            self._joint_data_buffer["joint_torque"].append(applied_torque.detach().clone())

            # 新增：采集基座线速度和命令速度
            root_lin_vel = self._robot.data.root_lin_vel_b[0].detach().clone()  # (3,)
            command_vel = self._commands[0, :2].detach().clone()  # (2,)
            if len(self._vel_data_buffer["root_lin_vel"]) >= self._max_datapoints:
                for key in self._vel_data_buffer:
                    self._vel_data_buffer[key].pop(0)
            self._vel_data_buffer["root_lin_vel"].append(root_lin_vel)
            self._vel_data_buffer["command_vel"].append(command_vel)
            
        except Exception as e:
            print(f"[TLR6Env] 更新关节数据时出错: {e}")
            import traceback
            traceback.print_exc()

    def _pre_physics_step(self, actions: torch.Tensor):
        # 保存之前的高度传感器值，用于估计z速度
        self._prev_height_sensor_value = self._height_sensor_value.clone()
        # 保存之前的关节速度，用于计算加速度
        self._prev_joint_vel = self._robot.data.joint_vel.clone()
        # 保存之前的动作
        self._previous_actions.copy_(self._actions)
        # 更新当前动作
        self._actions = actions.clone()
        # 应用动作
        self._apply_action()

    def _post_physics_step(self):
        """物理模拟步骤后的处理"""
        # 更新关节数据可视化
        self._update_joint_visualization()

    def _apply_action(self):
        # 应用动作到机器人关节
        processed_actions = self._actions * self.cfg.action_scale + self._default_joint_pos
        self._robot.set_joint_position_target(processed_actions)
        
        # 更新传感器数据
        self._simulate_height_sensor()

    def _simulate_height_sensor(self):
        """在模拟环境中模拟高度传感器的数值
        
        实际机器人中，这个方法会被实际传感器读数替代
        这里只是为了在模拟环境中提供数据
        """
        # 在模拟中，我们使用全局坐标来模拟传感器
        # 这只是为了模拟目的，实际实现时会被替换
        if hasattr(self._robot.data, "root_pos_w"):
            self._height_sensor_value = self._robot.data.root_pos_w[:, 2].clone()
            
        # 计算估计的z速度 (高度变化/时间)
        self._estimated_z_vel = (self._height_sensor_value - self._prev_height_sensor_value) / self.cfg.sim.dt

    def _get_observations(self) -> dict:
        # 获取观察和状态
        obs_dict = {}
        # 基本信息
        obs_dict["robot_obs"] = self._get_robot_obs()
        
        # 如果使用粗糙地形，添加高度扫描信息
        if isinstance(self.cfg, TLR6RoughEnvCfg):
            # 直接使用接口属性，不调用方法
            if hasattr(self._height_scanner, "data"):
                # 以通用方式提取高度数据
                height_data = torch.zeros((self.num_envs, 1), device=self.device)
                for attr in ["hit_pos", "heights", "height_map"]:
                    if hasattr(self._height_scanner.data, attr):
                        data = getattr(self._height_scanner.data, attr)
                        if isinstance(data, torch.Tensor):
                            if data.dim() > 1 and data.size(0) == self.num_envs:
                                height_data = data
                                break
                obs_dict["height_scan"] = height_data
            else:
                # 如果传感器数据不存在，使用空值代替
                obs_dict["height_scan"] = torch.zeros((self.num_envs, 1), device=self.device)
            
        # 状态信息，若需要
        if hasattr(self.cfg, "state_space") and isinstance(self.cfg.state_space, int) and self.cfg.state_space > 0:
            obs_dict["states"] = self._get_state_obs()
        
        # 添加policy键，这是RSL-RL所需的
        obs_dict["policy"] = obs_dict["robot_obs"]
            
        return obs_dict

    def _get_robot_obs(self) -> torch.Tensor:
        # 获取机器人的观察
        robot_obs = []
        
        # 基座方向(4)
        robot_obs.append(self._robot.data.root_quat_w)
        # 基座线速度和角速度(6)
        robot_obs.append(self._robot.data.root_lin_vel_b)
        robot_obs.append(self._robot.data.root_ang_vel_b)
        # 关节位置和速度(12 = 6个关节 x 2)
        robot_obs.append(self._robot.data.joint_pos)
        robot_obs.append(self._robot.data.joint_vel)
        # 使用高度传感器测量的高度(1)
        robot_obs.append(self._height_sensor_value.unsqueeze(1))
        # 上一帧高度传感器值(1) - 用于在嵌入式系统中计算垂直速度
        robot_obs.append(self._prev_height_sensor_value.unsqueeze(1))
        # 上一帧关节速度(6) - 用于在嵌入式系统中计算关节加速度 
        robot_obs.append(self._prev_joint_vel)
        # 添加关节力矩观测(6) - 在实际机器人中对应电机电流
        robot_obs.append(self._robot.data.applied_torque)
        # 上一动作(6)
        robot_obs.append(self._previous_actions)
        # 命令(3)
        robot_obs.append(self._commands)
        
        return torch.cat(robot_obs, dim=-1)

    def _get_state_obs(self) -> torch.Tensor:
        # 获取状态观察，如有需要
        if hasattr(self.cfg, "state_space") and isinstance(self.cfg.state_space, int) and self.cfg.state_space > 0:
            return torch.zeros((self.num_envs, self.cfg.state_space), device=self.device)
        else:
            # 默认返回一个小的空观测
            return torch.zeros((self.num_envs, 1), device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        # 计算奖励
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 从观测空间获取当前数据
        # 假设观测空间已按照以下顺序组织：
        # [0:4] - root_quat_w
        # [4:7] - root_lin_vel_b
        # [7:10] - root_ang_vel_b
        # [10:16] - joint_pos - BFS顺序: [大腿1, 大腿2, 大腿3, 小腿1, 小腿2, 小腿3]
        # [16:22] - joint_vel - BFS顺序: [大腿1, 大腿2, 大腿3, 小腿1, 小腿2, 小腿3]
        # [22:23] - height_sensor_value
        # [23:24] - prev_height_sensor_value
        # [24:30] - prev_joint_vel
        # [30:36] - applied_torque
        # [36:42] - previous_actions
        # [42:45] - commands
        obs = self._get_robot_obs()
        
        # 提取需要的数据
        root_quat_w = obs[:, 0:4]
        root_lin_vel_b = obs[:, 4:7]
        root_ang_vel_b = obs[:, 7:10]
        # joint_pos = obs[:, 10:16]
        joint_vel = obs[:, 16:22]
        # current_height = obs[:, 22:23].squeeze(1)  # 高度传感器值
        # prev_height_sensor_value = obs[:, 23:24].squeeze(1)  # 上一帧高度传感器值
        prev_joint_vel = obs[:, 24:30]
        applied_torque = obs[:, 30:36]
        commands = obs[:, 42:45]
        
        # 线速度跟踪奖励
        lin_vel_error = torch.sum(torch.abs(commands[:, :2] - root_lin_vel_b[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale
        rewards += lin_vel_reward
        self._episode_sums["lin_vel_reward"] += lin_vel_reward
        
        # 偏航角速度跟踪奖励
        yaw_rate_error = torch.abs(commands[:, 2] - root_ang_vel_b[:, 2])
        yaw_rate_reward = torch.exp(-yaw_rate_error / 0.25) * self.cfg.yaw_rate_reward_scale
        rewards += yaw_rate_reward
        self._episode_sums["yaw_rate_reward"] += yaw_rate_reward
        
        # 角速度惩罚(x,y)
        ang_vel = torch.sum(torch.abs(root_ang_vel_b[:, :2]), dim=1)
        ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale
        rewards += ang_vel_reward
        self._episode_sums["ang_vel_reward"] += ang_vel_reward
        
        # z方向速度惩罚 - 使用估计的z速度代替直接获取的速度
        # 分别处理向上和向下的z速度
        z_vel_up = torch.clamp(self._estimated_z_vel, min=0.0)    # 只保留向上的z速度（正值）
        z_vel_down = torch.clamp(-self._estimated_z_vel, min=0.0) # 只保留向下的z速度（取负后为正值）
        
        # 向上z速度惩罚（主要惩罚跳跃行为）- 使用更大的惩罚系数
        z_vel_up_penalty = z_vel_up * self.cfg.z_vel_reward_scale  # 强烈惩罚向上跳跃
        rewards += z_vel_up_penalty
        # 向下z速度的适度惩罚 - 使用较小的惩罚系数，防止过快坠落
        z_vel_down_reward = torch.exp(-z_vel_down / 0.25) * self.cfg.z_vel_reward_scale
        rewards += z_vel_down_reward
        # 记录总的z速度奖励
        self._episode_sums["z_vel_reward"] += (z_vel_up_penalty + z_vel_down_reward)
        
        # 关节扭矩惩罚 - 使用观测空间中的数据
        joint_torque_reward = torch.sum(torch.abs(applied_torque), dim=1) * self.cfg.joint_torque_reward_scale
        rewards += joint_torque_reward
        self._episode_sums["joint_torque_reward"] += joint_torque_reward
        
        # 关节加速度惩罚 - 使用当前速度与上一帧速度的差值估计加速度
        joint_acc_estimate = (joint_vel - prev_joint_vel) / self.cfg.sim.dt
        # 计算加速度惩罚
        joint_accel_reward = torch.sum(torch.abs(joint_acc_estimate), dim=1) * self.cfg.joint_accel_reward_scale
        rewards += joint_accel_reward
        self._episode_sums["joint_accel_reward"] += joint_accel_reward
        
        # 动作变化率惩罚 - 使用当前动作和上一动作
        action_rate_reward = torch.sum(torch.abs(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_reward_scale
        rewards += action_rate_reward
        self._episode_sums["action_rate_reward"] += action_rate_reward
        
        # 足部空气时间奖励 - 基于力矩和关节状态推断接触
        # 小腿关节索引 - 三条腿的小腿关节
        shin_joint_indices = torch.tensor([3, 4, 5], device=self.device)  # 三条腿的小腿关节
        # 获取小腿关节力矩
        shin_joint_torques = torch.abs(applied_torque[:, shin_joint_indices])
        # 获取小腿关节速度
        shin_joint_vels = torch.abs(joint_vel[:, shin_joint_indices])
        
        # 推断接触状态：当关节力矩大且关节速度小时，可能表示接触
        torque_threshold = 4.0  # 需要根据实际情况调整
        vel_threshold = 0.1     # 需要根据实际情况调整
        feet_contact = (shin_joint_torques > torque_threshold) & (shin_joint_vels < vel_threshold)
        
        # 保存当前帧的接触状态到历史记录
        self._contact_history.append(feet_contact.clone())
        if len(self._contact_history) > 3:
            self._contact_history.pop(0)
            
        # 判断连续三帧都接触才认为是真的接触
        if len(self._contact_history) == 3:
            contact_stack = torch.stack(self._contact_history, dim=0)  # shape: (3, num_envs, num_feet)
            confirmed_feet_contact = torch.min(contact_stack, dim=0).values.bool()  # 取三帧最小值
        else:
            confirmed_feet_contact = feet_contact  # 历史不足时直接用当前帧
        
        # 计算每个环境的接触状态平均值 - 使用确认后的接触状态
        contact_phases = torch.mean(confirmed_feet_contact.float(), dim=1)
        # 修改奖励计算逻辑：鼓励接近0.6的接触比例（约60%时间接触）
        feet_phase_reward = (0.6 - torch.abs(contact_phases - 0.6)) * self.cfg.feet_air_time_reward_scale
        rewards += feet_phase_reward
        self._episode_sums["feet_air_time_reward"] += feet_phase_reward
        
        # 基于身体高度的惩罚 - 惩罚身体高度小于阈值的情况
        height_threshold = 0.07  # 7厘米的身体高度阈值，跳跃环境使用更高的阈值
        too_low = self._height_sensor_value < height_threshold
        height_penalty = too_low.float() * self.cfg.undesired_contact_reward_scale  # 使用与之前相同的奖励系数
        rewards += height_penalty
        self._episode_sums["undesired_contact_reward"] += height_penalty
        
        # 平坦姿态奖励（仅平地环境）
        if self.cfg.flat_orientation_reward_scale != 0.0:
            # 使用观测空间中的四元数计算投影重力向量
            # 全局重力向量 (指向下方)
            gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
            # 使用四元数将重力向量从世界坐标系转换到机器人坐标系
            gravity_vec_b = math_utils.quat_rotate_inverse(root_quat_w, gravity_vec_w)
            # 计算投影重力向量的x和y分量（这些应该在平坦姿态时接近0）
            projected_gravity_b = gravity_vec_b[:, :2]
            
            flat_orientation_reward = torch.sum(torch.square(projected_gravity_b), dim=1) * self.cfg.flat_orientation_reward_scale
            rewards += flat_orientation_reward
            self._episode_sums["flat_orientation_reward"] += flat_orientation_reward
        
        # 接触地面惩罚 - 使用理想接触传感器
        if not hasattr(self, '_undesired_contact_body_ids'):
            self._undesired_contact_body_ids = [0]  
        if hasattr(self, '_contact_sensor') and hasattr(self._contact_sensor, 'data') \
            and hasattr(self._contact_sensor.data, 'net_forces_w_history') \
            and self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            # net_contact_forces: (num_envs, history, num_bodies, 3)
            # 取最近一帧
            last_forces = net_contact_forces[:, -1, :, :] if net_contact_forces.dim() == 4 else net_contact_forces
            # 计算大腿部分的受力范数
            thigh_forces = torch.norm(last_forces[:, self._undesired_contact_body_ids, :], dim=-1)
            # 判断是否有大腿接触
            is_contact = torch.any(thigh_forces > 1.0, dim=1)
            undesired_contact_reward = is_contact.float() * self.cfg.undesired_contact_reward_scale
            rewards += undesired_contact_reward
            self._episode_sums["undesired_contact_reward"] += undesired_contact_reward
        else:
            # 如果接触传感器数据不可用，输出警告
            if hasattr(self, 'episode_length_buf') and self.episode_length_buf[0] == 0:
                print("警告: 接触传感器数据不可用，无法计算大腿接触惩罚")
            undesired_contact_reward = torch.zeros(self.num_envs, device=self.device)
            rewards += undesired_contact_reward
        
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 获取结束信号
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # 使用观测空间中的四元数检测机器人是否倒下
        # 检查基座方向是否远离正常姿态
        quat = self._robot.data.root_quat_w  # 这在观测空间中
        # 计算与垂直方向的偏差
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        # 将向上向量从世界坐标系转换到机器人坐标系
        up_vec_in_robot = math_utils.quat_rotate_inverse(quat, up_vec.repeat(self.num_envs, 1))
        # 计算与理想向上方向的夹角余弦值
        cos_angle = torch.sum(up_vec_in_robot * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1), dim=1)
        # 如果夹角余弦值小于某个阈值，认为机器人倒下
        died = cos_angle < 0.5  # 对应约60度的偏差
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # 重置指定环境
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids_list = self._robot._ALL_INDICES
        else:
            # 注意：不要将张量转换为列表，保持原有张量类型
            env_ids_list = env_ids
        
        # 重置机器人
        self._robot.reset(env_ids_list)
        super()._reset_idx(env_ids)
        
        if env_ids is not None and len(env_ids) == self.num_envs:
            # 分散重置以避免训练中的峰值
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            
        # 初始化命令（线速度x,y和角速度）
        if env_ids is not None:
            self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3  # vx in [-0.3, 0.3]
            self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * 0.4 - 0.2  # vy in [-0.2, 0.2]
            self._commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)  # wz设为0
        
        # 重置机器人位置和关节状态
        if env_ids is not None:
            default_joint_pos = self._robot.data.default_joint_pos[env_ids]
            default_joint_vel = torch.zeros_like(default_joint_pos)
            
            # 更新存储的默认关节位置
            self._default_joint_pos[env_ids] = default_joint_pos.clone()
            
            # 将机器人放置在地形上
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            
            # 使用无旋转的四元数（w=1, x=y=z=0）
            rot_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # 无旋转
            
            # 应用旋转到所有环境
            default_root_state[:, 3:7] = rot_quat
            
            # 写入根姿态和速度 - 使用原始张量，不做类型转换
            self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_list)
            self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_list)
            self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, None, env_ids_list)
            
            # 重置其他状态变量
            self._prev_joint_vel[env_ids] = torch.zeros_like(self._prev_joint_vel[env_ids])
            self._prev_height_sensor_value[env_ids] = self._height_sensor_value[env_ids].clone()
            
            # 重置接触历史记录
            self._contact_history = []
            
            # 增加回合完成计数
            self._episode_counter += 1
        
        # 重置传感器数据
        self._simulate_height_sensor()
        
        # 日志记录
        extras = dict()
        if env_ids is not None:
            for key in self._episode_sums.keys():
                extras[f"episode_{key}"] = self._episode_sums[key][env_ids].mean().item()
                self._episode_sums[key][env_ids] = 0
            
        return extras

    def close(self):
        """关闭环境并清理资源"""
        # 清空关节数据缓冲区
        self._joint_data_buffer = {
            "joint_pos": [],
            "joint_vel": [], 
            "joint_torque": []
        }
        # 清空线速度与命令速度数据缓冲区
        self._vel_data_buffer = {
            "root_lin_vel": [],
            "command_vel": []
        }
            
        # 调用父类方法
        super().close()

    def _get_joint_data_from_obs(self, obs_dict=None):
        """从观察字典中提取关节数据，用于关节数据记录

        Args:
            obs_dict: 观察字典，如果为None则使用当前状态生成

        Returns:
            tuple: (joint_pos, joint_vel, joint_torque) 关节位置、速度和力矩数据
        """
        if obs_dict is None:
            # 如果没有提供观察字典，获取当前观察
            obs_dict = self._get_observations()
        
        # 从robot_obs中提取关节数据
        if "robot_obs" in obs_dict:
            # 根据观察空间布局提取关节数据 - 使用BFS顺序: [大腿1, 大腿2, 大腿3, 小腿1, 小腿2, 小腿3]
            # [10:16] - joint_pos
            # [16:22] - joint_vel
            # [30:36] - applied_torque
            robot_obs = obs_dict["robot_obs"]
            
            if robot_obs.dim() > 1:
                # 提取第一个环境的数据用于可视化
                joint_pos = robot_obs[0, 10:16]
                joint_vel = robot_obs[0, 16:22]
                joint_torque = robot_obs[0, 30:36]
            else:
                # 如果是单一环境数据
                joint_pos = robot_obs[10:16]
                joint_vel = robot_obs[16:22]
                joint_torque = robot_obs[30:36]
            
            return joint_pos, joint_vel, joint_torque
        
        # 如果没有robot_obs，但有单独的关节数据
        elif "joint_pos" in obs_dict and "joint_vel" in obs_dict and "applied_torque" in obs_dict:
            joint_pos = obs_dict["joint_pos"][0] if obs_dict["joint_pos"].dim() > 1 else obs_dict["joint_pos"]
            joint_vel = obs_dict["joint_vel"][0] if obs_dict["joint_vel"].dim() > 1 else obs_dict["joint_vel"]
            joint_torque = obs_dict["applied_torque"][0] if obs_dict["applied_torque"].dim() > 1 else obs_dict["applied_torque"]
            
            return joint_pos, joint_vel, joint_torque
        
        # 如果可以直接从机器人属性获取
        elif hasattr(self, "_robot") and hasattr(self._robot, "data"):
            joint_pos = self._robot.data.joint_pos[0] if self._robot.data.joint_pos.dim() > 1 else self._robot.data.joint_pos
            joint_vel = self._robot.data.joint_vel[0] if self._robot.data.joint_vel.dim() > 1 else self._robot.data.joint_vel
            
            # 力矩可能不总是可用
            if hasattr(self._robot.data, "applied_torque"):
                joint_torque = self._robot.data.applied_torque[0] if self._robot.data.applied_torque.dim() > 1 else self._robot.data.applied_torque
            else:
                joint_torque = torch.zeros_like(joint_vel)
            
            return joint_pos, joint_vel, joint_torque
        
        return None
        
    def step(self, actions):
        """环境步进，执行一次模拟"""
        # 调用父类的step方法
        observations, rewards, terminations, truncations, infos = super().step(actions)
        
        # 添加关节数据到infos中，便于JointMonitorRunner使用
        if "observations" not in infos:
            infos["observations"] = {}
        
        # 提取关节数据
        joint_data = self._get_joint_data_from_obs(observations)
        if joint_data:
            joint_pos, joint_vel, joint_torque = joint_data
            
            # 将关节数据添加到infos中
            infos["observations"]["joint_pos"] = joint_pos.unsqueeze(0) if joint_pos.dim() == 1 else joint_pos
            infos["observations"]["joint_vel"] = joint_vel.unsqueeze(0) if joint_vel.dim() == 1 else joint_vel
            infos["observations"]["joint_torque"] = joint_torque.unsqueeze(0) if joint_torque.dim() == 1 else joint_torque
            
            # 添加到单独的joint_data字段，而不是episode字典
            # 这样日志系统不会尝试记录这些复杂数据
            infos["joint_data"] = {
                "pos": joint_pos.unsqueeze(0) if joint_pos.dim() == 1 else joint_pos,
                "vel": joint_vel.unsqueeze(0) if joint_vel.dim() == 1 else joint_vel,
                "torque": joint_torque.unsqueeze(0) if joint_torque.dim() == 1 else joint_torque
            }
            
            # 添加一些数字统计信息到episode中供日志系统使用
            if "episode" not in infos:
                infos["episode"] = {}
            
            # 添加关节统计信息，这些是简单的标量，可以被日志系统处理
            infos["episode"]["joint_pos_mean"] = float(torch.mean(joint_pos).detach().cpu().numpy())
            infos["episode"]["joint_vel_mean"] = float(torch.mean(joint_vel).detach().cpu().numpy())
            infos["episode"]["joint_torque_mean"] = float(torch.mean(joint_torque).detach().cpu().numpy())
            infos["episode"]["joint_pos_max"] = float(torch.max(joint_pos).detach().cpu().numpy())
            infos["episode"]["joint_vel_max"] = float(torch.max(joint_vel).detach().cpu().numpy())
            infos["episode"]["joint_torque_max"] = float(torch.max(joint_torque).detach().cpu().numpy())
        else:
            print("[TLR6Env] 警告: 无法从观察获取关节数据")
        
        # 更新关节可视化数据
        self._update_joint_visualization()
        
        return observations, rewards, terminations, truncations, infos

class TLR6JumpEnv(TLR6Env):
    """三足机器人TLR6跳跃环境实现"""
    cfg: TLR6JumpEnvCfg

    def __init__(self, cfg: TLR6JumpEnvCfg, render_mode: str | None = None, **kwargs):
        # 首先调用父类的初始化方法，确保scene等基本属性已设置
        super().__init__(cfg, render_mode, **kwargs)
        
        # 跳跃状态跟踪
        self._jump_phase = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)  # 0:准备, 1:起跳, 2:空中, 3:着陆
        self._max_height = torch.zeros(self.num_envs, device=self.device)  # 跳跃过程中的最大高度
        self._air_time = torch.zeros(self.num_envs, device=self.device)  # 在空中的时间
        self._on_ground = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  # 是否在地面上
        self._prev_height = torch.zeros(self.num_envs, device=self.device)  # 上一帧的高度
        
        # 新增：静止状态检测
        self._prev_velocity = torch.zeros(self.num_envs, 3, device=self.device)  # 上一帧速度，用于检测静止
        self._static_time = torch.zeros(self.num_envs, device=self.device)  # 静止时间计数
        
        # 新增：高度数据缓冲区（只记录第一个环境的z高度）
        self._height_data_buffer = []
        
        # 确保命令初始化，即使在跳跃环境中不用于奖励计算
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 跳跃相关的日志记录
        self._episode_sums.update({
            "jump_height_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "jump_landing_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "air_time_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            # 新增日志记录
            "static_penalty": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "symmetry_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
            "phase_sync_reward": torch.zeros(self.num_envs, dtype=torch.float, device=self.device),
        })
        
        # 直接设置已知的大腿链接索引
        self._undesired_contact_body_ids = [0]  

    def _get_rewards(self) -> torch.Tensor:
        """计算跳跃环境的奖励"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 从观测空间获取数据
        obs = self._get_robot_obs()
        
        # 提取需要的数据
        root_quat_w = obs[:, 0:4]
        root_lin_vel_b = obs[:, 4:7]
        root_ang_vel_b = obs[:, 7:10]
        joint_pos = obs[:, 10:16]
        joint_vel = obs[:, 16:22]
        current_height = obs[:, 22:23].squeeze(1)  # 高度传感器值
        prev_height_sensor_value = obs[:, 23:24].squeeze(1)  # 上一帧高度传感器值
        prev_joint_vel = obs[:, 24:30]  # 上一帧关节速度
        applied_torque = obs[:, 30:36]  # 关节力矩观测
        # 以下观测空间在跳跃环境中不用于奖励计算，但为了保持一致性而包含
        # previous_actions = obs[:, 36:42]  # 上一帧动作
        # commands = obs[:, 42:45]  # 速度命令
        
        # 当前速度 - 从观测空间获取
        current_velocity = root_lin_vel_b.clone()
        
        # 检测是否在地面上 - 使用高度和垂直速度来推断，不使用接触传感器
        height_threshold = 0.05  # 接近地面的高度阈值
        vel_threshold = 0.02     # 垂直速度阈值
        
        # 使用小腿关节数据检测接触
        shin_joint_indices = torch.tensor([3, 4, 5], device=self.device)
        shin_joint_torques = torch.abs(applied_torque[:, shin_joint_indices])
        shin_joint_vels = torch.abs(joint_vel[:, shin_joint_indices])
        
        # 推断接触状态：当关节力矩大且关节速度小时，可能表示接触
        torque_threshold = 0.2  # 需要根据实际情况调整
        vel_threshold = 0.1     # 需要根据实际情况调整
        feet_contact = (shin_joint_torques > torque_threshold) & (shin_joint_vels < vel_threshold)
        
        # 保存当前帧的接触状态到历史记录
        self._contact_history.append(feet_contact.clone())
        if len(self._contact_history) > 3:
            self._contact_history.pop(0)
            
        # 判断连续三帧都接触才认为是真的接触
        if len(self._contact_history) == 3:
            contact_stack = torch.stack(self._contact_history, dim=0)  # shape: (3, num_envs, num_feet)
            confirmed_feet_contact = torch.min(contact_stack, dim=0).values.bool()  # 取三帧最小值
        else:
            confirmed_feet_contact = feet_contact  # 历史不足时直接用当前帧
        
        # # 计算每个环境的接触状态平均值 - 使用确认后的接触状态
        # contact_phases = torch.mean(confirmed_feet_contact.float(), dim=1)
        # # 修改奖励计算逻辑：鼓励接近0.6的接触比例（约60%时间接触）
        # feet_phase_reward = (0.6 - torch.abs(contact_phases - 0.6)) * self.cfg.feet_air_time_reward_scale
        # rewards += feet_phase_reward
        # self._episode_sums["feet_air_time_reward"] += feet_phase_reward
        
        # 估计垂直速度 - 使用当前高度和上一帧高度的差值
        estimated_z_vel = (current_height - self._prev_height) / self.cfg.sim.dt
        
        # 如果高度接近地面且垂直速度很小，或者有足部接触，认为在地面上
        self._on_ground = ((current_height < height_threshold) & (torch.abs(estimated_z_vel) < vel_threshold)) | torch.any(confirmed_feet_contact, dim=1)
        
        # 更新跳跃阶段
        jumping_start = self._on_ground & (current_height - self._prev_height > 0.01)
        in_air = (~self._on_ground) & (self._jump_phase == 1)
        landing = self._on_ground & (self._jump_phase == 2)
        
        # 更新跳跃阶段
        self._jump_phase[jumping_start] = 1  # 起跳
        self._jump_phase[in_air] = 2  # 空中
        self._jump_phase[landing] = 3  # 着陆
        # 着陆后，重置为准备阶段
        self._jump_phase[self._jump_phase == 3] = 0
        
        # 更新最大高度
        self._max_height = torch.maximum(self._max_height, current_height)
        
        # 更新空中时间
        sim_dt = self.cfg.sim.dt
        self._air_time[~self._on_ground] += sim_dt
        self._air_time[self._on_ground] = 0
        
        # 检测静止状态（速度变化小）
        velocity_change = torch.norm(current_velocity - self._prev_velocity, dim=1)
        velocity_magnitude = torch.norm(current_velocity, dim=1)
        is_static = (velocity_change < 0.05) & (velocity_magnitude < 0.1) & self._on_ground
        
        # 更新静止时间计数
        self._static_time[is_static] += sim_dt
        self._static_time[~is_static] = 0
        
        # 保存当前速度供下一帧使用
        self._prev_velocity = current_velocity
        
        # 计算奖励 - 简化版，只保留必要的奖励项
        
        # 1. 对称性奖励（保留，因为这是我们之前新增的重点功能）
        # 获取大腿和小腿关节的位置和速度
        thigh_joints = joint_pos[:, [0, 1, 2]]  # 三个大腿关节
        thigh_vels = joint_vel[:, [0, 1, 2]]   # 三个大腿关节速度
        shin_joints = joint_pos[:, [3, 4, 5]]  # 三个小腿关节
        shin_vels = joint_vel[:, [3, 4, 5]]    # 三个小腿关节速度
        
        # 计算对称性奖励
        def calculate_symmetry_reward(joints, vels):
            # 计算关节位置的平均值
            mean_pos = torch.mean(joints, dim=1, keepdim=True)
            # 计算关节速度的平均值
            mean_vel = torch.mean(vels, dim=1, keepdim=True)
            
            # 计算位置和速度的方差（越小越对称）
            pos_variance = torch.sum(torch.square(joints - mean_pos), dim=1)
            vel_variance = torch.sum(torch.square(vels - mean_vel), dim=1)
            
            # 使用指数函数将方差转换为奖励（方差越小，奖励越大）
            pos_reward = torch.exp(-pos_variance / 0.1)
            vel_reward = torch.exp(-vel_variance / 0.1)
            
            return pos_reward + vel_reward
        
        # 计算相位同步奖励
        def calculate_phase_sync_reward(joints, vels):
            # 计算相邻关节之间的相位差
            phase_diff1 = torch.abs(joints[:, 0] - joints[:, 1])
            phase_diff2 = torch.abs(joints[:, 1] - joints[:, 2])
            phase_diff3 = torch.abs(joints[:, 2] - joints[:, 0])
            
            # 计算速度同步性
            vel_sync1 = torch.abs(vels[:, 0] - vels[:, 1])
            vel_sync2 = torch.abs(vels[:, 1] - vels[:, 2])
            vel_sync3 = torch.abs(vels[:, 2] - vels[:, 0])
            
            # 合并相位差和速度同步性
            total_phase_diff = phase_diff1 + phase_diff2 + phase_diff3
            total_vel_sync = vel_sync1 + vel_sync2 + vel_sync3
            
            # 转换为奖励
            phase_reward = torch.exp(-total_phase_diff / 0.2)
            vel_sync_reward = torch.exp(-total_vel_sync / 0.2)
            
            return phase_reward + vel_sync_reward
        
        # 计算大腿和小腿的对称性奖励
        thigh_symmetry_reward = calculate_symmetry_reward(thigh_joints, thigh_vels)
        shin_symmetry_reward = calculate_symmetry_reward(shin_joints, shin_vels)
        
        # 计算大腿和小腿的相位同步奖励
        thigh_phase_reward = calculate_phase_sync_reward(thigh_joints, thigh_vels)
        shin_phase_reward = calculate_phase_sync_reward(shin_joints, shin_vels)
        
        # 根据跳跃阶段调整奖励权重
        symmetry_weight = torch.ones(self.num_envs, device=self.device)
        symmetry_weight[self._jump_phase == 0] = 1.5  # 准备阶段
        symmetry_weight[self._jump_phase == 3] = 1.5  # 着陆阶段
        symmetry_weight[self._jump_phase == 1] = 0.5  # 起跳阶段降低对称性要求
        
        # 添加对称性奖励
        rewards += (thigh_symmetry_reward + shin_symmetry_reward) * self.cfg.symmetry_reward_scale * symmetry_weight
        
        # 添加相位同步奖励
        rewards += (thigh_phase_reward + shin_phase_reward) * self.cfg.phase_sync_reward_scale * symmetry_weight
        
        # 更新日志记录
        self._episode_sums["symmetry_reward"] += (thigh_symmetry_reward + shin_symmetry_reward) * symmetry_weight
        self._episode_sums["phase_sync_reward"] += (thigh_phase_reward + shin_phase_reward) * symmetry_weight
        
        # 2. 跳跃高度奖励 - 鼓励机器人跳得更高
        jump_height_reward = self._max_height * self.cfg.jump_height_reward_scale
        jump_height_reward = torch.clamp(jump_height_reward, min=0.0)
        rewards += jump_height_reward
        self._episode_sums["jump_height_reward"] += jump_height_reward
        
        # 3. 快速的高度变化奖励 - 只对向上的高度变化做奖励（替代z方向速度奖励）
        height_change_rate = (current_height - self._prev_height) / sim_dt  # 高度变化率等同于z方向速度
        height_change_reward = torch.clamp(height_change_rate, min=0.0) * self.cfg.height_change_reward_scale
        rewards += height_change_reward
        
        # 4. 角速度惩罚(x,y) - 防止倾倒
        ang_vel = torch.sum(torch.abs(root_ang_vel_b[:, :2]), dim=1)
        ang_vel_reward = ang_vel * self.cfg.ang_vel_reward_scale  # 负值作为惩罚
        rewards += ang_vel_reward
        self._episode_sums["ang_vel_reward"] += ang_vel_reward
        
        # 5. 关节扭矩惩罚 - 避免过大的关节力矩
        joint_torque_reward = torch.sum(torch.abs(applied_torque), dim=1) * self.cfg.joint_torque_reward_scale
        rewards += joint_torque_reward
        self._episode_sums["joint_torque_reward"] += joint_torque_reward
        
        # 6. 关节加速度惩罚 - 使用当前关节速度与上一帧关节速度的差值估计加速度
        joint_acc_estimate = (joint_vel - prev_joint_vel) / self.cfg.sim.dt
        joint_accel_reward = torch.sum(torch.abs(joint_acc_estimate), dim=1) * self.cfg.joint_accel_reward_scale
        rewards += joint_accel_reward
        self._episode_sums["joint_accel_reward"] += joint_accel_reward
        
        # 7. 平坦姿态奖励 - 保持身体水平
        if self.cfg.flat_orientation_reward_scale != 0.0:
            # 使用四元数计算投影重力向量
            # 全局重力向量 (指向下方)
            gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
            # 使用四元数将重力向量从世界坐标系转换到机器人坐标系
            gravity_vec_b = math_utils.quat_rotate_inverse(root_quat_w, gravity_vec_w)
            # 计算投影重力向量的x和y分量（这些应该在平坦姿态时接近0）
            projected_gravity_b = gravity_vec_b[:, :2]
            
            flat_orientation_reward = torch.sum(torch.square(projected_gravity_b), dim=1) * self.cfg.flat_orientation_reward_scale
            rewards += flat_orientation_reward
            self._episode_sums["flat_orientation_reward"] += flat_orientation_reward
        
        # 8. 新增：静止状态惩罚 - 惩罚长时间静止不动
        if hasattr(self.cfg, 'static_penalty_scale') and self.cfg.static_penalty_scale != 0.0:
            # 静止时间超过阈值时给予惩罚 (0.3秒)
            static_time_normalized = torch.clamp(self._static_time / 0.3, max=1.0)
            static_penalty = -static_time_normalized * self.cfg.static_penalty_scale
            
            # 只在着陆后的准备阶段施加更强的静止惩罚
            landing_phase_multiplier = torch.ones_like(static_penalty)
            landing_phase_multiplier[self._jump_phase == 0] = 2.0  # 准备阶段的惩罚加倍
            
            static_penalty = static_penalty * landing_phase_multiplier
            rewards += static_penalty
            self._episode_sums["static_penalty"] += static_penalty
        
        # 9. 大腿接触惩罚 - 使用接触传感器数据直接判断大腿接触
        if hasattr(self._contact_sensor.data, "net_forces_w_history") and self._contact_sensor.data.net_forces_w_history is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
            undesired_contact_reward = contacts * self.cfg.undesired_contact_reward_scale
            rewards += undesired_contact_reward
            self._episode_sums["undesired_contact_reward"] += undesired_contact_reward
        else:
            # 如果接触传感器数据不可用，输出警告
            if self.episode_length_buf[0] == 0:  # 只在第一步输出警告，避免刷屏
                print("警告: 接触传感器数据不可用，无法计算大腿接触惩罚")
            undesired_contact_reward = torch.zeros(self.num_envs, device=self.device)
            rewards += undesired_contact_reward
        
        # 更新上一帧高度（用于下一帧计算高度变化）
        self._prev_height = current_height
        
        return rewards

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """重置指定环境"""
        # 调用父类的重置方法
        extras = super()._reset_idx(env_ids)
        
        # 重置跳跃相关的状态
        if env_ids is not None:
            self._jump_phase[env_ids] = 0
            self._max_height[env_ids] = 0.0
            self._air_time[env_ids] = 0.0
            self._on_ground[env_ids] = True
            self._prev_height[env_ids] = self._height_sensor_value[env_ids]
            
            # 重置静止时间跟踪
            self._static_time[env_ids] = 0.0
            self._prev_velocity[env_ids] = 0.0
            
            # 重置接触历史记录
            self._contact_history = []
            
            # 虽然在跳跃环境中不使用命令，但为了保持一致性而初始化
            # 设置为零值，表示不需要特定方向的移动
            self._commands[env_ids, 0] = torch.zeros(len(env_ids), device=self.device)  # x方向速度
            self._commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)  # y方向速度
            self._commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device)  # wz设为0
        
        return extras

    def _update_joint_visualization(self):
        """更新关节可视化数据 - 现在只收集数据，不进行实时可视化"""
        try:
            # 获取当前步骤
            current_step = self.episode_length_buf[0].item() if hasattr(self, "episode_length_buf") else 0
            # 只在关键点输出调试信息
            should_debug = current_step == 0 or current_step % 100 == 0
            
            # 从第一个环境中获取数据
            joint_pos = self._robot.data.joint_pos[0]
            joint_vel = self._robot.data.joint_vel[0]
            applied_torque = self._robot.data.applied_torque[0] if hasattr(self._robot.data, "applied_torque") else torch.zeros_like(joint_vel)
            
            # 存储数据到缓冲区
            if len(self._joint_data_buffer["joint_pos"]) >= self._max_datapoints:
                for key in self._joint_data_buffer:
                    self._joint_data_buffer[key].pop(0)
            
            self._joint_data_buffer["joint_pos"].append(joint_pos.detach().clone())
            self._joint_data_buffer["joint_vel"].append(joint_vel.detach().clone())
            self._joint_data_buffer["joint_torque"].append(applied_torque.detach().clone())

            # 新增：采集基座z高度
            height = self._height_sensor_value[0].detach().clone()  # 标量
            if len(self._height_data_buffer) >= self._max_datapoints:
                self._height_data_buffer.pop(0)
            self._height_data_buffer.append(height)
        except Exception as e:
            print(f"[TLR6JumpEnv] 更新关节/高度数据时出错: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """关闭环境并清理资源"""
        # 清空关节数据缓冲区
        self._joint_data_buffer = {
            "joint_pos": [],
            "joint_vel": [], 
            "joint_torque": []
        }
        # 清空高度数据缓冲区
        self._height_data_buffer = []
        # 调用父类方法
        super().close()