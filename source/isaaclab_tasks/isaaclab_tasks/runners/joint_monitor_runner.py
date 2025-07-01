from __future__ import annotations

import os
import time
import torch
from collections import deque
import statistics
from typing import List, Dict, Any, Optional, Union, Sequence

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import store_code_state
from rsl_rl.env import VecEnv

# 导入TensorBoard的SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

from isaaclab_tasks.runners.joint_data_logger import JointDataLogger


class JointMonitorRunner(OnPolicyRunner):
    """扩展OnPolicyRunner类，增加关节数据记录和图表生成功能"""
    
    def __init__(self, env: VecEnv, train_cfg: Dict[str, Any], log_dir: Optional[str] = None, device: str = "cpu"):
        """初始化JointMonitorRunner
        
        Args:
            env: 环境实例
            train_cfg: 训练配置
            log_dir: 日志目录
            device: 设备
        """
        super().__init__(env, train_cfg, log_dir, device)
        
        # 关节数据记录设置
        joint_names = None
        
        # 尝试从环境中获取关节名称
        try:
            # 尝试多种方式获取关节名称
            if hasattr(env, "_joint_names"):
                joint_names = env._joint_names
                print(f"从env._joint_names获取关节名称成功: {joint_names}")
            elif hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_joint_names"):
                joint_names = env.unwrapped._joint_names
                print(f"从env.unwrapped._joint_names获取关节名称成功: {joint_names}")
            elif hasattr(env, "env") and hasattr(env.env, "_joint_names"):
                joint_names = env.env._joint_names
                print(f"从env.env._joint_names获取关节名称成功: {joint_names}")
            else:
                # 使用我们想要的默认关节名称，使用BFS顺序：所有大腿关节在前，所有小腿关节在后
                joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
                print(f"未能从环境获取关节名称，使用默认关节名称: {joint_names}")
        except Exception as e:
            print(f"警告: 获取关节名称时出错: {e}")
            # 使用我们想要的默认关节名称
            joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
            print(f"由于错误使用默认关节名称: {joint_names}")
        
        # 初始化关节数据记录器
        self.joint_data_logger = None
        if log_dir is not None:
            try:
                self.joint_data_logger = JointDataLogger(
                    joint_names=joint_names,
                    log_dir=log_dir,
                    save_interval=250  # 修改为250
                )
                print(f"JointMonitorRunner初始化完成，将每250个迭代保存一次关节数据")
                print(f"使用的关节名称: {joint_names}")
            except Exception as e:
                print(f"警告: 初始化关节数据记录器时出错: {e}")
    
    def _get_joint_data(self, infos: Dict[str, Any]) -> Optional[tuple]:
        """从infos字典中获取关节数据
        
        Args:
            infos: 环境信息字典
        
        Returns:
            tuple: (joint_pos, joint_vel, joint_torque) 或 None
        """
        joint_pos = None
        joint_vel = None
        joint_torque = None
        
        try:
            # 只在第一次迭代和之后每500次迭代输出调试信息
            should_debug = self.current_learning_iteration == 0 or self.current_learning_iteration % 500 == 0
            
            # 方法1: 直接从joint_data字段获取数据（新增的首选路径）
            if isinstance(infos, dict) and "joint_data" in infos:
                joint_data = infos["joint_data"]
                if isinstance(joint_data, dict) and "pos" in joint_data and "vel" in joint_data:
                    joint_pos = joint_data["pos"]
                    joint_vel = joint_data["vel"]
                    joint_torque = joint_data.get("torque", torch.zeros_like(joint_vel))
                    return joint_pos, joint_vel, joint_torque
            
            # 方法2: 从TLR6的观察结构"observations"中获取
            if isinstance(infos, dict) and "observations" in infos:
                obs_data = infos["observations"]
                
                # 方法2.1: 使用环境提供的方法
                if hasattr(self.env, "_get_joint_data_from_obs"):
                    try:
                        joint_data = self.env._get_joint_data_from_obs(obs_data)
                        if joint_data and len(joint_data) == 3:
                            return joint_data
                    except Exception as e:
                        if should_debug:
                            print(f"从环境方法提取关节数据时出错: {e}")
                
                # 方法2.2: 直接从observations中提取
                if isinstance(obs_data, dict) and "joint_pos" in obs_data and "joint_vel" in obs_data:
                    joint_pos = obs_data["joint_pos"][0] if obs_data["joint_pos"].dim() > 1 else obs_data["joint_pos"]
                    joint_vel = obs_data["joint_vel"][0] if obs_data["joint_vel"].dim() > 1 else obs_data["joint_vel"]
                    joint_torque = obs_data["applied_torque"][0] if "applied_torque" in obs_data and obs_data["applied_torque"].dim() > 1 else \
                                obs_data.get("applied_torque", torch.zeros_like(joint_vel))
                    return joint_pos, joint_vel, joint_torque
            
            # 方法3: 从episode字段中的关节数据中获取（旧方法，但出于兼容性保留）
            if isinstance(infos, dict) and "episode" in infos:
                episode_info = infos["episode"]
                
                # 检查旧的joint_data结构是否存在
                if isinstance(episode_info, dict) and "joint_data" in episode_info:
                    joint_data = episode_info["joint_data"]
                    if isinstance(joint_data, dict) and "pos" in joint_data and "vel" in joint_data:
                        joint_pos = joint_data["pos"]
                        joint_vel = joint_data["vel"]
                        joint_torque = joint_data.get("torque", torch.zeros_like(joint_vel))
                        return joint_pos, joint_vel, joint_torque
            
            # 方法4: 从环境对象上直接获取
            if hasattr(self.env, "_robot") and hasattr(self.env._robot, "data"):
                try:
                    robot_data = self.env._robot.data
                    if hasattr(robot_data, "joint_pos") and hasattr(robot_data, "joint_vel"):
                        joint_pos = robot_data.joint_pos[0] if robot_data.joint_pos.dim() > 1 else robot_data.joint_pos
                        joint_vel = robot_data.joint_vel[0] if robot_data.joint_vel.dim() > 1 else robot_data.joint_vel
                        
                        # 力矩可能不总是可用
                        if hasattr(robot_data, "applied_torque"):
                            joint_torque = robot_data.applied_torque[0] if robot_data.applied_torque.dim() > 1 else robot_data.applied_torque
                        else:
                            joint_torque = torch.zeros_like(joint_vel)
                        
                        return joint_pos, joint_vel, joint_torque
                except Exception as e:
                    if should_debug:
                        print(f"从_robot.data中获取关节数据时出错: {e}")
            
            # 方法5: 尝试访问环境的关节数据缓冲区
            if hasattr(self.env, "_joint_data_buffer"):
                try:
                    buffer = self.env._joint_data_buffer
                    if isinstance(buffer, dict) and "joint_pos" in buffer and len(buffer["joint_pos"]) > 0:
                        # 获取最新的关节数据
                        joint_pos = buffer["joint_pos"][-1]
                        joint_vel = buffer["joint_vel"][-1]
                        joint_torque = buffer["joint_torque"][-1]
                        return joint_pos, joint_vel, joint_torque
                except Exception as e:
                    if should_debug:
                        print(f"从_joint_data_buffer获取关节数据时出错: {e}")
                    
            # 所有方法都失败，尝试从env.unwrapped获取
            if hasattr(self.env, "unwrapped"):
                try:
                    unwrapped = self.env.unwrapped
                    if hasattr(unwrapped, "_get_joint_data_from_obs"):
                        joint_data = unwrapped._get_joint_data_from_obs()
                        if joint_data and len(joint_data) == 3:
                            return joint_data
                            
                    if hasattr(unwrapped, "_joint_data_buffer"):
                        buffer = unwrapped._joint_data_buffer
                        if isinstance(buffer, dict) and "joint_pos" in buffer and len(buffer["joint_pos"]) > 0:
                            joint_pos = buffer["joint_pos"][-1]
                            joint_vel = buffer["joint_vel"][-1]
                            joint_torque = buffer["joint_torque"][-1]
                            return joint_pos, joint_vel, joint_torque
                except Exception as e:
                    if should_debug:
                        print(f"从unwrapped环境获取关节数据时出错: {e}")
                
            if should_debug:
                print("无法从任何来源获取关节数据")
            
        except Exception as e:
            if should_debug:
                print(f"获取关节数据时出错: {e}")
        
        return None
    
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """重写learn方法，添加关节数据收集逻辑"""
        
        # 初始化writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter
                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        # 初始环境长度随机化
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # 获取初始观察
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()

        # Book keeping
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            cur_ireward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        # 主训练循环
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        
        for it in range(start_iter, tot_iter):
            start = time.time()
            
            # 添加collection_time变量初始化，避免KeyError
            collection_time = 0.0
            
            # Rollout
            collection_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # 从策略采样动作
                    actions = self.alg.act(obs, critic_obs)
                    # 执行环境步进
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))

                    # 移回设备
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # 归一化观察
                    obs = self.obs_normalizer(obs)
                    # 安全地提取critic_obs
                    if isinstance(infos, dict) and "observations" in infos and "critic" in infos["observations"]:
                        critic_obs_data = infos["observations"]["critic"]
                        if critic_obs_data is not None:
                            critic_obs = self.critic_obs_normalizer(critic_obs_data.to(self.device))
                        else:
                            critic_obs = obs
                    else:
                        critic_obs = obs

                    # 收集关节数据和线速度数据（同步）
                    root_lin_vel = None
                    command_vel = None
                    env_ = self.env
                    if hasattr(env_, "unwrapped"):
                        env_ = env_.unwrapped
                    if hasattr(env_, "_vel_data_buffer"):
                        if env_._vel_data_buffer["root_lin_vel"]:
                            root_lin_vel = env_._vel_data_buffer["root_lin_vel"][0]
                        if env_._vel_data_buffer["command_vel"]:
                            command_vel = env_._vel_data_buffer["command_vel"][0]
                    if self.joint_data_logger is not None:
                        joint_data = self._get_joint_data(infos)
                        if joint_data:
                            joint_pos, joint_vel, joint_torque = joint_data
                            self.joint_data_logger.update(it, joint_pos, joint_vel, joint_torque, root_lin_vel, command_vel)

                    # 采集高度（仅跳跃任务）
                    if hasattr(env_, "_height_data_buffer"):
                        if env_._height_data_buffer:
                            height = env_._height_data_buffer[0]
                        else:
                            height = None
                        if height is not None and hasattr(self.joint_data_logger, "update_height"):
                            self.joint_data_logger.update_height(it, height)

                    # 内部奖励（仅用于日志记录）
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None

                    # 处理环境步骤
                    self.alg.process_env_step(rewards, dones, infos)

                    # Book keeping
                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                            
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                            
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        
                        if self.alg.rnd:
                            erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                # 计算返回
                self.alg.compute_returns(critic_obs)
            
            # 更新collection_time
            collection_time = time.time() - collection_start

            # 更新策略
            learn_start = time.time()
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
            learn_time = time.time() - learn_start
            
            stop = time.time()
            total_time = stop - start
            self.current_learning_iteration = it

            # 记录信息并保存检查点
            if self.log_dir is not None:
                # 创建一个包含必要变量的日志数据字典
                log_data = {}
                
                # 处理ep_infos，移除复杂数据结构
                filtered_ep_infos = []
                for ep_info in ep_infos:
                    filtered_info = {}
                    for k, v in ep_info.items():
                        if isinstance(v, (int, float, bool, str)) or (
                            isinstance(v, torch.Tensor) and len(v.shape) <= 1
                        ):
                            filtered_info[k] = v
                        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                            filtered_info[k] = v
                    filtered_ep_infos.append(filtered_info)
                
                # 包含原始基本变量
                local_vars = locals().copy()
                for key in local_vars:
                    if key not in ["self", "obs", "critic_obs", "infos", "actions", "rewards", "dones", "local_vars"]:
                        if isinstance(local_vars[key], (int, float, bool, str, list, torch.Tensor, deque)) or local_vars[key] is None:
                            log_data[key] = local_vars[key]
                
                # 确保包含这些关键变量
                important_keys = ["rewbuffer", "lenbuffer", "it", "collection_time", "learn_time", "total_time",
                                  "mean_value_loss", "mean_surrogate_loss", "mean_entropy"]
                for key in important_keys:
                    if key in local_vars:
                        log_data[key] = local_vars[key]
                    else:
                        print(f"警告: 未找到日志所需的变量: {key}")
                
                log_data["ep_infos"] = filtered_ep_infos
                
                # 添加RND相关变量
                if hasattr(self.alg, "rnd") and self.alg.rnd:
                    for key in ["erewbuffer", "irewbuffer", "mean_rnd_loss"]:
                        if key in local_vars:
                            log_data[key] = local_vars[key]
                
                # 日志记录
                self.log(log_data)
                
                # 保存模型
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # 清除episode infos
            ep_infos.clear()
            
            # 保存代码状态
            if it == start_iter:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # 保存最终模型和图表
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
            # 确保最后一次图表保存
            if self.joint_data_logger is not None:
                self.joint_data_logger.save_plots(self.current_learning_iteration)
                
    def close(self):
        """关闭Runner并清理资源"""
        # 清理关节数据记录器
        if hasattr(self, "joint_data_logger") and self.joint_data_logger is not None:
            self.joint_data_logger.clear()
            
        # 调用父类的close方法
        super().close() 