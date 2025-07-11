import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional
import traceback  # 添加导入


class JointDataLogger:
    """关节数据记录器，用于收集和保存机器人关节电机输出的信息（角度、速度、力矩）"""
    
    def __init__(
        self,
        joint_names: List[str],
        log_dir: str,
        max_datapoints: int = 200,
        save_interval: int = 250
    ):
        """初始化关节数据记录器
        
        Args:
            joint_names: 关节名称列表
            log_dir: 日志保存目录
            max_datapoints: 最大数据点数量
            save_interval: 保存间隔（迭代次数）
        """
        # 打印传入的关节名称，确认数据传递正确
        print(f"JointDataLogger接收到的关节名称: {joint_names}")
        
        # 默认关节名称，使用BFS顺序：所有大腿关节在前，所有小腿关节在后
        default_joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
        
        # 优先使用传入的关节名称，如果没有则使用默认值
        self.joint_names = joint_names if joint_names else default_joint_names.copy()
        self.num_joints = len(self.joint_names)
        self.log_dir = log_dir
        self.max_datapoints = max_datapoints
        self.save_interval = save_interval
        
        # 确保主日志目录存在
        self.joint_data_dir = os.path.join(log_dir, "joint_data")
        os.makedirs(self.joint_data_dir, exist_ok=True)
        
        # 直接使用self.joint_names作为描述性名称，不再重新创建
        self.descriptive_names = self.joint_names
        print(f"实际使用的描述性关节名称: {self.descriptive_names}")
        
        # 为每个关节创建单独的子目录
        self.joint_subdirs = {}
        for i in range(self.num_joints):
            joint_subdir = os.path.join(self.joint_data_dir, self.descriptive_names[i])
            os.makedirs(joint_subdir, exist_ok=True)
            self.joint_subdirs[i] = joint_subdir
            print(f"创建关节{i}子目录: {joint_subdir}")
        
        # 初始化数据缓冲区
        self.data_buffer = {
            "iterations": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_torque": []
        }
        
        # 线速度与命令速度数据缓冲区
        self.vel_data_buffer = {
            "iterations": [],
            "root_lin_vel": [],
            "command_vel": []
        }
        
        # 高度数据缓冲区
        self.height_data_buffer = {
            "iterations": [],
            "height": []
        }
        
        # 最近的迭代次数
        self.current_iteration = 0
        
        # 记录上次保存图表的迭代次数
        self.last_saved_iteration = -1
        
        # 记录图表尺寸和样式
        self.figsize = (12, 8)
        self.dpi = 120
        
        print(f"Joint data logger initialized, will save to {self.joint_data_dir}")
        print(f"Joint names: {self.descriptive_names}")
        print(f"Created subdirectories for each joint")
    
    def update(self, iteration: int, joint_pos: torch.Tensor, joint_vel: torch.Tensor, joint_torque: torch.Tensor, root_lin_vel: torch.Tensor = None, command_vel: torch.Tensor = None):
        """更新关节数据，并同步采集线速度数据（如有）
        Args:
            iteration: 当前迭代次数
            joint_pos: 关节位置张量
            joint_vel: 关节速度张量
            joint_torque: 关节力矩张量
            root_lin_vel: 基座线速度张量（可选）
            command_vel: 命令速度张量（可选）
        """
        try:
            self.current_iteration = iteration
            if joint_pos is None or joint_vel is None or joint_torque is None:
                if iteration % 100 == 0:
                    print(f"警告: 迭代 {iteration} 收到无效的关节数据")
                return
            should_debug = iteration % 100 == 0
            self.data_buffer["iterations"].append(iteration)
            self.data_buffer["joint_pos"].append(joint_pos.detach().cpu().numpy())
            self.data_buffer["joint_vel"].append(joint_vel.detach().cpu().numpy())
            self.data_buffer["joint_torque"].append(joint_torque.detach().cpu().numpy())
            # 同步采集线速度数据
            if root_lin_vel is not None and command_vel is not None:
                self.vel_data_buffer["iterations"].append(iteration)
                self.vel_data_buffer["root_lin_vel"].append(root_lin_vel.detach().cpu().numpy())
                self.vel_data_buffer["command_vel"].append(command_vel.detach().cpu().numpy())
                if len(self.vel_data_buffer["iterations"]) > self.max_datapoints:
                    for key in self.vel_data_buffer:
                        self.vel_data_buffer[key].pop(0)
            current_datapoints = len(self.data_buffer["iterations"])
            if current_datapoints > self.max_datapoints:
                for key in self.data_buffer:
                    self.data_buffer[key].pop(0)
                current_datapoints = self.max_datapoints
            is_save_interval = iteration % self.save_interval == 0
            is_new_iteration = iteration != self.last_saved_iteration
            if is_save_interval and is_new_iteration and current_datapoints > 0:
                if should_debug:
                    print(f"迭代 {iteration} - 保存关节/速度数据图表")
                self.save_plots(iteration)
                self.last_saved_iteration = iteration
        except Exception as e:
            if iteration % 100 == 0:
                print(f"更新关节/速度数据时出错: {e}")
                print(traceback.format_exc())
    
    def update_velocity(self, iteration, root_lin_vel, command_vel):
        # 兼容旧接口，但不再负责保存，只做数据缓存
        self.vel_data_buffer["iterations"].append(iteration)
        self.vel_data_buffer["root_lin_vel"].append(root_lin_vel.detach().cpu().numpy())
        self.vel_data_buffer["command_vel"].append(command_vel.detach().cpu().numpy())
        if len(self.vel_data_buffer["iterations"]) > self.max_datapoints:
            for key in self.vel_data_buffer:
                self.vel_data_buffer[key].pop(0)

    def update_height(self, iteration, height):
        """记录基座高度"""
        self.height_data_buffer["iterations"].append(iteration)
        self.height_data_buffer["height"].append(float(height.detach().cpu().numpy()))
        if len(self.height_data_buffer["iterations"]) > self.max_datapoints:
            for key in self.height_data_buffer:
                self.height_data_buffer[key].pop(0) 
                
    def save_plots(self, iteration: Optional[int] = None):
        print(f"[DEBUG] save_plots called at iter {iteration}")
        try:
            if iteration is None:
                iteration = self.current_iteration
            if not self.data_buffer["iterations"] or len(self.data_buffer["iterations"]) == 0:
                if iteration is not None and iteration % 100 == 0:
                    print(f"Warning: No data to save for iteration {iteration}")
                return
            iterations = np.array(self.data_buffer["iterations"])
            joint_pos = np.array(self.data_buffer["joint_pos"])
            joint_vel = np.array(self.data_buffer["joint_vel"])
            joint_torque = np.array(self.data_buffer["joint_torque"])
            self._save_joint_plot(
                iterations, joint_pos, 
                title="Joint Positions", 
                ylabel="Position (rad)", 
                filename=f"joint_positions_combined_{iteration}.png"
            )
            self._save_joint_plot(
                iterations, joint_vel, 
                title="Joint Velocities", 
                ylabel="Velocity (rad/s)", 
                filename=f"joint_velocities_combined_{iteration}.png"
            )
            self._save_joint_plot(
                iterations, joint_torque, 
                title="Joint Torques", 
                ylabel="Torque (N·m)", 
                filename=f"joint_torques_combined_{iteration}.png"
            )
            self._save_individual_joint_plots(
                iterations, joint_pos, joint_vel, joint_torque, iteration
            )
            # 只在这里保存速度跟踪图，频率与关节图完全一致
            self.save_velocity_plots(iteration)
            self.save_height_plots(iteration)
        except Exception as e:
            if iteration is not None and iteration % 100 == 0:
                print(f"Error saving plots: {e}")
                print(traceback.format_exc())
    
    def _save_individual_joint_plots(self, iterations, joint_pos, joint_vel, joint_torque, iteration):
        """为每个关节单独绘制图表
        
        Args:
            iterations: 迭代次数数组
            joint_pos: 关节位置数据
            joint_vel: 关节速度数据
            joint_torque: 关节力矩数据
            iteration: 当前迭代次数
        """
        try:
            # 检查数据形状并修复维度问题
            if joint_pos.ndim == 3:
                joint_pos = joint_pos[:, 0, :]
            if joint_vel.ndim == 3:
                joint_vel = joint_vel[:, 0, :]
            if joint_torque.ndim == 3:
                joint_torque = joint_torque[:, 0, :]
                
            # 创建数据点的索引作为横坐标
            data_indices = np.arange(len(joint_pos))
            
            # 为每个关节单独绘制三种图表
            for i in range(self.num_joints):
                joint_name = self.descriptive_names[i]
                
                # 获取该关节的专用子目录
                joint_dir = self.joint_subdirs[i]
                
                # 颜色映射 - 与合并图表一致，适用于BFS顺序
                joint_colors = {
                    'L1_thigh': 'blue',      # 腿1大腿 
                    'L2_thigh': 'green',     # 腿2大腿
                    'L3_thigh': 'red',       # 腿3大腿
                    'L1_shin': 'lightblue',  # 腿1小腿
                    'L2_shin': 'lightgreen', # 腿2小腿 
                    'L3_shin': 'salmon'      # 腿3小腿
                }
                
                # 设置颜色 - 使用一致的颜色方案
                color = joint_colors.get(joint_name, 'blue')  # 从颜色映射中获取颜色
                
                # 1. 位置图
                plt.figure(figsize=(10, 6), dpi=self.dpi)
                plt.plot(data_indices, joint_pos[:, i], label=joint_name, color=color)
                plt.title(f"{joint_name} Position (Iter: {min(iterations)}-{max(iterations)})")
                plt.xlabel("Data Point Index")
                plt.ylabel("Position (rad)")
                plt.grid(True)
                plt.legend()
                save_path = os.path.join(joint_dir, f"position_{iteration}.png")
                plt.savefig(save_path)
                plt.close()
                
                # 2. 速度图
                plt.figure(figsize=(10, 6), dpi=self.dpi)
                plt.plot(data_indices, joint_vel[:, i], label=joint_name, color=color)
                plt.title(f"{joint_name} Velocity (Iter: {min(iterations)}-{max(iterations)})")
                plt.xlabel("Data Point Index")
                plt.ylabel("Velocity (rad/s)")
                plt.grid(True)
                plt.legend()
                save_path = os.path.join(joint_dir, f"velocity_{iteration}.png")
                plt.savefig(save_path)
                plt.close()
                
                # 3. 力矩图
                plt.figure(figsize=(10, 6), dpi=self.dpi)
                plt.plot(data_indices, joint_torque[:, i], label=joint_name, color=color)
                plt.title(f"{joint_name} Torque (Iter: {min(iterations)}-{max(iterations)})")
                plt.xlabel("Data Point Index")
                plt.ylabel("Torque (N·m)")
                plt.grid(True)
                plt.legend()
                save_path = os.path.join(joint_dir, f"torque_{iteration}.png")
                plt.savefig(save_path)
                plt.close()
                
        except Exception as e:
            print(f"Error saving individual joint plots: {e}")
            print(traceback.format_exc())
        
    def _save_joint_plot(self, iterations, data, title, ylabel, filename):
        """保存特定类型的关节数据图表（所有关节合并在一张图上）
        
        Args:
            iterations: 迭代次数数组(仅用于标题显示，不再用作横坐标)
            data: 关节数据数组
            title: 图表标题
            ylabel: Y轴标签
            filename: 保存的文件名
        """
        try:
            plt.figure(figsize=self.figsize, dpi=self.dpi)
            
            # 检查数据形状并修复维度问题
            # 如果是三维数据 (样本数, 批次维度, 关节数)，则需要压缩或重塑
            if data.ndim == 3:
                # 提取第一个批次的数据并转为二维 (样本数, 关节数)
                data = data[:, 0, :]
            
            # 创建数据点的索引作为横坐标
            data_indices = np.arange(len(data))
            
            # 颜色映射 - 适用于BFS顺序：所有大腿关节在前，所有小腿关节在后
            joint_colors = {
                'L1_thigh': 'blue',      # 腿1大腿
                'L2_thigh': 'green',     # 腿2大腿
                'L3_thigh': 'red',       # 腿3大腿
                'L1_shin': 'lightblue',  # 腿1小腿
                'L2_shin': 'lightgreen', # 腿2小腿 
                'L3_shin': 'salmon'      # 腿3小腿
            }
            
            # 线条样式映射 - 大腿关节使用实线，小腿关节使用虚线，适用于BFS顺序
            joint_styles = {
                'L1_thigh': '-',   # 大腿使用实线
                'L2_thigh': '-',
                'L3_thigh': '-',
                'L1_shin': '--',   # 小腿使用虚线
                'L2_shin': '--',
                'L3_shin': '--'
            }
            
            # 为每个关节绘制一条线
            for i in range(self.num_joints):
                joint_name = self.descriptive_names[i]
                plt.plot(data_indices, data[:, i], label=joint_name, 
                         color=joint_colors[joint_name], 
                         linestyle=joint_styles[joint_name])
            
            plt.title(f"{title} (Iter: {min(iterations)}-{max(iterations)})")
            plt.xlabel("Data Point Index")
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            save_path = os.path.join(self.joint_data_dir, filename)
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error saving combined plot: {e}")
            print(traceback.format_exc())
        
    def save_velocity_plots(self, iteration=None):
        print(f"[DEBUG] save_velocity_plots called, buffer size: {len(self.vel_data_buffer['iterations'])}")
        if not self.vel_data_buffer["iterations"]:
            # 测试：强制画一张假图
            import matplotlib.pyplot as plt
            import numpy as np
            plt.figure()
            plt.plot([0,1,2],[0,1,0])
            plt.title("TEST")
            plt.savefig(os.path.join(self.joint_data_dir, "test_velocity_plot.png"))
            plt.close()
            print("[DEBUG] test_velocity_plot.png saved")
            return
        import matplotlib.pyplot as plt
        import numpy as np
        N = self.max_datapoints
        iterations = np.array(self.vel_data_buffer["iterations"])[-N:]
        root_lin_vel = np.array(self.vel_data_buffer["root_lin_vel"])[-N:]
        command_vel = np.array(self.vel_data_buffer["command_vel"])[-N:]
        data_indices = np.arange(len(root_lin_vel))
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.plot(data_indices, root_lin_vel[:, 0], label="Base Vx", color="blue")
        plt.plot(data_indices, command_vel[:, 0], label="Cmd Vx", color="cyan", linestyle="--")
        plt.plot(data_indices, root_lin_vel[:, 1], label="Base Vy", color="green")
        plt.plot(data_indices, command_vel[:, 1], label="Cmd Vy", color="lime", linestyle="--")
        plt.title(f"Base Linear Velocity Tracking (Iter: {min(iterations)}-{max(iterations)})")
        plt.xlabel("Data Point Index")
        plt.ylabel("Velocity (m/s)")
        plt.grid(True)
        plt.legend()
        save_path = os.path.join(self.joint_data_dir, f"velocity_tracking_{iteration if iteration is not None else self.current_iteration}.png")
        plt.savefig(save_path)
        plt.close()

    def save_height_plots(self, iteration=None):
        """保存基座高度曲线图"""
        if not self.height_data_buffer["iterations"]:
            return
        import matplotlib.pyplot as plt
        import numpy as np
        iterations = np.array(self.height_data_buffer["iterations"])
        height = np.array(self.height_data_buffer["height"])
        data_indices = np.arange(len(height))
        plt.figure(figsize=(10, 6), dpi=self.dpi)
        plt.plot(data_indices, height, label="Base Height (z)", color="purple")
        plt.title(f"Base Height Curve (Iter: {min(iterations)}-{max(iterations)})")
        plt.xlabel("Data Point Index")
        plt.ylabel("Height (m)")
        plt.grid(True)
        plt.legend()
        save_path = os.path.join(self.joint_data_dir, f"base_height_{iteration if iteration is not None else self.current_iteration}.png")
        plt.savefig(save_path)
        plt.close()
    
    def clear(self):
        """清除数据缓冲区"""
        for key in self.data_buffer:
            self.data_buffer[key].clear()
        for key in self.vel_data_buffer:
            self.vel_data_buffer[key].clear()
        for key in self.height_data_buffer:
            self.height_data_buffer[key].clear() 

   