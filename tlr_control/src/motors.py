import sys
import time
import numpy as np
import os

# 添加正确的路径到library_funcs目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
lib_path = os.path.join(script_dir, "..", "library_funcs")  # 向上一级然后进入library_funcs
sys.path.append(os.path.abspath(lib_path))

from DrEmpower_socketcan import DrEmpower

class MotorInterface:
    def __init__(self, default_positions, can_channel="can0", motor_ids=[3, 5, 1, 4, 6, 2], action_scale=-1.0, shin_gear_ratio=1.159):
        """初始化电机控制接口
        
        Args:
            default_positions: 默认关节位置列表(6个关节) - 弧度制，与仿真训练一致
            can_channel: CAN总线通道
            motor_ids: 电机ID列表，顺序对应[L1_thigh, L2_thigh, L3_thigh, L1_shin, L2_shin, L3_shin]
            action_scale: 大腿电机的动作缩放系数
            shin_gear_ratio: 小腿电机的减速比（如果小腿电机有减速器）
        """
        # 初始化电机控制器
        self.motor_ids = motor_ids
        self.can_channel = can_channel
        self.motor_driver = DrEmpower(can_channel=can_channel, enable_reply_state=1, motor_num=max(motor_ids))
        
        # CAN通信控制
        self.last_send_time = 0
        self.min_send_interval = 0.005  # 最小发送间隔5ms，避免缓冲区溢出
        self.can_error_count = 0
        self.max_can_errors = 10
        
        self.setup_motors()
        
        # 存储默认位置（弧度制，与仿真训练保持一致）
        self.default_positions = default_positions
        
        # 为每个电机设置独立的缩放系数
        # 前3个是大腿电机，后3个是小腿电机
        self.action_scales = [
            action_scale,                    # L1_thigh
            action_scale,                    # L2_thigh  
            action_scale,                    # L3_thigh
            shin_gear_ratio,  # L1_shin (考虑减速比)
            shin_gear_ratio,  # L2_shin (考虑减速比)
            shin_gear_ratio   # L3_shin (考虑减速比)
        ]
        
        print(f"电机缩放系数设置:")
        joint_names = ["L1_thigh", "L2_thigh", "L3_thigh", "L1_shin", "L2_shin", "L3_shin"]
        for i, (name, scale) in enumerate(zip(joint_names, self.action_scales)):
            print(f"  {name}: {scale}")
        
        print(f"默认位置 (弧度制): {[f'{pos:.3f}' for pos in self.default_positions]}")
        print(f"默认位置 (角度制): {[f'{pos * 180.0 / np.pi:.1f}°' for pos in self.default_positions]}")
        print(f"CAN通信设置: 最小发送间隔={self.min_send_interval*1000:.1f}ms")
        
        # 存储历史数据（弧度制，与仿真训练保持一致）
        self.prev_positions = [0.0] * len(motor_ids)
        self.prev_velocities = [0.0] * len(motor_ids)
        self.prev_actions = [0.0] * len(motor_ids)
        self.actions = [0.0] * len(motor_ids)
    
    def setup_motors(self):
        """设置电机初始状态"""
        print(f"初始化{len(self.motor_ids)}个电机...")
        
        for motor_id in self.motor_ids:
            # 清除可能的错误
            self.motor_driver.clear_error(id_num=motor_id)
            time.sleep(0.05)
            
            # 设置为闭环控制模式
            self.motor_driver.set_mode(id_num=motor_id, mode=2)  # 2表示闭环控制模式
            time.sleep(0.05)
    
    def read_joint_states(self):
        """读取关节状态，返回位置、速度和力矩
        
        Returns:
            positions: 关节位置列表 (弧度制，与仿真训练一致)
            velocities: 关节速度列表 (弧度/秒)
            torques: 关节力矩列表 (N·m)
        """
        positions = []
        velocities = []
        torques = []
        
        for motor_id in self.motor_ids:
            # 使用get_state获取位置和速度
            state = self.motor_driver.get_state(id_num=motor_id)
            if state:
                pos_deg, vel_rpm = state
                
                # 单位转换: 电机硬件 -> 仿真训练单位
                # 角度: 度 -> 弧度 (电机驱动器输出角度制，仿真训练使用弧度制)
                pos_rad = pos_deg * np.pi / 180.0
                
                # 速度: 转/分 -> 弧度/秒 (电机驱动器输出rpm，仿真训练使用rad/s)
                vel_rad_per_sec = vel_rpm * (2 * np.pi / 60.0)
                
                positions.append(pos_rad)
                velocities.append(vel_rad_per_sec)
                
                # 保存当前状态为历史状态
                idx = self.motor_ids.index(motor_id)
                self.prev_positions[idx] = pos_rad
                self.prev_velocities[idx] = vel_rad_per_sec
            else:
                # 如果读取失败，使用上次的值
                idx = self.motor_ids.index(motor_id)
                positions.append(self.prev_positions[idx])
                velocities.append(self.prev_velocities[idx])
            
            # 读取电流(扭矩)
            current_data = self.motor_driver.read_property(
                id_num=motor_id, 
                property='axis0.motor.current_control.Iq_measured'
            )
            
            if current_data and self.motor_driver.READ_FLAG == 1:
                torques.append(current_data)
            else:
                torques.append(0.0)
        
        return positions, velocities, torques
    
    def set_positions(self, action_commands):
        """设置电机位置目标
        
        Args:
            action_commands: 目标位置列表 (弧度制，与仿真训练一致)
            
        Returns:
            motor_debug_info: 电机控制的调试信息字典
        """
        # 控制CAN发送频率，避免缓冲区溢出
        current_time = time.time()
        if current_time - self.last_send_time < self.min_send_interval:
            # 如果发送太频繁，稍微等待
            time.sleep(self.min_send_interval - (current_time - self.last_send_time))
            current_time = time.time()
        
        # 保存上一次动作
        self.prev_actions = self.actions.copy()
        self.actions = action_commands.copy()
        
        # 将动作转换为实际关节角度（弧度制）
        target_positions_rad = [a * scale + default for a, scale, default in 
                               zip(action_commands, self.action_scales, self.default_positions)]
        
        # 准备调试信息
        target_degrees = []
        can_success_count = 0
        
        # 控制每个电机到目标位置
        for i, motor_id in enumerate(self.motor_ids):
            if i < len(target_positions_rad):
                # 单位转换: 仿真训练 -> 电机硬件
                # 弧度 -> 角度 (仿真训练使用弧度制，电机驱动器需要角度制)
                target_deg = target_positions_rad[i] * 180.0 / np.pi
                target_degrees.append(target_deg)
                
                try:
                    # 使用梯形轨迹模式
                    result = self.motor_driver.set_angle(
                        id_num=motor_id,
                        angle=float(target_deg),  # 目标角度（角度制）
                        speed=20.0,               # 速度 r/min
                        param=50.0,               # 加速度 (r/min)/s
                        mode=1                    # 梯形轨迹模式
                    )
                    
                    if result is not False:  # 成功发送
                        can_success_count += 1
                        self.can_error_count = max(0, self.can_error_count - 1)  # 错误计数减少
                    else:
                        self.can_error_count += 1
                        
                except Exception as e:
                    print(f"电机{motor_id}控制错误: {e}")
                    self.can_error_count += 1
        
        # 更新发送时间
        self.last_send_time = current_time
        
        # 检查CAN通信健康状况
        if self.can_error_count > self.max_can_errors:
            print(f"⚠️  CAN通信错误过多({self.can_error_count})，尝试重置...")
            self.reset_can_communication()
        
        # 返回调试信息
        debug_info = {
            'target_positions_rad': target_positions_rad,
            'target_degrees': target_degrees,
            'action_scales': self.action_scales,
            'default_positions': self.default_positions,
            'action_commands': action_commands,
            'can_success_count': can_success_count,
            'can_error_count': self.can_error_count
        }
        
        return debug_info
    
    def reset_can_communication(self):
        """重置CAN通信"""
        try:
            print("正在重置CAN通信...")
            # 重新初始化CAN接口
            self.motor_driver.init_can_interface(channel=self.can_channel, baudrate=1000000)
            self.can_error_count = 0
            time.sleep(0.1)  # 短暂等待
            print("CAN通信重置完成")
        except Exception as e:
            print(f"CAN通信重置失败: {e}")
    
    def emergency_stop(self):
        """紧急停止所有电机"""
        for motor_id in self.motor_ids:
            self.motor_driver.estop(id_num=motor_id)
    
    def zero_positions(self):
        """将当前位置设为零点"""
        for motor_id in self.motor_ids:
            self.motor_driver.set_zero_position(id_num=motor_id)
    
    def get_errors(self):
        """读取所有电机错误状态"""
        errors = {}
        for motor_id in self.motor_ids:
            errors[motor_id] = self.motor_driver.dump_error(id_num=motor_id)
        return errors