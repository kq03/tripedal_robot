#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
电机位置控制测试脚本
用途：控制单个电机旋转到指定角度
"""
import time
# import sys
# sys.path.append("tlr_control/library_funcs")  # 确保能找到电机库
from DrEmpower_socketcan import DrEmpower

def motor_position_test(motor_id=1, target_angle=90.0, speed=10.0, mode=1):
    """
    控制电机旋转到指定角度
    参数:
        motor_id: 电机ID
        target_angle: 目标角度(度)
        speed: 最大旋转速度(r/min)
        mode: 控制模式 (0:轨迹跟踪, 1:梯形轨迹, 2:前馈控制)
    """
    try:
        # 初始化CAN连接
        print("初始化电机驱动...")
        motor = DrEmpower(can_channel="can0", enable_reply_state=1, motor_num=motor_id)
        
        # 清除可能存在的错误
        print("清除电机错误...")
        motor.clear_error(id_num=motor_id)
        time.sleep(0.1)
        
        # 设置为闭环控制模式
        print("设置为闭环控制模式...")
        motor.set_mode(id_num=motor_id, mode=2)  # 2表示闭环控制模式
        time.sleep(0.5)
        
        # 读取当前位置
        print("读取当前位置...")
        current_state = motor.get_state(id_num=motor_id)
        if current_state:
            current_angle = current_state[0]
            print(f"当前角度: {current_angle:.2f}度")
        else:
            print("无法读取当前角度，使用0作为初始值")
            current_angle = 0
        
        # 设置目标角度
        print(f"旋转电机到 {target_angle}度，速度: {speed}r/min，模式: {mode}")
        if mode == 1:
            # 梯形轨迹模式需要加速度参数
            param = 50.0  # 加速度 (r/min)/s
        elif mode == 0:
            # 轨迹跟踪模式需要滤波带宽参数
            param = 20.0  # 角度输入滤波带宽
        else:
            # 前馈控制模式需要前馈扭矩
            param = 0.1   # 前馈扭矩(Nm)
        
        # 执行位置控制
        motor.set_angle(id_num=motor_id, angle=target_angle, speed=speed, param=param, mode=mode)
        
        # 等待运动完成
        print("等待运动完成...")
        time.sleep(2)  # 简单等待
        
        # 判断是否到达目标位置
        final_state = motor.get_state(id_num=motor_id)
        if final_state:
            final_angle = final_state[0]
            print(f"最终角度: {final_angle:.2f}度")
            print(f"角度误差: {abs(target_angle - final_angle):.2f}度")
        else:
            print("无法读取最终角度")
        
        print("测试完成!")
        
    except KeyboardInterrupt:
        print("用户中断测试")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='电机位置控制测试')
    parser.add_argument('--id', type=int, default=1, help='电机ID (默认: 1)')
    parser.add_argument('--angle', type=float, default=90.0, help='目标角度 (默认: 90.0)')
    parser.add_argument('--speed', type=float, default=10.0, help='旋转速度 r/min (默认: 10.0)')
    parser.add_argument('--mode', type=int, default=1, choices=[0, 1, 2], 
                        help='控制模式: 0=轨迹跟踪, 1=梯形轨迹, 2=前馈控制 (默认: 1)')
    
    args = parser.parse_args()
    
    print(f"开始测试 - 电机ID: {args.id}, 目标角度: {args.angle}度, 速度: {args.speed}r/min, 模式: {args.mode}")
    motor_position_test(motor_id=args.id, target_angle=args.angle, speed=args.speed, mode=args.mode)