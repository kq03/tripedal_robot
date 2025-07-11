#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU测试程序 - 用于验证Yesense IMU连接和数据读取
"""

import sys
import time
import numpy as np
from tlr_control.src.sensor import SensorInterface

def test_imu(port='/dev/ttyUSB0', baud=460800):
    """
    测试IMU传感器连接和数据读取
    
    Args:
        port: IMU串口设备路径，默认为/dev/ttyUSB0
        baud: 波特率，默认为460800
    """
    print(f"开始测试IMU，端口={port}，波特率={baud}")
    
    # 创建传感器接口
    sensors = SensorInterface(imu_port=port, imu_baud=baud)
    
    # 记录测试开始时间
    start_time = time.time()
    data_count = 0
    
    try:
        while True:
            # 读取IMU数据
            quaternion, angular_vel = sensors.read_imu()
            linear_acc = sensors.get_linear_acceleration()
            
            # 数据计数
            data_count += 1
            
            # 计算运行时间
            elapsed = time.time() - start_time
            
            # 打印数据
            print("\n" + "="*50)
            print(f"运行时间: {elapsed:.2f}秒 | 数据帧数: {data_count}")
            print("-"*50)
            print(f"四元数: q0={quaternion[0]:.4f}, q1={quaternion[1]:.4f}, "
                  f"q2={quaternion[2]:.4f}, q3={quaternion[3]:.4f}")
            print(f"角速度(rad/s): x={angular_vel[0]:.4f}, y={angular_vel[1]:.4f}, "
                  f"z={angular_vel[2]:.4f}")
            print(f"加速度(m/s²): x={linear_acc[0]:.4f}, y={linear_acc[1]:.4f}, "
                  f"z={linear_acc[2]:.4f}")
            print("="*50)
            
            # 控制输出频率
            time.sleep(0.1)  # 100ms间隔，即10Hz显示频率
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n\n程序出错: {e}")
    finally:
        # 关闭传感器连接
        sensors.close()
        print(f"\nIMU测试完成，共收到{data_count}帧数据，运行时间{time.time()-start_time:.2f}秒")

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) > 1:
        port = sys.argv[1]
        baud = int(sys.argv[2]) if len(sys.argv) > 2 else 460800
        test_imu(port, baud)
    else:
        test_imu() 