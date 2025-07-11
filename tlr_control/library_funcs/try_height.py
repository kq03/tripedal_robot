#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STP23高度测距传感器测试程序
简单直观地显示传感器数据
"""

import serial
import time

# 串口参数设置
PORT = '/dev/wheeltec_STP23'  # 串口设备路径
BAUDRATE = 921600  # 波特率

def parse_data(data):
    """解析传感器数据包
    
    Args:
        data: 原始字节数据列表
        
    Returns:
        list: 测量点列表，每个点为(距离, 强度)元组
    """
    # 检查起始符
    if not data or data[0] != 0x54:
        return []
    
    measurements = []
    
    # 从第7个字节开始解析每个测量数据点
    for i in range(6, len(data)-5, 3):
        if i + 2 < len(data):  # 确保不会越界
            # 解析高字节
            distance1 = data[i + 1]
            distance1 = hex(distance1)
            interval1 = distance1[2:].upper()
            str1 = str(interval1)
            
            # 解析低字节
            distance2 = data[i]
            distance2 = hex(distance2)
            interval2 = distance2[2:].upper()
            str2 = str(interval2)
            
            # 拼接高低字节
            string1 = str1 + str2
            
            # 转换为整数距离值（毫米）
            distance = int(string1, 16)
            
            # 信号强度
            intensity = data[i + 2]
            
            measurements.append((distance, intensity))
    
    return measurements

def test_height_sensor():
    """测试高度传感器功能"""
    print("正在连接STP23高度传感器...")
    
    try:
        # 打开串口连接
        with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
            print(f"成功连接到传感器: {PORT}，波特率: {BAUDRATE}")
            print("等待设备准备...")
            time.sleep(2)  # 等待设备准备
            
            # 清空缓冲区
            ser.reset_input_buffer()
            
            print("开始读取数据(按Ctrl+C退出)...")
            print("=" * 50)
            
            # 统计变量
            last_time = time.time()
            frame_count = 0
            distance_sum = 0
            valid_count = 0
            
            while True:
                if ser.in_waiting > 0:
                    # 读取数据
                    raw_data = ser.read(ser.in_waiting)
                    data = list(bytearray(raw_data))
                    
                    # 解析数据
                    measurements = parse_data(data)
                    frame_count += 1
                    
                    if measurements:
                        # 计算有效距离测量的平均值
                        valid_points = [(d, i) for d, i in measurements if i > 10 and 0 < d < 2000]
                        
                        if valid_points:
                            current_distance = sum(d for d, _ in valid_points) / len(valid_points)
                            distance_sum += current_distance
                            valid_count += 1
                            
                            # 将毫米转换为米显示
                            current_distance_m = current_distance / 1000.0
                            
                            # 显示当前距离
                            print(f"当前高度: {current_distance_m:.3f} 米 (原始数据: {current_distance:.1f} 毫米)")
                            print(f"测量点数量: {len(measurements)}, 有效点: {len(valid_points)}")
                            
                            # 显示一些原始测量点
                            if len(valid_points) > 0:
                                print("有效测量点示例:")
                                for idx, (dist, inten) in enumerate(valid_points[:3]):
                                    print(f"  点 {idx+1}: 距离 = {dist} 毫米, 强度 = {inten}")
                    
                    # 计算数据速率
                    current_time = time.time()
                    if current_time - last_time >= 1.0:
                        elapsed = current_time - last_time
                        # 计算平均距离（如果有效）
                        avg_distance = (distance_sum / valid_count) / 1000.0 if valid_count > 0 else 0
                        
                        print("-" * 50)
                        print(f"数据帧率: {frame_count/elapsed:.1f} 帧/秒")
                        print(f"平均高度: {avg_distance:.3f} 米")
                        print("-" * 50)
                        
                        # 重置统计
                        last_time = current_time
                        frame_count = 0
                        distance_sum = 0
                        valid_count = 0
                
                # 短暂等待，避免CPU使用率过高
                time.sleep(0.01)
                
    except serial.SerialException as e:
        print(f"串口错误: {e}")
        print("请检查传感器连接和端口设置")
        return False
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"发生错误: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print(" STP23高度测距传感器测试程序")
    print("=" * 50)
    
    # 允许用户修改端口
    user_port = input(f"请输入传感器端口 (直接回车使用默认值 {PORT}): ")
    if user_port.strip():
        PORT = user_port
    
    print(f"使用端口: {PORT}")
    test_result = test_height_sensor()
    
    if test_result:
        print("\n测试完成")
    else:
        print("\n测试失败")
        print("如果遇到权限问题，请尝试执行:")
        print(f"  sudo chmod 666 {PORT}") 