#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print("🔍 逐步调试主程序")
print("=" * 50)

import sys
import os

# 步骤1: 检查Python路径和工作目录
print(f"1. 当前工作目录: {os.getcwd()}")
print(f"2. Python路径: {sys.path[:3]}...")

# 步骤2: 检查模型文件
print("\n3. 检查模型文件...")
model_paths = [
    "../models/model_walk.onnx",
    "models/model_walk.onnx", 
    "tlr_control/models/model_walk.onnx"
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        print(f"   ✅ 找到模型: {path}")
        break

if not model_path:
    print("   ❌ 未找到模型文件")
    print("   💡 请检查模型文件位置")
    sys.exit(1)

# 步骤3: 测试模块导入
print("\n4. 测试模块导入...")

print("   导入PolicyModel...")
try:
    from model import PolicyModel
    print("   ✅ PolicyModel导入成功")
except Exception as e:
    print(f"   ❌ PolicyModel导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("   导入MotorInterface...")
try:
    from motors import MotorInterface
    print("   ✅ MotorInterface导入成功")
except Exception as e:
    print(f"   ❌ MotorInterface导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("   导入SensorInterface...")
try:
    from sensor import SensorInterface
    print("   ✅ SensorInterface导入成功")
except Exception as e:
    print(f"   ❌ SensorInterface导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 步骤4: 测试电机初始化
print("\n5. 测试电机初始化...")
try:
    print("   创建MotorInterface对象...")
    motors = MotorInterface(default_positions=[0, 0, 0, 0, 0, 0])
    print("   ✅ 电机接口创建成功")
except Exception as e:
    print(f"   ❌ 电机接口创建失败: {e}")
    import traceback
    traceback.print_exc()

# 步骤5: 测试传感器初始化
print("\n6. 测试传感器初始化...")
try:
    print("   创建SensorInterface对象...")
    sensors = SensorInterface(
        imu_port='/dev/ttyACM1',
        imu_baud=460800,
        height_port='/dev/ttyACM0', 
        height_baud=921600
    )
    print("   ✅ 传感器接口创建成功")
except Exception as e:
    print(f"   ❌ 传感器接口创建失败: {e}")
    import traceback
    traceback.print_exc()

# 步骤6: 测试模型初始化
print("\n7. 测试模型初始化...")
try:
    print("   创建PolicyModel对象...")
    policy = PolicyModel(model_path, device="cuda", fp16=True)
    print("   ✅ 策略模型创建成功")
except Exception as e:
    print(f"   ❌ 策略模型创建失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("🎉 所有组件测试完成!")
print("如果到这里都成功了，说明初始化没问题")
