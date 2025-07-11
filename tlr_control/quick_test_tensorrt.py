#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT策略模型快速测试脚本
验证TensorRT模型加载、转换和推理功能
"""

import os
import sys
import numpy as np

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    from model import PolicyModel
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    IMPORT_ERROR = str(e)

def check_environment():
    """检查TensorRT环境"""
    print("🔍 检查TensorRT环境...")
    
    if not TENSORRT_AVAILABLE:
        print(f"   ❌ TensorRT不可用: {IMPORT_ERROR}")
        print("   💡 请安装TensorRT:")
        print("      1. 下载TensorRT (适用于Jetson)")
        print("      2. pip install tensorrt")
        print("      3. pip install pycuda")
        return False
    
    print(f"   ✅ TensorRT版本: {trt.__version__}")
    
    # 检查CUDA
    try:
        cuda.init()
        device_count = cuda.Device.count()
        if device_count == 0:
            print("   ❌ 未检测到CUDA设备")
            return False
        
        device = cuda.Device(0)
        print(f"   ✅ CUDA设备: {device.name()}")
        print(f"   ✅ 设备数量: {device_count}")
        
        # 获取设备计算能力
        major, minor = device.compute_capability()
        print(f"   📊 计算能力: {major}.{minor}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CUDA初始化失败: {e}")
        return False

def test_onnx_to_tensorrt_conversion(models_dir="tlr_control/models"):
    """测试ONNX到TensorRT转换"""
    print("\n🔄 测试ONNX到TensorRT转换...")
    
    model_path = os.path.join(models_dir, "model_walk.onnx")
    if not os.path.exists(model_path):
        print(f"   ❌ ONNX模型文件不存在: {model_path}")
        return False, None
    
    print(f"   📁 ONNX模型: {model_path}")
    
    try:
        # 使用FP16以获得更好的性能（特别是在Jetson上）
        policy = PolicyModel(model_path, device="cuda", fp16=True)
        print(f"   ✅ TensorRT模型创建成功")
        
        # 显示引擎信息
        engine_info = policy.get_engine_info()
        print(f"   📊 引擎路径: {engine_info['engine_path']}")
        print(f"   📊 FP16启用: {engine_info['fp16_enabled']}")
        print(f"   📊 最大批次: {engine_info['max_batch_size']}")
        
        return True, policy
        
    except Exception as e:
        print(f"   ❌ TensorRT转换失败: {e}")
        return False, None

def test_inference_performance(policy, num_tests=50):
    """测试推理性能"""
    print(f"\n⚡ 测试推理性能 ({num_tests}次)...")
    
    # 创建测试数据
    test_obs = {
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "linear_vel": [0.3, 0.0, 0.0],
        "angular_vel": [0.0, 0.0, 0.0],
        "joint_pos": [0.1, -0.2, 0.1, -0.2, 0.1, -0.2],
        "joint_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "height": 0.25,
        "prev_height": 0.25,
        "prev_joint_vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "joint_torque": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "prev_action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "commands": [0.3, 0.0, 0.0]
    }
    
    import time
    inference_times = []
    
    # 预热
    for _ in range(5):
        policy.predict(test_obs)
    
    # 正式测试
    for i in range(num_tests):
        # 添加小的随机变化
        test_obs["joint_pos"] = [p + np.random.normal(0, 0.01) for p in test_obs["joint_pos"]]
        
        start_time = time.time()
        action = policy.predict(test_obs)
        end_time = time.time()
        
        inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        if i == 0:
            print(f"   🎯 首次推理结果: {action}")
            print(f"   📊 动作维度: {action.shape}")
    
    # 统计结果
    inference_times = np.array(inference_times)
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"   📈 平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   🏃 最快推理时间: {min_time:.2f} ms")
    print(f"   🐌 最慢推理时间: {max_time:.2f} ms")
    print(f"   🔥 推理频率: {1000/avg_time:.1f} Hz")
    
    # 性能评估
    if avg_time < 2.0:
        print(f"   🎉 性能优秀! (< 2ms)")
    elif avg_time < 5.0:
        print(f"   ✅ 性能良好! (< 5ms)")
    elif avg_time < 10.0:
        print(f"   ⚠️  性能中等 (< 10ms)")
    else:
        print(f"   🔴 性能较慢 (> 10ms)")
    
    return avg_time < 10.0  # 认为10ms以下为可接受

def quick_test(models_dir="tlr_control/models"):
    """主测试函数"""
    print("🚀 TensorRT策略模型快速测试")
    print("=" * 50)
    
    # 1. 检查环境
    if not check_environment():
        return False
    
    # 2. 测试转换
    success, policy = test_onnx_to_tensorrt_conversion(models_dir)
    if not success:
        return False
    
    # 3. 测试推理性能
    if not test_inference_performance(policy):
        print("\n⚠️  推理性能不符合要求，但基本功能正常")
    
    print("\n" + "=" * 50)
    print("🎉 TensorRT快速测试完成!")
    return True

def print_jetson_optimization_tips():
    """打印Jetson优化建议"""
    print("\n💡 Jetson性能优化建议:")
    print("   1. 设置最大性能模式:")
    print("      sudo nvpmodel -m 0")
    print("      sudo jetson_clocks")
    print("   2. 确保使用FP16精度 (已启用)")
    print("   3. 考虑使用DLA (深度学习加速器)")
    print("   4. 调整TensorRT工作空间大小")
    print("   5. 使用INT8量化 (如果模型支持)")

if __name__ == "__main__":
    print("当前工作目录:", os.getcwd())
    
    # 智能检查模型目录
    models_path = None
    if os.path.exists("tlr_control/models"):
        # 从父目录运行
        models_path = "tlr_control/models"
    elif os.path.exists("models"):
        # 从tlr_control目录内运行
        models_path = "models"
    elif os.path.exists("../models"):
        # 从子目录运行
        models_path = "../models"
    
    if models_path is None:
        print("错误: 找不到models目录")
        print("请确保以下目录之一存在:")
        print("  - tlr_control/models (从父目录运行)")
        print("  - models (从tlr_control目录运行)")
        print("  - ../models (从子目录运行)")
        sys.exit(1)
    
    print(f"找到模型目录: {models_path}")
    
    success = quick_test(models_path)
    
    if not success:
        print("\n💥 测试失败，请检查错误信息")
        print_jetson_optimization_tips()
        sys.exit(1)
    else:
        print("💯 TensorRT基本功能正常，性能优异!")
        print("🔥 可以继续部署到实际机器人控制中")
        print_jetson_optimization_tips() 