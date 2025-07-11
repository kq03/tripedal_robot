#!/usr/bin/env python3
"""
将Isaac Lab训练的.pt模型文件转换为ONNX格式

使用方法:
python convert_pt_to_onnx.py <input_pt_file> [output_dir] [--verbose]

示例:
python convert_pt_to_onnx.py model.pt ./models/ --verbose
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# 导入Isaac Lab的导出器
try:
    from source.isaaclab_rl.isaaclab_rl.rsl_rl.exporter import export_policy_as_onnx
    print("✅ 成功导入Isaac Lab导出器")
except ImportError as e:
    print(f"❌ 无法导入Isaac Lab导出器: {e}")
    print("请确保您在Isaac Lab环境中运行此脚本")
    sys.exit(1)

def load_model_checkpoint(pt_file_path):
    """加载.pt模型文件"""
    try:
        print(f"📂 加载模型文件: {pt_file_path}")
        checkpoint = torch.load(pt_file_path, map_location='cpu')
        print(f"✅ 模型文件加载成功")
        
        # 打印checkpoint的结构信息
        print(f"📋 Checkpoint包含的键: {list(checkpoint.keys())}")
        
        return checkpoint
    except Exception as e:
        print(f"❌ 加载模型文件失败: {e}")
        return None

def extract_policy_components(checkpoint):
    """从checkpoint中提取策略组件"""
    try:
        # 常见的键名模式
        possible_keys = [
            'model',           # 通常的模型键
            'actor_critic',    # Isaac Lab常用
            'policy',          # 策略键
            'state_dict',      # 状态字典
        ]
        
        model_state = None
        for key in possible_keys:
            if key in checkpoint:
                model_state = checkpoint[key]
                print(f"✅ 找到模型状态: {key}")
                break
        
        if model_state is None:
            print("❌ 未找到模型状态")
            return None, None
        
        # 查找normalizer
        normalizer = None
        if 'obs_normalizer' in checkpoint:
            normalizer = checkpoint['obs_normalizer']
            print("✅ 找到观测归一化器")
        elif 'normalizer' in checkpoint:
            normalizer = checkpoint['normalizer']
            print("✅ 找到归一化器")
        else:
            print("⚠️ 未找到归一化器，将使用Identity")
        
        return model_state, normalizer
        
    except Exception as e:
        print(f"❌ 提取策略组件失败: {e}")
        return None, None

def create_dummy_actor_critic(state_dict):
    """根据state_dict创建一个虚拟的actor_critic对象"""
    
    class DummyActorCritic:
        def __init__(self, state_dict):
            self.actor = torch.nn.Sequential()
            self.is_recurrent = False
            
            # 尝试重建网络结构
            self._build_network_from_state_dict(state_dict)
            
        def _build_network_from_state_dict(self, state_dict):
            """从state_dict重建网络"""
            # 这里需要根据您的具体模型结构来调整
            # 假设是标准的全连接网络
            
            layers = []
            weight_keys = [k for k in state_dict.keys() if 'weight' in k and 'actor' in k]
            weight_keys.sort()
            
            print(f"📋 找到权重层: {weight_keys}")
            
            for i, weight_key in enumerate(weight_keys):
                weight = state_dict[weight_key]
                bias_key = weight_key.replace('weight', 'bias')
                
                if len(weight.shape) == 2:  # 全连接层
                    in_features, out_features = weight.shape[1], weight.shape[0]
                    layer = torch.nn.Linear(in_features, out_features)
                    
                    # 加载权重和偏置
                    layer.weight.data = weight
                    if bias_key in state_dict:
                        layer.bias.data = state_dict[bias_key]
                    
                    layers.append(layer)
                    
                    # 添加激活函数（除了最后一层）
                    if i < len(weight_keys) - 1:
                        layers.append(torch.nn.ReLU())
            
            if layers:
                self.actor = torch.nn.Sequential(*layers)
                print(f"✅ 重建网络结构: {len(layers)}层")
            else:
                print("❌ 无法重建网络结构")
    
    return DummyActorCritic(state_dict)

def convert_pt_to_onnx(pt_file_path, output_dir="./", filename=None, verbose=False):
    """将.pt文件转换为ONNX格式"""
    
    # 1. 加载checkpoint
    checkpoint = load_model_checkpoint(pt_file_path)
    if checkpoint is None:
        return False
    
    # 2. 提取组件
    model_state, normalizer = extract_policy_components(checkpoint)
    if model_state is None:
        return False
    
    # 3. 创建actor_critic对象
    print("🔧 重建模型结构...")
    actor_critic = create_dummy_actor_critic(model_state)
    
    # 4. 确定输出文件名
    if filename is None:
        pt_filename = Path(pt_file_path).stem
        filename = f"{pt_filename}.onnx"
    
    # 5. 执行转换
    try:
        print(f"🔄 开始转换为ONNX格式...")
        print(f"📁 输出目录: {output_dir}")
        print(f"📄 输出文件: {filename}")
        
        export_policy_as_onnx(
            actor_critic=actor_critic,
            path=output_dir,
            normalizer=normalizer,
            filename=filename,
            verbose=verbose
        )
        
        output_path = os.path.join(output_dir, filename)
        print(f"✅ 转换成功! ONNX文件已保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将Isaac Lab .pt模型转换为ONNX格式")
    parser.add_argument("input_file", help="输入的.pt文件路径")
    parser.add_argument("output_dir", nargs='?', default="./", help="输出目录 (默认: ./)")
    parser.add_argument("--filename", help="输出ONNX文件名 (默认: 基于输入文件名)")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 输入文件不存在: {args.input_file}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行转换
    print("🚀 开始模型转换...")
    success = convert_pt_to_onnx(
        pt_file_path=args.input_file,
        output_dir=args.output_dir,
        filename=args.filename,
        verbose=args.verbose
    )
    
    if success:
        print("🎉 转换完成!")
    else:
        print("💥 转换失败!")
        sys.exit(1)

if __name__ == "__main__":
    main() 