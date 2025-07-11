#!/usr/bin/env python3
"""
简单的.pt到ONNX转换脚本
适用于大多数Isaac Lab训练的模型

使用方法:
python convert_simple.py model.pt

这个脚本会:
1. 加载.pt文件
2. 分析模型结构
3. 转换为ONNX格式
"""

import os
import sys
import torch
import torch.onnx

def inspect_checkpoint(pt_file):
    """检查checkpoint文件的结构"""
    print(f"🔍 检查文件: {pt_file}")
    
    try:
        checkpoint = torch.load(pt_file, map_location='cpu', weights_only=False)
        print("✅ 文件加载成功")
        
        print(f"\n📋 Checkpoint结构:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict (包含 {len(checkpoint[key])} 个键)")
                # 显示前几个子键
                sub_keys = list(checkpoint[key].keys())[:5]
                for sub_key in sub_keys:
                    print(f"    - {sub_key}")
                if len(checkpoint[key]) > 5:
                    print(f"    ... 还有 {len(checkpoint[key])-5} 个键")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: Tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        return None

def extract_model_info(checkpoint):
    """提取模型信息"""
    
    # 查找actor网络权重
    actor_weights = {}
    
    # 尝试不同的可能路径
    possible_paths = [
        checkpoint,  # 直接在根目录
        checkpoint.get('model', {}),  # 在model键下
        checkpoint.get('model_state_dict', {}),  # 在model_state_dict键下 - 新增
        checkpoint.get('actor_critic', {}),  # 在actor_critic键下
        checkpoint.get('policy', {}),  # 在policy键下
        checkpoint.get('state_dict', {}),  # 在state_dict键下
    ]
    
    for state_dict in possible_paths:
        if not isinstance(state_dict, dict):
            continue
            
        # 查找actor相关的权重
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and 'actor' in key.lower() and ('weight' in key or 'bias' in key):
                actor_weights[key] = value
                print(f"🔍 找到权重: {key} {value.shape}")
    
    if not actor_weights:
        print("❌ 未找到actor权重")
        print("📋 尝试显示所有包含'actor'的键:")
        for state_dict in possible_paths:
            if not isinstance(state_dict, dict):
                continue
            for key in state_dict.keys():
                if 'actor' in key.lower():
                    print(f"  发现: {key}")
        return None
    
    print(f"✅ 找到 {len([k for k in actor_weights.keys() if 'weight' in k])} 个actor层")
    
    # 分析网络结构
    layer_sizes = []
    weight_keys = sorted([k for k in actor_weights.keys() if 'weight' in k])
    
    for key in weight_keys:
        weight = actor_weights[key]
        if len(weight.shape) == 2:
            in_size, out_size = weight.shape[1], weight.shape[0]
            layer_sizes.append((in_size, out_size))
            print(f"  层 {key}: {in_size} -> {out_size}")
    
    if not layer_sizes:
        print("❌ 无法确定网络结构")
        return None
    
    return {
        'layer_sizes': layer_sizes,
        'weights': actor_weights,
        'input_size': layer_sizes[0][0],
        'output_size': layer_sizes[-1][1]
    }

def create_simple_model(model_info):
    """创建简单的PyTorch模型"""
    
    layers = []
    layer_sizes = model_info['layer_sizes']
    
    for i, (in_size, out_size) in enumerate(layer_sizes):
        # 创建线性层
        layer = torch.nn.Linear(in_size, out_size)
        layers.append(layer)
        
        # 添加激活函数（最后一层除外）
        if i < len(layer_sizes) - 1:
            layers.append(torch.nn.ReLU())
    
    model = torch.nn.Sequential(*layers)
    
    # 加载权重 - 改进的匹配逻辑
    weights = model_info['weights']
    weight_keys = sorted([k for k in weights.keys() if 'weight' in k])
    
    print(f"🔧 开始加载权重，找到的权重层: {weight_keys}")
    
    with torch.no_grad():
        linear_layer_idx = 0
        for i, (layer_name, layer) in enumerate(model.named_children()):
            if isinstance(layer, torch.nn.Linear):
                if linear_layer_idx < len(weight_keys):
                    # 直接按顺序匹配权重
                    weight_key = weight_keys[linear_layer_idx]
                    bias_key = weight_key.replace('weight', 'bias')
                    
                    if weight_key in weights:
                        layer.weight.data = weights[weight_key]
                        print(f"✅ 加载权重: {weight_key} -> 层{linear_layer_idx}")
                    
                    if bias_key in weights:
                        layer.bias.data = weights[bias_key]
                        print(f"✅ 加载偏置: {bias_key} -> 层{linear_layer_idx}")
                    else:
                        print(f"⚠️ 未找到偏置: {bias_key}")
                
                linear_layer_idx += 1
    
    return model

def convert_to_onnx(model, input_size, output_file):
    """转换模型为ONNX格式"""
    
    try:
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(1, input_size)
        
        print(f"🔄 转换为ONNX...")
        print(f"📊 输入尺寸: {dummy_input.shape}")
        
        # 导出ONNX
        torch.onnx.export(
            model,
            (dummy_input,),
            output_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX文件已保存: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ ONNX转换失败: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("用法: python convert_simple.py <model.pt>")
        print("示例: python convert_simple.py model_walk.pt")
        return
    
    pt_file = sys.argv[1]
    
    if not os.path.exists(pt_file):
        print(f"❌ 文件不存在: {pt_file}")
        return
    
    # 1. 检查文件
    checkpoint = inspect_checkpoint(pt_file)
    if checkpoint is None:
        return
    
    # 2. 提取模型信息
    model_info = extract_model_info(checkpoint)
    if model_info is None:
        return
    
    # 3. 创建模型
    print("\n🔧 重建模型...")
    model = create_simple_model(model_info)
    
    # 4. 转换为ONNX
    output_file = pt_file.replace('.pt', '.onnx')
    print(f"\n🎯 输出文件: {output_file}")
    
    success = convert_to_onnx(model, model_info['input_size'], output_file)
    
    if success:
        print("🎉 转换完成!")
        print(f"📁 ONNX文件: {output_file}")
    else:
        print("💥 转换失败!")

if __name__ == "__main__":
    main() 