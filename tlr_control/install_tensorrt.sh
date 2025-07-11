#!/bin/bash
# TensorRT环境安装脚本 - 适用于Jetson平台
# 运行方式: bash tlr_control/install_tensorrt.sh

set -e  # 遇到错误立即退出

echo "🚀 开始安装TensorRT环境..."
echo "================================"

# 检查是否为Jetson设备
if [ -f /etc/nv_tegra_release ]; then
    echo "✅ 检测到Jetson设备"
    JETSON_VERSION=$(cat /etc/nv_tegra_release | grep "R" | cut -d' ' -f2)
    echo "   Jetson版本: $JETSON_VERSION"
else
    echo "⚠️  未检测到Jetson设备，将尝试通用安装"
fi

# 更新系统包
echo "📦 更新系统包..."
sudo apt update

# 检查JetPack是否已安装
echo "🔍 检查JetPack安装状态..."
if dpkg -l | grep -q "nvidia-jetpack"; then
    echo "✅ JetPack已安装"
else
    echo "❌ JetPack未安装"
    echo "💡 请先安装JetPack SDK，包含TensorRT库"
    echo "   下载地址: https://developer.nvidia.com/jetpack"
    exit 1
fi

# 检查CUDA
echo "🔍 检查CUDA安装..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "✅ CUDA已安装: $CUDA_VERSION"
else
    echo "❌ CUDA未安装或未在PATH中"
    exit 1
fi

# 安装Python依赖
echo "🐍 安装Python依赖..."

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "安装pip3..."
    sudo apt install -y python3-pip
fi

# 更新pip
echo "更新pip..."
pip3 install --upgrade pip

# 安装基础依赖
echo "安装基础依赖..."
pip3 install numpy

# 安装PyCUDA
echo "📦 安装PyCUDA..."
if pip3 list | grep -q pycuda; then
    echo "✅ PyCUDA已安装"
else
    echo "正在安装PyCUDA..."
    # Jetson需要特殊的PyCUDA安装方式
    sudo apt install -y python3-dev
    pip3 install pycuda
fi

# 检查TensorRT Python绑定
echo "🔍 检查TensorRT Python绑定..."
python3 -c "import tensorrt; print(f'TensorRT版本: {tensorrt.__version__}')" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ TensorRT Python绑定已可用"
else
    echo "❌ TensorRT Python绑定不可用"
    echo "💡 尝试修复..."
    
    # 查找TensorRT库
    TRT_LIB_PATH="/usr/lib/python3.*/dist-packages/tensorrt"
    if ls $TRT_LIB_PATH 2>/dev/null; then
        echo "✅ 找到TensorRT库路径"
        # 添加到Python路径
        echo "export PYTHONPATH=\$PYTHONPATH:/usr/lib/python3.*/dist-packages" >> ~/.bashrc
        source ~/.bashrc
    else
        echo "❌ 未找到TensorRT库，请检查JetPack安装"
        exit 1
    fi
fi

# 设置环境变量
echo "⚙️  设置环境变量..."
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export PATH=\$PATH:\$CUDA_HOME/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$CUDA_HOME/lib64" >> ~/.bashrc

# 创建测试脚本
echo "📝 创建验证脚本..."
cat > /tmp/test_tensorrt.py << 'EOF'
#!/usr/bin/env python3
import sys

def test_imports():
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy导入失败:", e)
        return False
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        print("✅ PyCUDA: 可用")
        
        # 检查CUDA设备
        cuda.init()
        device_count = cuda.Device.count()
        if device_count > 0:
            device = cuda.Device(0)
            print(f"   GPU设备: {device.name()}")
            major, minor = device.compute_capability()
            print(f"   计算能力: {major}.{minor}")
        else:
            print("❌ 未检测到CUDA设备")
            return False
            
    except ImportError as e:
        print("❌ PyCUDA导入失败:", e)
        return False
    except Exception as e:
        print("❌ CUDA初始化失败:", e)
        return False
    
    try:
        import tensorrt as trt
        print("✅ TensorRT:", trt.__version__)
    except ImportError as e:
        print("❌ TensorRT导入失败:", e)
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 验证TensorRT环境...")
    success = test_imports()
    if success:
        print("\n🎉 TensorRT环境安装成功!")
        print("现在可以运行TensorRT测试脚本:")
        print("  python tlr_control/quick_test_tensorrt.py")
    else:
        print("\n💥 环境验证失败，请检查安装")
        sys.exit(1)
EOF

# 运行验证
echo "🧪 验证安装..."
python3 /tmp/test_tensorrt.py

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "🎉 TensorRT环境安装完成!"
    echo "================================"
    echo ""
    echo "💡 建议的下一步操作:"
    echo "1. 重启终端或运行: source ~/.bashrc"
    echo "2. 设置Jetson性能模式:"
    echo "   sudo nvpmodel -m 0"
    echo "   sudo jetson_clocks"
    echo "3. 运行TensorRT测试:"
    echo "   python tlr_control/quick_test_tensorrt.py"
    echo ""
    echo "🚀 准备好进行高性能机器人控制了!"
else
    echo ""
    echo "💥 安装验证失败，请检查错误信息"
    exit 1
fi 