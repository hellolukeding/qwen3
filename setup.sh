#!/bin/bash

# Qwen3-0.6B 项目快速设置脚本
# 使用方法: bash setup.sh

set -e  # 遇到错误时退出

echo "🚀 Qwen3-0.6B 项目设置开始..."

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📋 检测到 Python 版本: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "❌ 错误: 需要 Python 3.8+ 版本"
    exit 1
fi

# 1. 创建虚拟环境
echo "🔨 创建虚拟环境..."
if [ ! -d "qwen-env" ]; then
    python3 -m venv qwen-env
    echo "✅ 虚拟环境创建完成"
else
    echo "⚠️  虚拟环境已存在，跳过创建"
fi

# 2. 激活虚拟环境并安装依赖
echo "📦 安装依赖包..."
source qwen-env/bin/activate

# 检查是否有 requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ 基础依赖安装完成"
else
    echo "⚠️  requirements.txt 不存在，手动安装基础包..."
    pip install torch transformers accelerate
fi

# 3. 检查模型文件
echo "📁 检查模型文件..."
if [ ! -d "Qwen3-0.6B" ]; then
    echo "📥 模型文件不存在，开始下载..."
    
    # 检查是否安装了 huggingface-hub
    if ! pip show huggingface-hub > /dev/null 2>&1; then
        echo "🔧 安装 huggingface-hub..."
        pip install huggingface-hub
    fi
    
    # 尝试下载模型
    echo "⏬ 下载 Qwen2.5-0.5B-Instruct 模型..."
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir Qwen3-0.6B
    
    # 清理 Git 信息
    echo "🧹 清理模型目录的 Git 信息..."
    rm -rf Qwen3-0.6B/.git
    rm -f Qwen3-0.6B/.gitattributes
    
    echo "✅ 模型下载和清理完成"
else
    echo "✅ 模型文件已存在"
    
    # 检查并清理 Git 信息（如果存在）
    if [ -d "Qwen3-0.6B/.git" ]; then
        echo "🧹 清理模型目录的 Git 信息..."
        rm -rf Qwen3-0.6B/.git
        rm -f Qwen3-0.6B/.gitattributes
        echo "✅ Git 信息清理完成"
    fi
fi

# 4. 检查 GPU 支持
echo "🔍 检查 GPU 支持..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'当前 GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存容量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  未检测到 CUDA，将使用 CPU 推理（速度较慢）')
"

# 5. 运行测试
echo "🧪 运行快速测试..."
if python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; print('✅ Transformers 导入成功')"; then
    echo "✅ 环境设置完成！"
    echo ""
    echo "🎉 设置完成！现在你可以："
    echo "   1. 激活环境: source qwen-env/bin/activate"
    echo "   2. 运行基础推理: python test.py"
    echo "   3. 运行优化推理: python final_test.py"
    echo "   4. 启动 vLLM 服务: bash start_vllm.sh"
    echo ""
    echo "📖 更多信息请查看 README.md"
else
    echo "❌ 环境测试失败，请检查安装"
    exit 1
fi

echo "🔄 请运行以下命令激活环境:"
echo "source qwen-env/bin/activate"
