#!/bin/bash
# 验证 uv 环境和依赖

echo "🔍 验证 uv 环境和 Python 依赖..."

echo "UV 版本："
uv --version

echo -e "\nPython 环境信息："
uv run python --version
uv run python -c "
import sys
print(f'Python 路径: {sys.executable}')
print(f'Python 版本: {sys.version}')
"

echo -e "\n📦 检查关键依赖包："

# 检查必需的包
declare -A packages=(
    ["torch"]="torch"
    ["transformers"]="transformers"
    ["datasets"]="datasets"
    ["peft"]="peft"
    ["accelerate"]="accelerate"
    ["bitsandbytes"]="bitsandbytes"
    ["tensorboard"]="tensorboard"
    ["wandb"]="wandb"
    ["sklearn"]="scikit-learn"
    ["pandas"]="pandas"
)

for import_name in "${!packages[@]}"; do
    package_name="${packages[$import_name]}"
    uv run python -c "
try:
    import $import_name
    print(f'✅ $package_name: {getattr($import_name, \"__version__\", \"已安装\")}')
except ImportError:
    print(f'❌ $package_name: 未安装')
    exit(1)
" || exit 1
done

echo -e "\n🔧 检查 CUDA 和 GPU 支持："
uv run python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️ CUDA 不可用，将使用 CPU 进行训练')
"

echo -e "\n📁 检查项目文件："

files_to_check=(
    "./models/Qwen3-0.6B"
    "./datasets/mao20250621.dataset.json"
    "./finetune/train.py"
    "./pyproject.toml"
)

for file in "${files_to_check[@]}"; do
    if [ -e "$file" ]; then
        echo "✅ $file: 存在"
    else
        echo "❌ $file: 不存在"
    fi
done

echo -e "\n🎉 环境验证完成！"
echo "现在可以使用以下命令启动训练："
echo "  ./scripts/quick_train.sh          # 使用默认参数快速训练"
echo "  ./scripts/train_with_uv.sh --help # 查看完整训练选项"
