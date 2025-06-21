#!/bin/bash
# Qwen3 训练启动脚本 - 使用 uv 环境

set -e

echo "使用 uv 运行 Qwen3 训练脚本..."
echo "Python 环境信息："

# 激活并显示环境信息
uv run python --version
uv run python -c "import sys; print(f'Python 可执行文件路径: {sys.executable}')"

echo "检查关键依赖包..."
uv run python -c "
try:
    import torch
    import transformers
    import datasets
    import peft
    print(f'✓ PyTorch 版本: {torch.__version__}')
    print(f'✓ Transformers 版本: {transformers.__version__}')
    print(f'✓ Datasets 版本: {datasets.__version__}')
    print(f'✓ PEFT 版本: {peft.__version__}')
    print('所有依赖包检查通过！')
except ImportError as e:
    print(f'✗ 依赖包导入失败: {e}')
    exit(1)
"

echo ""
echo "开始训练..."

# 使用 uv run 执行训练脚本
uv run python finetune/train.py "$@"
