#!/bin/bash
# 简化的小显存测试训练脚本

set -e

echo "🧪 测试 Qwen3 简化小显存训练..."

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 创建临时输出目录
TEST_OUTPUT="./output/test-simple-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEST_OUTPUT"

echo "使用输出目录：$TEST_OUTPUT"

# 清理 GPU 缓存
echo "清理 GPU 缓存..."
uv run python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU 内存已清理. 总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'当前可用内存: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB')
"

# 运行简化配置训练测试 - 不使用梯度检查点
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir "$TEST_OUTPUT" \
    --do_train \
    --use_lora \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1 \
    --max_steps 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --max_seq_length 128 \
    --logging_steps 1 \
    --save_steps 2 \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --report_to none \
    --torch_empty_cache_steps 1 \
    --dataloader_pin_memory false \
    --overwrite_output_dir

if [ $? -eq 0 ]; then
    echo "✅ 简化小显存测试成功！"
    echo "📁 测试输出保存在：$TEST_OUTPUT"
    echo "现在可以尝试更大的配置。"
else
    echo "❌ 简化小显存测试失败！需要进一步诊断。"
    exit 1
fi
