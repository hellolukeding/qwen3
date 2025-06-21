#!/bin/bash
# 测试训练环境 - 运行 1 步训练来验证配置

set -e

echo "🧪 测试 Qwen3 训练环境（运行 1 步验证）..."

# 创建临时输出目录
TEST_OUTPUT="./output/test-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEST_OUTPUT"

echo "使用输出目录：$TEST_OUTPUT"

# 运行 1 步训练进行测试
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir "$TEST_OUTPUT" \
    --do_train \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --max_seq_length 512 \
    --logging_steps 1 \
    --save_steps 1 \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --report_to none \
    --overwrite_output_dir

if [ $? -eq 0 ]; then
    echo "✅ 测试成功！训练环境配置正确。"
    echo "📁 测试输出保存在：$TEST_OUTPUT"
    echo "现在可以运行完整训练：./scripts/quick_train.sh"
else
    echo "❌ 测试失败！请检查配置。"
    exit 1
fi
