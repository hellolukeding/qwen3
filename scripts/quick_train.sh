#!/bin/bash
# Qwen3 快速训练启动脚本

set -e

echo "🚀 启动 Qwen3-0.6B LoRA 微调训练..."

# 检查数据集文件
if [ ! -f "./datasets/mao20250621.dataset.json" ]; then
    echo "❌ 错误：未找到训练数据集文件 ./datasets/mao20250621.dataset.json"
    exit 1
fi

# 检查模型文件
if [ ! -d "./models/Qwen3-0.6B" ]; then
    echo "❌ 错误：未找到模型目录 ./models/Qwen3-0.6B"
    exit 1
fi

# 创建输出目录
mkdir -p ./output/qwen3-lora-$(date +%Y%m%d-%H%M%S)

# 使用 uv 运行训练，带默认参数
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir ./output/qwen3-lora-$(date +%Y%m%d-%H%M%S) \
    --do_train \
    --use_lora \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 1024 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --fp16 \
    --dataloader_num_workers 4 \
    --remove_unused_columns false \
    --report_to none \
    --overwrite_output_dir \
    "$@"

echo "✅ 训练完成！"
