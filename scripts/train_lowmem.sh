#!/bin/bash
# Qwen3 小显存优化训练脚本 (适用于 6GB 显存)

set -e

echo "🚀 启动 Qwen3-0.6B LoRA 微调训练 (小显存优化)..."

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
OUTPUT_DIR="./output/qwen3-lora-lowmem-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "📁 输出目录: $OUTPUT_DIR"

# 设置环境变量优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 使用 uv 运行训练，针对小显存优化
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_seq_length 256 \
    --logging_steps 5 \
    --save_steps 25 \
    --save_total_limit 2 \
    --warmup_steps 20 \
    --lr_scheduler_type cosine \
    --fp16 \
    --dataloader_num_workers 0 \
    --remove_unused_columns false \
    --report_to none \
    --torch_empty_cache_steps 5 \
    --dataloader_pin_memory false \
    --overwrite_output_dir \
    "$@"

echo "✅ 训练完成！输出保存在: $OUTPUT_DIR"
