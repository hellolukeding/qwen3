#!/bin/bash

# Qwen3-0.6B 微调启动脚本
# 使用LoRA进行高效微调

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练（速度较慢）"
fi

# 激活虚拟环境
if [ -d "/home/ubuntu/qwen3/qwen-env" ]; then
    echo "激活虚拟环境..."
    source /home/ubuntu/qwen3/qwen-env/bin/activate
else
    echo "未找到虚拟环境，请先运行 scripts/setup.sh"
    exit 1
fi

# 检查必要文件
if [ ! -f "./models/Qwen3-0.6B/config.json" ]; then
    echo "错误: 未找到模型文件，请确保模型在 ./models/Qwen3-0.6B/ 目录下"
    exit 1
fi

if [ ! -f "./datasets/mao20250621.dataset.json" ]; then
    echo "错误: 未找到数据集文件"
    exit 1
fi

# 创建输出目录
mkdir -p ./finetune/output
mkdir -p ./finetune/logs

# 设置训练参数
export CUDA_VISIBLE_DEVICES=0  # 使用第一张GPU
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告

echo "开始Qwen3-0.6B微调训练..."
echo "输出目录: ./finetune/output"
echo "日志目录: ./finetune/logs"

# 启动训练
python ./finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir ./finetune/output \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir ./finetune/logs \
    --report_to "tensorboard" \
    --fp16 \
    --dataloader_pin_memory \
    --dataloader_num_workers 2 \
    --max_seq_length 1024 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    2>&1 | tee ./finetune/logs/training.log

echo "训练完成！"
echo "模型保存在: ./finetune/output"
echo "LoRA适配器保存在: ./finetune/output/lora_adapter"
echo "训练日志: ./finetune/logs/training.log"
echo ""
echo "您可以使用以下命令查看训练进度:"
echo "tensorboard --logdir ./finetune/logs"
