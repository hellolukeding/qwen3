# Qwen3-0.6B 微调指南

## 概述

本项目提供了对 Qwen3-0.6B 模型进行微调的完整流程，使用 LoRA (Low-Rank Adaptation) 技术进行高效微调。

## 文件结构

```
finetune/
├── train.py              # 主训练脚本
├── start_training.sh     # 训练启动脚本
├── test_model.py         # 模型测试脚本
├── requirements.txt      # 微调依赖包
├── output/              # 训练输出目录
├── logs/                # 训练日志目录
└── README.md            # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
# 激活虚拟环境
source ./qwen-env/bin/activate

# 安装微调依赖
pip install -r ./finetune/requirements.txt
```

### 2. 开始训练

```bash
# 使用默认参数开始训练
./finetune/start_training.sh
```

### 3. 测试微调模型

```bash
# 交互式测试
python ./finetune/test_model.py --mode interactive

# 测试数据集样本
python ./finetune/test_model.py --mode test

# 两种模式都运行
python ./finetune/test_model.py --mode both
```

## 数据集格式

数据集应为 JSON 格式，每个样本包含以下字段：

```json
{
  "instruction": "问题或指令",
  "input": "可选的额外输入",
  "output": "期望的回答",
  "system": "可选的系统提示"
}
```

## 训练参数说明

### 基础训练参数

- `num_train_epochs`: 训练轮数 (默认: 3)
- `per_device_train_batch_size`: 每设备批量大小 (默认: 2)
- `gradient_accumulation_steps`: 梯度累积步数 (默认: 8)
- `learning_rate`: 学习率 (默认: 2e-4)
- `max_seq_length`: 最大序列长度 (默认: 2048)

### LoRA 参数

- `lora_r`: LoRA 秩 (默认: 64)
- `lora_alpha`: LoRA alpha (默认: 128)
- `lora_dropout`: LoRA dropout (默认: 0.1)
- `lora_target_modules`: 目标模块 (默认: 注意力和前馈网络层)

## 自定义训练

### 修改训练参数

编辑 `start_training.sh` 中的参数：

```bash
python ./finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/your_dataset.json \
    --output_dir ./finetune/output \
    --num_train_epochs 5 \              # 增加训练轮数
    --per_device_train_batch_size 4 \   # 增加批量大小
    --learning_rate 1e-4 \              # 调整学习率
    --lora_r 32 \                       # 调整LoRA秩
    # ... 其他参数
```

### 使用不同数据集

```bash
python ./finetune/train.py \
    --data_path ./datasets/your_custom_dataset.json \
    # ... 其他参数
```

## 监控训练进度

### TensorBoard

```bash
# 启动 TensorBoard
tensorboard --logdir ./finetune/logs

# 在浏览器中访问 http://localhost:6006
```

### 日志文件

训练日志保存在 `./finetune/logs/training.log`

## 模型推理

### 使用微调模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("./models/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-0.6B")

# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./finetune/output/lora_adapter")
model = model.merge_and_unload()  # 合并权重

# 推理
messages = [
    {"role": "user", "content": "你的问题"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
```

### 交互式测试

```bash
# 使用微调模型进行交互
python ./finetune/test_model.py --mode interactive

# 对比原始模型和微调模型
python ./finetune/test_model.py --no_lora --mode interactive  # 原始模型
python ./finetune/test_model.py --mode interactive            # 微调模型
```

## 常见问题

### 内存不足

1. 减少批量大小：`--per_device_train_batch_size 1`
2. 增加梯度累积：`--gradient_accumulation_steps 16`
3. 使用 gradient checkpointing：`--gradient_checkpointing True`
4. 降低LoRA秩：`--lora_r 32`

### 训练速度慢

1. 增加批量大小（如果内存允许）
2. 使用多GPU：设置 `CUDA_VISIBLE_DEVICES=0,1,2,3`
3. 使用混合精度：`--fp16` 或 `--bf16`
4. 减少序列长度：`--max_seq_length 1024`

### 训练不收敛

1. 调整学习率：`--learning_rate 1e-4` 或 `--learning_rate 5e-4`
2. 增加训练轮数：`--num_train_epochs 5`
3. 调整 warmup：`--warmup_ratio 0.1`
4. 检查数据质量和格式

## 高级功能

### 使用量化训练

```bash
pip install bitsandbytes

# 在训练脚本中添加
--load_in_4bit True \
--bnb_4bit_compute_dtype bfloat16 \
--bnb_4bit_use_double_quant True
```

### WandB 实验跟踪

```bash
pip install wandb
wandb login

# 在训练参数中添加
--report_to "wandb" \
--run_name "qwen3-finetune-experiment"
```

### 多GPU训练

```bash
# 使用 torchrun 进行分布式训练
torchrun --nproc_per_node=2 ./finetune/train.py \
    # ... 训练参数
```

## 最佳实践

1. **数据质量**：确保训练数据高质量、格式一致
2. **参数调优**：从小的学习率开始，逐步调整
3. **验证集**：使用验证集监控过拟合
4. **保存检查点**：定期保存模型检查点
5. **实验记录**：记录每次实验的参数和结果

## 输出文件说明

- `./finetune/output/`：完整的训练输出
- `./finetune/output/lora_adapter/`：LoRA适配器文件
- `./finetune/logs/`：TensorBoard日志
- `./finetune/logs/training.log`：训练日志文件
