# 使用 UV 包管理器运行 Qwen3 训练

## 环境配置

本项目使用 **uv** 作为 Python 包管理器，确保依赖管理的一致性和效率。

### 1. 验证环境

首先验证 uv 环境和所有依赖是否正确安装：

```bash
./scripts/verify_env.sh
```

这将检查：
- uv 版本和 Python 环境
- 所有必需的 Python 包（torch、transformers、datasets 等）
- CUDA 和 GPU 支持
- 项目文件完整性

### 2. 快速开始训练

使用预设的优化参数快速开始训练：

```bash
./scripts/quick_train.sh
```

这将使用以下默认配置：
- LoRA 微调 (r=64, alpha=128, dropout=0.1)
- 训练 3 个 epoch
- 批次大小：2，梯度累积：4 步
- 学习率：2e-4，余弦学习率调度
- FP16 混合精度训练
- 序列长度：1024

### 3. 自定义训练参数

查看完整的训练选项：

```bash
./scripts/train_with_uv.sh --help
```

或直接使用 uv 运行：

```bash
uv run python finetune/train.py --help
```

### 4. 训练示例

#### 基础 LoRA 训练
```bash
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir ./output/qwen3-lora \
    --do_train \
    --use_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-4
```

#### 高级配置训练
```bash
uv run python finetune/train.py \
    --model_name_or_path ./models/Qwen3-0.6B \
    --data_path ./datasets/mao20250621.dataset.json \
    --output_dir ./output/qwen3-advanced \
    --do_train \
    --use_lora \
    --lora_r 128 \
    --lora_alpha 256 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_steps 200 \
    --save_steps 250 \
    --logging_steps 25 \
    --fp16
```

## 环境说明

### 依赖包版本
- Python: 3.12.10
- PyTorch: 2.7.1+cu126
- Transformers: 4.52.4
- Datasets: 3.6.0
- PEFT: 0.15.2
- Accelerate: 1.8.1
- 其他深度学习和数据科学包

## 小显存训练优化 (RTX 2060 6GB)

对于显存较小的GPU（如RTX 2060），建议使用优化配置：

### 1. 测试环境
```bash
./scripts/test_simple.sh      # 极简测试 (2步训练)
```

### 2. 小显存训练
```bash
./scripts/train_lowmem.sh     # 优化的小显存训练
```

### 3. GPU内存监控
```bash
./scripts/monitor_gpu.sh      # 实时监控GPU内存使用
```

### 小显存优化配置说明
- **LoRA rank**: 8 (而非64) - 大幅减少训练参数
- **序列长度**: 256 (而非1024) - 减少内存占用
- **批次大小**: 1 - 最小批次
- **梯度累积**: 4步 - 模拟更大批次
- **数据类型**: FP16 - 节省内存
- **缓存清理**: 每5步清理一次缓存

### 环境变量优化
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

### 输出目录结构
```
output/
├── qwen3-lora-YYYYMMDD-HHMMSS/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── pytorch_model.bin
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_args.bin
```

## 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保使用 `uv run` 命令或激活虚拟环境
2. **CUDA 内存不足**: 减少 `per_device_train_batch_size` 或增加 `gradient_accumulation_steps`
3. **训练中断**: 使用 `--resume_from_checkpoint` 恢复训练

### 调试命令

检查 Python 环境：
```bash
uv run python -c "import sys; print(sys.executable)"
```

检查 GPU 状态：
```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

查看包版本：
```bash
uv pip list
```
