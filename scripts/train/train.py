import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
# /*---------------------------------------  ------------------------------------------*/
# rtx 3060适用于rtx3060 6gb
# /*---------------------------------------  ------------------------------------------*/


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置工作目录
work_dir = os.getcwd()
# 设置模型名称
model_name = "/home/ubuntu/Desktop/qwen3/models/Qwen3-0.6B"
# 数据集路径
data_file = "/home/ubuntu/Desktop/qwen3/datasets/datasets-jbUvvCzNBAt7-alpaca-2025-07-21.json"
# 输出地址
output_dir = "/home/ubuntu/Desktop/qwen3/finetune/vet-qwen3-lora"

logger.info(f"Loading tokenizer and model from: {model_name}")

# 清理GPU缓存
torch.cuda.empty_cache()

# 配置4位量化以节省显存
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 使用4位量化加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    trust_remote_code=True, 
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# 为量化模型准备训练
model = prepare_model_for_kbit_training(model)
# 设置 LoRA 参数 - 使用更小的配置以节省内存
peft_config = LoraConfig(
    r=8,  # 减小rank以节省内存
    lora_alpha=16,  # 相应调整alpha
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # 只对部分模块应用LoRA以节省内存
    target_modules=["q_proj", "v_proj"]  
)

logger.info("Applying LoRA configuration to model")
# 获取 PEFT 模型
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# 数据加载和格式化
def format_example(example):
    """格式化训练样本"""
    instruction = example["instruction"]
    input_ = example.get("input", "")
    output = example["output"]

    # 拼接成 prompt
    if input_:
        prompt = f"用户：{instruction}\n{input_}\n助手：{output}"
    else:
        prompt = f"用户：{instruction}\n助手：{output}"

    # 添加特殊token并tokenize
    tokenized = tokenizer(
        prompt + tokenizer.eos_token,  # 添加结束token
        truncation=True, 
        max_length=256,  # 减少序列长度以节省内存
        padding="max_length",
        return_tensors=None
    )
    
    # 设置labels，用于计算loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

logger.info(f"Loading dataset from: {data_file}")
try:
    dataset = load_dataset("json", data_files=data_file)["train"]
    logger.info(f"Dataset loaded successfully with {len(dataset)} samples")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# 划分训练集和验证集 - 使用较小的数据集以节省内存
logger.info("Splitting dataset into train and validation sets")
# 只使用部分数据进行训练以节省内存
dataset_size = min(len(dataset), 1000)  # 限制数据集大小
small_dataset = dataset.select(range(dataset_size))
logger.info(f"Using {dataset_size} samples from {len(dataset)} total samples")

split_dataset = small_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"].map(
    format_example, 
    remove_columns=small_dataset.column_names,
    desc="Formatting training examples"
)
eval_dataset = split_dataset["test"].map(
    format_example, 
    remove_columns=small_dataset.column_names,
    desc="Formatting evaluation examples"
)

logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(eval_dataset)}")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 训练参数 - 优化显存使用
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,  # 保持最小batch size
    gradient_accumulation_steps=32,  # 增加梯度累积步数以补偿小batch size
    learning_rate=1e-4,  # 稍微降低学习率
    num_train_epochs=3,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=1,  # 只保存1个检查点以节省空间
    logging_steps=10,
    logging_dir=f"{output_dir}/logs",
    report_to="none",
    eval_strategy="no",  # 暂时关闭评估以节省内存
    # eval_steps=100,  # 暂时注释掉
    # load_best_model_at_end=True,  # 暂时注释掉
    # metric_for_best_model="loss",
    # greater_is_better=False,
    warmup_steps=50,  # 减少warmup步数
    weight_decay=0.01,
    dataloader_num_workers=0,  # 设为0以节省内存
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # 关闭pin memory
    gradient_checkpointing=True,  # 开启梯度检查点以节省内存
)

# 训练
logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training completed successfully!")
    
    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise