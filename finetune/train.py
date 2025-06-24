#!/usr/bin/env -S uv run python
"""
Qwen3-0.6B 微调训练脚本
支持LoRA微调，针对对话格式数据集优化
"""

import os
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import transformers

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="./models/Qwen3-0.6B",
        metadata={"help": "模型路径"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "缓存目录"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用快速tokenizer"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程代码"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default="./datasets/mao20250621.dataset.json",
        metadata={"help": "训练数据路径"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "最大序列长度"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "预处理工作进程数"}
    )

@dataclass
class LoraArguments:
    """LoRA微调参数"""
    use_lora: bool = field(
        default=True,
        metadata={"help": "是否使用LoRA微调"}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules"}
    )

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def load_and_preprocess_data(self, data_path: str) -> Dataset:
        """加载并预处理数据"""
        logger.info(f"加载数据: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"原始数据量: {len(raw_data)}")
        
        # 转换为训练格式
        processed_data = []
        for item in raw_data:
            # 构建对话格式
            messages = [
                {"role": "system", "content": item.get("system", "")},
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
            
            # 如果有input字段且不为空，添加到user消息中
            if item.get("input", "").strip():
                messages[1]["content"] = f"{item['instruction']}\n{item['input']}"
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            processed_data.append({"text": text})
        
        logger.info(f"处理后数据量: {len(processed_data)}")
        
        # 创建Dataset
        dataset = Dataset.from_list(processed_data)
        
        # 分词处理
        def tokenize_function(examples):
            # 分词
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None,
            )
            
            # 设置labels为input_ids的副本
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # 应用分词
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        logger.info(f"分词完成，样本数: {len(dataset)}")
        return dataset

def setup_model_and_tokenizer(model_args: ModelArguments):
    """设置模型和tokenizer"""
    logger.info(f"加载模型: {model_args.model_name_or_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型 - 优化内存使用
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,  # 使用 float16 而不是 bfloat16 来节省内存
        device_map="auto",
        low_cpu_mem_usage=True,  # 优化CPU内存使用
        # max_memory={0: "5GB", "cpu": "8GB"},  # 限制GPU内存使用
    )
    
    return model, tokenizer

def setup_lora(model, lora_args: LoraArguments):
    """设置LoRA"""
    if not lora_args.use_lora:
        return model
    
    logger.info("设置LoRA配置")
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.lora_target_modules.split(","),
        bias="none",
    )
    
    # 确保模型参数需要梯度
    model.enable_input_require_grads()
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def main():
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    transformers.utils.logging.set_verbosity_info()
    
    # 设置模型和tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # 设置LoRA
    model = setup_lora(model, lora_args)
    
    # 预处理数据
    preprocessor = DataPreprocessor(tokenizer, data_args.max_seq_length)
    train_dataset = preprocessor.load_and_preprocess_data(data_args.data_path)
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存模型
    logger.info("保存模型...")
    trainer.save_model()
    
    # 如果使用LoRA，单独保存适配器
    if lora_args.use_lora:
        model.save_pretrained(os.path.join(training_args.output_dir, "lora_adapter"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "lora_adapter"))
    
    logger.info("训练完成!")

if __name__ == "__main__":
    main()
