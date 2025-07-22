import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 路径设置
base_model_path = "/home/ubuntu/Desktop/qwen3/models/Qwen3-0.6B"
lora_model_path = "/home/ubuntu/Desktop/qwen3/finetune/vet-qwen3-lora"  # 指向主目录
merged_model_path = "/home/ubuntu/Desktop/qwen3/merged/vet-qwen3-lora"

logger.info(f"Loading LoRA model from: {lora_model_path}")

try:
    # 加载 LoRA 模型，使用CPU避免显存问题
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_path, 
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    logger.info("LoRA model loaded successfully")

    # 合并 adapter
    logger.info("Merging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    logger.info("Model merged successfully")

    # 确保输出目录存在
    os.makedirs(merged_model_path, exist_ok=True)

    # 保存合并后的模型
    logger.info(f"Saving merged model to: {merged_model_path}")
    merged_model.save_pretrained(merged_model_path)
    
    # 同时保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(merged_model_path)
    
    logger.info("Merged model and tokenizer saved successfully!")
    
except Exception as e:
    logger.error(f"Error during model merging: {e}")
    
    # 如果自动加载失败，尝试手动修复配置
    logger.info("Attempting manual fix...")
    try:
        import json
        config_path = os.path.join(lora_model_path, "adapter_config.json")
        
        # 读取并修复配置文件
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 删除problematic字段
        if 'corda_config' in config:
            del config['corda_config']
        if 'eva_config' in config:
            del config['eva_config']
            
        # 临时保存修复的配置
        fixed_config_path = config_path + ".fixed"
        with open(fixed_config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info("Config file fixed, please manually replace adapter_config.json with the fixed version")
        
    except Exception as fix_error:
        logger.error(f"Manual fix also failed: {fix_error}")
        raise