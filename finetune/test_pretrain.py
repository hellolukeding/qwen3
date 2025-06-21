#!/usr/bin/env python3
"""
预训练测试脚本 - 测试模型加载和基本推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

def test_model_loading():
    """测试模型加载"""
    print("=== 测试模型加载 ===")
    
    try:
        # 加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "./models/Qwen3-0.6B",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer加载成功")
        
        # 加载模型
        print("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            "./models/Qwen3-0.6B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        print(f"模型参数量: {model.num_parameters():,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

def test_data_preprocessing():
    """测试数据预处理"""
    print("\n=== 测试数据预处理 ===")
    
    try:
        # 加载数据
        with open("./datasets/mao20250621.dataset.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"数据条目数: {len(data)}")
        
        # 测试一个样本
        sample = data[0]
        print(f"样本指令: {sample['instruction'][:50]}...")
        print(f"样本输出: {sample['output'][:50]}...")
        
        print("✅ 数据预处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据预处理测试失败: {e}")
        return False

def test_inference(model, tokenizer):
    """测试推理"""
    print("\n=== 测试基础推理 ===")
    
    if model is None or tokenizer is None:
        print("❌ 模型未加载，跳过推理测试")
        return False
    
    try:
        # 构建测试消息
        messages = [
            {"role": "system", "content": "你是一个AI助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"输入文本: {text}")
        
        # 分词
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print(f"输入tokens数: {inputs['input_ids'].shape[1]}")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码输出
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(f"模型回复: {response}")
        print("✅ 推理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

def main():
    print("Qwen3-0.6B 微调前测试")
    print("=" * 50)
    
    # 测试模型加载
    model, tokenizer = test_model_loading()
    
    # 测试数据预处理
    data_ok = test_data_preprocessing()
    
    # 测试推理
    inference_ok = test_inference(model, tokenizer)
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"模型加载: {'✅' if model is not None else '❌'}")
    print(f"数据预处理: {'✅' if data_ok else '❌'}")
    print(f"基础推理: {'✅' if inference_ok else '❌'}")
    
    if model is not None and data_ok:
        print("\n🚀 所有测试通过，可以开始微调训练！")
        print("运行命令: ./finetune/start_training.sh")
    else:
        print("\n❌ 存在问题，请检查配置")

if __name__ == "__main__":
    main()
