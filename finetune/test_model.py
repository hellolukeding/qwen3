#!/usr/bin/env python3
"""
微调后模型推理测试脚本
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model_and_tokenizer(base_model_path, lora_adapter_path=None):
    """加载模型和tokenizer"""
    print(f"加载基础模型: {base_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 如果有LoRA适配器，加载它
    if lora_adapter_path:
        print(f"加载LoRA适配器: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()  # 合并LoRA权重
    
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_length=2048, temperature=0.7):
    """生成回复"""
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 分词
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def interactive_chat(model, tokenizer):
    """交互式对话"""
    print("=== Qwen3-0.6B 微调模型测试 ===")
    print("输入 'quit' 退出，输入 'clear' 清空对话历史")
    print("=" * 40)
    
    conversation_history = []
    
    while True:
        user_input = input("\n用户: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("对话历史已清空")
            continue
        elif not user_input:
            continue
        
        # 构建消息
        messages = [
            {"role": "system", "content": "你是一个基于毛泽东思想训练的AI助手，请根据相关理论和实践回答问题。"},
        ]
        
        # 添加历史对话
        messages.extend(conversation_history)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 生成回复
        try:
            response = generate_response(model, tokenizer, messages)
            print(f"助手: {response}")
            
            # 更新对话历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # 限制历史长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
                
        except Exception as e:
            print(f"生成错误: {e}")

def test_dataset_samples(model, tokenizer, dataset_path, num_samples=5):
    """测试数据集样本"""
    print(f"=== 测试数据集样本 ===")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data[:num_samples]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {item['instruction']}")
        if item.get('input', '').strip():
            print(f"输入: {item['input']}")
        
        # 构建消息
        messages = [
            {"role": "system", "content": item.get("system", "")},
            {"role": "user", "content": item["instruction"]}
        ]
        
        if item.get('input', '').strip():
            messages[1]["content"] = f"{item['instruction']}\n{item['input']}"
        
        # 生成回复
        try:
            response = generate_response(model, tokenizer, messages)
            print(f"模型回复: {response}")
            print(f"期望回复: {item['output'][:200]}...")
        except Exception as e:
            print(f"生成错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="微调模型推理测试")
    parser.add_argument("--base_model", default="./models/Qwen3-0.6B", help="基础模型路径")
    parser.add_argument("--lora_adapter", default="./finetune/output/lora_adapter", help="LoRA适配器路径")
    parser.add_argument("--dataset", default="./datasets/mao20250621.dataset.json", help="测试数据集路径")
    parser.add_argument("--mode", choices=["interactive", "test", "both"], default="both", help="运行模式")
    parser.add_argument("--no_lora", action="store_true", help="不使用LoRA适配器")
    
    args = parser.parse_args()
    
    # 加载模型
    lora_path = None if args.no_lora else args.lora_adapter
    model, tokenizer = load_model_and_tokenizer(args.base_model, lora_path)
    
    if args.mode in ["test", "both"]:
        test_dataset_samples(model, tokenizer, args.dataset)
    
    if args.mode in ["interactive", "both"]:
        interactive_chat(model, tokenizer)

if __name__ == "__main__":
    main()
