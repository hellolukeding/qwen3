#!/usr/bin/env python3
"""
微调模型功能测试脚本
测试 LoRA 微调模型的各项功能
"""

import torch
import sys
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(adapter_path):
    """加载微调模型"""
    base_model_path = "./models/Qwen3-0.6B"
    
    print("📦 加载基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("🔧 加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def generate_response(model, tokenizer, user_input, system_prompt="你是一个有用的AI助手。"):
    """生成回复"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response, generation_time

def test_model(adapter_path):
    """测试微调模型的功能"""
    print(f"🧪 测试模型: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"❌ 模型路径不存在: {adapter_path}")
        return False
    
    try:
        # 加载模型
        model, tokenizer = load_model(adapter_path)
        
        # 显示模型信息
        print("\n📊 模型信息:")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        
        # GPU 内存信息
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU 内存 - 已分配: {memory_allocated:.2f} GB, 已保留: {memory_reserved:.2f} GB")
        
        # 测试案例
        test_cases = [
            {
                "input": "你好，请介绍一下自己。",
                "description": "自我介绍测试"
            },
            {
                "input": "1+1等于多少？",
                "description": "数学计算测试"
            },
            {
                "input": "请解释一下什么是人工智能。",
                "description": "知识问答测试"
            },
            {
                "input": "写一首关于春天的诗。",
                "description": "创作能力测试"
            },
            {
                "input": "如何学习编程？",
                "description": "建议咨询测试"
            }
        ]
        
        print("\n💬 开始对话测试...")
        total_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}: {test_case['description']} ---")
            print(f"用户: {test_case['input']}")
            
            try:
                response, gen_time = generate_response(model, tokenizer, test_case['input'])
                total_time += gen_time
                
                print(f"AI: {response}")
                print(f"⏱️  生成时间: {gen_time:.2f}秒")
                
                # 简单的质量检查
                if len(response.strip()) > 10:
                    print("✅ 回复长度正常")
                else:
                    print("⚠️  回复较短")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                continue
        
        # 性能总结
        print(f"\n📈 性能总结:")
        print(f"总测试数: {len(test_cases)}")
        print(f"总生成时间: {total_time:.2f}秒")
        print(f"平均生成时间: {total_time/len(test_cases):.2f}秒/次")
        
        # 内存使用总结
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            print(f"最终GPU内存使用: {final_memory:.2f} GB")
        
        print("\n✅ 模型测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python scripts/test_finetuned_model.py <模型路径>")
        print("示例: python scripts/test_finetuned_model.py ./output/qwen3-lora-lowmem-20250621-195657")
        
        # 尝试自动查找最新模型
        import glob
        models = glob.glob("./output/qwen3-lora-*")
        if models:
            latest_model = max(models, key=os.path.getctime)
            print(f"\n💡 找到最新模型: {latest_model}")
            print(f"可以使用: python scripts/test_finetuned_model.py {latest_model}")
        
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🚀 开始模型功能测试...")
    success = test_model(model_path)
    
    if success:
        print("\n🎉 测试成功完成！")
        print("\n📝 下一步建议:")
        print(f"  # 交互式对话测试")
        print(f"  python scripts/chat_with_finetuned.py {model_path}")
    else:
        print("\n❌ 测试失败，请检查模型文件和环境配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
