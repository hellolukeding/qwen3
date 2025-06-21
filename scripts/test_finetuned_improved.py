#!/usr/bin/env python3
"""
改进的微调模型测试脚本
处理模型输出中的思考过程标签
"""

import torch
import sys
import os
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def clean_response(response):
    """清理回复中的思考过程标签"""
    # 移除 <think> 和 </think> 标签及其内容
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # 移除 <think> 开始但没有结束的内容
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL)
    
    # 清理多余的空白
    response = re.sub(r'\n\s*\n', '\n', response)
    response = response.strip()
    
    # 检查是否回复被截断（以不完整的句子结尾）
    if response and not response.endswith(('.', '。', '!', '！', '?', '？', ':', '：', ';', '；')):
        response += "..."  # 添加省略号表示内容被截断
    
    return response

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
            max_new_tokens=512,  # 增加最大生成token数
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            length_penalty=1.0,  # 添加长度惩罚
            no_repeat_ngram_size=3,  # 避免重复n-gram
            early_stopping=False  # 禁用早停
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # 清理回复
    cleaned_response = clean_response(response)
    
    return cleaned_response, generation_time

def test_model_improved(adapter_path):
    """改进的模型测试"""
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
                "input": "你好，请简单介绍一下自己。",
                "description": "自我介绍测试"
            },
            {
                "input": "3+5等于多少？请直接回答。",
                "description": "数学计算测试"
            },
            {
                "input": "什么是人工智能？请简洁回答。",
                "description": "知识问答测试"
            },
            {
                "input": "写一首简短的关于春天的诗。",
                "description": "创作能力测试"
            },
            {
                "input": "被推翻的地主买办阶级残余对社会有何影响",
                "description": "微调测试"
            }
        ]
        
        print("\n💬 开始对话测试...")
        total_time = 0
        successful_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}: {test_case['description']} ---")
            print(f"用户: {test_case['input']}")
            
            try:
                response, gen_time = generate_response(model, tokenizer, test_case['input'])
                total_time += gen_time
                successful_tests += 1
                
                print(f"AI: {response}")
                print(f"⏱️  生成时间: {gen_time:.2f}秒")
                
                # 质量评估
                if len(response.strip()) > 10:
                    print("✅ 回复长度正常")
                else:
                    print("⚠️  回复较短")
                
                if not response.strip():
                    print("❌ 回复为空")
                elif response.endswith("..."):
                    print("⚠️  回复可能被截断")
                elif not response.endswith(('.', '。', '!', '！', '?', '？')):
                    print("⚠️  回复可能不完整")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                continue
        
        # 性能总结
        print(f"\n📈 性能总结:")
        print(f"成功测试: {successful_tests}/{len(test_cases)}")
        print(f"总生成时间: {total_time:.2f}秒")
        if successful_tests > 0:
            print(f"平均生成时间: {total_time/successful_tests:.2f}秒/次")
        
        # 内存使用总结
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            print(f"最终GPU内存使用: {final_memory:.2f} GB")
        
        print("\n✅ 改进测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 2:
        print("用法: python scripts/test_finetuned_improved.py <模型路径>")
        print("示例: python scripts/test_finetuned_improved.py ./output/qwen3-lora-lowmem-20250621-195657")
        
        # 尝试自动查找最新模型
        import glob
        models = glob.glob("./output/qwen3-lora-*")
        if models:
            latest_model = max(models, key=os.path.getctime)
            print(f"\n💡 找到最新模型: {latest_model}")
            print(f"可以使用: python scripts/test_finetuned_improved.py {latest_model}")
        
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🚀 开始改进的模型功能测试...")
    success = test_model_improved(model_path)
    
    if success:
        print("\n🎉 测试成功完成！")
        print("\n📝 下一步建议:")
        print(f"  # 交互式对话测试")
        print(f"  python scripts/chat_with_finetuned.py {model_path}")
        print(f"  # 模型对比测试")
        print(f"  python scripts/compare_models.py {model_path}")
    else:
        print("\n❌ 测试失败，请检查模型文件和环境配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
