#!/usr/bin/env python3
"""
模型对比测试脚本
对比基础模型和微调模型的性能差异
"""

import torch
import sys
import os
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ModelComparator:
    def __init__(self, base_model_path, finetuned_model_path):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.tokenizer = None
        self.base_model = None
        self.finetuned_model = None
        
    def load_models(self):
        """加载基础模型和微调模型"""
        print("📦 加载分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        print("📦 加载基础模型...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("🔧 加载微调模型...")
        base_for_peft = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.finetuned_model = PeftModel.from_pretrained(
            base_for_peft, 
            self.finetuned_model_path
        )
        
        print("✅ 模型加载完成！")
    
    def generate_response(self, model, prompt, max_tokens=200):
        """生成回复"""
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response, generation_time
    
    def compare_responses(self, test_cases):
        """对比两个模型的回复"""
        results = []
        
        print("🔄 开始对比测试...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}: {test_case['description']} ---")
            print(f"问题: {test_case['input']}")
            
            # 基础模型回复
            print("\n🤖 基础模型回复:")
            try:
                base_response, base_time = self.generate_response(
                    self.base_model, test_case['input']
                )
                print(f"回复: {base_response}")
                print(f"时间: {base_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                base_response, base_time = "生成失败", 0
            
            # 微调模型回复
            print("\n🚀 微调模型回复:")
            try:
                ft_response, ft_time = self.generate_response(
                    self.finetuned_model, test_case['input']
                )
                print(f"回复: {ft_response}")
                print(f"时间: {ft_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                ft_response, ft_time = "生成失败", 0
            
            # 保存结果
            result = {
                "question": test_case['input'],
                "description": test_case['description'],
                "base_model": {
                    "response": base_response,
                    "time": base_time,
                    "length": len(base_response)
                },
                "finetuned_model": {
                    "response": ft_response,
                    "time": ft_time,
                    "length": len(ft_response)
                }
            }
            results.append(result)
            
            print("-" * 60)
        
        return results
    
    def analyze_results(self, results):
        """分析对比结果"""
        print("\n📊 对比分析结果:")
        
        base_times = [r['base_model']['time'] for r in results if r['base_model']['time'] > 0]
        ft_times = [r['finetuned_model']['time'] for r in results if r['finetuned_model']['time'] > 0]
        
        base_lengths = [r['base_model']['length'] for r in results]
        ft_lengths = [r['finetuned_model']['length'] for r in results]
        
        if base_times and ft_times:
            print(f"\n⏱️  响应时间对比:")
            print(f"  基础模型平均: {sum(base_times)/len(base_times):.2f}秒")
            print(f"  微调模型平均: {sum(ft_times)/len(ft_times):.2f}秒")
        
        print(f"\n📝 回复长度对比:")
        print(f"  基础模型平均: {sum(base_lengths)/len(base_lengths):.0f}字符")
        print(f"  微调模型平均: {sum(ft_lengths)/len(ft_lengths):.0f}字符")
        
        # 保存详细结果
        output_file = f"./output/model_comparison_{int(time.time())}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {output_file}")
    
    def run_comparison(self):
        """运行完整对比测试"""
        # 测试用例
        test_cases = [
            {
                "input": "你好，请介绍一下自己。",
                "description": "自我介绍"
            },
            {
                "input": "什么是人工智能？",
                "description": "知识问答"
            },
            {
                "input": "如何学习编程？",
                "description": "建议咨询"
            },
            {
                "input": "写一首关于秋天的诗。",
                "description": "创作能力"
            },
            {
                "input": "解释一下量子计算的原理。",
                "description": "技术解释"
            }
        ]
        
        # 加载模型
        self.load_models()
        
        # 运行对比
        results = self.compare_responses(test_cases)
        
        # 分析结果
        self.analyze_results(results)

def main():
    if len(sys.argv) != 2:
        print("用法: python scripts/compare_models.py <微调模型路径>")
        print("示例: python scripts/compare_models.py ./output/qwen3-lora-lowmem-20250621-195657")
        sys.exit(1)
    
    finetuned_path = sys.argv[1]
    base_path = "./models/Qwen3-0.6B"
    
    if not os.path.exists(finetuned_path):
        print(f"❌ 微调模型路径不存在: {finetuned_path}")
        sys.exit(1)
    
    if not os.path.exists(base_path):
        print(f"❌ 基础模型路径不存在: {base_path}")
        sys.exit(1)
    
    print("🔍 开始模型对比测试...")
    print(f"基础模型: {base_path}")
    print(f"微调模型: {finetuned_path}")
    
    try:
        comparator = ModelComparator(base_path, finetuned_path)
        comparator.run_comparison()
        
        print("\n🎉 对比测试完成！")
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
