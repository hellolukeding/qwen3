#!/usr/bin/env -S uv run python
"""
高级模型测试脚本 - 处理截断问题的优化版本
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc

def find_latest_model():
    """查找最新的模型目录"""
    output_dir = "./output"
    if not os.path.exists(output_dir):
        return None
    
    # 查找所有 qwen3-lora-lowmem- 开头的目录
    model_dirs = [d for d in os.listdir(output_dir) 
                  if d.startswith("qwen3-lora-lowmem-") and 
                  os.path.isdir(os.path.join(output_dir, d))]
    
    if not model_dirs:
        return None
    
    # 按时间戳排序，返回最新的
    model_dirs.sort(reverse=True)
    return os.path.join(output_dir, model_dirs[0])

def load_model(adapter_path):
    """加载带有 LoRA 适配器的模型"""
    print(f"🔄 加载模型: {adapter_path}")
    
    # 基础模型路径
    base_model_path = "./models/Qwen3-0.6B"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"✅ 模型加载成功!")
    return model, tokenizer

def detect_truncation(text):
    """检测文本是否被截断"""
    # 检查是否以不完整的句子结尾
    truncation_indicators = [
        lambda t: len(t.strip()) == 0,  # 空文本
        lambda t: t.strip().endswith(('...', '。。。')),  # 省略号结尾
        lambda t: not t.strip().endswith(('.', '!', '?', '。', '！', '？', ':', '：', ';', '；')),  # 没有标点结尾
        lambda t: len([c for c in t if c in '"\'""''']) % 2 != 0,  # 引号不匹配
        lambda t: len([c for c in t if c in '()（）']) % 2 != 0,  # 括号不匹配
    ]
    
    return any(indicator(text) for indicator in truncation_indicators)

def generate_with_retry(model, tokenizer, messages, max_retries=3):
    """带重试机制的生成函数"""
    
    for attempt in range(max_retries):
        print(f"  🔄 生成尝试 {attempt + 1}/{max_retries}")
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 动态调整参数
        max_tokens = min(512 + attempt * 256, 1024)  # 逐次增加最大token数
        temperature = 0.7 - attempt * 0.1  # 逐次降低温度
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.3),
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.2,  # 鼓励较长回复
                no_repeat_ngram_size=3,
                early_stopping=False,
                # 添加更多控制参数
                top_p=0.9,
                top_k=50,
                num_beams=1,  # 贪心搜索更稳定
            )
        
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 清理回复
        cleaned_response = clean_response(response)
        
        # 检查是否被截断
        is_truncated = detect_truncation(cleaned_response)
        
        print(f"    📝 生成长度: {len(cleaned_response)} 字符")
        print(f"    ⏱️  生成时间: {generation_time:.2f} 秒")
        print(f"    ✂️  截断状态: {'是' if is_truncated else '否'}")
        
        if not is_truncated or attempt == max_retries - 1:
            return cleaned_response, generation_time, is_truncated
        
        print(f"    🔄 检测到截断，进行第 {attempt + 2} 次尝试...")
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
    
    return cleaned_response, generation_time, True

def clean_response(response):
    """增强的回复清理函数"""
    if not response:
        return ""
    
    # 移除开头的空白和特殊字符
    response = response.strip()
    
    # 移除可能的重复开头
    lines = response.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:
            # 移除明显的重复模式
            if not (len(cleaned_lines) > 0 and line == cleaned_lines[-1]):
                cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    
    # 修复常见的格式问题
    result = result.replace('  ', ' ')  # 多余空格
    result = result.replace('\n\n\n', '\n\n')  # 多余换行
    
    return result

def evaluate_response_quality(question, response, generation_time, is_truncated):
    """评估回复质量"""
    score = 0
    details = []
    
    # 基础分数
    if len(response.strip()) > 0:
        score += 20
        details.append("✅ 有回复内容")
    else:
        details.append("❌ 回复为空")
        return score, details
    
    # 长度评估
    if len(response) >= 50:
        score += 20
        details.append("✅ 回复长度充足")
    elif len(response) >= 20:
        score += 10
        details.append("⚠️  回复略短")
    else:
        details.append("❌ 回复过短")
    
    # 截断检测
    if not is_truncated:
        score += 20
        details.append("✅ 回复完整")
    else:
        details.append("❌ 回复被截断")
    
    # 相关性检测（简单关键词匹配）
    question_lower = question.lower()
    response_lower = response.lower()
    
    if any(keyword in response_lower for keyword in ['你好', '介绍', '自己']) and '介绍' in question_lower:
        score += 15
        details.append("✅ 回复相关")
    elif any(keyword in response_lower for keyword in ['解释', '说明', '原理']) and any(k in question_lower for k in ['解释', '说明', '什么']):
        score += 15
        details.append("✅ 回复相关")
    elif len(set(response_lower.split()) & set(question_lower.split())) >= 2:
        score += 10
        details.append("⚠️  回复部分相关")
    else:
        details.append("❌ 回复可能不相关")
    
    # 语言质量
    if not any(char in response for char in ['<', '>', '[', ']']):
        score += 10
        details.append("✅ 无特殊字符")
    else:
        details.append("⚠️  包含特殊字符")
    
    # 生成速度
    if generation_time < 5:
        score += 15
        details.append("✅ 生成速度快")
    elif generation_time < 10:
        score += 10
        details.append("⚠️  生成速度一般")
    else:
        details.append("❌ 生成速度慢")
    
    return min(score, 100), details

def test_model_advanced(adapter_path):
    """高级模型测试"""
    print(f"🧪 高级测试模型: {adapter_path}")
    
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
                "input": "你好，请详细介绍一下自己的能力和特点。",
                "description": "自我介绍测试（详细版）"
            },
            {
                "input": "请解释一下人工智能和机器学习的区别，并给出具体例子。",
                "description": "知识解释测试"
            },
            {
                "input": "如果你是一位旅行向导，请为我推荐北京三日游的详细行程安排。",
                "description": "场景应用测试"
            },
            {
                "input": "写一首关于春天的现代诗，要求至少四句话，表达对生命的热爱。",
                "description": "创意写作测试"
            }
        ]
        
        print(f"\n🎯 开始测试 {len(test_cases)} 个案例:")
        
        total_score = 0
        successful_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(test_cases)}: {test_case['description']}")
            print(f"{'='*60}")
            print(f"📝 输入: {test_case['input']}")
            
            try:
                messages = [{"role": "user", "content": test_case['input']}]
                response, gen_time, is_truncated = generate_with_retry(model, tokenizer, messages)
                
                print(f"\n🤖 输出:")
                print(f"{response}")
                
                # 评估质量
                score, details = evaluate_response_quality(
                    test_case['input'], response, gen_time, is_truncated
                )
                
                print(f"\n📊 质量评估:")
                print(f"评分: {score}/100")
                for detail in details:
                    print(f"  {detail}")
                
                total_score += score
                successful_tests += 1
                
                # 清理显存
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                continue
        
        # 总结
        print(f"\n{'='*60}")
        print(f"🎯 测试总结")
        print(f"{'='*60}")
        if successful_tests > 0:
            avg_score = total_score / successful_tests
            print(f"✅ 成功测试: {successful_tests}/{len(test_cases)}")
            print(f"📊 平均得分: {avg_score:.1f}/100")
            
            if avg_score >= 80:
                print("🏆 模型表现优秀!")
            elif avg_score >= 60:
                print("👍 模型表现良好!")
            elif avg_score >= 40:
                print("⚠️  模型表现一般，需要改进")
            else:
                print("❌ 模型表现较差，建议重新训练")
        else:
            print("❌ 所有测试都失败了")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 Qwen3 LoRA 高级模型测试")
    print("=" * 50)
    
    # 获取模型路径
    if len(sys.argv) > 1:
        adapter_path = sys.argv[1]
    else:
        adapter_path = find_latest_model()
        if not adapter_path:
            print("❌ 未找到微调模型，请先运行训练脚本")
            print("💡 使用方法: uv run test_finetuned_advanced.py [模型路径]")
            return
    
    print(f"📁 使用模型: {adapter_path}")
    
    # 测试模型
    success = test_model_advanced(adapter_path)
    
    if success:
        print("\n✅ 高级测试完成!")
    else:
        print("\n❌ 高级测试失败!")

if __name__ == "__main__":
    main()
