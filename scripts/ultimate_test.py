#!/usr/bin/env -S uv run python
"""
终极优化测试脚本 - 使用最佳配置解决截断和质量问题
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
import re
import json

# 导入配置
sys.path.append('./config')
try:
    from generation_config import generation_configs, task_to_config, select_config_by_input, truncation_patterns
except ImportError:
    print("⚠️  未找到配置文件，使用默认配置")
    generation_configs = {
        "standard": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "early_stopping": False,
        }
    }

def find_latest_model():
    """查找最新的模型目录"""
    output_dir = "./output"
    if not os.path.exists(output_dir):
        return None
    
    model_dirs = [d for d in os.listdir(output_dir) 
                  if d.startswith("qwen3-lora-lowmem-") and 
                  os.path.isdir(os.path.join(output_dir, d))]
    
    if not model_dirs:
        return None
    
    model_dirs.sort(reverse=True)
    return os.path.join(output_dir, model_dirs[0])

def load_model(adapter_path):
    """加载模型"""
    print(f"🔄 加载模型: {adapter_path}")
    
    base_model_path = "./models/Qwen3-0.6B"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print(f"✅ 模型加载成功!")
    return model, tokenizer

def advanced_truncation_detector(text):
    """高级截断检测器"""
    if not text or len(text.strip()) == 0:
        return True, "空文本"
    
    text = text.strip()
    
    # 正常结束模式
    normal_endings = [
        r'[。！？\.!?]\s*$',  # 标点结尾
        r'[。！？\.!?]["\'""'')\]】）]\s*$',  # 标点+引号结尾
    ]
    
    for pattern in normal_endings:
        if re.search(pattern, text):
            return False, "正常结束"
    
    # 截断模式检测
    for pattern_name, pattern in truncation_patterns.items():
        if callable(pattern):
            if pattern(text):
                return True, f"截断检测: {pattern_name}"
        else:
            if re.search(pattern, text):
                return True, f"截断检测: {pattern_name}"
    
    return True, "可能截断"

def optimized_generate(model, tokenizer, question, config_name="standard", max_attempts=3):
    """优化的生成函数"""
    
    # 选择配置
    if config_name not in generation_configs:
        config_name = select_config_by_input(question)
    
    config = generation_configs.get(config_name, generation_configs["standard"])
    
    print(f"🎛️  使用配置: {config_name}")
    
    for attempt in range(max_attempts):
        print(f"  🔄 第 {attempt + 1}/{max_attempts} 次生成")
        
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 动态调整参数
        dynamic_config = config.copy()
        if attempt > 0:
            # 增加最大token数
            dynamic_config["max_new_tokens"] = min(
                config["max_new_tokens"] + attempt * 256, 1024
            )
            # 降低温度提高确定性
            dynamic_config["temperature"] = max(
                config["temperature"] - attempt * 0.1, 0.3
            )
        
        start_time = time.time()
        
        with torch.no_grad():
            # 过滤掉不支持的参数
            generation_kwargs = {
                k: v for k, v in dynamic_config.items() 
                if k in [
                    "max_new_tokens", "temperature", "do_sample", "top_p", "top_k",
                    "repetition_penalty", "no_repeat_ngram_size", "early_stopping",
                    "num_beams", "pad_token_id", "eos_token_id"
                ]
            }
            
            generation_kwargs.update({
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            })
            
            outputs = model.generate(**inputs, **generation_kwargs)
        
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 清理回复
        cleaned_response = advanced_clean_response(response)
        
        # 检测截断
        is_truncated, reason = advanced_truncation_detector(cleaned_response)
        
        print(f"    📝 长度: {len(cleaned_response)} 字符")
        print(f"    ⏱️  时间: {generation_time:.2f} 秒")
        print(f"    🔍 状态: {reason}")
        
        if not is_truncated:
            return cleaned_response, generation_time, False, config_name
        
        # 如果最后一次尝试仍然截断，尝试续写
        if attempt == max_attempts - 1 and len(cleaned_response) > 50:
            print(f"    🔧 尝试续写修复...")
            final_response = attempt_continuation(
                model, tokenizer, question, cleaned_response
            )
            final_truncated, final_reason = advanced_truncation_detector(final_response)
            return final_response, generation_time, final_truncated, config_name
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return cleaned_response, generation_time, True, config_name

def advanced_clean_response(response):
    """高级回复清理"""
    if not response:
        return ""
    
    # 移除开头的think标签内容
    response = re.sub(r'^<think>.*?</think>\s*', '', response, flags=re.DOTALL)
    
    # 基础清理
    response = response.strip()
    
    # 移除重复的换行
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # 移除多余的空格
    response = re.sub(r' {2,}', ' ', response)
    
    # 移除行首的多余空格
    lines = []
    for line in response.split('\n'):
        lines.append(line.strip())
    
    response = '\n'.join(lines)
    
    return response

def attempt_continuation(model, tokenizer, original_question, partial_response):
    """尝试续写未完成的回复"""
    continue_prompt = f"""请直接继续完成以下回答，不要重复已有内容：

原问题：{original_question}

已有回答：{partial_response}

继续："""
    
    messages = [{"role": "user", "content": continue_prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.6,
            do_sample=True,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    continuation = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # 智能合并
    if continuation and len(continuation) > 10:
        # 避免重复开头
        continuation = re.sub(r'^[，,。！？\.!?]*\s*', '', continuation)
        return partial_response + continuation
    
    return partial_response

def comprehensive_quality_evaluation(question, response, generation_time, is_truncated, config_name):
    """综合质量评估"""
    score = 0
    details = []
    
    # 1. 内容存在性 (20分)
    if len(response.strip()) > 0:
        score += 20
        details.append("✅ 生成了回复内容")
    else:
        details.append("❌ 没有生成内容")
        return score, details
    
    # 2. 长度充足性 (20分)
    response_length = len(response)
    if response_length >= 200:
        score += 20
        details.append(f"✅ 回复长度充足 ({response_length} 字符)")
    elif response_length >= 100:
        score += 15
        details.append(f"⚠️  回复长度适中 ({response_length} 字符)")
    elif response_length >= 50:
        score += 10
        details.append(f"⚠️  回复较短 ({response_length} 字符)")
    else:
        details.append(f"❌ 回复过短 ({response_length} 字符)")
    
    # 3. 完整性 (25分)
    if not is_truncated:
        score += 25
        details.append("✅ 回复完整，无截断")
    else:
        details.append("❌ 回复被截断或不完整")
    
    # 4. 相关性检测 (20分)
    relevance_score = calculate_relevance(question, response)
    score += relevance_score
    if relevance_score >= 15:
        details.append("✅ 回复高度相关")
    elif relevance_score >= 10:
        details.append("⚠️  回复部分相关")
    else:
        details.append("❌ 回复相关性不足")
    
    # 5. 语言质量 (10分)
    quality_issues = check_language_quality(response)
    language_score = max(10 - len(quality_issues), 0)
    score += language_score
    if language_score >= 8:
        details.append("✅ 语言质量良好")
    elif quality_issues:
        details.append(f"⚠️  语言质量问题: {', '.join(quality_issues)}")
    
    # 6. 生成效率 (5分)
    if generation_time < 10:
        score += 5
        details.append(f"✅ 生成速度快 ({generation_time:.1f}秒)")
    elif generation_time < 20:
        score += 3
        details.append(f"⚠️  生成速度一般 ({generation_time:.1f}秒)")
    else:
        details.append(f"❌ 生成速度慢 ({generation_time:.1f}秒)")
    
    return min(score, 100), details

def calculate_relevance(question, response):
    """计算问题和回复的相关性"""
    question_lower = question.lower()
    response_lower = response.lower()
    
    # 关键词匹配
    question_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', question_lower))
    response_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', response_lower))
    
    if not question_words:
        return 10
    
    # 计算词汇重叠度
    overlap = len(question_words & response_words)
    overlap_ratio = overlap / len(question_words)
    
    # 特定任务关键词检测
    task_keywords = {
        "介绍": ["我", "能力", "特点", "擅长"],
        "解释": ["解释", "说明", "原理", "区别"],
        "推荐": ["推荐", "建议", "安排", "行程"],
        "写": ["诗", "创作", "文字", "表达"]
    }
    
    task_bonus = 0
    for task, keywords in task_keywords.items():
        if task in question_lower:
            if any(kw in response_lower for kw in keywords):
                task_bonus = 5
                break
    
    # 计算最终相关性分数
    base_score = min(overlap_ratio * 15, 15)
    return min(base_score + task_bonus, 20)

def check_language_quality(response):
    """检查语言质量问题"""
    issues = []
    
    # 检查特殊字符
    if re.search(r'[<>{}[\]\\]', response):
        issues.append("包含特殊字符")
    
    # 检查重复短语
    if re.search(r'(.{3,})\1{2,}', response):
        issues.append("存在重复短语")
    
    # 检查过多的标点符号
    punct_ratio = len(re.findall(r'[，。！？、：；,.!?:;]', response)) / max(len(response), 1)
    if punct_ratio > 0.2:
        issues.append("标点密度过高")
    
    return issues

def ultimate_test(adapter_path):
    """终极测试函数"""
    print(f"🚀 终极优化测试")
    print(f"📁 模型路径: {adapter_path}")
    
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
        
        # 测试案例
        test_cases = [
            {
                "input": "请详细介绍一下你的能力、特长和工作方式，让我更好地了解如何与你协作。",
                "description": "详细自我介绍测试",
                "config": "analytical"
            },
            {
                "input": "请深入解释人工智能和机器学习的本质区别，包括它们的工作原理、应用场景和发展趋势。",
                "description": "深度知识解释测试", 
                "config": "analytical"
            },
            {
                "input": "作为专业旅行规划师，为第一次来北京的游客设计一个完整的五日深度游方案，包括景点推荐、时间安排、交通方式、餐饮建议和注意事项。",
                "description": "复杂规划任务测试",
                "config": "long_form"
            },
            {
                "input": "写一首表达对生命热爱和对未来憧憬的现代诗，要求至少八行，运用丰富的意象和修辞手法。",
                "description": "创意写作测试",
                "config": "creative"
            },
            {
                "input": "从历史唯物主义角度分析，被推翻的地主买办阶级残余对社会有何影响？请详细阐述其在政治、经济、思想文化等方面的表现和危害，以及如何彻底清除这些消极影响。",
                "description": "微调专题测试 - 地主买办阶级残余影响分析",
                "config": "analytical"
            },
            {
                "input": "被推翻的地主买办阶级残余对社会有何影响？请从政治、经济、文化等多个层面进行深入分析。",
                "description": "微调专题测试",
                "config": "analytical"
            }
        ]
        
        print(f"\n🎯 开始 {len(test_cases)} 项终极测试:")
        
        total_score = 0
        successful_tests = 0
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"测试 {i}/{len(test_cases)}: {test_case['description']}")
            print(f"{'='*80}")
            print(f"📝 问题: {test_case['input']}")
            
            try:
                response, gen_time, truncated, used_config = optimized_generate(
                    model, tokenizer, test_case['input'], 
                    test_case.get('config', 'standard')
                )
                
                print(f"\n🤖 回复:")
                print(f"{response}")
                
                # 质量评估
                score, details = comprehensive_quality_evaluation(
                    test_case['input'], response, gen_time, truncated, used_config
                )
                
                print(f"\n📊 详细评估:")
                print(f"总分: {score}/100")
                print(f"配置: {used_config}")
                for detail in details:
                    print(f"  {detail}")
                
                results.append({
                    "test": test_case['description'],
                    "score": score,
                    "config": used_config,
                    "truncated": truncated,
                    "time": gen_time,
                    "length": len(response)
                })
                
                total_score += score
                successful_tests += 1
                
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
                results.append({
                    "test": test_case['description'],
                    "score": 0,
                    "error": str(e)
                })
                continue
        
        # 生成详细报告
        print(f"\n{'='*80}")
        print(f"🎯 终极测试报告")
        print(f"{'='*80}")
        
        if successful_tests > 0:
            avg_score = total_score / successful_tests
            avg_length = sum(r.get('length', 0) for r in results if 'length' in r) / successful_tests
            avg_time = sum(r.get('time', 0) for r in results if 'time' in r) / successful_tests
            truncation_rate = sum(1 for r in results if r.get('truncated', False)) / successful_tests * 100
            
            print(f"✅ 成功测试: {successful_tests}/{len(test_cases)}")
            print(f"📊 平均得分: {avg_score:.1f}/100")
            print(f"📏 平均长度: {avg_length:.0f} 字符")
            print(f"⏱️  平均用时: {avg_time:.1f} 秒")
            print(f"✂️  截断率: {truncation_rate:.1f}%")
            
            # 评级
            if avg_score >= 85:
                grade = "🏆 优秀"
            elif avg_score >= 75:
                grade = "🥇 良好"
            elif avg_score >= 65:
                grade = "🥈 中等"
            elif avg_score >= 50:
                grade = "🥉 及格"
            else:
                grade = "❌ 不及格"
            
            print(f"🎖️  综合评级: {grade}")
            
            # 保存详细结果
            result_file = f"./test_results_ultimate_{int(time.time())}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "success_rate": successful_tests / len(test_cases),
                        "average_score": avg_score,
                        "average_length": avg_length,
                        "average_time": avg_time,
                        "truncation_rate": truncation_rate,
                        "grade": grade
                    },
                    "detailed_results": results
                }, f, ensure_ascii=False, indent=2)
            
            print(f"📄 详细结果保存至: {result_file}")
            
        else:
            print("❌ 所有测试都失败了")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"❌ 终极测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 Qwen3 LoRA 终极优化测试系统")
    print("=" * 60)
    
    # 获取模型路径
    if len(sys.argv) > 1:
        adapter_path = sys.argv[1]
    else:
        adapter_path = find_latest_model()
        if not adapter_path:
            print("❌ 未找到微调模型，请先运行训练脚本")
            print("💡 使用方法: uv run ultimate_test.py [模型路径]")
            return
    
    # 运行终极测试
    success = ultimate_test(adapter_path)
    
    if success:
        print("\n🎉 终极测试完成! 检查结果文件获取详细报告。")
    else:
        print("\n❌ 终极测试失败!")

if __name__ == "__main__":
    main()
