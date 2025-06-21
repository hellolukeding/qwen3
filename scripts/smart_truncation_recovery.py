#!/usr/bin/env -S uv run python
"""
智能截断恢复脚本 - 自动检测并修复模型回复截断问题
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gc
import re

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

def smart_truncation_detector(text):
    """智能截断检测器"""
    if not text or len(text.strip()) == 0:
        return True, "空文本"
    
    text = text.strip()
    
    # 检测模式
    patterns = [
        (r'[。！？\.!?]\s*$', False, "正常结束"),  # 正常句号结尾
        (r'[：:]\s*$', True, "冒号结尾（可能截断）"),
        (r'[，,]\s*$', True, "逗号结尾（可能截断）"),
        (r'[的地得]\s*$', True, "助词结尾（可能截断）"),
        (r'[是在]\s*$', True, "动词结尾（可能截断）"),
        (r'[一二三四五六七八九十]\s*$', True, "数字结尾（可能截断）"),
        (r'[而且但是因为所以]\s*$', True, "连词结尾（可能截断）"),
        (r'\.{3,}', True, "省略号"),
        (r'[^。！？\.!?]\s*$', True, "非标点结尾"),
    ]
    
    for pattern, is_truncated, reason in patterns:
        if re.search(pattern, text):
            return is_truncated, reason
    
    # 检查引号和括号匹配
    quote_chars = ['"', "'", '"', '"', "'", "'"]
    bracket_chars = ['(', ')', '（', '）', '[', ']', '【', '】']
    
    for char in quote_chars:
        if text.count(char) % 2 != 0:
            return True, f"引号不匹配: {char}"
    
    open_brackets = ['(', '（', '[', '【']
    close_brackets = [')', '）', ']', '】']
    
    for i, open_b in enumerate(open_brackets):
        close_b = close_brackets[i]
        if text.count(open_b) != text.count(close_b):
            return True, f"括号不匹配: {open_b}{close_b}"
    
    # 检查长度异常
    if len(text) < 10:
        return True, "回复过短"
    
    return False, "完整回复"

def continue_generation(model, tokenizer, original_question, partial_response, max_attempts=3):
    """继续生成未完成的回复"""
    print(f"🔧 尝试继续生成...")
    
    for attempt in range(max_attempts):
        print(f"  📝 续写尝试 {attempt + 1}/{max_attempts}")
        
        # 构建续写提示
        continue_prompt = f"""继续完成下面的回答，保持语言风格一致，不要重复已有内容：

问题：{original_question}

已有回答：{partial_response}

请继续："""
        
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
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                length_penalty=1.1,
                no_repeat_ngram_size=4,
                early_stopping=False,
                top_p=0.85,
            )
        
        continuation = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        if continuation and len(continuation) > 5:
            # 合并回复
            combined = partial_response + continuation
            
            # 检查新的组合是否完整
            is_truncated, reason = smart_truncation_detector(combined)
            
            print(f"    📏 续写长度: {len(continuation)} 字符")
            print(f"    🔍 检测结果: {reason}")
            
            if not is_truncated:
                return combined, True
            
            # 如果仍然截断，更新部分回复继续尝试
            partial_response = combined
        
        torch.cuda.empty_cache()
        gc.collect()
    
    return partial_response, False

def smart_generate(model, tokenizer, question, max_retries=3):
    """智能生成，自动处理截断"""
    print(f"🧠 智能生成模式")
    
    for retry in range(max_retries):
        print(f"\n🔄 第 {retry + 1}/{max_retries} 轮生成")
        
        messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # 逐步增加生成长度
        max_tokens = 256 + retry * 256
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=False,
                top_p=0.9,
                top_k=50,
            )
        
        generation_time = time.time() - start_time
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        print(f"  📝 生成长度: {len(response)} 字符")
        print(f"  ⏱️  生成时间: {generation_time:.2f} 秒")
        
        # 检测截断
        is_truncated, reason = smart_truncation_detector(response)
        print(f"  🔍 检测结果: {reason}")
        
        if not is_truncated:
            print(f"  ✅ 生成完整!")
            return response, generation_time, False
        
        # 如果截断，尝试续写
        if len(response) > 20:  # 只有在有足够内容时才尝试续写
            print(f"  🔧 尝试续写修复...")
            continued_response, success = continue_generation(
                model, tokenizer, question, response
            )
            
            if success:
                print(f"  ✅ 续写修复成功!")
                return continued_response, generation_time, False
            else:
                print(f"  ⚠️  续写修复失败，继续下一轮")
        
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"  ❌ 所有尝试都未能生成完整回复")
    return response, generation_time, True

def interactive_test(adapter_path):
    """交互式测试模式"""
    print(f"🎮 智能截断恢复 - 交互模式")
    print(f"📁 模型路径: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"❌ 模型路径不存在: {adapter_path}")
        return
    
    try:
        # 加载模型
        model, tokenizer = load_model(adapter_path)
        
        print("\n🎯 交互式对话开始!")
        print("💡 输入 'quit' 或 'exit' 退出")
        print("💡 输入 'test' 运行预设测试案例")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见!")
                    break
                
                if user_input.lower() == 'test':
                    # 运行预设测试
                    test_cases = [
                        "请详细介绍一下人工智能的发展历史和未来趋势。",
                        "如何制作一道美味的红烧肉？请提供详细步骤。",
                        "解释一下量子计算的基本原理和应用前景。"
                    ]
                    
                    for i, test_q in enumerate(test_cases, 1):
                        print(f"\n🧪 测试案例 {i}: {test_q}")
                        response, gen_time, truncated = smart_generate(model, tokenizer, test_q)
                        
                        print(f"\n🤖 回复:")
                        print(response)
                        print(f"\n📊 统计: {len(response)} 字符, {gen_time:.2f} 秒, 截断: {'是' if truncated else '否'}")
                        print("-" * 50)
                    
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n🤖 正在思考...")
                response, gen_time, truncated = smart_generate(model, tokenizer, user_input)
                
                print(f"\n🤖 回复:")
                print(response)
                
                print(f"\n📊 统计信息:")
                print(f"  📝 回复长度: {len(response)} 字符")
                print(f"  ⏱️  生成时间: {gen_time:.2f} 秒")
                print(f"  ✂️  截断状态: {'是' if truncated else '否'}")
                
                # GPU 内存监控
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"  💾 GPU 内存: {memory_used:.2f} GB")
                
            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，退出对话...")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                continue
    
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 Qwen3 智能截断恢复系统")
    print("=" * 50)
    
    # 获取模型路径
    if len(sys.argv) > 1:
        adapter_path = sys.argv[1]
    else:
        adapter_path = find_latest_model()
        if not adapter_path:
            print("❌ 未找到微调模型，请先运行训练脚本")
            print("💡 使用方法: uv run smart_truncation_recovery.py [模型路径]")
            return
    
    # 启动交互模式
    interactive_test(adapter_path)

if __name__ == "__main__":
    main()
