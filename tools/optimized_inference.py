import requests
import re
import time

class OptimizedInference:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_response(self, prompt, task_type="general"):
        """根据任务类型优化生成参数"""
        
        # 根据任务类型选择参数
        if task_type == "factual":
            params = {
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.3,
                "max_tokens": 100,
                "stop": ["\n\n", "问题：", "另外", "此外", "补充："]
            }
        elif task_type == "creative":
            params = {
                "temperature": 0.8,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "max_tokens": 250,
                "stop": ["\n\n问题", "\n\n---", "总结："]
            }
        else:  # general
            params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "max_tokens": 150,
                "stop": ["\n\n", "问题：", "用户：", "助手："]
            }
        
        # 构建请求
        data = {
            "model": "./models/Qwen3-0.6B",
            "prompt": prompt,
            **params
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_text = result["choices"][0]["text"]
                
                # 后处理优化
                cleaned_text = self.clean_response(raw_text)
                quality = self.evaluate_quality(cleaned_text)
                
                return {
                    "success": True,
                    "response": cleaned_text,
                    "raw_response": raw_text,
                    "tokens_used": result["usage"]["total_tokens"],
                    "quality_score": quality["quality_score"],
                    "issues": quality["issues"]
                }
            else:
                return {"success": False, "error": f"API错误: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clean_response(self, text):
        """清理回复文本"""
        # 移除开头的重复prompt内容
        text = text.strip()
        
        # 移除常见的重复开头
        prefixes = ["答案：", "回答：", "解答：", "答："]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 移除重复句子
        sentences = text.split('。')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        text = '。'.join(unique_sentences)
        if text and not text.endswith(('。', '！', '？')):
            text += '。'
        
        # 控制长度，保持完整性
        if len(text) > 200:
            sentences = text.split('。')
            result = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= 180:
                    result.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
            
            text = '。'.join(result)
            if text and not text.endswith(('。', '！', '？')):
                text += '。'
        
        return text
    
    def evaluate_quality(self, response):
        """评估回复质量"""
        issues = []
        
        # 检查重复
        sentences = response.split('。')
        if len(sentences) != len(set(sentences)):
            issues.append("包含重复句子")
        
        # 检查长度
        if len(response) < 20:
            issues.append("回复过短")
        elif len(response) > 300:
            issues.append("回复过长")
        
        # 检查完整性
        if not response.strip().endswith(('。', '！', '？')):
            issues.append("回复不完整")
        
        # 检查关键词重复
        words = response.split()
        word_freq = {}
        for word in words:
            if len(word) > 1:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [word for word, freq in word_freq.items() if freq > 3]
        if repeated_words:
            issues.append(f"词汇重复: {repeated_words}")
        
        return {
            "quality_score": max(0, 10 - len(issues)),
            "issues": issues,
            "is_good": len(issues) == 0
        }

def create_optimized_prompt(question, task_type="general"):
    """创建优化的prompt"""
    
    if task_type == "factual":
        return f"""请简洁、准确地回答以下问题：

问题：{question}

要求：
- 回答要事实准确
- 控制在100字以内
- 避免重复表述

回答："""
    
    elif task_type == "creative":
        return f"""请发挥创造力回答以下问题：

问题：{question}

要求：
- 可以有想象力和创意
- 语言生动有趣
- 内容丰富但不冗余

回答："""
    
    else:  # general
        return f"""请友好、清晰地回答以下问题：

问题：{question}

请提供有用的回答："""

def main():
    """主函数：演示优化推理的使用"""
    
    print("🚀 启动优化推理系统...")
    inference = OptimizedInference()
    
    # 测试不同类型的问题
    test_cases = [
        ("请简要介绍人工智能的定义。", "factual"),
        ("描述一个美丽的春天早晨。", "creative"),
        ("你好，今天天气怎么样？", "general"),
        ("什么是量子计算？", "factual"),
        ("如何学好编程？", "general")
    ]
    
    for i, (question, task_type) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"📝 测试 {i}/{len(test_cases)}")
        print(f"🎯 任务类型: {task_type}")
        print(f"❓ 问题: {question}")
        
        # 创建优化的prompt
        optimized_prompt = create_optimized_prompt(question, task_type)
        
        # 生成回复
        start_time = time.time()
        result = inference.generate_response(optimized_prompt, task_type)
        end_time = time.time()
        
        if result["success"]:
            print(f"✅ 回答: {result['response']}")
            print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
            print(f"🔢 Token使用: {result['tokens_used']}")
            print(f"⭐ 质量评分: {result['quality_score']}/10")
            
            if result['issues']:
                print(f"⚠️  发现问题: {', '.join(result['issues'])}")
            else:
                print("✨ 回复质量良好！")
                
        else:
            print(f"❌ 生成失败: {result['error']}")
        
        # 添加延迟避免请求过快
        if i < len(test_cases):
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print("🎉 测试完成！")

def interactive_mode():
    """交互模式：用户可以输入问题并选择任务类型"""
    
    print("🤖 进入交互式优化推理模式")
    print("输入 'quit' 退出程序")
    
    inference = OptimizedInference()
    
    while True:
        print(f"\n{'-'*40}")
        question = input("请输入问题: ").strip()
        
        if question.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break
        
        if not question:
            continue
        
        print("\n请选择任务类型:")
        print("1. factual (事实性问答)")
        print("2. creative (创意性回答)")
        print("3. general (一般对话)")
        
        task_choice = input("请输入选择 (1-3, 默认为3): ").strip()
        
        task_map = {"1": "factual", "2": "creative", "3": "general"}
        task_type = task_map.get(task_choice, "general")
        
        print(f"\n🎯 任务类型: {task_type}")
        print("🔄 正在生成回复...")
        
        # 创建优化prompt并生成回复
        optimized_prompt = create_optimized_prompt(question, task_type)
        start_time = time.time()
        result = inference.generate_response(optimized_prompt, task_type)
        end_time = time.time()
        
        if result["success"]:
            print(f"\n✅ 回答: {result['response']}")
            print(f"⏱️  耗时: {end_time - start_time:.2f}秒")
            print(f"⭐ 质量评分: {result['quality_score']}/10")
            
            if result['issues']:
                print(f"⚠️  注意: {', '.join(result['issues'])}")
        else:
            print(f"❌ 生成失败: {result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
