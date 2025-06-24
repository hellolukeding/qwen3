"""
回复质量评估和优化工具（包含响应时间评估）
"""

import re
import time
import requests
from typing import Dict, List, Tuple, Optional

class ResponseOptimizer:
    """回复内容优化器"""
    
    def __init__(self):
        self.common_prefixes = ["答案：", "回答：", "解答：", "答：", "回复："]
        self.common_suffixes = ["以上", "希望", "如果", "需要更多"]
        
    def clean_response(self, text: str) -> str:
        """清理和优化回复文本"""
        
        # 1. 基础清理
        text = text.strip()
        
        # 2. 移除常见前缀
        for prefix in self.common_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        
        # 3. 移除重复句子
        text = self._remove_duplicate_sentences(text)
        
        # 4. 移除重复短语
        text = self._remove_duplicate_phrases(text)
        
        # 5. 规范化标点
        text = self._normalize_punctuation(text)
        
        # 6. 确保完整性
        text = self._ensure_completeness(text)
        
        # 7. 控制长度
        text = self._control_length(text, max_length=200)
        
        return text
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """移除重复的句子"""
        sentences = re.split(r'[。！？]', text)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        return '。'.join(unique_sentences).rstrip('。') + '。' if unique_sentences else text
    
    def _remove_duplicate_phrases(self, text: str) -> str:
        """移除重复的短语"""
        # 检查并移除重复的3-gram短语
        words = text.split()
        seen_phrases = set()
        cleaned_words = []
        
        for i, word in enumerate(words):
            if i >= 2:
                phrase = ' '.join(words[i-2:i+1])
                if phrase in seen_phrases:
                    continue
                seen_phrases.add(phrase)
            cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号"""
        # 合并重复的标点
        text = re.sub(r'[，,]{2,}', '，', text)
        text = re.sub(r'[。]{2,}', '。', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text)
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _ensure_completeness(self, text: str) -> str:
        """确保回复的完整性"""
        if not text:
            return text
            
        # 如果没有结束标点，尝试在最后一个完整句子处截断
        if not text[-1] in '。！？':
            last_punct_pos = max(
                text.rfind('。'), 
                text.rfind('！'), 
                text.rfind('？')
            )
            if last_punct_pos > len(text) * 0.7:  # 如果截断点不会损失太多内容
                text = text[:last_punct_pos + 1]
            else:
                text += '。'  # 简单添加句号
        
        return text
    
    def _control_length(self, text: str, max_length: int = 200) -> str:
        """控制文本长度，保持完整性"""
        if len(text) <= max_length:
            return text
        
        # 在句号处截断
        sentences = re.split(r'([。！？])', text)
        result = ""
        
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence_with_punct = sentences[i] + sentences[i + 1]
                if len(result + sentence_with_punct) <= max_length:
                    result += sentence_with_punct
                else:
                    break
        
        return result if result else text[:max_length].rsplit('。', 1)[0] + '。'

class TimeEvaluator:
    """响应时间评估器"""
    
    def __init__(self):
        self.time_standards = {
            "excellent": 2.0,    # 优秀：2秒以内
            "good": 5.0,         # 良好：5秒以内
            "acceptable": 10.0,  # 可接受：10秒以内
            "slow": 20.0,        # 较慢：20秒以内
            # 很慢：20秒以上
        }
    
    def evaluate_time(self, response_time: float) -> Dict:
        """评估响应时间"""
        
        if response_time <= self.time_standards["excellent"]:
            score = 10
            level = "优秀"
            description = "响应速度非常快，用户体验极佳"
            suggestions = []
        elif response_time <= self.time_standards["good"]:
            score = 8
            level = "良好"
            description = "响应速度良好，符合用户期望"
            suggestions = ["可以尝试进一步优化以达到优秀水平"]
        elif response_time <= self.time_standards["acceptable"]:
            score = 6
            level = "可接受"
            description = "响应速度可接受，但有改进空间"
            suggestions = [
                "考虑减少max_tokens参数",
                "优化模型推理参数",
                "检查网络连接状况"
            ]
        elif response_time <= self.time_standards["slow"]:
            score = 4
            level = "较慢"
            description = "响应速度较慢，可能影响用户体验"
            suggestions = [
                "显著减少max_tokens参数",
                "使用更快的推理引擎（如vLLM）",
                "检查GPU性能和显存使用",
                "考虑使用更小的模型"
            ]
        else:
            score = 2
            level = "很慢"
            description = "响应速度过慢，严重影响用户体验"
            suggestions = [
                "大幅减少max_tokens参数",
                "切换到vLLM推理引擎",
                "升级硬件配置",
                "检查系统资源占用",
                "考虑使用量化模型"
            ]
        
        return {
            "time_score": score,
            "time_level": level,
            "description": description,
            "response_time": response_time,
            "suggestions": suggestions,
            "percentile": self._get_percentile(response_time)
        }
    
    def _get_percentile(self, response_time: float) -> str:
        """获取响应时间百分位描述"""
        if response_time <= 1.0:
            return "前5%（极快）"
        elif response_time <= 2.0:
            return "前10%（很快）"
        elif response_time <= 5.0:
            return "前30%（较快）"
        elif response_time <= 10.0:
            return "前60%（中等）"
        elif response_time <= 20.0:
            return "前80%（偏慢）"
        else:
            return "后20%（很慢）"

class QualityEvaluator:
    """综合质量评估器（内容+时间）"""
    
    def __init__(self):
        self.time_evaluator = TimeEvaluator()
        self.content_weight = 0.7  # 内容质量权重
        self.time_weight = 0.3     # 时间性能权重
    
    def evaluate_content(self, response: str) -> Dict:
        """评估内容质量"""
        
        issues = []
        score = 10
        
        # 1. 检查长度
        if len(response) < 10:
            issues.append("回复过短")
            score -= 3
        elif len(response) > 500:
            issues.append("回复过长")
            score -= 1
        
        # 2. 检查重复
        sentences = re.split(r'[。！？]', response)
        unique_sentences = set(s.strip() for s in sentences if s.strip())
        if len(sentences) - len(unique_sentences) > 1:
            issues.append("包含重复句子")
            score -= 2
        
        # 3. 检查完整性
        if not response.strip().endswith(('。', '！', '？')):
            issues.append("回复不完整")
            score -= 2
        
        # 4. 检查词汇重复
        words = [w for w in response.split() if len(w) > 1]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        high_freq_words = [w for w, freq in word_freq.items() if freq > 3]
        if high_freq_words:
            issues.append(f"词汇过度重复: {', '.join(high_freq_words[:3])}")
            score -= min(2, len(high_freq_words) * 0.5)
        
        # 5. 检查无意义重复
        if re.search(r'(.{2,})\1{2,}', response):
            issues.append("包含无意义重复内容")
            score -= 2
        
        # 6. 检查语言流畅性
        if len(re.findall(r'[，,]', response)) > len(response) * 0.1:
            issues.append("句子结构可能过于复杂")
            score -= 1
        
        score = max(0, score)
        
        return {
            "content_score": score,
            "issues": issues,
            "word_count": len(words),
            "sentence_count": len(unique_sentences),
            "avg_sentence_length": len(words) / max(1, len(unique_sentences))
        }
    
    def evaluate_comprehensive(self, response: str, response_time: float) -> Dict:
        """综合评估（内容质量+响应时间）"""
        
        # 评估内容质量
        content_eval = self.evaluate_content(response)
        
        # 评估响应时间
        time_eval = self.time_evaluator.evaluate_time(response_time)
        
        # 计算加权综合评分
        comprehensive_score = (
            content_eval["content_score"] * self.content_weight +
            time_eval["time_score"] * self.time_weight
        )
        
        # 确定综合等级
        if comprehensive_score >= 9:
            overall_level = "优秀"
            overall_desc = "回复质量和响应速度都非常出色"
        elif comprehensive_score >= 7:
            overall_level = "良好"
            overall_desc = "回复质量和响应速度整体良好"
        elif comprehensive_score >= 5:
            overall_level = "一般"
            overall_desc = "回复质量或响应速度需要改进"
        else:
            overall_level = "较差"
            overall_desc = "回复质量和响应速度都需要显著优化"
        
        # 生成改进建议
        suggestions = []
        if content_eval["content_score"] < 7:
            suggestions.extend([
                "优化prompt结构以提高回复相关性",
                "调整repetition_penalty参数减少重复",
                "使用更合适的max_tokens长度"
            ])
        
        if time_eval["time_score"] < 7:
            suggestions.extend(time_eval["suggestions"])
        
        return {
            "comprehensive_score": round(comprehensive_score, 1),
            "overall_level": overall_level,
            "overall_desc": overall_desc,
            "content_evaluation": content_eval,
            "time_evaluation": time_eval,
            "suggestions": suggestions,
            "metrics": {
                "content_weight": self.content_weight,
                "time_weight": self.time_weight,
                "weighted_content_score": round(content_eval["content_score"] * self.content_weight, 1),
                "weighted_time_score": round(time_eval["time_score"] * self.time_weight, 1)
            }
        }
    
    def evaluate(self, response: str, response_time: Optional[float] = None) -> Dict:
        """兼容旧版本的评估方法"""
        if response_time is not None:
            return self.evaluate_comprehensive(response, response_time)
        else:
            content_eval = self.evaluate_content(response)
            return {
                "quality_score": content_eval["content_score"],
                "issues": content_eval["issues"],
                "is_good": len(content_eval["issues"]) == 0
            }

class PerformanceBenchmark:
    """性能基准测试工具"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.evaluator = QualityEvaluator()
        self.session = requests.Session()
    
    def single_test(self, prompt: str, **api_params) -> Dict:
        """单次测试"""
        
        data = {
            "model": "./models/Qwen3-0.6B",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            **api_params
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=data,
                timeout=30
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["text"].strip()
                
                evaluation = self.evaluator.evaluate_comprehensive(generated_text, response_time)
                
                return {
                    "success": True,
                    "prompt": prompt,
                    "response": generated_text,
                    "response_time": response_time,
                    "evaluation": evaluation,
                    "api_usage": result.get("usage", {})
                }
            else:
                return {
                    "success": False,
                    "error": f"API错误: {response.status_code}",
                    "response_time": response_time
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def batch_test(self, prompts: List[str], iterations: int = 3, **api_params) -> Dict:
        """批量基准测试"""
        
        results = {
            "total_tests": len(prompts) * iterations,
            "successful_tests": 0,
            "failed_tests": 0,
            "response_times": [],
            "quality_scores": [],
            "time_scores": [],
            "detailed_results": []
        }
        
        print(f"🧪 开始批量测试 - {len(prompts)} 个问题 × {iterations} 次迭代")
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 测试问题 {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            prompt_results = {
                "prompt": prompt,
                "iterations": [],
                "avg_response_time": 0,
                "avg_quality_score": 0,
                "avg_time_score": 0,
                "success_rate": 0
            }
            
            successful_runs = 0
            total_response_time = 0
            total_quality_score = 0
            total_time_score = 0
            
            for j in range(iterations):
                print(f"  🔄 迭代 {j+1}/{iterations}...", end=" ")
                
                test_result = self.single_test(prompt, **api_params)
                
                if test_result["success"]:
                    successful_runs += 1
                    
                    response_time = test_result["response_time"]
                    evaluation = test_result["evaluation"]
                    
                    total_response_time += response_time
                    total_quality_score += evaluation["comprehensive_score"]
                    total_time_score += evaluation["time_evaluation"]["time_score"]
                    
                    results["response_times"].append(response_time)
                    results["quality_scores"].append(evaluation["comprehensive_score"])
                    results["time_scores"].append(evaluation["time_evaluation"]["time_score"])
                    
                    print(f"✅ {response_time:.2f}s (综合: {evaluation['comprehensive_score']}/10)")
                    
                else:
                    results["failed_tests"] += 1
                    print(f"❌ 失败: {test_result['error']}")
                
                prompt_results["iterations"].append(test_result)
            
            # 计算该问题的平均值
            if successful_runs > 0:
                prompt_results["avg_response_time"] = total_response_time / successful_runs
                prompt_results["avg_quality_score"] = total_quality_score / successful_runs
                prompt_results["avg_time_score"] = total_time_score / successful_runs
                prompt_results["success_rate"] = successful_runs / iterations
                
                results["successful_tests"] += successful_runs
            
            results["detailed_results"].append(prompt_results)
        
        # 计算总体统计
        if results["response_times"]:
            results["statistics"] = self._calculate_statistics(results)
        
        return results
    
    def _calculate_statistics(self, results: Dict) -> Dict:
        """计算统计数据"""
        
        response_times = results["response_times"]
        quality_scores = results["quality_scores"]
        time_scores = results["time_scores"]
        
        response_times.sort()
        quality_scores.sort()
        time_scores.sort()
        
        def percentile(data, p):
            index = int(len(data) * p / 100)
            return data[min(index, len(data) - 1)]
        
        return {
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "p50": percentile(response_times, 50),
                "p95": percentile(response_times, 95),
                "p99": percentile(response_times, 99)
            },
            "quality_score": {
                "avg": sum(quality_scores) / len(quality_scores),
                "min": min(quality_scores),
                "max": max(quality_scores)
            },
            "time_score": {
                "avg": sum(time_scores) / len(time_scores),
                "min": min(time_scores),
                "max": max(time_scores)
            },
            "overall_success_rate": results["successful_tests"] / results["total_tests"]
        }
    
    def print_report(self, results: Dict):
        """打印测试报告"""
        
        print(f"\n{'='*60}")
        print("📊 性能基准测试报告")
        print(f"{'='*60}")
        
        print(f"\n📈 总体统计:")
        print(f"  总测试数: {results['total_tests']}")
        print(f"  成功数: {results['successful_tests']}")
        print(f"  失败数: {results['failed_tests']}")
        print(f"  成功率: {results['statistics']['overall_success_rate']:.1%}")
        
        stats = results["statistics"]
        
        print(f"\n⏱️  响应时间统计:")
        print(f"  平均值: {stats['response_time']['avg']:.2f}s")
        print(f"  最小值: {stats['response_time']['min']:.2f}s")
        print(f"  最大值: {stats['response_time']['max']:.2f}s")
        print(f"  中位数: {stats['response_time']['p50']:.2f}s")
        print(f"  95分位: {stats['response_time']['p95']:.2f}s")
        print(f"  99分位: {stats['response_time']['p99']:.2f}s")
        
        print(f"\n🎯 质量评分统计:")
        print(f"  平均综合评分: {stats['quality_score']['avg']:.1f}/10")
        print(f"  最高评分: {stats['quality_score']['max']:.1f}/10")
        print(f"  最低评分: {stats['quality_score']['min']:.1f}/10")
        
        print(f"\n⚡ 时间评分统计:")
        print(f"  平均时间评分: {stats['time_score']['avg']:.1f}/10")
        print(f"  最高时间评分: {stats['time_score']['max']:.1f}/10")
        print(f"  最低时间评分: {stats['time_score']['min']:.1f}/10")
        
        # 响应时间分布
        response_times = results["response_times"]
        excellent = sum(1 for t in response_times if t <= 2.0)
        good = sum(1 for t in response_times if 2.0 < t <= 5.0)
        acceptable = sum(1 for t in response_times if 5.0 < t <= 10.0)
        slow = len(response_times) - excellent - good - acceptable
        
        print(f"\n📊 响应时间分布:")
        print(f"  优秀 (≤2s): {excellent} ({excellent/len(response_times):.1%})")
        print(f"  良好 (2-5s): {good} ({good/len(response_times):.1%})")
        print(f"  可接受 (5-10s): {acceptable} ({acceptable/len(response_times):.1%})")
        print(f"  较慢 (>10s): {slow} ({slow/len(response_times):.1%})")

def main():
    """主测试函数"""
    print("🚀 启动回复质量评估和性能测试工具")
    
    # 创建评估器
    optimizer = ResponseOptimizer()
    evaluator = QualityEvaluator()
    benchmark = PerformanceBenchmark()
    
    # 测试回复优化
    print("\n📝 回复优化测试:")
    test_response = "答案：人工智能是一门科学。人工智能是一门科学。它很有用，它很有用，它很有用。"
    print(f"原始回复: {test_response}")
    
    optimized = optimizer.clean_response(test_response)
    print(f"优化后: {optimized}")
    
    # 内容质量评估
    content_eval = evaluator.evaluate_content(optimized)
    print(f"内容评分: {content_eval['content_score']}/10")
    if content_eval['issues']:
        print(f"发现问题: {', '.join(content_eval['issues'])}")
    
    # 单次综合测试
    print("\n🧪 单次推理测试:")
    test_result = benchmark.single_test("请简要介绍机器学习的基本概念。")
    
    if test_result["success"]:
        evaluation = test_result["evaluation"]
        print(f"问题: {test_result['prompt']}")
        print(f"回答: {test_result['response']}")
        print(f"响应时间: {test_result['response_time']:.2f}s")
        print(f"综合评分: {evaluation['comprehensive_score']}/10 ({evaluation['overall_level']})")
        print(f"内容评分: {evaluation['content_evaluation']['content_score']}/10")
        print(f"时间评分: {evaluation['time_evaluation']['time_score']}/10 ({evaluation['time_evaluation']['time_level']})")
        
        if evaluation['suggestions']:
            print(f"改进建议: {', '.join(evaluation['suggestions'][:2])}")
    else:
        print(f"测试失败: {test_result['error']}")
    
    # 批量基准测试
    print("\n🔬 批量基准测试:")
    test_prompts = [
        "什么是深度学习？",
        "请解释神经网络的工作原理。",
        "人工智能有哪些应用领域？"
    ]
    
    batch_results = benchmark.batch_test(test_prompts, iterations=2)
    benchmark.print_report(batch_results)

if __name__ == "__main__":
    main()
