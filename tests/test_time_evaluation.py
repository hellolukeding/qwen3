#!/usr/bin/env python3
"""
快速时间评估测试工具
"""

from response_optimizer import QualityEvaluator, TimeEvaluator

def test_time_evaluation():
    """测试时间评估功能"""
    
    time_evaluator = TimeEvaluator()
    
    # 测试不同响应时间
    test_times = [1.5, 3.2, 7.8, 15.6, 25.3]
    
    print("⏱️  响应时间评估测试\n")
    print(f"{'时间(s)':<8} {'评分':<6} {'等级':<8} {'百分位':<12} {'描述'}")
    print("-" * 60)
    
    for response_time in test_times:
        eval_result = time_evaluator.evaluate_time(response_time)
        
        print(f"{response_time:<8} {eval_result['time_score']:<6} {eval_result['time_level']:<8} "
              f"{eval_result['percentile']:<12} {eval_result['description']}")
        
        if eval_result['suggestions']:
            print(f"         建议: {eval_result['suggestions'][0]}")
        print()

def test_comprehensive_evaluation():
    """测试综合评估功能"""
    
    evaluator = QualityEvaluator()
    
    # 测试数据：回复内容和响应时间
    test_cases = [
        ("这是一个很好的问题。机器学习是人工智能的重要分支，它让计算机能够从数据中学习。", 1.8),
        ("AI是很好的。AI是很好的。AI是很好的技术。", 3.5),
        ("人工智能", 0.5),
        ("人工智能是一个非常复杂的领域，涉及到计算机科学、数学、统计学等多个学科的知识。它的应用范围很广，包括自然语言处理、计算机视觉、机器学习等。", 12.0)
    ]
    
    print("\n🔍 综合评估测试\n")
    
    for i, (response, response_time) in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"回复: {response}")
        print(f"时间: {response_time}s")
        
        evaluation = evaluator.evaluate_comprehensive(response, response_time)
        
        print(f"综合评分: {evaluation['comprehensive_score']}/10 ({evaluation['overall_level']})")
        print(f"内容评分: {evaluation['content_evaluation']['content_score']}/10")
        print(f"时间评分: {evaluation['time_evaluation']['time_score']}/10")
        
        if evaluation['content_evaluation']['issues']:
            print(f"内容问题: {', '.join(evaluation['content_evaluation']['issues'])}")
        
        if evaluation['suggestions']:
            print(f"改进建议: {evaluation['suggestions'][0]}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_time_evaluation()
    test_comprehensive_evaluation()
