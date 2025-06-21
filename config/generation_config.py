# Qwen3 LoRA 生成参数优化配置
# 针对不同场景的最佳参数设置

generation_configs = {
    # 标准对话配置 - 平衡质量和速度
    "standard": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": False,
        "num_beams": 1,
    },
    
    # 高质量配置 - 追求最佳输出质量
    "high_quality": {
        "max_new_tokens": 768,
        "temperature": 0.6,
        "do_sample": True,
        "top_p": 0.85,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "length_penalty": 1.2,
        "no_repeat_ngram_size": 4,
        "early_stopping": False,
        "num_beams": 2,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
    },
    
    # 快速配置 - 优先速度
    "fast": {
        "max_new_tokens": 256,
        "temperature": 0.8,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 60,
        "repetition_penalty": 1.05,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "num_beams": 1,
    },
    
    # 创意写作配置 - 提高创造性
    "creative": {
        "max_new_tokens": 512,
        "temperature": 0.9,
        "do_sample": True,
        "top_p": 0.92,
        "top_k": 80,
        "repetition_penalty": 1.08,
        "length_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "early_stopping": False,
        "num_beams": 1,
    },
    
    # 专业分析配置 - 逻辑性和准确性
    "analytical": {
        "max_new_tokens": 640,
        "temperature": 0.5,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "repetition_penalty": 1.2,
        "length_penalty": 1.3,
        "no_repeat_ngram_size": 4,
        "early_stopping": False,
        "num_beams": 2,
    },
    
    # 长文本配置 - 生成长篇内容
    "long_form": {
        "max_new_tokens": 1024,
        "temperature": 0.65,
        "do_sample": True,
        "top_p": 0.88,
        "top_k": 45,
        "repetition_penalty": 1.12,
        "length_penalty": 1.4,
        "no_repeat_ngram_size": 5,
        "early_stopping": False,
        "num_beams": 1,
    }
}

# 任务类型到配置的映射
task_to_config = {
    "自我介绍": "analytical",
    "知识解释": "analytical", 
    "旅行规划": "standard",
    "创意写作": "creative",
    "诗歌创作": "creative",
    "问题分析": "analytical",
    "故事创作": "creative",
    "技术解答": "analytical",
    "日常对话": "standard",
    "长篇内容": "long_form",
}

# 基于输入长度和复杂度的动态配置选择
def select_config_by_input(input_text, user_preference=None):
    """根据输入自动选择最佳配置"""
    
    if user_preference and user_preference in generation_configs:
        return user_preference
    
    input_length = len(input_text)
    input_lower = input_text.lower()
    
    # 检测任务类型关键词
    keywords = {
        "creative": ["写", "创作", "诗", "故事", "想象", "创意"],
        "analytical": ["解释", "分析", "比较", "原理", "为什么", "如何"],
        "long_form": ["详细", "完整", "全面", "深入", "具体"],
        "fast": ["简单", "快速", "简要", "概括"]
    }
    
    for config_type, words in keywords.items():
        if any(word in input_lower for word in words):
            return config_type
    
    # 基于长度选择
    if input_length > 100:
        return "analytical"
    elif input_length > 50:
        return "standard"
    else:
        return "fast"

# 截断检测模式
truncation_patterns = {
    "sentence_incomplete": r'[^。！？\.!?]\s*$',
    "punctuation_missing": r'[，,：:；;]\s*$',
    "word_incomplete": r'[的地得在是]\s*$',
    "number_incomplete": r'[一二三四五六七八九十]\s*$',
    "conjunction_incomplete": r'[而且但是因为所以]\s*$',
    "quote_mismatch": lambda text: len([c for c in text if c in '"\'""''']) % 2 != 0,
    "bracket_mismatch": lambda text: any(
        text.count(open_b) != text.count(close_b) 
        for open_b, close_b in [('(', ')'), ('（', '）'), ('[', ']'), ('【', '】')]
    ),
    "too_short": lambda text: len(text.strip()) < 10,
    "ellipsis": r'\.{3,}|。{3,}',
}

# 质量评估权重
quality_weights = {
    "has_content": 20,
    "sufficient_length": 20,
    "completeness": 20,
    "relevance": 20,
    "language_quality": 10,
    "generation_speed": 10,
}

# 生成模式
generation_modes = {
    "single": "单次生成",
    "retry": "重试生成", 
    "continue": "续写生成",
    "beam": "束搜索生成",
    "interactive": "交互式生成"
}
