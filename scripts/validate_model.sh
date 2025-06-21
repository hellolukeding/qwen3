#!/bin/bash
# 微调模型验证脚本

MODEL_PATH=${1:-""}

if [ -z "$MODEL_PATH" ]; then
    echo "🔍 自动查找最新的微调模型..."
    MODEL_PATH=$(find ./output -name "qwen3-lora-*" -type d | sort -r | head -1)
    
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ 未找到微调模型目录"
        echo "请手动指定模型路径，例如:"
        echo "  ./scripts/validate_model.sh ./output/qwen3-lora-lowmem-20250621-195657"
        exit 1
    fi
    
    echo "✅ 找到模型: $MODEL_PATH"
fi

echo "🔍 验证微调模型: $MODEL_PATH"

# 检查模型目录
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型目录不存在: $MODEL_PATH"
    echo "请指定正确的模型路径"
    exit 1
fi

echo "✅ 模型目录存在"

# 检查必需文件
required_files=("adapter_config.json" "adapter_model.safetensors")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
        missing_files+=("$file")
    fi
done

# 检查可选文件
optional_files=("README.md" "training_args.bin")
for file in "${optional_files[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✅ $file 存在"
    else
        echo "⚠️  $file 缺失（可选）"
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ 缺少必需文件，模型可能不完整"
    exit 1
fi

# 显示模型信息
echo ""
echo "📊 模型信息:"
echo "模型路径: $MODEL_PATH"
echo "创建时间: $(stat -c %y "$MODEL_PATH" 2>/dev/null || stat -f %Sm "$MODEL_PATH" 2>/dev/null || echo "未知")"

if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    echo "LoRA 配置:"
    cat "$MODEL_PATH/adapter_config.json" | python3 -m json.tool 2>/dev/null || echo "无法解析配置文件"
fi

# 运行 Python 验证
echo ""
echo "🧪 运行功能测试..."

uv run python -c "
import sys
import os
sys.path.append('.')

try:
    # 简单的模型加载测试
    from transformers import AutoTokenizer
    from peft import PeftConfig
    
    model_path = '$MODEL_PATH'
    base_model_path = './models/Qwen3-0.6B'
    
    print('📦 检查 LoRA 配置...')
    config = PeftConfig.from_pretrained(model_path)
    print(f'✅ LoRA 配置加载成功')
    print(f'   基础模型: {config.base_model_name_or_path}')
    print(f'   任务类型: {config.task_type}')
    print(f'   LoRA rank: {config.r}')
    print(f'   LoRA alpha: {config.lora_alpha}')
    
    print('📝 检查分词器...')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f'✅ 分词器加载成功，词汇表大小: {len(tokenizer)}')
    
    print('🎉 模型验证通过!')
    
except Exception as e:
    print(f'❌ 验证失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 模型验证完成！"
    echo ""
    echo "📝 使用建议:"
    echo "  # 测试模型功能"
    echo "  uv run python scripts/test_finetuned_model.py $MODEL_PATH"
    echo ""
    echo "  # 交互式对话"
    echo "  uv run python scripts/chat_with_finetuned.py $MODEL_PATH"
else
    echo "❌ 模型验证失败"
    exit 1
fi
