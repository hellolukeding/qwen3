#!/bin/bash
# 验证模型路径更新脚本

echo "🔍 验证 Qwen3-0.6B 模型路径更新..."

# 检查模型文件是否存在
if [ -d "./models/Qwen3-0.6B" ]; then
    echo "✅ 模型目录存在: ./models/Qwen3-0.6B"
    
    # 检查关键模型文件
    if [ -f "./models/Qwen3-0.6B/config.json" ]; then
        echo "✅ config.json 存在"
    else
        echo "❌ config.json 缺失"
    fi
    
    if [ -f "./models/Qwen3-0.6B/model.safetensors" ]; then
        echo "✅ model.safetensors 存在"
    else
        echo "❌ model.safetensors 缺失"
    fi
    
    if [ -f "./models/Qwen3-0.6B/tokenizer.json" ]; then
        echo "✅ tokenizer.json 存在"
    else
        echo "❌ tokenizer.json 缺失"
    fi
else
    echo "❌ 模型目录不存在: ./models/Qwen3-0.6B"
    exit 1
fi

echo ""
echo "🔍 检查文件中的路径引用..."

# 检查脚本文件
echo "检查启动脚本..."
if grep -q "./models/Qwen3-0.6B" scripts/start_vllm.sh; then
    echo "✅ start_vllm.sh 路径已更新"
else
    echo "❌ start_vllm.sh 路径未更新"
fi

# 检查测试文件
echo "检查测试文件..."
if grep -q "./models/Qwen3-0.6B" tests/test.py; then
    echo "✅ test.py 路径已更新"
else
    echo "❌ test.py 路径未更新"
fi

if grep -q "./models/Qwen3-0.6B" tests/final_test_vllm.py; then
    echo "✅ final_test_vllm.py 路径已更新"
else
    echo "❌ final_test_vllm.py 路径未更新"
fi

# 检查工具文件
echo "检查工具文件..."
if grep -q "./models/Qwen3-0.6B" tools/optimized_inference.py; then
    echo "✅ optimized_inference.py 路径已更新"
else
    echo "❌ optimized_inference.py 路径未更新"
fi

echo ""
echo "🎯 路径更新验证完成！"
