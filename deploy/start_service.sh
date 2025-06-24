#!/bin/bash

# vLLM 服务启动脚本

set -e

# 读取配置
if [ -f "config.json" ]; then
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['model_path'])" 2>/dev/null || echo "")
    BASE_MODEL_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['base_model_path'])" 2>/dev/null || echo "../models/Qwen3-0.6B")
    PORT=$(python3 -c "import json; print(json.load(open('config.json'))['port'])" 2>/dev/null || echo "8000")
    HOST=$(python3 -c "import json; print(json.load(open('config.json'))['host'])" 2>/dev/null || echo "0.0.0.0")
    
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ 无法读取模型路径，配置文件可能损坏"
        exit 1
    fi
else
    echo "❌ 配置文件不存在，请先运行 deploy.sh"
    exit 1
fi

# 激活虚拟环境
source ../.venv/bin/activate

# 检测GPU类型和显存
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')

echo "🖥️  GPU信息: $GPU_INFO"
echo "💾 显存大小: ${GPU_MEMORY}MB"
echo "📁 模型路径: $MODEL_PATH"

# 根据显存大小设置参数
if [ $GPU_MEMORY -lt 6000 ]; then
    # 小于6GB显存
    DTYPE="half"
    MAX_LEN="2048"
    GPU_UTIL="0.7"
    echo "📉 使用小显存配置"
elif [ $GPU_MEMORY -lt 12000 ]; then
    # 6-12GB显存
    DTYPE="half"
    MAX_LEN="4096"
    GPU_UTIL="0.8"
    echo "📊 使用中等显存配置"
else
    # 大于12GB显存
    DTYPE="bfloat16"
    MAX_LEN="8192"
    GPU_UTIL="0.9"
    echo "📈 使用大显存配置"
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 已被占用，正在停止现有服务..."
    pkill -f "vllm.*serve" || true
    sleep 3
fi

echo "🚀 启动 vLLM 服务..."
echo "   📡 地址: $HOST:$PORT"
echo "   🔧 参数: --dtype $DTYPE --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_UTIL"

# 检查是否是 LoRA 微调模型
if [ -f "$MODEL_PATH/adapter_config.json" ]; then
    echo "🔧 检测到 LoRA 适配器，使用基础模型 + LoRA 方式启动"
    
    # 使用基础模型 + LoRA 适配器
    nohup vllm serve "$BASE_MODEL_PATH" \
        --enable-lora \
        --lora-modules qwen3-lora="$MODEL_PATH" \
        --dtype $DTYPE \
        --host $HOST \
        --port $PORT \
        --max-model-len $MAX_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --disable-log-requests \
        --trust-remote-code \
        > service.log 2>&1 &
else
    echo "🔧 使用完整模型方式启动"
    
    # 直接使用完整模型
    nohup vllm serve "$MODEL_PATH" \
        --dtype $DTYPE \
        --host $HOST \
        --port $PORT \
        --max-model-len $MAX_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --disable-log-requests \
        --trust-remote-code \
        > service.log 2>&1 &
fi

PID=$!
echo "🎯 vLLM 服务正在启动，进程 ID: $PID"
echo "📋 日志文件: service.log"

# 保存进程ID
echo $PID > service.pid

echo "⏳ 等待服务启动..."
