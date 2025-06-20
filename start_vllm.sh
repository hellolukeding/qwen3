#!/bin/bash

# vLLM 服务启动脚本
# 根据GPU类型自动选择最优配置

# 激活虚拟环境
source qwen-env/bin/activate

# 检测GPU类型和显存
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')

echo "检测到GPU: $GPU_INFO"
echo "显存大小: ${GPU_MEMORY}MB"

# 根据显存大小设置参数
if [ $GPU_MEMORY -lt 6000 ]; then
    # 小于6GB显存
    DTYPE="half"
    MAX_LEN="2048"
    GPU_UTIL="0.7"
    echo "使用小显存配置"
elif [ $GPU_MEMORY -lt 12000 ]; then
    # 6-12GB显存
    DTYPE="half"
    MAX_LEN="8192"
    GPU_UTIL="0.8"
    echo "使用中等显存配置"
else
    # 大于12GB显存
    DTYPE="bfloat16"
    MAX_LEN="16384"
    GPU_UTIL="0.9"
    echo "使用大显存配置"
fi

# 检查端口是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
    echo "端口8000已被占用，正在尝试停止现有服务..."
    pkill -f "vllm.*serve"
    sleep 3
fi

echo "启动vLLM服务..."
echo "参数: --dtype $DTYPE --max-model-len $MAX_LEN --gpu-memory-utilization $GPU_UTIL"

# 启动vLLM服务
nohup vllm serve ./Qwen3-0.6B \
    --dtype $DTYPE \
    --port 8000 \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --disable-log-requests \
    > vllm.log 2>&1 &

echo "vLLM服务正在启动，进程ID: $!"
echo "日志文件: vllm.log"
echo "等待30秒后检查服务状态..."

sleep 30

# 检查服务是否启动成功
if curl -s http://localhost:8000/v1/models > /dev/null; then
    echo "✅ vLLM服务启动成功！"
    echo "📡 API地址: http://localhost:8000"
    echo "🔍 查看日志: tail -f vllm.log"
    echo "🛑 停止服务: ./stop_vllm.sh"
else
    echo "❌ vLLM服务启动失败，请检查日志: tail -f vllm.log"
    exit 1
fi
