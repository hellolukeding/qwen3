#!/bin/bash

# vLLM 服务停止脚本

echo "正在停止vLLM服务..."

# 查找并停止vLLM进程
PIDS=$(pgrep -f "vllm.*serve")

if [ -z "$PIDS" ]; then
    echo "未找到运行中的vLLM服务"
else
    echo "找到vLLM进程: $PIDS"
    
    # 优雅停止
    kill $PIDS
    sleep 5
    
    # 检查是否还在运行
    REMAINING=$(pgrep -f "vllm.*serve")
    if [ ! -z "$REMAINING" ]; then
        echo "强制停止剩余进程: $REMAINING"
        kill -9 $REMAINING
    fi
    
    echo "✅ vLLM服务已停止"
fi

# 清理端口
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null; then
    echo "端口8000仍被占用，尝试释放..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
fi

echo "🧹 清理完成"
