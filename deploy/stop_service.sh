#!/bin/bash

# vLLM 服务停止脚本

echo "🛑 正在停止 vLLM 服务..."

# 从 PID 文件停止服务
if [ -f "service.pid" ]; then
    PID=$(cat service.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "🔄 停止进程 $PID..."
        kill $PID
        
        # 等待进程结束
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "✅ 服务已成功停止"
                break
            fi
            echo "⏳ 等待进程结束... ($i/10)"
            sleep 1
        done
        
        # 如果进程仍在运行，强制终止
        if ps -p $PID > /dev/null 2>&1; then
            echo "⚡ 强制终止进程..."
            kill -9 $PID
        fi
    else
        echo "⚠️  进程 $PID 已不存在"
    fi
    rm -f service.pid
fi

# 通过进程名停止所有 vLLM 进程
echo "🔍 检查并停止所有 vLLM 进程..."
pkill -f "vllm.*serve" || echo "📭 没有找到运行中的 vLLM 进程"

# 检查端口占用
if [ -f "config.json" ]; then
    PORT=$(python3 -c "import json; print(json.load(open('config.json'))['port'])" 2>/dev/null || echo "8000")
else
    PORT=8000
fi

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  端口 $PORT 仍被占用"
    PIDS=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    echo "🔫 终止占用端口的进程: $PIDS"
    kill -9 $PIDS 2>/dev/null || true
fi

echo "✅ vLLM 服务停止完成"
