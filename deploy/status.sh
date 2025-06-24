#!/bin/bash

# vLLM 服务状态检查脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "📊 vLLM 服务状态检查"
echo "========================"

# 检查配置文件
if [ -f "config.json" ]; then
    echo "📋 配置信息:"
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('config.json'))['model_path'])")
    PORT=$(python3 -c "import json; print(json.load(open('config.json'))['port'])")
    HOST=$(python3 -c "import json; print(json.load(open('config.json'))['host'])")
    DEPLOY_TIME=$(python3 -c "import json; print(json.load(open('config.json'))['deployment_time'])")
    
    echo -e "   📁 模型路径: ${BLUE}$MODEL_PATH${NC}"
    echo -e "   📡 服务地址: ${BLUE}$HOST:$PORT${NC}"
    echo -e "   🕐 部署时间: ${BLUE}$DEPLOY_TIME${NC}"
else
    echo -e "${RED}❌ 配置文件不存在${NC}"
    PORT=8000
    HOST="localhost"
fi

echo

# 检查进程状态
echo "🔍 进程状态:"
if [ -f "service.pid" ]; then
    PID=$(cat service.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "   ✅ 服务进程运行中 (PID: ${GREEN}$PID${NC})"
        
        # 显示进程信息
        PROCESS_INFO=$(ps -p $PID -o pid,ppid,cmd,etime,pcpu,pmem --no-headers)
        echo "   📈 进程详情:"
        echo "      PID: $(echo $PROCESS_INFO | awk '{print $1}')"
        echo "      运行时间: $(echo $PROCESS_INFO | awk '{print $4}')"
        echo "      CPU使用率: $(echo $PROCESS_INFO | awk '{print $5}')%"
        echo "      内存使用率: $(echo $PROCESS_INFO | awk '{print $6}')%"
    else
        echo -e "   ${RED}❌ 服务进程未运行 (PID 文件存在但进程不存在)${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  PID 文件不存在${NC}"
fi

# 检查端口占用
echo
echo "🌐 网络状态:"
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "   ✅ 端口 ${GREEN}$PORT${NC} 已监听"
    LISTENING_PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
    echo "   📡 监听进程 PID: $LISTENING_PID"
else
    echo -e "   ${RED}❌ 端口 $PORT 未被监听${NC}"
fi

# 检查 API 可访问性
echo
echo "🔗 API 状态:"
if curl -s --max-time 5 http://$HOST:$PORT/v1/models > /dev/null 2>&1; then
    echo -e "   ✅ API ${GREEN}可访问${NC}"
    
    # 获取模型信息
    MODELS=$(curl -s http://$HOST:$PORT/v1/models | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = [model['id'] for model in data.get('data', [])]
    print('可用模型: ' + ', '.join(models))
except:
    print('无法解析模型信息')
" 2>/dev/null)
    echo "   🤖 $MODELS"
else
    echo -e "   ${RED}❌ API 不可访问${NC}"
fi

# 检查 GPU 使用情况
echo
echo "🖥️  GPU 状态:"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    echo "   📊 GPU信息: $GPU_INFO"
    
    MEMORY_USED=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    MEMORY_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | tr -d ' ')
    GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f4 | tr -d ' ')
    
    MEMORY_PERCENT=$((MEMORY_USED * 100 / MEMORY_TOTAL))
    echo "   💾 显存使用: ${MEMORY_USED}MB / ${MEMORY_TOTAL}MB (${MEMORY_PERCENT}%)"
    echo "   ⚡ GPU利用率: ${GPU_UTIL}%"
else
    echo -e "   ${YELLOW}⚠️  未检测到 NVIDIA GPU${NC}"
fi

# 检查日志
echo
echo "📋 日志状态:"
if [ -f "service.log" ]; then
    LOG_SIZE=$(stat -c%s service.log 2>/dev/null || echo "0")
    LOG_SIZE_MB=$((LOG_SIZE / 1024 / 1024))
    echo "   📄 日志文件大小: ${LOG_SIZE_MB}MB"
    
    # 显示最新的错误或警告
    RECENT_ERRORS=$(tail -n 100 service.log | grep -i -E "(error|exception|failed)" | tail -n 3)
    if [ ! -z "$RECENT_ERRORS" ]; then
        echo -e "   ${RED}🚨 最近的错误:${NC}"
        echo "$RECENT_ERRORS" | sed 's/^/      /'
    fi
    
    RECENT_WARNINGS=$(tail -n 100 service.log | grep -i "warning" | tail -n 3)
    if [ ! -z "$RECENT_WARNINGS" ]; then
        echo -e "   ${YELLOW}⚠️  最近的警告:${NC}"
        echo "$RECENT_WARNINGS" | sed 's/^/      /'
    fi
else
    echo -e "   ${YELLOW}⚠️  日志文件不存在${NC}"
fi

echo
echo "========================"

# 总体状态评估
if [ -f "service.pid" ] && ps -p $(cat service.pid) > /dev/null 2>&1 && curl -s --max-time 5 http://$HOST:$PORT/v1/models > /dev/null 2>&1; then
    echo -e "🎉 总体状态: ${GREEN}健康${NC}"
    echo
    echo "🔧 管理命令:"
    echo "   📊 查看实时日志: tail -f service.log"
    echo "   🛑 停止服务: ./stop_service.sh"
    echo "   🔄 重启服务: ./deploy.sh restart"
    echo "   🧪 测试推理: ./test_inference.sh"
else
    echo -e "🚨 总体状态: ${RED}异常${NC}"
    echo
    echo "🔧 修复建议:"
    echo "   🚀 重新部署: ./deploy.sh"
    echo "   📋 查看日志: tail -f service.log"
    echo "   🛑 停止服务: ./stop_service.sh"
fi
