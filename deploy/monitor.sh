#!/bin/bash

# vLLM 服务监控脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 读取配置
if [ -f "config.json" ]; then
    PORT=$(python3 -c "import json; print(json.load(open('config.json'))['port'])")
    HOST=$(python3 -c "import json; print(json.load(open('config.json'))['host'])")
else
    PORT=8000
    HOST="localhost"
fi

# 监控间隔（秒）
INTERVAL=${1:-10}

echo "🔍 vLLM 服务监控启动"
echo "监控间隔: ${INTERVAL}秒"
echo "按 Ctrl+C 停止监控"
echo "========================"

# 创建监控日志文件
MONITOR_LOG="monitor.log"
echo "$(date): 监控开始" >> $MONITOR_LOG

# 信号处理
trap 'echo ""; echo "监控停止"; exit 0' INT

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # 清屏并显示时间
    clear
    echo -e "${BLUE}🕐 $TIMESTAMP${NC}"
    echo "========================"
    
    # 检查进程状态
    if [ -f "service.pid" ]; then
        PID=$(cat service.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "✅ 进程状态: ${GREEN}运行中${NC} (PID: $PID)"
            
            # 获取进程资源使用情况
            PROCESS_INFO=$(ps -p $PID -o pcpu,pmem,etime --no-headers)
            CPU_USAGE=$(echo $PROCESS_INFO | awk '{print $1}')
            MEM_USAGE=$(echo $PROCESS_INFO | awk '{print $2}')
            RUNTIME=$(echo $PROCESS_INFO | awk '{print $3}')
            
            echo "   CPU: ${CPU_USAGE}%"
            echo "   内存: ${MEM_USAGE}%"
            echo "   运行时间: $RUNTIME"
        else
            echo -e "❌ 进程状态: ${RED}未运行${NC}"
            echo "$(date): 服务进程未运行" >> $MONITOR_LOG
        fi
    else
        echo -e "❌ 进程状态: ${RED}PID文件不存在${NC}"
    fi
    
    echo
    
    # 检查端口状态
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "✅ 端口状态: ${GREEN}监听中${NC} (:$PORT)"
    else
        echo -e "❌ 端口状态: ${RED}未监听${NC} (:$PORT)"
        echo "$(date): 端口未监听" >> $MONITOR_LOG
    fi
    
    # 检查 API 可访问性
    API_STATUS=$(curl -s --max-time 3 http://$HOST:$PORT/v1/models > /dev/null 2>&1 && echo "可访问" || echo "不可访问")
    if [ "$API_STATUS" = "可访问" ]; then
        echo -e "✅ API状态: ${GREEN}$API_STATUS${NC}"
    else
        echo -e "❌ API状态: ${RED}$API_STATUS${NC}"
        echo "$(date): API不可访问" >> $MONITOR_LOG
    fi
    
    echo
    
    # GPU 监控
    if command -v nvidia-smi &> /dev/null; then
        echo "🖥️  GPU 状态:"
        
        # 获取 GPU 信息
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
        
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1)
        MEMORY_USED=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
        MEMORY_TOTAL=$(echo $GPU_INFO | cut -d',' -f3 | tr -d ' ')
        GPU_UTIL=$(echo $GPU_INFO | cut -d',' -f4 | tr -d ' ')
        GPU_TEMP=$(echo $GPU_INFO | cut -d',' -f5 | tr -d ' ')
        
        MEMORY_PERCENT=$((MEMORY_USED * 100 / MEMORY_TOTAL))
        
        echo "   名称: $GPU_NAME"
        echo "   显存: ${MEMORY_USED}MB / ${MEMORY_TOTAL}MB (${MEMORY_PERCENT}%)"
        echo "   利用率: ${GPU_UTIL}%"
        echo "   温度: ${GPU_TEMP}°C"
        
        # 显存使用率颜色提示
        if [ $MEMORY_PERCENT -gt 90 ]; then
            echo -e "   ${RED}⚠️  显存使用率过高${NC}"
        elif [ $MEMORY_PERCENT -gt 70 ]; then
            echo -e "   ${YELLOW}⚠️  显存使用率较高${NC}"
        fi
        
        # GPU 温度提示
        if [ $GPU_TEMP -gt 80 ]; then
            echo -e "   ${RED}🔥 GPU 温度过高${NC}"
        elif [ $GPU_TEMP -gt 70 ]; then
            echo -e "   ${YELLOW}🌡️  GPU 温度较高${NC}"
        fi
    fi
    
    echo
    
    # 系统资源监控
    echo "💻 系统资源:"
    
    # CPU 使用率
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "   CPU总使用率: ${CPU_USAGE}%"
    
    # 内存使用率
    MEM_INFO=$(free | grep Mem)
    MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
    MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
    MEM_PERCENT=$((MEM_USED * 100 / MEM_TOTAL))
    echo "   内存使用率: ${MEM_PERCENT}%"
    
    # 磁盘使用率
    DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    echo "   磁盘使用率: ${DISK_USAGE}%"
    
    echo
    
    # 日志监控（显示最新的错误）
    if [ -f "service.log" ]; then
        RECENT_ERRORS=$(tail -n 50 service.log | grep -i -E "(error|exception|failed)" | tail -n 3)
        if [ ! -z "$RECENT_ERRORS" ]; then
            echo -e "${RED}🚨 最近错误:${NC}"
            echo "$RECENT_ERRORS" | sed 's/^/   /'
            echo
        fi
        
        RECENT_WARNINGS=$(tail -n 50 service.log | grep -i "warning" | tail -n 2)
        if [ ! -z "$RECENT_WARNINGS" ]; then
            echo -e "${YELLOW}⚠️  最近警告:${NC}"
            echo "$RECENT_WARNINGS" | sed 's/^/   /'
            echo
        fi
    fi
    
    # 请求统计（如果有日志）
    if [ -f "service.log" ]; then
        RECENT_REQUESTS=$(tail -n 100 service.log | grep -c "POST /v1/chat/completions" || echo "0")
        echo "📊 最近请求数: $RECENT_REQUESTS (最近100行日志)"
    fi
    
    echo
    echo "按 Ctrl+C 停止监控 | 下次刷新: ${INTERVAL}秒后"
    echo "========================"
    
    # 记录关键指标到日志
    echo "$(date): CPU:${CPU_USAGE}% MEM:${MEM_PERCENT}% GPU_UTIL:${GPU_UTIL}% GPU_MEM:${MEMORY_PERCENT}% API:$API_STATUS" >> $MONITOR_LOG
    
    sleep $INTERVAL
done
