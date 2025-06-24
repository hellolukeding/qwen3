#!/bin/bash

# Qwen3 微调模型部署脚本
# 自动检测并部署最新的微调模型

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查部署依赖..."
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    # 检查 CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA GPU 驱动未安装"
        exit 1
    fi
    
    # 检查 uv
    if ! command -v uv &> /dev/null; then
        log_warning "uv 未安装，正在安装..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc
    fi
    
    log_success "依赖检查完成"
}

# 查找最新微调模型
find_latest_model() {
    log_info "查找最新的微调模型..."
    
    LATEST_MODEL=$(find ../output -name "qwen3-lora-lowmem-*" -type d | sort -V | tail -n 1)
    
    if [ -z "$LATEST_MODEL" ]; then
        log_error "未找到微调模型，请先运行训练"
        exit 1
    fi
    
    log_success "找到最新模型: $LATEST_MODEL"
    return 0
}

# 环境检查
check_environment() {
    log_info "检查运行环境..."
    
    # 检查 GPU 信息
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "未检测到GPU")
    log_info "GPU信息: $GPU_INFO"
    
    # 检查虚拟环境
    if [ ! -d "../.venv" ]; then
        log_warning "虚拟环境不存在，正在创建..."
        cd .. && uv venv .venv && cd deploy
    fi
    
    # 检查依赖是否安装
    if [ ! -f "../uv.lock" ]; then
        log_warning "依赖未安装，正在安装..."
        cd .. && uv sync && cd deploy
    fi
    
    log_success "环境检查完成"
}

# 部署模型
deploy_model() {
    local model_path=$1
    log_info "开始部署模型: $model_path"
    
    # 停止现有服务
    log_info "停止现有服务..."
    ./stop_service.sh 2>/dev/null || true
    
    # 更新配置
    # 转义 GPU 信息中的特殊字符
    GPU_INFO_ESCAPED=$(echo "$GPU_INFO" | sed 's/"/\\"/g')
    
    cat > config.json << EOF
{
    "model_path": "$model_path",
    "base_model_path": "../models/Qwen3-0.6B",
    "port": 8000,
    "host": "0.0.0.0",
    "deployment_time": "$(date -Iseconds)",
    "gpu_info": "$GPU_INFO_ESCAPED"
}
EOF
    
    # 启动服务
    log_info "启动 vLLM 服务..."
    ./start_service.sh
    
    log_success "模型部署完成"
}

# 健康检查
health_check() {
    log_info "进行健康检查..."
    
    # 等待服务启动，最多等待 3 分钟
    local max_wait=180
    local wait_time=0
    local check_interval=10
    
    log_info "等待服务启动（最多 ${max_wait} 秒）..."
    
    while [ $wait_time -lt $max_wait ]; do
        if curl -s --max-time 5 http://localhost:8000/v1/models > /dev/null 2>&1; then
            log_success "服务启动成功，API 可访问"
            
            # 测试推理
            log_info "测试模型推理..."
            if ./test_inference.sh; then
                log_success "推理测试通过"
            else
                log_warning "推理测试失败，但服务已启动"
            fi
            
            log_success "部署完成！"
            echo
            echo "🚀 服务信息:"
            echo "   📡 API地址: http://localhost:8000"
            echo "   🔍 查看日志: tail -f service.log"
            echo "   🛑 停止服务: ./stop_service.sh"
            echo "   📊 服务状态: ./status.sh"
            return 0
        fi
        
        echo "⏳ 等待中... ($((wait_time + check_interval))/${max_wait}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    log_error "服务启动超时（${max_wait}秒），请检查日志: tail -f service.log"
    echo "💡 提示: vLLM 首次启动需要编译模型，可能需要较长时间"
    echo "📋 可以运行 'tail -f service.log' 查看启动进度"
    exit 1
}

# 主函数
main() {
    echo "🚀 Qwen3 微调模型部署开始..."
    echo
    
    check_dependencies
    check_environment
    
    find_latest_model
    
    deploy_model "$LATEST_MODEL"
    health_check
    
    log_success "🎉 部署完成！"
}

# 解析命令行参数
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        ./status.sh
        ;;
    "stop")
        ./stop_service.sh
        ;;
    "restart")
        ./stop_service.sh
        sleep 5
        ./start_service.sh
        ;;
    "help")
        echo "用法: $0 [deploy|status|stop|restart|help]"
        echo "  deploy  - 部署最新的微调模型 (默认)"
        echo "  status  - 查看服务状态"
        echo "  stop    - 停止服务"
        echo "  restart - 重启服务"
        echo "  help    - 显示帮助"
        ;;
    *)
        log_error "未知命令: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac
