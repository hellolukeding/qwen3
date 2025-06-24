#!/bin/bash

# Qwen3 一键部署脚本
# 自动安装依赖、部署模型、启动服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🚀 Qwen3 微调模型一键部署"
echo "=========================="

# 检查是否在项目根目录
if [ ! -f "../pyproject.toml" ]; then
    echo -e "${RED}❌ 请在 deploy 目录中运行此脚本${NC}"
    exit 1
fi

# 步骤1: 安装系统依赖
echo -e "${BLUE}📦 步骤1: 检查系统依赖${NC}"
echo "--------------------"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装，请先安装 Python3${NC}"
    exit 1
fi
echo "✅ Python3 已安装: $(python3 --version)"

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}⚠️  NVIDIA GPU 驱动未检测到，将使用 CPU 模式${NC}"
    USE_GPU=false
else
    echo "✅ NVIDIA GPU 已检测到: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    USE_GPU=true
fi

# 安装 uv
if ! command -v uv &> /dev/null; then
    echo "📥 安装 uv 包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    export PATH="$HOME/.cargo/bin:$PATH"
fi
echo "✅ uv 已安装: $(uv --version)"

echo

# 步骤2: 设置 Python 环境
echo -e "${BLUE}🐍 步骤2: 设置 Python 环境${NC}"
echo "--------------------"

cd ..

# 创建虚拟环境
if [ ! -d ".venv" ]; then
    echo "🔨 创建虚拟环境..."
    uv venv .venv
fi
echo "✅ 虚拟环境已准备"

# 安装依赖
echo "📦 安装项目依赖..."
if $USE_GPU; then
    # GPU 版本
    UV_HTTP_TIMEOUT=120 uv sync
else
    # CPU 版本（如果需要的话，可以添加特殊处理）
    UV_HTTP_TIMEOUT=120 uv sync
fi
echo "✅ 依赖安装完成"

echo

# 步骤3: 检查模型文件
echo -e "${BLUE}🤖 步骤3: 检查模型文件${NC}"
echo "--------------------"

# 检查基础模型
if [ ! -d "models/Qwen3-0.6B" ]; then
    echo -e "${RED}❌ 基础模型不存在: models/Qwen3-0.6B${NC}"
    echo "请先下载 Qwen3-0.6B 模型到 models/ 目录"
    exit 1
fi
echo "✅ 基础模型已存在: models/Qwen3-0.6B"

# 检查微调模型
FINETUNED_MODELS=$(find output -name "qwen3-lora-lowmem-*" -type d 2>/dev/null | wc -l)
if [ $FINETUNED_MODELS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  未找到微调模型，将使用基础模型${NC}"
    echo "如需使用微调模型，请先运行训练脚本"
    USE_FINETUNED=false
else
    LATEST_MODEL=$(find output -name "qwen3-lora-lowmem-*" -type d | sort -V | tail -n 1)
    echo "✅ 找到微调模型: $LATEST_MODEL"
    USE_FINETUNED=true
fi

echo

# 步骤4: 安装 vLLM
echo -e "${BLUE}⚡ 步骤4: 安装 vLLM${NC}"
echo "--------------------"

source .venv/bin/activate

# 检查 vLLM 是否已安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "📥 安装 vLLM..."
    if $USE_GPU; then
        pip install vllm
    else
        pip install vllm[cpu]
    fi
else
    echo "✅ vLLM 已安装"
fi

# 检查其他推理依赖
echo "📦 安装推理相关依赖..."
pip install transformers accelerate torch

echo "✅ vLLM 安装完成"

echo

# 步骤5: 创建部署配置
echo -e "${BLUE}⚙️  步骤5: 创建部署配置${NC}"
echo "--------------------"

cd deploy

# 创建配置文件
if $USE_FINETUNED; then
    MODEL_PATH="../$LATEST_MODEL"
else
    MODEL_PATH="../models/Qwen3-0.6B"
fi

cat > config.json << EOF
{
    "model_path": "$MODEL_PATH",
    "base_model_path": "../models/Qwen3-0.6B",
    "port": 8000,
    "host": "0.0.0.0",
    "use_gpu": $USE_GPU,
    "deployment_time": "$(date -Iseconds)",
    "deployment_type": "$(if $USE_FINETUNED; then echo 'finetuned'; else echo 'base'; fi)"
}
EOF

echo "✅ 配置文件已创建"

echo

# 步骤6: 启动服务
echo -e "${BLUE}🚀 步骤6: 启动服务${NC}"
echo "--------------------"

# 设置脚本权限
chmod +x *.sh

# 停止现有服务
echo "🛑 停止现有服务..."
./stop_service.sh 2>/dev/null || true

# 启动新服务
echo "🚀 启动 vLLM 服务..."
./start_service.sh

echo "⏳ 等待服务启动..."
sleep 30

echo

# 步骤7: 验证部署
echo -e "${BLUE}🧪 步骤7: 验证部署${NC}"
echo "--------------------"

# 运行测试
echo "🔍 运行服务测试..."
if ./test_inference.sh; then
    echo -e "${GREEN}✅ 部署验证成功！${NC}"
    
    echo
    echo "🎉 一键部署完成！"
    echo "=================="
    echo -e "📡 服务地址: ${BLUE}http://localhost:8000${NC}"
    echo -e "🤖 模型类型: $(if $USE_FINETUNED; then echo -e "${GREEN}微调模型${NC}"; else echo -e "${YELLOW}基础模型${NC}"; fi)"
    echo -e "💻 运行模式: $(if $USE_GPU; then echo -e "${GREEN}GPU${NC}"; else echo -e "${YELLOW}CPU${NC}"; fi)"
    echo
    echo "🔧 管理命令:"
    echo "   📊 查看状态: ./status.sh"
    echo "   📋 查看日志: tail -f service.log"
    echo "   🔍 实时监控: ./monitor.sh"
    echo "   🧪 测试推理: ./test_inference.sh"
    echo "   🛑 停止服务: ./stop_service.sh"
    echo "   🔄 重启服务: ./deploy.sh restart"
    echo
    echo "🌐 API 使用示例:"
    echo "curl -X POST http://localhost:8000/v1/chat/completions \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"model\": \"$(if $USE_FINETUNED; then echo "qwen3-lora"; else echo "Qwen3-0.6B"; fi)\", \"messages\": [{\"role\": \"user\", \"content\": \"你好\"}]}'"
    
else
    echo -e "${RED}❌ 部署验证失败${NC}"
    echo "请检查日志文件: tail -f service.log"
    exit 1
fi
