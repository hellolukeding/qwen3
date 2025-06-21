#!/bin/bash

# Qwen3-0.6B MCP 工具调用系统安装脚本

echo "🚀 开始安装 Qwen3-0.6B MCP 工具调用系统"
echo "=" * 60

# 检查虚拟环境
if [ ! -d "qwen-env" ]; then
    echo "❌ 未找到 qwen-env 虚拟环境，请先运行基础安装"
    echo "创建虚拟环境: python -m venv qwen-env"
    exit 1
fi

# 激活虚拟环境
echo "📦 激活虚拟环境..."
source qwen-env/bin/activate

# 安装 MCP 工具调用依赖
echo "📥 安装 MCP 系统依赖..."
pip install -r mcp_tools/requirements.txt

# 检查基础依赖
echo "🔍 检查基础依赖..."
python -c "import requests; print('✅ requests 已安装')" || pip install requests

# 设置权限
echo "🔧 设置执行权限..."
chmod +x mcp_tools/test_mcp.py
chmod +x mcp_tools/mcp_client.py

# 验证安装
echo "✅ 验证安装..."
cd mcp_tools
python -c "from mcp_server import MCPServer; print('✅ MCP 服务器模块加载成功')"
python -c "from mcp_client import MCPClient; print('✅ MCP 客户端模块加载成功')"

echo ""
echo "🎉 MCP 工具调用系统安装完成！"
echo ""
echo "📋 使用方法:"
echo "  1. 启动 vLLM 服务: ./start_vllm.sh"
echo "  2. 运行快速测试: python mcp_tools/test_mcp.py"
echo "  3. 交互模式: python mcp_tools/mcp_client.py"
echo ""
echo "🛠️  已支持的工具:"
echo "  • calculate: 数学计算器"
echo "  • get_time: 时间查询"
echo "  • text_analysis: 文本分析"
echo ""