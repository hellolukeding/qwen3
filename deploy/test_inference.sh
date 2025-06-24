#!/bin/bash

# vLLM 推理测试脚本

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

API_URL="http://$HOST:$PORT"

echo "🧪 vLLM 推理测试"
echo "================"
echo -e "🔗 API地址: ${BLUE}$API_URL${NC}"
echo

# 测试1: 检查模型列表
echo "📋 测试1: 获取模型列表"
echo "------------------------"

MODELS_RESPONSE=$(curl -s --max-time 10 "$API_URL/v1/models")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 成功获取模型列表${NC}"
    echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('可用模型:')
    for model in data.get('data', []):
        print(f'  - {model[\"id\"]}')
except Exception as e:
    print(f'解析失败: {e}')
"
    MODEL_ID=$(echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data.get('data'):
        print(data['data'][0]['id'])
except:
    pass
")
else
    echo -e "${RED}❌ 获取模型列表失败${NC}"
    exit 1
fi

echo

# 测试2: 简单对话测试
echo "💬 测试2: 简单对话测试"
echo "------------------------"

if [ -z "$MODEL_ID" ]; then
    MODEL_ID="qwen3-lora"  # 默认模型名
fi

# 准备测试消息
TEST_MESSAGES='[
    {"role": "user", "content": "你好，请简单介绍一下自己"}
]'

CHAT_REQUEST=$(cat <<EOF
{
    "model": "$MODEL_ID",
    "messages": $TEST_MESSAGES,
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": false
}
EOF
)

echo -e "🤖 使用模型: ${BLUE}$MODEL_ID${NC}"
echo "📝 测试消息: 你好，请简单介绍一下自己"
echo "⏳ 发送请求..."

CHAT_RESPONSE=$(curl -s --max-time 30 \
    -H "Content-Type: application/json" \
    -d "$CHAT_REQUEST" \
    "$API_URL/v1/chat/completions")

if [ $? -eq 0 ]; then
    # 解析响应
    RESPONSE_TEXT=$(echo "$CHAT_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if 'choices' in data and len(data['choices']) > 0:
        content = data['choices'][0]['message']['content']
        print(content)
    elif 'error' in data:
        print(f'API错误: {data[\"error\"]}')
    else:
        print('未知响应格式')
except Exception as e:
    print(f'解析失败: {e}')
    print('原始响应:', sys.stdin.read())
")
    
    if [[ "$RESPONSE_TEXT" =~ "API错误" ]] || [[ "$RESPONSE_TEXT" =~ "解析失败" ]]; then
        echo -e "${RED}❌ 对话测试失败${NC}"
        echo "$RESPONSE_TEXT"
    else
        echo -e "${GREEN}✅ 对话测试成功${NC}"
        echo -e "${YELLOW}🤖 模型回复:${NC}"
        echo "$RESPONSE_TEXT"
    fi
else
    echo -e "${RED}❌ 请求发送失败${NC}"
fi

echo

# 测试3: 性能测试
echo "⚡ 测试3: 性能测试"
echo "------------------------"

echo "🕐 测试响应时间..."

# 简单的性能测试
PERF_MESSAGES='[
    {"role": "user", "content": "1+1等于多少？"}
]'

PERF_REQUEST=$(cat <<EOF
{
    "model": "$MODEL_ID",
    "messages": $PERF_MESSAGES,
    "max_tokens": 50,
    "temperature": 0.1
}
EOF
)

START_TIME=$(date +%s.%N)
PERF_RESPONSE=$(curl -s --max-time 20 \
    -H "Content-Type: application/json" \
    -d "$PERF_REQUEST" \
    "$API_URL/v1/chat/completions")
END_TIME=$(date +%s.%N)

if [ $? -eq 0 ]; then
    RESPONSE_TIME=$(echo "$END_TIME - $START_TIME" | bc -l)
    RESPONSE_TIME_MS=$(echo "$RESPONSE_TIME * 1000" | bc -l | cut -d'.' -f1)
    
    echo -e "${GREEN}✅ 性能测试完成${NC}"
    echo "⏱️  响应时间: ${RESPONSE_TIME_MS}ms"
    
    # 检查响应时间
    if [ $RESPONSE_TIME_MS -lt 5000 ]; then
        echo -e "${GREEN}🚀 响应速度: 优秀${NC}"
    elif [ $RESPONSE_TIME_MS -lt 10000 ]; then
        echo -e "${YELLOW}⚡ 响应速度: 良好${NC}"
    else
        echo -e "${RED}🐌 响应速度: 较慢${NC}"
    fi
else
    echo -e "${RED}❌ 性能测试失败${NC}"
fi

echo

# 测试4: 流式输出测试
echo "🌊 测试4: 流式输出测试"
echo "------------------------"

STREAM_REQUEST=$(cat <<EOF
{
    "model": "$MODEL_ID",
    "messages": [{"role": "user", "content": "请数数从1到5"}],
    "max_tokens": 100,
    "stream": true
}
EOF
)

echo "📡 测试流式输出..."
STREAM_RESPONSE=$(curl -s --max-time 15 \
    -H "Content-Type: application/json" \
    -d "$STREAM_REQUEST" \
    "$API_URL/v1/chat/completions")

if [ $? -eq 0 ] && [[ "$STREAM_RESPONSE" =~ "data:" ]]; then
    echo -e "${GREEN}✅ 流式输出测试成功${NC}"
    echo "📥 接收到流式数据"
else
    echo -e "${YELLOW}⚠️  流式输出可能不支持或配置问题${NC}"
fi

echo

# 总结
echo "📊 测试总结"
echo "================"

# 统计成功的测试
SUCCESS_COUNT=0
TOTAL_COUNT=4

# 模型列表测试
if echo "$MODELS_RESPONSE" | grep -q "data"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo -e "✅ 模型列表: ${GREEN}通过${NC}"
else
    echo -e "❌ 模型列表: ${RED}失败${NC}"
fi

# 对话测试
if [[ ! "$RESPONSE_TEXT" =~ "API错误" ]] && [[ ! "$RESPONSE_TEXT" =~ "解析失败" ]] && [ ! -z "$RESPONSE_TEXT" ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo -e "✅ 对话功能: ${GREEN}通过${NC}"
else
    echo -e "❌ 对话功能: ${RED}失败${NC}"
fi

# 性能测试
if [ ! -z "$RESPONSE_TIME_MS" ] && [ $RESPONSE_TIME_MS -gt 0 ]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo -e "✅ 性能测试: ${GREEN}通过${NC} (${RESPONSE_TIME_MS}ms)"
else
    echo -e "❌ 性能测试: ${RED}失败${NC}"
fi

# 流式输出测试
if [[ "$STREAM_RESPONSE" =~ "data:" ]]; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    echo -e "✅ 流式输出: ${GREEN}通过${NC}"
else
    echo -e "❌ 流式输出: ${RED}失败${NC}"
fi

echo
echo "📈 测试结果: $SUCCESS_COUNT/$TOTAL_COUNT 通过"

if [ $SUCCESS_COUNT -eq $TOTAL_COUNT ]; then
    echo -e "🎉 ${GREEN}所有测试通过！模型部署成功！${NC}"
    exit 0
elif [ $SUCCESS_COUNT -ge 2 ]; then
    echo -e "⚠️  ${YELLOW}部分测试通过，模型基本可用${NC}"
    exit 0
else
    echo -e "🚨 ${RED}多数测试失败，请检查部署${NC}"
    exit 1
fi
