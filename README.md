# Qwen3-0.6B 本地推理部署指南

本项目演示如何在本地部署和使用 Qwen3-0.6B 大语言模型，支持传统 Transformers 推理和高性能 vLLM 推理。

## 📁 项目结构

```
qwen3/
├── README.md                 # 项目主文档
├── requirements.txt         # 依赖包列表
├── models/                  # 模型文件目录
│   └── Qwen3-0.6B/          # Qwen3-0.6B 模型
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── ...
├── qwen-env/                # Python 虚拟环境
├── scripts/                 # 脚本文件
│   ├── setup.sh             # 环境安装脚本
│   ├── start_vllm.sh        # vLLM 服务启动脚本
│   └── stop_vllm.sh         # vLLM 服务停止脚本
├── tests/                   # 测试文件
│   ├── test.py              # 传统 Transformers 推理测试
│   ├── final_test.py        # 优化的 Transformers 推理测试
│   ├── final_test_vllm.py   # vLLM API 推理测试
│   ├── debug_test.py        # 调试测试脚本
│   ├── improved_test.py     # 改进测试脚本
│   ├── simple_time_test.py  # 简单时间测试
│   └── test_time_evaluation.py # 时间评估测试
├── tools/                   # 工具文件
│   ├── optimized_inference.py  # 优化推理引擎
│   ├── response_optimizer.py   # 回复质量优化工具
│   └── mcp_tools/           # MCP 工具调用系统
│       ├── __init__.py
│       ├── mcp_client.py    # MCP 客户端
│       ├── mcp_server.py    # MCP 服务器
│       ├── requirements.txt # MCP 专用依赖
│       └── test_mcp.py      # MCP 测试脚本
└── docs/                    # 文档目录
    └── git_setup.md         # Git 设置指南
```

## �️ 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU（推荐 6GB+ 显存）
- **显存**: 最少 2GB，推荐 6GB+
- **内存**: 最少 8GB，推荐 16GB+
- **存储**: 2GB+ 可用空间

### 软件要求
- Python 3.8+
- CUDA 11.8+ 或 12.x（如使用 GPU）
- Ubuntu/Linux 系统

## 📦 安装步骤

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv qwen-env

# 激活虚拟环境
source qwen-env/bin/activate
```

### 2. 安装基础依赖

```bash
# 更新 pip
pip install --upgrade pip

# 安装 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装基础包
pip install -r requirements.txt
```

### 3. 安装 vLLM（可选，用于高性能推理）

```bash
pip install vllm
```

## 🚀 快速开始

### 环境安装

1. **运行安装脚本**:
```bash
# 使用自动化安装脚本
./scripts/setup.sh

# 或手动安装
source qwen-env/bin/activate
pip install -r requirements.txt
```

### 方法一：使用 Transformers（简单）

1. **运行基础推理脚本**:
```bash
source qwen-env/bin/activate
python tests/test.py
```

2. **运行优化推理脚本**:
```bash
python tests/final_test.py
```

### 方法二：使用 vLLM（高性能）

1. **启动 vLLM 服务**:
```bash
# 使用脚本启动（推荐）
./scripts/start_vllm.sh

# 或手动启动
source qwen-env/bin/activate
vllm serve ./models/Qwen3-0.6B --dtype half --port 8000 --max-model-len 8192
```

2. **等待服务启动完成**（约30-60秒），看到类似输出：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

3. **运行 API 推理**:
```bash
python tests/final_test_vllm.py
```

4. **停止 vLLM 服务**:
```bash
./scripts/stop_vllm.sh
```

### 方法三：使用 MCP 工具调用系统

1. **安装 MCP 工具**:
```bash
# 运行 MCP 系统安装
./scripts/setup.sh

# 安装 MCP 专用依赖
pip install -r tools/mcp_tools/requirements.txt
```

2. **启动 vLLM 服务**:
```bash
./scripts/start_vllm.sh
```

3. **测试 MCP 工具调用**:
```bash
# 快速测试
python tools/mcp_tools/test_mcp.py

# 交互模式
python tools/mcp_tools/mcp_client.py
```

## 🔍 使用优化工具的快速指南

### 回复质量和性能优化

#### 1. 运行完整的评估测试
```bash
# 运行综合评估工具（包含时间评估）
python tools/response_optimizer.py
```

#### 2. 快速时间评估演示
```bash
# 查看不同响应时间的评估标准
python tests/simple_time_test.py
```

#### 3. 优化推理引擎
```bash
# 使用优化推理引擎
python tools/optimized_inference.py

# 交互模式
python tools/optimized_inference.py --interactive
```

#### 4. MCP 工具调用系统

**支持的工具:**
- 🧮 `calculate`: 数学计算器
- 🕒 `get_time`: 时间查询  
- 📊 `text_analysis`: 文本分析
- 🌐 `web_search`: 网络搜索（扩展）
- 📧 `email_tools`: 邮件工具（扩展）

**使用示例:**
```python
from tools.mcp_tools.mcp_client import MCPClient

client = MCPClient()

# 数学计算
result = client.call_tool("calculate", {"expression": "25 * 4 + 10"})

# 获取时间
time_info = client.call_tool("get_time", {})

# 文本分析
analysis = client.call_tool("text_analysis", {
    "text": "这是一段需要分析的文本"
})
```

#### 3. 在代码中使用评估工具
```python
from tools.response_optimizer import QualityEvaluator

evaluator = QualityEvaluator()

# 评估单个回复（包含响应时间）
response = "您的模型回复内容"
response_time = 2.5  # 秒

evaluation = evaluator.evaluate_comprehensive(response, response_time)

print(f"综合评分: {evaluation['comprehensive_score']}/10")
print(f"时间评分: {evaluation['time_evaluation']['time_score']}/10")
print(f"内容评分: {evaluation['content_evaluation']['content_score']}/10")
```

#### 4. 实际测试结果展示

基于当前vLLM配置的实际测试结果：

```
📊 性能基准测试报告
============================================================

📈 总体统计:
  成功率: 100.0%

⏱️  响应时间统计:
  平均值: 0.75s     ⭐ 优秀水平
  95分位: 0.76s     ⭐ 稳定性极佳
  
🎯 质量评分统计:
  平均综合评分: 8.4/10  ✅ 良好水平
  
📊 响应时间分布:
  优秀 (≤2s): 100.0%   🚀 全部请求都是优秀级别
```

### 针对不同场景的优化建议

| 场景     | 目标响应时间 | 推荐配置                        | 预期质量 |
| -------- | ------------ | ------------------------------- | -------- |
| 实时对话 | ≤ 2秒        | max_tokens=60, temperature=0.6  | 8/10     |
| 知识问答 | ≤ 5秒        | max_tokens=120, temperature=0.7 | 9/10     |
| 内容生成 | ≤ 10秒       | max_tokens=200, temperature=0.8 | 9.5/10   |

## 📈 性能监控和优化

### RTX 2060/1660 系列（6GB 显存）
```bash
# 使用 float16，减少序列长度
vllm serve ./models/Qwen3-0.6B --dtype half --max-model-len 4096 --gpu-memory-utilization 0.8
```

### RTX 3060/4060 系列（8-12GB 显存）
```bash
# 可以使用更长序列
vllm serve ./models/Qwen3-0.6B --dtype half --max-model-len 8192 --gpu-memory-utilization 0.9
```

### RTX 3080/4080 系列（10-16GB 显存）
```bash
# 支持 bfloat16 和更长序列
vllm serve ./models/Qwen3-0.6B --dtype bfloat16 --max-model-len 16384
```

### RTX 4090/A100 系列（24GB+ 显存）
```bash
# 最大性能配置
vllm serve ./models/Qwen3-0.6B --dtype bfloat16 --max-model-len 32768 --tensor-parallel-size 1
```

### 计算能力对照表
| GPU 系列 | 计算能力 | 推荐数据类型 | 最大序列长度 |
| -------- | -------- | ------------ | ------------ |
| GTX 10xx | 6.1      | half         | 2048-4096    |
| RTX 20xx | 7.5      | half         | 4096-8192    |
| RTX 30xx | 8.6      | bfloat16     | 8192-16384   |
| RTX 40xx | 8.9      | bfloat16     | 16384-32768  |

## 📋 推理参数说明

### Transformers 参数
```python
model.generate(
    max_new_tokens=100,      # 生成的最大 token 数
    temperature=0.7,         # 随机性控制 (0.1-1.0)
    top_p=0.9,              # 核采样参数 (0.1-1.0)
    top_k=50,               # 候选词数量限制
    repetition_penalty=1.1,  # 重复惩罚 (1.0-1.3)
    do_sample=True,         # 是否使用采样
)
```

### vLLM API 参数
```python
{
    "max_tokens": 100,           # 最大生成长度
    "temperature": 0.7,          # 温度参数
    "top_p": 0.9,               # 核采样
    "repetition_penalty": 1.1,   # 重复惩罚
    "stop": ["\n\n"],           # 停止词
}
```

## 🔧 常见问题解决

### 1. CUDA 内存不足
```bash
# 错误: CUDA out of memory
# 解决: 减少 max_model_len 或增加 gpu_memory_utilization
vllm serve ./models/Qwen3-0.6B --dtype half --max-model-len 2048 --gpu-memory-utilization 0.7
```

### 2. bfloat16 不支持
```bash
# 错误: Bfloat16 is only supported on GPUs with compute capability >= 8.0
# 解决: 使用 half 代替 bfloat16
vllm serve ./models/Qwen3-0.6B --dtype half
```

### 3. 模型加载失败
```bash
# 检查模型文件是否完整
ls -la Qwen3-0.6B/
# 重新下载模型或检查路径
```

### 4. 端口被占用
```bash
# 检查端口占用
lsof -i :8000
# 杀死占用进程或更换端口
vllm serve ./models/Qwen3-0.6B --port 8001
```

## 📊 性能对比

| 推理方式     | 首次响应时间 | 吞吐量 | 显存占用 | 并发支持 |
| ------------ | ------------ | ------ | -------- | -------- |
| Transformers | 2-5秒        | 低     | 2GB      | 不支持   |
| vLLM         | 1-2秒        | 高     | 1.5GB    | 支持     |

## 🔍 使用示例

### 1. 简单问答
```python
# 使用 vLLM API
import requests

response = requests.post("http://localhost:8000/v1/completions", json={
    "model": "./models/Qwen3-0.6B",
    "prompt": "什么是人工智能？",
    "max_tokens": 100,
    "temperature": 0.7
})

print(response.json()["choices"][0]["text"])
```

### 2. 对话模式
```python
prompt = """用户: 你好，请介绍一下自己。
助手: 您好！我是Qwen，一个AI助手。
用户: 你能帮我解释什么是机器学习吗？
助手:"""

# 发送请求...
```

### 3. 批量推理
```python
prompts = [
    "请解释什么是深度学习",
    "机器学习有哪些应用",
    "人工智能的发展趋势"
]

for prompt in prompts:
    # 并发发送请求
    pass
```

## 📈 性能优化建议

### 1. 系统级优化
- 使用 SSD 存储模型文件
- 确保充足的系统内存
- 关闭不必要的后台程序

### 2. 模型级优化
- 根据 GPU 选择合适的数据类型
- 调整 `max_model_len` 平衡性能和内存
- 使用量化技术（4-bit, 8-bit）

### 3. 推理级优化
- 减少 `max_tokens` 提高响应速度
- 调整温度参数控制输出质量
- 使用适当的停止词避免无效输出

## 🚨 注意事项

1. **首次启动较慢**: vLLM 首次启动需要加载模型，约需 30-60 秒
2. **显存监控**: 使用 `nvidia-smi` 监控 GPU 使用情况
3. **服务管理**: 及时停止 vLLM 服务释放资源
4. **参数调优**: 根据具体任务调整生成参数
5. **错误处理**: 实现适当的异常处理和重试机制

## 📝 更新日志

- **v1.0**: 初始版本，支持基础推理
- **v1.1**: 添加 vLLM 支持
- **v1.2**: 优化 GPU 兼容性配置
- **v1.3**: 添加详细的故障排除指南

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目！

## 📄 许可证

本项目遵循 MIT 许可证。
- [vLLM高性能推理](#vllm高性能推理)
- [GPU适配说明](#gpu适配说明)
- [性能对比](#性能对比)
- [常见问题](#常见问题)

## 🛠️ 环境准备

### 系统要求
- **操作系统**: Linux (推荐 Ubuntu 18.04+)
- **Python**: 3.8+
- **CUDA**: 11.8+ (用于GPU推理)
- **显存**: 至少4GB (推荐6GB+)

### 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv qwen-env

# 激活虚拟环境
source qwen-env/bin/activate

# 升级pip
pip install --upgrade pip
```

## 📦 安装依赖

### 基础依赖 (传统推理)
```bash
# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1+
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装transformers和其他依赖
pip install transformers accelerate
pip install safetensors sentencepiece
```

### vLLM高性能推理 (可选但推荐)
```bash
# 安装vLLM
pip install vllm

# 安装API调用依赖
pip install requests
```

## 📥 模型下载

### 方式1: 使用git lfs (推荐)
```bash
# 安装git lfs
git lfs install

# 克隆模型仓库
git clone https://huggingface.co/Qwen/Qwen3-0.6B

# 重命名文件夹
mv Qwen3-0.6B ./models/Qwen3-0.6B
```

### 方式2: 使用huggingface-hub
```bash
# 安装huggingface-hub
pip install huggingface-hub

# Python脚本下载
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-0.6B', local_dir='./models/Qwen3-0.6B')
"
```

## 🔧 传统推理方式

### 基础推理脚本 (`tests/test.py`)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16,  # RTX 2060用torch.float16
    low_cpu_mem_usage=True
)

# 设置pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = "请简要介绍引力的原理。"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=100,        # 控制生成长度
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    no_repeat_ngram_size=3,
)

# 只显示新生成的内容
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"问题：{prompt}")
print(f"回答：{generated_text.strip()}")
```

### 运行传统推理
```bash
source qwen-env/bin/activate
python tests/test.py
```

## 🚀 vLLM高性能推理

### 启动vLLM服务

#### 针对不同GPU的启动命令:

**RTX 3090/4090 等 (计算能力 ≥ 8.0)**
```bash
vllm serve ./models/Qwen3-0.6B --dtype bfloat16 --port 8000 --max-model-len 8192
```

**RTX 2060/2070/2080 等 (计算能力 7.5)**
```bash
vllm serve ./models/Qwen3-0.6B --dtype half --port 8000 --max-model-len 8192
```

**显存不足时 (4GB以下)**
```bash
vllm serve ./models/Qwen3-0.6B --dtype half --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.8
```

### vLLM API调用脚本 (`tests/final_test_vllm.py`)
```python
import requests

url = "http://localhost:8000/v1/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "./models/Qwen3-0.6B",
    "prompt": "请简要介绍引力的原理。",
    "max_tokens": 100,
    "temperature": 0.6,
    "top_p": 0.8,
    "repetition_penalty": 1.1,
    "stop": None
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["text"].strip()
    print(f"问题：请简要介绍引力的原理。")
    print(f"回答：{answer}")
else:
    print(f"请求失败: {response.status_code}")
    print(f"错误信息: {response.text}")
```

### 运行vLLM推理
```bash
# 终端1: 启动vLLM服务
source qwen-env/bin/activate
vllm serve ./models/Qwen3-0.6B --dtype half --port 8000 --max-model-len 8192

# 终端2: 调用API推理
source qwen-env/bin/activate
python tests/final_test_vllm.py
```

## 🎯 GPU适配说明

### 按计算能力分类

| GPU系列       | 计算能力 | 推荐dtype | max_model_len | 备注         |
| ------------- | -------- | --------- | ------------- | ------------ |
| RTX 4090/3090 | 8.9/8.6  | bfloat16  | 16384         | 最佳性能     |
| RTX 4080/3080 | 8.9/8.6  | bfloat16  | 12288         | 良好性能     |
| RTX 4070/3070 | 8.9/8.6  | bfloat16  | 8192          | 中等性能     |
| RTX 2080 Ti   | 7.5      | half      | 8192          | 需要half精度 |
| RTX 2070/2060 | 7.5      | half      | 4096-8192     | 显存限制     |
| GTX 1080 Ti   | 6.1      | half      | 4096          | 较老架构     |

### 显存使用估算

| 配置            | 模型加载 | KV Cache | 总显存需求 |
| --------------- | -------- | -------- | ---------- |
| bfloat16 + 8192 | ~1.2GB   | ~2.5GB   | ~4GB       |
| half + 8192     | ~1.1GB   | ~2.5GB   | ~3.8GB     |
| half + 4096     | ~1.1GB   | ~1.2GB   | ~2.5GB     |

## ⚡ 性能对比

### 推理速度对比 (RTX 2060)

| 方式         | 首次响应 | 生成速度        | 内存使用 | 并发支持 |
| ------------ | -------- | --------------- | -------- | -------- |
| Transformers | ~3-5秒   | ~10 tokens/s    | 高       | 否       |
| vLLM         | ~1-2秒   | ~20-30 tokens/s | 低       | 是       |

### 质量对比
- **Transformers**: 更灵活的参数调节，适合研究和开发
- **vLLM**: 高度优化，适合生产环境，支持批处理

## 🔧 参数调优建议

### 生成质量参数
```python
# 平衡模式 (推荐)
{
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1
}

# 创造性模式
{
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.05
}

# 保守模式
{
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.2
}
```

## 📝 优化回复内容的策略

### 1. 回复质量评估工具（包含响应时间）

我们提供了全面的回复质量评估工具，包括内容质量和响应时间的综合评估：

#### 响应时间评估标准
| 时间范围 | 评级   | 评分 | 用户体验 | 优化建议     |
| -------- | ------ | ---- | -------- | ------------ |
| ≤ 2秒    | 优秀   | 10分 | 极佳     | 保持当前配置 |
| 2-5秒    | 良好   | 8分  | 良好     | 可进一步优化 |
| 5-10秒   | 可接受 | 6分  | 一般     | 需要优化参数 |
| 10-20秒  | 较慢   | 4分  | 较差     | 显著优化配置 |
| > 20秒   | 很慢   | 2分  | 很差     | 更换推理方案 |

#### 综合评估公式
```
综合评分 = 内容质量评分 × 0.7 + 响应时间评分 × 0.3
```

#### 使用回复质量评估工具
```python
from response_optimizer import QualityEvaluator, ResponseOptimizer, PerformanceBenchmark

# 创建评估器
evaluator = QualityEvaluator()
optimizer = ResponseOptimizer()

# 评估单个回复（包含时间）
response = "人工智能是计算机科学的一个分支..."
response_time = 3.5  # 秒

evaluation = evaluator.evaluate_comprehensive(response, response_time)

print(f"综合评分: {evaluation['comprehensive_score']}/10")
print(f"内容评分: {evaluation['content_evaluation']['content_score']}/10")
print(f"时间评分: {evaluation['time_evaluation']['time_score']}/10")
print(f"响应等级: {evaluation['time_evaluation']['time_level']}")
print(f"改进建议: {evaluation['suggestions']}")
```

#### 性能基准测试
```python
# 批量性能测试
benchmark = PerformanceBenchmark()

test_prompts = [
    "什么是机器学习？",
    "请解释深度学习原理。",
    "AI有哪些应用？"
]

# 运行基准测试
results = benchmark.batch_test(test_prompts, iterations=3)
benchmark.print_report(results)
```

#### 测试报告示例
```
📊 性能基准测试报告
============================================================

📈 总体统计:
  总测试数: 9
  成功数: 9
  失败数: 0
  成功率: 100.0%

⏱️  响应时间统计:
  平均值: 2.45s
  最小值: 1.23s
  最大值: 4.67s
  中位数: 2.34s
  95分位: 4.12s
  99分位: 4.67s

🎯 质量评分统计:
  平均综合评分: 8.2/10
  最高评分: 9.5/10
  最低评分: 6.8/10

📊 响应时间分布:
  优秀 (≤2s): 4 (44.4%)
  良好 (2-5s): 5 (55.6%)
  可接受 (5-10s): 0 (0.0%)
  较慢 (>10s): 0 (0.0%)
```

### 2. 针对不同响应时间的优化策略

#### 响应时间 > 10秒的优化方案
```python
# 激进优化配置
aggressive_params = {
    "max_tokens": 50,           # 大幅减少生成长度
    "temperature": 0.3,         # 降低随机性
    "top_p": 0.8,              # 减少候选词
    "do_sample": False,         # 使用贪心解码
    "stop": ["\n", "。", "！", "？"]  # 早期停止
}
```

#### 响应时间 5-10秒的优化方案
```python
# 平衡优化配置
balanced_params = {
    "max_tokens": 80,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 30,
    "repetition_penalty": 1.1
}
```

#### 响应时间 < 5秒的维持方案
```python
# 质量优先配置
quality_params = {
    "max_tokens": 120,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.05
}
```

### 3. 实时响应时间监控

### 创建一个实时监控脚本 `tools/time_monitor.py`：

```python
import time
import requests
from collections import deque
from typing import Deque, Dict, List

class ResponseTimeMonitor:
    """响应时间实时监控器"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.response_times: Deque[float] = deque(maxlen=window_size)
        self.quality_scores: Deque[float] = deque(maxlen=window_size)
        
    def add_measurement(self, response_time: float, quality_score: float):
        """添加测量数据"""
        self.response_times.append(response_time)
        self.quality_scores.append(quality_score)
    
    def get_current_metrics(self) -> Dict:
        """获取当前性能指标"""
        if not self.response_times:
            return {"status": "no_data"}
        
        avg_time = sum(self.response_times) / len(self.response_times)
        avg_quality = sum(self.quality_scores) / len(self.quality_scores)
        
        # 判断趋势
        if len(self.response_times) >= 5:
            recent_avg = sum(list(self.response_times)[-3:]) / 3
            early_avg = sum(list(self.response_times)[:3]) / 3
            trend = "improving" if recent_avg < early_avg else "degrading" if recent_avg > early_avg else "stable"
        else:
            trend = "unknown"
        
        # 性能等级
        if avg_time <= 2.0:
            performance_level = "excellent"
        elif avg_time <= 5.0:
            performance_level = "good"
        elif avg_time <= 10.0:
            performance_level = "acceptable"
        else:
            performance_level = "poor"
        
        return {
            "avg_response_time": round(avg_time, 2),
            "avg_quality_score": round(avg_quality, 1),
            "performance_level": performance_level,
            "trend": trend,
            "sample_count": len(self.response_times),
            "latest_time": self.response_times[-1],
            "time_stability": round(max(self.response_times) - min(self.response_times), 2)
        }
    
    def suggest_optimizations(self) -> List[str]:
        """基于监控数据提供优化建议"""
        metrics = self.get_current_metrics()
        suggestions = []
        
        if metrics["performance_level"] == "poor":
            suggestions.extend([
                "立即减少max_tokens到50以下",
                "切换到greedy decoding (do_sample=False)",
                "考虑升级到vLLM推理引擎",
                "检查GPU性能和显存使用"
            ])
        elif metrics["performance_level"] == "acceptable":
            suggestions.extend([
                "适度减少max_tokens参数",
                "优化temperature和top_p参数",
                "检查网络延迟"
            ])
        
        if metrics["trend"] == "degrading":
            suggestions.append("性能正在下降，建议重启推理服务")
        
        if metrics["time_stability"] > 5.0:
            suggestions.append("响应时间不稳定，检查系统资源占用")
        
        return suggestions

# 使用示例
monitor = ResponseTimeMonitor()

# 模拟添加监控数据
test_data = [
    (2.1, 8.5), (1.8, 9.0), (3.2, 7.8), 
    (2.5, 8.2), (4.1, 7.5), (2.9, 8.0)
]

for response_time, quality in test_data:
    monitor.add_measurement(response_time, quality)

metrics = monitor.get_current_metrics()
print(f"当前性能: {metrics}")

suggestions = monitor.suggest_optimizations()
print(f"优化建议: {suggestions}")
```

### 4. 自适应参数调整

根据响应时间自动调整推理参数：

```python
class AdaptiveInference:
    """自适应推理引擎"""
    
    def __init__(self, target_response_time: float = 3.0):
        self.target_time = target_response_time
        self.current_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9
        }
        self.performance_history = []
    
    def adjust_parameters(self, last_response_time: float):
        """根据上次响应时间调整参数"""
        
        if last_response_time > self.target_time * 1.5:
            # 响应过慢，激进优化
            self.current_params["max_tokens"] = max(30, self.current_params["max_tokens"] - 20)
            self.current_params["temperature"] = max(0.1, self.current_params["temperature"] - 0.1)
            self.current_params["top_p"] = max(0.7, self.current_params["top_p"] - 0.1)
            
        elif last_response_time > self.target_time:
            # 响应较慢，适度优化
            self.current_params["max_tokens"] = max(50, self.current_params["max_tokens"] - 10)
            self.current_params["temperature"] = max(0.3, self.current_params["temperature"] - 0.05)
            
        elif last_response_time < self.target_time * 0.7:
            # 响应很快，可以提高质量
            self.current_params["max_tokens"] = min(150, self.current_params["max_tokens"] + 10)
            self.current_params["temperature"] = min(0.8, self.current_params["temperature"] + 0.05)
            self.current_params["top_p"] = min(0.95, self.current_params["top_p"] + 0.05)
    
    def get_optimized_params(self) -> Dict:
        """获取当前优化的参数"""
        return self.current_params.copy()

# 使用示例
adaptive_engine = AdaptiveInference(target_response_time=2.5)

# 模拟多次推理和调整
response_times = [4.2, 3.1, 2.8, 2.1, 1.9, 1.7]

for i, rt in enumerate(response_times):
    print(f"第{i+1}次推理时间: {rt}s")
    adaptive_engine.adjust_parameters(rt)
    params = adaptive_engine.get_optimized_params()
    print(f"调整后参数: max_tokens={params['max_tokens']}, temp={params['temperature']}")
    print()
```

#### 结构化Prompt模板
```python
# 问答模式模板
def create_qa_prompt(question):
    return f"""请基于以下问题给出简洁、准确的回答：

问题：{question}

要求：
1. 回答要简洁明了，控制在100字以内
2. 重点突出，逻辑清晰
3. 避免重复表述

回答："""

# 对话模式模板
def create_chat_prompt(context, user_input):
    return f"""以下是一段对话记录：

{context}

用户：{user_input}
助手："""

# 专业领域模板
def create_expert_prompt(domain, question):
    return f"""你是一位{domain}专家，请专业地回答以下问题：

{question}

请用专业但易懂的语言回答："""
```

#### Prompt优化技巧
```python
# 1. 添加角色设定
prompt = """你是一位知识渊博的AI助手，擅长用简洁的语言解释复杂概念。

用户问题：{question}

请提供清晰、准确的回答："""

# 2. 设定输出格式
prompt = """请按以下格式回答问题：

问题：{question}

回答要点：
1. 核心概念：
2. 主要原理：
3. 实际应用：

详细说明："""

# 3. 添加示例引导
prompt = """请参考以下示例风格回答问题：

示例问题：什么是人工智能？
示例回答：人工智能（AI）是计算机科学的一个分支，目标是创建能够模拟人类智能行为的系统。AI包括机器学习、深度学习、自然语言处理等技术，广泛应用于图像识别、语音助手、自动驾驶等领域。

现在请回答：{question}"""
```

### 2. 生成参数优化

#### 控制重复的参数组合
```python
# 严格控制重复（用于事实性问答）
strict_params = {
    "max_tokens": 150,
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 4,
    "stop": ["\n\n", "问题：", "用户：", "助手："]
}

# 平衡重复和创造性（通用对话）
balanced_params = {
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "stop": ["\n\n问题", "\n\n用户", "---"]
}

# 鼓励创造性（创意写作）
creative_params = {
    "max_tokens": 300,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.05,
    "no_repeat_ngram_size": 2
}
```

#### 停止词策略
```python
# 通用停止词
general_stops = [
    "\n\n",           # 防止多段落
    "问题：",         # 防止生成新问题
    "用户：",         # 防止对话角色混乱
    "助手：",         # 同上
    "答案：",         # 防止重复答案标记
    "总结：",         # 防止自动总结
    "例如：",         # 在简短回答中防止过多举例
]

# 专业领域停止词
technical_stops = general_stops + [
    "参考文献：",
    "注意：",
    "警告：",
    "补充：",
    "详见：",
]

# 对话系统停止词
chat_stops = [
    "\n用户",
    "\n助手",
    "下一个问题",
    "还有什么",
    "您还想"
]
```

### 3. 后处理优化

#### 文本清理函数
```python
import re

def clean_response(text):
    """清理和优化生成的回复"""
    
    # 1. 移除多余的空行
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # 2. 移除重复的句子
    sentences = text.split('。')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in unique_sentences:
            unique_sentences.append(sentence)
    text = '。'.join(unique_sentences)
    
    # 3. 移除重复的短语（3个字符以上）
    words = text.split()
    seen_phrases = set()
    cleaned_words = []
    
    for i, word in enumerate(words):
        # 检查3-gram重复
        if i >= 2:
            phrase = ' '.join(words[i-2:i+1])
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
        cleaned_words.append(word)
    
    # 4. 移除末尾的不完整句子
    text = ' '.join(cleaned_words)
    if text and not text[-1] in '。！？':
        last_punct = max(text.rfind('。'), text.rfind('！'), text.rfind('？'))
        if last_punct > 0:
            text = text[:last_punct+1]
    
    # 5. 规范化标点符号
    text = re.sub(r'[,，]{2,}', '，', text)
    text = re.sub(r'[。]{2,}', '。', text)
    
    return text.strip()

def format_response(text, max_length=200):
    """格式化回复，确保长度适中"""
    
    # 清理文本
    text = clean_response(text)
    
    # 控制长度
    if len(text) > max_length:
        # 在句号处截断
        sentences = text.split('。')
        formatted_text = ""
        for sentence in sentences:
            if len(formatted_text + sentence + '。') <= max_length:
                formatted_text += sentence + '。'
            else:
                break
        text = formatted_text.rstrip('。') + '。'
    
    return text
```

#### 质量评估函数
```python
def evaluate_response_quality(response):
    """评估回复质量"""
    
    issues = []
    
    # 检查重复
    sentences = response.split('。')
    if len(sentences) != len(set(sentences)):
        issues.append("包含重复句子")
    
    # 检查长度
    if len(response) < 20:
        issues.append("回复过短")
    elif len(response) > 500:
        issues.append("回复过长")
    
    # 检查完整性
    if not response.strip().endswith(('。', '！', '？')):
        issues.append("回复不完整")
    
    # 检查关键词重复
    words = response.split()
    word_freq = {}
    for word in words:
        if len(word) > 1:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    repeated_words = [word for word, freq in word_freq.items() if freq > 3]
    if repeated_words:
        issues.append(f"词汇重复: {repeated_words}")
    
    return {
        "quality_score": max(0, 10 - len(issues)),
        "issues": issues,
        "is_good": len(issues) == 0
    }
```

### 4. 实时优化脚本

创建一个优化的推理脚本 `tools/optimized_inference.py`：

```python
import requests
import re
import time

class OptimizedInference:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_response(self, prompt, task_type="general"):
        """根据任务类型优化生成参数"""
        
        # 根据任务类型选择参数
        if task_type == "factual":
            params = {
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.3,
                "max_tokens": 100,
                "stop": ["\n\n", "问题：", "另外", "此外"]
            }
        elif task_type == "creative":
            params = {
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50,
                "repetition_penalty": 1.05,
                "max_tokens": 250,
                "stop": ["\n\n问题", "\n\n---"]
            }
        else:  # general
            params = {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.1,
                "max_tokens": 150,
                "stop": ["\n\n", "问题：", "用户："]
            }
        
        # 构建请求
        data = {
            "model": "./models/Qwen3-0.6B",
            "prompt": prompt,
            **params
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_text = result["choices"][0]["text"]
                
                # 后处理优化
                cleaned_text = self.clean_response(raw_text)
                
                return {
                    "success": True,
                    "response": cleaned_text,
                    "raw_response": raw_text,
                    "tokens_used": result["usage"]["total_tokens"]
                }
            else:
                return {"success": False, "error": f"API错误: {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def clean_response(self, text):
        """清理回复文本"""
        # 移除开头的重复prompt内容
        text = text.strip()
        
        # 移除常见的重复开头
        prefixes = ["答案：", "回答：", "解答：", "答："]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 移除重复句子
        sentences = text.split('。')
        unique_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in unique_sentences:
                unique_sentences.append(sentence)
        
        text = '。'.join(unique_sentences)
        if text and not text.endswith('。'):
            text += '。'
        
        # 控制长度，保持完整性
        if len(text) > 200:
            sentences = text.split('。')
            result = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) <= 180:
                    result.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break
            
            text = '。'.join(result)
            if text and not text.endswith('。'):
                text += '。'
        
        return text

# 使用示例
def main():
    inference = OptimizedInference()
    
    test_prompts = [
        ("请简要介绍人工智能。", "factual"),
        ("写一个关于春天的短诗。", "creative"),
        ("你好，最近怎么样？", "general")
    ]
    
    for prompt, task_type in test_prompts:
        print(f"\n{'='*50}")
        print(f"问题类型: {task_type}")
        print(f"问题: {prompt}")
        
        start_time = time.time()
        result = inference.generate_response(prompt, task_type)
        end_time = time.time()
        
        if result["success"]:
            print(f"回答: {result['response']}")
            print(f"耗时: {end_time - start_time:.2f}秒")
            print(f"Token使用: {result['tokens_used']}")
        else:
            print(f"错误: {result['error']}")

if __name__ == "__main__":
    main()
```

### 5. 常见回复问题及解决方案

#### 问题1: 回复重复
```python
# 解决方案：增加重复惩罚和n-gram限制
{
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "stop": ["\n\n", "重复的短语"]
}
```

#### 问题2: 回复不完整
```python
# 解决方案：增加max_tokens，改进停止词
{
    "max_tokens": 200,  # 增加长度
    "stop": ["问题："]  # 避免过早停止
}
```

#### 问题3: 回复偏离主题
```python
# 解决方案：优化prompt结构
prompt = f"""请专注回答以下问题，不要涉及其他话题：

问题：{question}

请直接回答上述问题："""
```

#### 问题4: 回复过于简短
```python
# 解决方案：在prompt中明确要求详细程度
prompt = f"""请详细回答以下问题（至少50字）：

{question}

请提供完整、详细的回答："""
```

### 6. A/B测试框架

```python
def ab_test_parameters():
    """测试不同参数组合的效果"""
    
    test_configs = {
        "conservative": {
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.3
        },
        "balanced": {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        },
        "creative": {
            "temperature": 0.8,
            "top_p": 0.95,
            "repetition_penalty": 1.05
        }
    }
    
    test_questions = [
        "什么是机器学习？",
        "请解释量子计算的原理。",
        "如何学好编程？"
    ]
    
    results = {}
    
    for config_name, config in test_configs.items():
        results[config_name] = []
        
        for question in test_questions:
            # 生成回复并评估质量
            response = generate_with_config(question, config)
            quality = evaluate_response_quality(response)
            
            results[config_name].append({
                "question": question,
                "response": response,
                "quality_score": quality["quality_score"]
            })
    
    return results
```

这些优化策略将帮助你获得更高质量、更相关且无重复的模型回复。

## 🎯 LoRA 微调模型使用指南

### 微调训练完成后的模型使用

#### 1. 快速开始微调训练

```bash
# 环境验证
./scripts/verify_env.sh

# 小显存优化训练（适用于 RTX 2060 6GB）
./scripts/train_lowmem.sh

# 简单测试训练
./scripts/test_simple.sh
```

#### 2. 加载和使用微调模型

**Python 代码示例：**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_finetuned_model(base_model_path, adapter_path):
    """加载微调后的 LoRA 模型"""
    print("加载基础模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # RTX 2060用torch.float16
        low_cpu_mem_usage=True
    )
    
    print("加载 LoRA 适配器...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model, tokenizer

def chat_with_model(model, tokenizer, user_input, system_prompt="你是一个有用的AI助手。"):
    """与微调模型对话"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response

# 使用示例
if __name__ == "__main__":
    base_model_path = "./models/Qwen3-0.6B"
    adapter_path = "./output/qwen3-lora-lowmem-20250621-195657"  # 替换为你的模型路径
    
    model, tokenizer = load_finetuned_model(base_model_path, adapter_path)
    
    # 测试对话
    response = chat_with_model(model, tokenizer, "你好，请介绍一下自己。")
    print(f"AI: {response}")
```

#### 3. 模型验证和测试脚本

**创建验证脚本：**

```bash
# 创建模型验证脚本
cat > scripts/validate_model.sh << 'EOF'
#!/bin/bash
# 微调模型验证脚本

MODEL_PATH=${1:-"./output/qwen3-lora-lowmem-$(date +%Y%m%d-*)*"}

echo "🔍 验证微调模型: $MODEL_PATH"

# 检查模型文件
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型目录不存在: $MODEL_PATH"
    echo "请指定正确的模型路径，例如:"
    echo "  ./scripts/validate_model.sh ./output/qwen3-lora-lowmem-20250621-195657"
    exit 1
fi

echo "✅ 模型目录存在"

# 检查必需文件
required_files=("adapter_config.json" "adapter_model.safetensors")
for file in "${required_files[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
    fi
done

# 运行 Python 验证
uv run python -c "
import sys
sys.path.append('.')

try:
    from scripts.test_finetuned_model import test_model
    test_model('$MODEL_PATH')
    print('🎉 模型验证完成!')
except Exception as e:
    print(f'❌ 验证失败: {e}')
    sys.exit(1)
"

echo "模型验证完成！"
EOF

chmod +x scripts/validate_model.sh
```

#### 4. 创建模型测试脚本

```python
# scripts/test_finetuned_model.py
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_model(adapter_path):
    """测试微调模型的功能"""
    print(f"🧪 测试模型: {adapter_path}")
    
    base_model_path = "./models/Qwen3-0.6B"
    
    try:
        # 加载模型
        print("📦 加载基础模型...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("🔧 加载 LoRA 适配器...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 测试对话
        test_cases = [
            "你好，请介绍一下自己。",
            "1+1等于多少？",
            "请解释一下什么是人工智能。",
            "写一首关于春天的诗。"
        ]
        
        print("💬 开始对话测试...")
        for i, user_input in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i} ---")
            print(f"用户: {user_input}")
            
            messages = [
                {"role": "system", "content": "你是一个有用的AI助手。"},
                {"role": "user", "content": user_input}
            ]
            
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"AI: {response}")
        
        print("\n✅ 模型测试完成！")
        
        # 显示模型信息
        print("\n📊 模型信息:")
        if hasattr(model, 'print_trainable_parameters'):
            model.print_trainable_parameters()
        
        # 内存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"GPU 内存使用: {memory_used:.2f} GB")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python scripts/test_finetuned_model.py <模型路径>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_model(model_path)
```

#### 5. 交互式对话脚本

```python
# scripts/chat_with_finetuned.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def interactive_chat(adapter_path):
    """交互式对话"""
    base_model_path = "./models/Qwen3-0.6B"
    
    print("🚀 加载微调模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("✅ 模型加载完成！")
    
    print("\n💬 开始对话（输入 'exit' 退出）:")
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            print("👋 对话结束！")
            break
        
        if not user_input:
            continue
        
        messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": user_input}
        ]
        
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        print("AI: ", end="", flush=True)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(response)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python scripts/chat_with_finetuned.py <模型路径>")
        print("示例: python scripts/chat_with_finetuned.py ./output/qwen3-lora-lowmem-20250621-195657")
        sys.exit(1)
    
    model_path = sys.argv[1]
    interactive_chat(model_path)
```

#### 6. 使用命令示例

```bash
# 验证微调模型
./scripts/validate_model.sh ./output/qwen3-lora-lowmem-20250621-195657

# 测试模型功能
uv run python scripts/test_finetuned_model.py ./output/qwen3-lora-lowmem-20250621-195657

# 交互式对话
uv run python scripts/chat_with_finetuned.py ./output/qwen3-lora-lowmem-20250621-195657

# 监控GPU使用
./scripts/monitor_gpu.sh
```

### 选项2: API服务
```bash
# 创建简单的FastAPI服务
cat > serve_finetuned.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

# 全局加载模型
model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    base_model_path = "./models/Qwen3-0.6B"
    adapter_path = "./output/qwen3-lora-lowmem-20250621-195657"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{"role": "user", "content": request.message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=request.temperature
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# 启动服务
uv run python serve_finetuned.py
```

### 选项3: Gradio界面
```bash
# 创建Web界面
cat > gradio_chat.py << 'EOF'
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载模型
base_model_path = "./models/Qwen3-0.6B"
adapter_path = "./output/qwen3-lora-lowmem-20250621-195657"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)

def chat_fn(message, history, temperature):
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=temperature
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    history.append([message, response])
    return "", history

with gr.Blocks(title="Qwen3 微调模型") as demo:
    gr.Markdown("# 🤖 Qwen3-0.6B LoRA 微调模型对话界面")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="输入消息")
    temperature = gr.Slider(0.1, 2.0, value=0.7, label="温度")
    clear = gr.Button("清除")
    
    msg.submit(chat_fn, [msg, chatbot, temperature], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860)
EOF

# 安装并启动
uv add gradio
uv run python gradio_chat.py
```

## 🚀 下一步优化建议

1. **增加训练数据**: 扩展数据集以提升模型能力
2. **调整超参数**: 尝试不同的学习率和LoRA配置
3. **多轮对话**: 实现上下文记忆功能
4. **专业化微调**: 针对特定领域进行专门训练
5. **量化部署**: 使用INT8/INT4量化进一步减少内存使用

恭喜你成功完成了 Qwen3-0.6B 的 LoRA 微调！🎉
