# Qwen3-0.6B Git 管理完整指南

## 项目结构说明

本项目包含两个主要部分：
1. **代码仓库** - 推理脚本、评估工具、配置文件等
2. **模型文件** - 大型模型文件（约1.5GB）需要特殊处理

## 方案1：使用Git LFS管理大文件（推荐）

### 1.1 安装Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# 或者直接下载安装
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 初始化Git LFS
git lfs install
```

### 1.2 配置项目
```bash
cd /home/ubuntu/qwen3

# 更新.gitignore - 移除对模型目录的忽略
echo "# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
qwen-env/
.venv/
venv/
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Temporary files
.tmp/
tmp/" > .gitignore

# 创建.gitattributes文件，配置LFS跟踪大文件
echo "# 模型文件使用Git LFS
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text

# 大型文本文件
tokenizer.json filter=lfs diff=lfs merge=lfs -text
vocab.json filter=lfs diff=lfs merge=lfs -text
merges.txt filter=lfs diff=lfs merge=lfs -text" > .gitattributes
```

### 1.3 推送到远程仓库
```bash
# 添加所有文件（包括模型）
git add .
git commit -m "Initial commit: Qwen3-0.6B deployment with optimization tools"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/your-username/qwen3-deployment.git

# 推送到远程
git push -u origin master
```

## 方案2：分离代码和模型（适合公开分享）

### 2.1 仅推送代码
```bash
cd /home/ubuntu/qwen3

# 保持.gitignore忽略模型目录
echo "/Qwen3-0.6B/
__pycache__/
*.pyc
qwen-env/
.venv/
*.log" > .gitignore

# 创建模型下载脚本
```

### 2.2 创建模型获取说明
在README.md中添加模型获取方式，用户可以：
- 从HuggingFace Hub下载
- 从百度网盘等云存储下载
- 使用提供的下载脚本

## 方案3：使用HuggingFace Hub（最佳实践）

### 3.1 上传到HuggingFace Hub
```bash
# 安装huggingface_hub
pip install huggingface_hub

# 登录（需要HuggingFace账号和token）
huggingface-cli login

# 创建模型仓库并上传
huggingface-cli repo create your-username/qwen3-0.6b-optimized --type model

# 上传模型文件
cd Qwen3-0.6B
huggingface-cli upload your-username/qwen3-0.6b-optimized . .
```

### 3.2 更新代码中的模型路径
修改推理脚本，支持从HuggingFace自动下载：
```python
model_name = "your-username/qwen3-0.6b-optimized"  # 或本地路径
```

## 推荐的最终项目结构

```
qwen3-deployment/
├── README.md                    # 完整文档
├── requirements.txt             # 依赖列表
├── .gitignore                   # Git忽略文件
├── .gitattributes              # Git LFS配置（方案1）
├── setup.py                    # 安装脚本（可选）
├── scripts/                    # 脚本目录
│   ├── download_model.py       # 模型下载脚本
│   ├── start_vllm.sh          # vLLM启动脚本
│   └── stop_vllm.sh           # vLLM停止脚本
├── src/                        # 源代码目录
│   ├── inference/
│   │   ├── transformers_inference.py
│   │   ├── vllm_inference.py
│   │   └── optimized_inference.py
│   ├── evaluation/
│   │   ├── response_optimizer.py
│   │   ├── time_evaluation.py
│   │   └── benchmark_tools.py
│   └── utils/
│       └── config.py
├── examples/                   # 示例目录
│   ├── basic_usage.py
│   ├── advanced_optimization.py
│   └── batch_evaluation.py
├── docs/                       # 文档目录
│   ├── installation.md
│   ├── optimization_guide.md
│   └── gpu_compatibility.md
├── tests/                      # 测试目录
│   ├── test_inference.py
│   └── test_evaluation.py
└── Qwen3-0.6B/                # 模型文件（方案1）
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## Git操作最佳实践

### 1. 提交消息规范
```bash
git commit -m "feat: add vLLM inference optimization"
git commit -m "fix: resolve GPU memory issue for RTX 2060"
git commit -m "docs: update README with optimization guide"
git commit -m "test: add response quality evaluation tests"
```

### 2. 分支管理
```bash
# 创建功能分支
git checkout -b feature/gpu-optimization
git checkout -b feature/response-evaluation
git checkout -b docs/installation-guide

# 合并到主分支
git checkout master
git merge feature/gpu-optimization
```

### 3. 版本标签
```bash
# 创建版本标签
git tag -a v1.0.0 -m "Initial release with Qwen3-0.6B optimization"
git tag -a v1.1.0 -m "Add vLLM support and evaluation tools"

# 推送标签
git push origin --tags
```

## 常见问题解决

### Q1: Git LFS推送失败
```bash
# 检查LFS配置
git lfs env

# 重新初始化LFS
git lfs install --force

# 手动追踪大文件
git lfs track "*.safetensors"
```

### Q2: 模型文件太大无法推送
- 使用Git LFS（推荐）
- 分离模型到云存储
- 使用HuggingFace Hub

### Q3: 仓库克隆速度慢
```bash
# 克隆时跳过LFS文件
git clone --filter=blob:none https://github.com/your-repo/qwen3.git

# 后续按需下载LFS文件
git lfs pull
```

## 自动化脚本

我们还将创建自动化脚本来简化Git管理流程。
