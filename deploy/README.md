# Qwen3 微调模型部署指南

这个目录包含了 Qwen3 微调模型的完整部署脚本集合，支持自动化部署、监控和管理。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `one_click_deploy.sh` | **一键部署脚本** - 自动安装依赖并部署模型 |
| `deploy.sh` | 主部署脚本 - 部署最新的微调模型 |
| `start_service.sh` | 服务启动脚本 |
| `stop_service.sh` | 服务停止脚本 |
| `status.sh` | 服务状态检查脚本 |
| `test_inference.sh` | 推理测试脚本 |
| `monitor.sh` | 实时监控脚本 |
| `config.json` | 部署配置文件（自动生成） |

## 🚀 快速开始

### 方式1: 一键部署（推荐）

```bash
cd deploy
chmod +x *.sh
./one_click_deploy.sh
```

### 方式2: 手动部署

```bash
cd deploy
chmod +x *.sh

# 部署模型
./deploy.sh

# 查看状态
./status.sh

# 测试推理
./test_inference.sh
```

## 📋 前置要求

### 系统要求
- Ubuntu 18.04+ 或其他 Linux 发行版
- Python 3.8+
- NVIDIA GPU（可选，支持 CPU 模式）

### 模型文件
确保以下文件/目录存在：
- `../models/Qwen3-0.6B/` - 基础模型目录
- `../output/qwen3-lora-lowmem-*/` - 微调模型目录（可选）

## 🔧 使用说明

### 部署命令

```bash
# 部署最新微调模型
./deploy.sh

# 查看服务状态
./deploy.sh status

# 停止服务
./deploy.sh stop

# 重启服务
./deploy.sh restart

# 显示帮助
./deploy.sh help
```

### 服务管理

```bash
# 启动服务
./start_service.sh

# 停止服务
./stop_service.sh

# 查看状态
./status.sh

# 测试推理
./test_inference.sh

# 实时监控（10秒间隔）
./monitor.sh

# 自定义监控间隔（30秒）
./monitor.sh 30
```

## 📊 监控和日志

### 日志文件
- `service.log` - 服务运行日志
- `monitor.log` - 监控记录日志

### 实时监控
```bash
# 启动实时监控
./monitor.sh

# 查看实时日志
tail -f service.log
```

## 🧪 测试验证

### API 测试
```bash
# 运行完整测试套件
./test_inference.sh

# 手动测试 API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-lora",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100
  }'
```

### 性能测试
测试脚本会自动检查：
- ✅ 模型列表获取
- ✅ 对话功能
- ✅ 响应时间
- ✅ 流式输出

## ⚙️ 配置说明

### 自动配置
脚本会根据以下条件自动调整配置：

#### GPU 显存优化
- **< 6GB**: 小显存配置（dtype=half, max_len=2048）
- **6-12GB**: 中等配置（dtype=half, max_len=4096）
- **> 12GB**: 大显存配置（dtype=bfloat16, max_len=8192）

#### 模型选择
- 自动检测并使用最新的微调模型
- 如无微调模型则使用基础模型
- 支持 LoRA 适配器自动加载

### 手动配置
编辑 `config.json` 文件：
```json
{
  "model_path": "../output/qwen3-lora-lowmem-xxx",
  "base_model_path": "../models/Qwen3-0.6B",
  "port": 8000,
  "host": "0.0.0.0"
}
```

## 🔍 故障排除

### 常见问题

#### 1. 端口被占用
```bash
# 检查端口占用
lsof -i :8000

# 强制停止服务
./stop_service.sh
```

#### 2. GPU 内存不足
```bash
# 查看 GPU 使用情况
nvidia-smi

# 调整配置使用更小的模型长度
# 编辑 start_service.sh 中的 MAX_LEN 参数
```

#### 3. 依赖安装失败
```bash
# 重新安装依赖
cd ..
rm -rf .venv
uv venv .venv
UV_HTTP_TIMEOUT=120 uv sync
```

#### 4. 模型加载失败
```bash
# 检查模型文件
ls -la ../models/Qwen3-0.6B/
ls -la ../output/

# 查看详细日志
tail -f service.log
```

### 日志分析
```bash
# 查看错误日志
grep -i error service.log

# 查看警告日志
grep -i warning service.log

# 查看最近的请求
grep "POST /v1/chat" service.log | tail -10
```

## 🔗 API 接口

### 聊天接口
```
POST http://localhost:8000/v1/chat/completions
```

### 模型列表
```
GET http://localhost:8000/v1/models
```

### 健康检查
```
GET http://localhost:8000/health
```

## 📈 性能优化

### GPU 优化
- 使用 FP16 精度减少显存占用
- 动态调整 batch size 和 sequence length
- 启用 CUDA 内存池优化

### CPU 优化
- 自动检测并使用所有可用 CPU 核心
- 优化 tokenizer 并行处理
- 禁用不必要的日志输出

## 🛡️ 安全注意事项

- 默认绑定到 `0.0.0.0:8000`，生产环境请修改为内网地址
- 建议在防火墙后部署
- 定期检查和更新依赖包

## 📞 支持

如遇到问题，请：
1. 查看 `service.log` 日志文件
2. 运行 `./status.sh` 检查服务状态
3. 运行 `./test_inference.sh` 进行诊断测试

---

🎉 **部署完成后，您的 Qwen3 微调模型就可以通过 API 提供服务了！**
