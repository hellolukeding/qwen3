# Qwen3-0.6B 项目总览

## 🎯 项目目标

将 Qwen3-0.6B 打造成一个具备 MCP 和工具调用功能的智能助手系统。

## 📂 项目结构说明

### 🎯 models/ - 模型文件
- `Qwen3-0.6B/` - Qwen3-0.6B 模型文件

### 🚀 scripts/ - 自动化脚本
- `setup.sh` - 一键环境安装
- `start_vllm.sh` - vLLM 服务启动
- `stop_vllm.sh` - vLLM 服务停止

### 🧪 tests/ - 测试套件
- `test.py` - 基础 Transformers 测试
- `final_test_vllm.py` - vLLM API 测试
- `simple_time_test.py` - 性能时间测试
- `debug_test.py` - 调试测试

### 🛠️ tools/ - 核心工具
- `optimized_inference.py` - 优化推理引擎
- `response_optimizer.py` - 回复质量优化
- `mcp_tools/` - MCP 工具调用系统
  - `mcp_server.py` - MCP 服务器
  - `mcp_client.py` - MCP 客户端
  - `test_mcp.py` - MCP 测试

### 📚 docs/ - 文档目录
- `git_setup.md` - Git 配置指南

## 🌟 核心特性

1. **高性能推理**: vLLM 引擎，支持并发
2. **质量优化**: 智能回复优化和评估
3. **工具调用**: MCP 协议兼容的工具系统
4. **易于使用**: 一键安装和启动脚本

## 🚀 快速开始

```bash
# 1. 安装环境
./scripts/setup.sh

# 2. 启动服务
./scripts/start_vllm.sh

# 3. 测试功能
python tests/final_test_vllm.py

# 4. 测试 MCP 工具
python tools/mcp_tools/test_mcp.py
```

## 🎯 使用场景

- **开发测试**: 使用 tests/ 中的脚本
- **生产部署**: 使用 scripts/ 中的启动脚本
- **功能扩展**: 在 tools/ 中添加新工具
- **性能优化**: 使用 response_optimizer.py

## 📈 项目优势

1. **结构清晰**: 按功能模块化组织
2. **易于维护**: 脚本自动化管理
3. **功能完整**: 从基础推理到高级工具调用
4. **性能优化**: 多种优化策略和工具
