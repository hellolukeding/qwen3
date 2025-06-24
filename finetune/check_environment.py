#!/usr/bin/env python3
"""
GPU和环境检查脚本
"""

import torch
import sys

def check_environment():
    """检查训练环境"""
    print("=== 环境检查 ===")
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA检查
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 内存检查
    import psutil
    memory = psutil.virtual_memory()
    print(f"系统内存: {memory.total / 1024**3:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")
    
    # 推荐配置
    print("\n=== 推荐训练配置 ===")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 24:
            print("✅ 高端GPU配置")
            print("推荐batch_size: 4-8")
            print("推荐lora_r: 64-128")
        elif gpu_memory >= 12:
            print("✅ 中端GPU配置")
            print("推荐batch_size: 2-4")
            print("推荐lora_r: 32-64")
        elif gpu_memory >= 8:
            print("⚠️ 入门GPU配置")
            print("推荐batch_size: 1-2")
            print("推荐lora_r: 16-32")
        else:
            print("❌ GPU内存不足，建议使用CPU训练")
    else:
        print("⚠️ CPU训练模式")
        print("推荐batch_size: 1")
        print("推荐lora_r: 16")
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    has_gpu = check_environment()
    print(f"\n{'🚀 准备开始GPU训练' if has_gpu else '⏳ 准备开始CPU训练（较慢）'}")
