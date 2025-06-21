#!/bin/bash
# GPU 内存监控脚本

echo "🔍 GPU 内存监控工具"
echo "按 Ctrl+C 退出监控"
echo ""

while true; do
    clear
    echo "=== GPU 内存状态 $(date) ==="
    
    # 使用 nvidia-smi 显示详细信息
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
        echo ""
        nvidia-smi
    else
        echo "❌ nvidia-smi 不可用"
    fi
    
    echo ""
    
    # 使用 Python 显示 PyTorch 内存信息
    uv run python -c "
import torch
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = props.total_memory / 1e9
    free = total - allocated
    
    print(f'PyTorch 内存状态:')
    print(f'  设备: {props.name}')
    print(f'  总内存: {total:.2f} GB')
    print(f'  已分配: {allocated:.2f} GB ({allocated/total*100:.1f}%)')
    print(f'  已保留: {reserved:.2f} GB ({reserved/total*100:.1f}%)')
    print(f'  可用: {free:.2f} GB ({free/total*100:.1f}%)')
else:
    print('PyTorch CUDA 不可用')
" 2>/dev/null || echo "无法获取 PyTorch 内存信息"
    
    sleep 2
done
