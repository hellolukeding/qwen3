#!/usr/bin/env python3
"""
数据集格式验证脚本
"""

import json
import sys
from pathlib import Path

def validate_dataset(dataset_path):
    """验证数据集格式"""
    print(f"验证数据集: {dataset_path}")
    
    if not Path(dataset_path).exists():
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON格式无效: {e}")
        return False
    except Exception as e:
        print(f"错误: 读取文件失败: {e}")
        return False
    
    if not isinstance(data, list):
        print("错误: 数据集应该是一个列表")
        return False
    
    print(f"数据集条目数: {len(data)}")
    
    # 检查必需字段
    required_fields = ['instruction', 'output']
    optional_fields = ['input', 'system']
    
    valid_count = 0
    error_count = 0
    
    for i, item in enumerate(data[:10]):  # 只检查前10条
        print(f"\n--- 样本 {i+1} ---")
        
        if not isinstance(item, dict):
            print(f"错误: 第{i+1}条不是字典格式")
            error_count += 1
            continue
        
        # 检查必需字段
        missing_fields = []
        for field in required_fields:
            if field not in item:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"错误: 缺少必需字段: {missing_fields}")
            error_count += 1
            continue
        
        # 显示字段内容
        print(f"instruction: {item['instruction'][:100]}...")
        print(f"output: {item['output'][:100]}...")
        
        if item.get('input', '').strip():
            print(f"input: {item['input'][:100]}...")
        
        if item.get('system', '').strip():
            print(f"system: {item['system'][:50]}...")
        
        valid_count += 1
    
    print(f"\n=== 验证结果 ===")
    print(f"总数据条目: {len(data)}")
    print(f"有效条目: {valid_count}")
    print(f"错误条目: {error_count}")
    
    if error_count == 0:
        print("✅ 数据集格式验证通过!")
        return True
    else:
        print(f"❌ 发现 {error_count} 个错误")
        return False

def main():
    dataset_path = "./datasets/mao20250621.dataset.json"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    success = validate_dataset(dataset_path)
    
    if success:
        print("\n🚀 数据集准备就绪，可以开始微调训练！")
        print("\n运行以下命令开始训练:")
        print("./finetune/start_training.sh")
    else:
        print("\n❌ 请修复数据集格式后再尝试训练")
        sys.exit(1)

if __name__ == "__main__":
    main()
