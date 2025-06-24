import requests
import time

def test_vllm_api():
    """测试vLLM API推理"""
    
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    # 优化的推理参数
    data = {
        "model": "./models/Qwen3-0.6B",
        "prompt": "请简要介绍引力的原理。",
        "max_tokens": 80,           # 减少生成长度，提高速度
        "temperature": 0.7,         # 适中的随机性
        "top_p": 0.8,              # 核采样
        "repetition_penalty": 1.1,  # 轻微重复惩罚
        "stop": ["\n\n", "问题:", "答案:", "用户:", "助手:"],  # 添加停止词
        "stream": False
    }
    
    try:
        print("🚀 开始vLLM推理...")
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"].strip()
            
            # 清理输出
            if generated_text.startswith("答案："):
                generated_text = generated_text[3:].strip()
            
            end_time = time.time()
            
            print(f"✅ 推理成功 (耗时: {end_time - start_time:.2f}秒)")
            print(f"📝 问题: 请简要介绍引力的原理。")
            print(f"🤖 回答: {generated_text}")
            print(f"📊 Token统计: {result['usage']}")
            
            return True
            
        else:
            print(f"❌ API请求失败")
            print(f"状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到vLLM服务")
        print("请确保vLLM服务已启动: ./start_vllm.sh")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False

if __name__ == "__main__":
    test_vllm_api()
