import requests

# 添加API Key认证头
headers = {
    "Authorization": "Bearer sk-vet-qwen3-lora-your-secret-key-here",
    "Content-Type": "application/json"
}

print(requests.get("http://127.0.0.1:9999/v1/models", headers=headers).json())

url = "http://127.0.0.1:9999/v1/chat/completions"
payload = {
    "model": "vet-logicstorm-lora",
    "messages": [
        {"role": "user", "content": "请介绍犬瘟热的主要症状。"}
    ]
}

response = requests.post(url, json=payload, headers=headers)
print(response.status_code)
print(response.text)ts
print(requests.get("http://127.0.0.1:9999/v1/models").json())
url = "http://127.0.0.1:9999/v1/chat/completions"
payload = {
    "model": "vet-qwen3-lora",
    "messages": [
        {"role": "user", "content": "请介绍犬瘟热的主要症状。"}
    ]
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.text)