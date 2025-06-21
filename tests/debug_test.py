import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("开始加载模型...")
start_time = time.time()

model_id = "./models/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f"Tokenizer加载完成，耗时: {time.time() - start_time:.2f}s")

model_start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16  # 明确指定数据类型
)
print(f"Model加载完成，耗时: {time.time() - model_start:.2f}s")
print(f"模型设备: {model.device}")
print(f"模型数据类型: {model.dtype}")

# 简单的推理测试
prompt = "你好"
print(f"\n测试prompt: {prompt}")

inference_start = time.time()
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=50,  # 减少生成长度
    do_sample=False,    # 关闭采样，使用贪心解码
    pad_token_id=tokenizer.pad_token_id,
)

inference_time = time.time() - inference_start
print(f"推理完成，耗时: {inference_time:.2f}s")

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"生成结果: {result}")
