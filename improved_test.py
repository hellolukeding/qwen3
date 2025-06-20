from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

# 设置pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 改进的prompt格式
prompt = "请简要介绍引力的原理。"  # 简化prompt格式
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("开始生成回答...")

outputs = model.generate(
    **inputs,
    max_new_tokens=150,          # 适中的长度
    do_sample=True,              
    temperature=0.8,             # 稍微提高创造性
    top_p=0.85,                  # 减少top_p值，提高聚焦度
    top_k=40,                    # 添加top_k限制
    repetition_penalty=1.05,     # 轻微的重复惩罚
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    no_repeat_ngram_size=2,      # 防止2-gram重复
)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"原始输出:\n{result}")

# 只输出新生成的部分
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\n生成的回答:\n{generated_text}")
