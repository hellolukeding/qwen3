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

# 更好的prompt格式，添加停止条件
prompt = "请简要介绍引力的原理。"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=120,         # 进一步减少长度
    do_sample=True,
    temperature=0.6,           # 降低随机性
    top_p=0.8,                 # 降低top_p，作用于提高聚焦度
    top_k=40,                  # 限制词汇选择 
    repetition_penalty=1.1,    # 适度的重复惩罚
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    no_repeat_ngram_size=3,
)

# 只显示生成的回答部分
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# 清理输出，只保留第一个完整的回答
answer = generated_text.strip()
# 如果有多个段落，只取第一个
if '\n\n' in answer:
    answer = answer.split('\n\n')[0]

print(f"问题：请简要介绍引力的原理。")
print(f"回答：{answer}")
