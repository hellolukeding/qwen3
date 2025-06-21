from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "./Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 使用bfloat16提高效率
    low_cpu_mem_usage=True       # 减少CPU内存使用
)

# 设置pad_token_id
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 更清晰的prompt格式
prompt = "问题：请简要介绍引力的原理。\n回答："
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(
    **inputs,
    max_new_tokens=120,        # 减少到120个token，提高速度
    do_sample=True,
    temperature=0.7,           # 保持适中的随机性
    top_p=0.9,
    top_k=50,                  # 添加top_k限制
    repetition_penalty=1.1,    # 适度的重复惩罚
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    no_repeat_ngram_size=3,    # 防止3-gram重复
    
    
)


# 只显示新生成的内容
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"问题：请简要介绍引力的原理。")
print(f"回答：{generated_text.strip()}")