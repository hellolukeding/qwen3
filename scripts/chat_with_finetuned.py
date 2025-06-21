#!/usr/bin/env python3
"""
微调模型交互式对话脚本
与 LoRA 微调后的 Qwen3 模型进行交互式对话
"""

import torch
import sys
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class ChatBot:
    def __init__(self, adapter_path):
        self.adapter_path = adapter_path
        self.base_model_path = "./models/Qwen3-0.6B"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型"""
        print("🚀 加载微调模型...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 加载 LoRA 适配器
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        print("✅ 模型加载完成！")
        
        # 显示模型信息
        if hasattr(self.model, 'print_trainable_parameters'):
            print("\n📊 模型参数信息:")
            self.model.print_trainable_parameters()
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"📈 GPU 内存使用: {memory_used:.2f} GB")
        
    def generate_response(self, user_input, system_prompt="你是一个有用的AI助手。"):
        """生成回复"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generation_time = time.time() - start_time
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response, generation_time
    
    def chat(self):
        """开始交互式对话"""
        print("\n💬 开始对话（输入指令查看帮助）:")
        print("  - 输入 '/help' 查看帮助")
        print("  - 输入 '/exit' 或 '/quit' 退出")
        print("  - 输入 '/clear' 清屏")
        print("  - 输入 '/stats' 查看统计信息")
        print("  - 输入 '/temp <数值>' 调整温度参数 (0.1-2.0)")
        
        conversation_count = 0
        total_time = 0
        current_temperature = 0.7
        
        while True:
            try:
                user_input = input(f"\n💭 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊指令
                if user_input.lower() in ['/exit', '/quit', '退出', 'exit', 'quit']:
                    print("👋 对话结束！")
                    break
                
                elif user_input.lower() in ['/help', '帮助']:
                    self.show_help()
                    continue
                
                elif user_input.lower() in ['/clear', '清屏']:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                
                elif user_input.lower() in ['/stats', '统计']:
                    self.show_stats(conversation_count, total_time)
                    continue
                
                elif user_input.lower().startswith('/temp'):
                    try:
                        temp = float(user_input.split()[1])
                        if 0.1 <= temp <= 2.0:
                            current_temperature = temp
                            print(f"🌡️  温度参数已设置为: {current_temperature}")
                        else:
                            print("❌ 温度参数范围: 0.1-2.0")
                    except (IndexError, ValueError):
                        print("❌ 用法: /temp <数值>，例如: /temp 0.8")
                    continue
                
                # 生成回复
                print("🤖 AI: ", end="", flush=True)
                
                try:
                    response, gen_time = self.generate_response(user_input)
                    total_time += gen_time
                    conversation_count += 1
                    
                    print(response)
                    print(f"⏱️  {gen_time:.2f}秒")
                    
                except Exception as e:
                    print(f"❌ 生成失败: {e}")
                    continue
                
            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，对话结束！")
                break
            except EOFError:
                print("\n\n👋 对话结束！")
                break
        
        # 显示对话统计
        if conversation_count > 0:
            print(f"\n📈 对话统计:")
            print(f"  对话轮数: {conversation_count}")
            print(f"  总时间: {total_time:.2f}秒")
            print(f"  平均响应时间: {total_time/conversation_count:.2f}秒")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 帮助信息:")
        print("  /help    - 显示此帮助")
        print("  /exit    - 退出对话")
        print("  /clear   - 清屏")
        print("  /stats   - 显示统计信息")
        print("  /temp <数值> - 调整生成温度 (0.1-2.0)")
        print("\n💡 使用技巧:")
        print("  - 温度越低回答越保守，越高越有创意")
        print("  - 可以要求AI扮演不同角色")
        print("  - 支持多轮对话，AI会记住上下文")
    
    def show_stats(self, count, total_time):
        """显示统计信息"""
        print(f"\n📊 当前统计:")
        print(f"  对话轮数: {count}")
        print(f"  总时间: {total_time:.2f}秒")
        if count > 0:
            print(f"  平均响应时间: {total_time/count:.2f}秒")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU内存使用: {memory_used:.2f} GB")
            print(f"  GPU内存保留: {memory_reserved:.2f} GB")

def main():
    if len(sys.argv) != 2:
        print("用法: python scripts/chat_with_finetuned.py <模型路径>")
        print("示例: python scripts/chat_with_finetuned.py ./output/qwen3-lora-lowmem-20250621-195657")
        
        # 尝试自动查找最新模型
        import glob
        models = glob.glob("./output/qwen3-lora-*")
        if models:
            latest_model = max(models, key=os.path.getctime)
            print(f"\n💡 找到最新模型: {latest_model}")
            print(f"可以使用: python scripts/chat_with_finetuned.py {latest_model}")
        
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    try:
        # 创建聊天机器人
        chatbot = ChatBot(model_path)
        
        # 加载模型
        chatbot.load_model()
        
        # 开始对话
        chatbot.chat()
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
