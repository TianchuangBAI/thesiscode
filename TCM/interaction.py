import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict(messages, model, tokenizer):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-14B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-14B", device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora模型
model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-14B/checkpoint-50400")

# 设置系统提示
instruction = "你是一个医学专家，你需要根据用户的问题，给出带有结合中医思考还有中医解决方案的回答。"
print("系统已启动，输入'退出'或'quit'结束对话。")

while True:
    # 获取用户输入
    user_input = input("\n用户: ")
    
    # 检查是否退出
    if user_input.lower() in ['退出', 'quit']:
        print("对话结束。")
        break
    
    # 构建消息
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    
    # 获取模型响应
    response = predict(messages, model, tokenizer)
    
    # 打印响应
    print("\n医学专家:", response)