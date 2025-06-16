import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import time

# 配置 DeepSeek API 密钥和端点
DEEPSEEK_API_KEY = "sk-fefc9e312daa4c68a3d4d00b68fc2886"  # 替换为你的 DeepSeek API Key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # DeepSeek API 地址

# 全局变量用于控制请求速率
MAX_WORKERS = 10  # 最大并发线程数
REQUEST_DELAY = 0.5  # 请求之间的延迟(秒)，避免触发API速率限制

def generate_think_process(query: str, response: str) -> str:
    """
    调用 DeepSeek API 生成思考过程
    """
    prompt = f"""
    你是一个专业的医学助手。请根据以下用户问题和回答，生成一个详细的思考过程：
    
    用户问题: {query}
    回答: {response}
    
    请详细说明从问题到回答的推理过程，包括：
    1. 可能的病因分析
    2. 为什么这样回答
    3. 回答中考虑的因素
    4. 任何需要注意的事项
    
    用中文输出思考过程，保持专业但易懂。
    """
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "deepseek-chat",  # DeepSeek 的模型名称
        "messages": [
            {"role": "system", "content": "你是一个专业的医学助手，能够详细分析医学问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # 检查请求是否成功
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"调用 DeepSeek API 出错: {e}")
        return f"API调用错误: {str(e)}"
    except Exception as e:
        print(f"处理 API 响应出错: {e}")
        return "无法生成思考过程"

def process_item(item: Dict) -> Dict:
    """
    处理单个数据项，生成思考过程并返回处理后的项
    """
    try:
        query = item.get("query", "")
        response = item.get("response", "")
        
        if query and response:
            think = generate_think_process(query, response)
            processed_item = {
                "query": query,
                "think": think,
                "response": response
            }
            return processed_item
        return None
    except Exception as e:
        print(f"处理数据项出错: {e}")
        return None

def process_jsonl_file(input_file: str, output_file: str):
    """
    处理 JSONL 文件，添加思考过程并保存到新文件
    """
    # 首先读取所有数据
    with open(input_file, "r", encoding="utf-8") as f_in:
        items = [json.loads(line.strip()) for line in f_in if line.strip()]
    
    print(f"开始处理 {len(items)} 条数据...")
    
    processed_data = []
    success_count = 0
    fail_count = 0
    first_item_printed = False  # 标记是否已打印第一条数据
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_item, item): item 
            for item in items
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                if result:
                    processed_data.append(result)
                    success_count += 1
                    
                    # 打印第一条处理完成的数据内容
                    if not first_item_printed:
                        print("\n=== 第一条处理完成的数据 ===")
                        print(f"问题: {result['query']}")
                        print(f"思考过程: {result['think']}")
                        print(f"原回答: {result['response']}")
                        print("="*40 + "\n")
                        first_item_printed = True
                    
                    print(f"成功处理: {result['query'][:50]}... ({success_count}/{len(items)})")
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                print(f"处理失败: {str(e)}")
            
            # 添加延迟以避免触发API速率限制
            time.sleep(REQUEST_DELAY)
    
    # 按原始顺序保存结果
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in processed_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n处理完成，结果已保存到 {output_file}")
    print(f"成功处理: {success_count} 条, 失败: {fail_count} 条")

def main():
    input_file = "val.jsonl"  # 修改这里，使用 train.jsonl 作为输入文件
    output_file = "processed_data.jsonl"  # 输出 JSONL 文件路径
    
    process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()