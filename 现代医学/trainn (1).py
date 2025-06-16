import json
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model
import os
import swanlab
from datetime import datetime

# 常量定义
PROMPT = "If you are a doctor, please answer the medical questions based on the patient's description."
MAX_LENGTH = 1024
MODEL_NAME = "Qwen/Qwen3-14B"
SWANLAB_API_KEY = os.getenv("SWANLAB_API_KEY", "6Y7FwqjEqksmgDalAb89M")  # 从环境变量获取或直接设置

# 初始化SwanLab（添加错误处理）
try:
    if SWANLAB_API_KEY and SWANLAB_API_KEY != "your_api_key_here":
        swanlab.login(api_key=SWANLAB_API_KEY)
        os.environ["SWANLAB_PROJECT"] = "qwen3-sft-medical"
    else:
        print("警告: 使用本地模式运行，实验记录不会上传到SwanLab云")
except Exception as e:
    print(f"SwanLab初始化错误: {e}")

# 分布式训练环境设置
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# [保持原有的 dataset_jsonl_transfer, create_process_func, predict 函数不变]

def train(rank, world_size, args):
    """分布式训练主函数"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # 只在主进程初始化SwanLab
    if rank == 0:
        try:
            swanlab.init(
                project="qwen3-sft-medical",
                config={
                    "model": MODEL_NAME,
                    "prompt": PROMPT,
                    "max_length": MAX_LENGTH,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "gpus": world_size
                },
                mode="online" if SWANLAB_API_KEY and SWANLAB_API_KEY != "your_api_key_here" else "local"
            )
        except Exception as e:
            print(f"SwanLab初始化失败: {e}")

    # [保持原有的模型加载、数据处理、训练代码不变]

    # 只在主进程保存和记录
    if rank == 0:
        try:
            trainer.save_model(args.output_dir)
            
            # 测试和记录代码
            test_df = pd.read_json(args.eval_path, lines=True)[:3]
            test_text_list = []
            
            for _, row in test_df.iterrows():
                messages = [
                    {"role": "system", "content": str(row['instruction'])},
                    {"role": "user", "content": str(row['input'])}
                ]
                response = predict(messages, model.module, tokenizer)
                response_text = f"问题: {row['input']}\n回答: {response}"
                test_text_list.append(swanlab.Text(response_text))
                print(response_text)

            if test_text_list:
                swanlab.log({"预测示例": test_text_list})
            swanlab.finish()
        except Exception as e:
            print(f"保存和记录时出错: {e}")
    
    cleanup()

class Args:
    model_dir = "./Qwen/Qwen3-7B"
    train_path = "train_format.jsonl"
    eval_path = "val_format.jsonl"
    output_dir = "./output/Qwen3-7B"
    batch_size = 2
    gradient_accumulation_steps = 2
    eval_steps = 500
    logging_steps = 100
    save_steps = 1000
    epochs = 2
    learning_rate = 1e-4

if __name__ == "__main__":
    args = Args()
    
    # 数据准备检查
    if not os.path.exists(args.train_path):
        dataset_jsonl_transfer("train.jsonl", args.train_path)
    if not os.path.exists(args.eval_path):
        dataset_jsonl_transfer("val.jsonl", args.eval_path)
    
    # 模型下载检查
    if not os.path.exists(args.model_dir):
        snapshot_download(MODEL_NAME, cache_dir="./", revision="master")
    
    # 启动分布式训练
    world_size = 2
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )