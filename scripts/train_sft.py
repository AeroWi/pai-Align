import os
import sys
import torch
import wandb
import json
from tqdm import tqdm
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from vllm import LLM, SamplingParams


# 确保根目录在路径中
sys.path.append(os.getcwd())

# 按照模块路径导入
try:
    from scripts.evaluate_vllm import run_evaluation
except ModuleNotFoundError:
    # 如果上面的失败，尝试直接导入同级目录下的模块
    import evaluate_vllm as ev
    run_evaluation = ev.run_evaluation

# 导入你之前 PASSED 的辅助函数
from cs336_alignment.sft import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

## --- 作业提供的 vLLM 初始化与权重加载函数 ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.2): # 默认改小
    from vllm.model_executor import set_random_seed as vllm_set_random_seed
    vllm_set_random_seed(seed)
    
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id, 
            device=device, 
            dtype=torch.bfloat16, 
            enable_prefix_caching=True, 
            gpu_memory_utilization=gpu_memory_utilization, # 关键：在这里传入参数
            enforce_eager=True # 建议加上，显存更省
        )        


def load_policy_into_vllm_instance(policy, llm):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

## --- 训练主逻辑 ---
def train():
    # 1. 配置参数
    # MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/SFT_model"
    TRAIN_DATA = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl"
    OUTPUT_DIR = "/root/autodl-tmp/assignment5-alignment/models/SFT_model"
    
    # 实验变量：根据 Problem 4.3 改这个值 {128, 256, 512, 1024}
    NUM_EXAMPLES = 7473
    LR = 5e-6
    BATCH_SIZE = 4
    GRAD_ACCUM = 8 # 有效 Batch Size = 4 * 8 = 32
    EPOCHS = 3

    # 2. 初始化 WandB
    wandb.init(project="cs336-sft", name=f"sft-{NUM_EXAMPLES}")
    wandb.define_metric("train_step")
    wandb.define_metric("train/*", step_metric="train_step")

    # 3. 加载模型与分词器 (GPU 0)
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)

    # 4. 初始化 vLLM 用于评估 (GPU 1)
    vllm_instance = init_vllm(MODEL_PATH, device="cuda:1", seed=42, gpu_memory_utilization=0.6)

    # 5. 准备数据
    with open(TRAIN_DATA, "r") as f:
        all_data = [json.loads(line) for line in f][:NUM_EXAMPLES]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 6. 训练循环 (Algorithm 1)
    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        import random
        random.shuffle(all_data)
        
        # --- 在这里添加 tqdm 进度条 ---
        pbar = tqdm(range(0, len(all_data), BATCH_SIZE), desc=f"Epoch {epoch}")
        for i in pbar:
            batch = all_data[i : i + BATCH_SIZE]
            prompts = [item["prompt"] for item in batch]
            outputs = [item["response"] for item in batch]

            # 处理数据
            tokenized = tokenize_prompt_and_output(prompts, outputs, tokenizer)
            input_ids = tokenized["input_ids"].to(device)
            labels = tokenized["labels"].to(device)
            response_mask = tokenized["response_mask"].to(device)

            # 前向传播
            res = get_response_log_probs(model, input_ids, labels)
            
            # 微批次训练步
            loss, metrics = sft_microbatch_train_step(
                res["log_probs"], response_mask, GRAD_ACCUM, normalize_constant=1.0
            )

            # 更新进度条显示的 Loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if (global_step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"train/loss": loss.item(), "train_step": global_step})

            global_step += 1

        # 7. 每个 Epoch 结束后同步权重并评估
        print(f"Epoch {epoch} finished. Evaluating...")
        load_policy_into_vllm_instance(model, vllm_instance)
        print(f"\nEpoch {epoch} finished. Synchronizing weights and evaluating...")
        load_policy_into_vllm_instance(model, vllm_instance)
        
        # 调用独立评估模块
        eval_results = run_evaluation(
            vllm_instance=vllm_instance,
            test_data_path="/root/autodl-tmp/assignment5-alignment/data/gsm8k/test.jsonl",
            prompt_template_path="/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
        )
        
        # 记录到 WandB
        current_acc = eval_results["accuracy"]
        print(f"Epoch {epoch} Accuracy: {current_acc:.2%} (Format Rate: {eval_results['format_rate']:.2%})")
        
        wandb.log({
            "eval/accuracy": current_acc,
            "eval/format_rate": eval_results["format_rate"],
            "train_step": global_step
        })
        epoch_save_dir = f"{OUTPUT_DIR}_epoch_{epoch}"
        model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        print(f"Epoch {epoch} checkpoint saved to {epoch_save_dir}")
        
        
    # 8. 保存模型
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()