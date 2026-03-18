import os
import sys
import torch
import json
import wandb
import random
from tqdm import tqdm
from typing import List, Dict, Any
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# 确保路径正确
sys.path.append(os.getcwd())

from cs336_alignment.sft import (
    tokenize_prompt_and_output, 
    get_response_log_probs, 
    sft_microbatch_train_step
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
 
try:
    from scripts.evaluate_vllm import run_evaluation
except ModuleNotFoundError:
    import evaluate_vllm as ev
    run_evaluation = ev.run_evaluation



# --- 1. 环境与参数配置 ---
# MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/SFT_model"
MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/expert_iteration_v1"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
TRAIN_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl"
VALID_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/test.jsonl"



# EI 超参数 (根据作业 Problem expert_iteration_experiment)
N_EI_STEPS = 5              # 总迭代次数 [cite: 425, 453]
DB_SIZE = 1024               # 每步采样的题目数 Db [cite: 453]
G = 8                       # 每个题目的 Rollout 数量 [cite: 434, 453]
SFT_EPOCHS_PER_STEP = 1     # 拿到 Dsft 后的微调轮数 [cite: 453]
LR = 1e-5
BATCH_SIZE = 2
GRAD_ACCUM = 16              # 有效 Batch Size = 32
MAX_TOKENS = 1024

# --- 2. 核心辅助函数 ---

def load_math_data(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def init_vllm(model_id: str, device: str, seed: int):
    """作业提供的 vLLM 初始化 [cite: 367, 385]"""
    from vllm.model_executor import set_random_seed
    set_random_seed(seed)
    
    # 解决设备冲突与内存占用的 Patch [cite: 377, 382]
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=0.5 # 留点呼吸空间防止 OOM
        )

def load_policy_into_vllm(policy, vllm_inst):
    """将训练好的权重加载到推理卡上 [cite: 392, 399]"""
    state_dict = policy.state_dict()
    llm_model = vllm_inst.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def collect_expert_data(vllm_inst, batch_questions, rollout_g, reward_fn):
    """Algorithm 2: 生成并过滤正确答案 [cite: 434, 435]"""
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=MAX_TOKENS,
        min_tokens=4,   # 必加：防止空响应 
        n=rollout_g,    # 每个问题生成 G 个结果 [cite: 447]
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    prompts = [item["prompt"] for item in batch_questions]
    # vLLM 批量生成
    outputs = vllm_inst.generate(prompts, sampling_params)

    dsft = []
    total_entropy = 0
    token_count = 0

    for idx, output in enumerate(outputs):
        gold = batch_questions[idx]["answer"]
        for res in output.outputs:
            # 记录奖励逻辑 [cite: 435]
            reward_info = reward_fn(res.text, gold)
            if reward_info["reward"] == 1.0:
                dsft.append({"prompt": prompts[idx], "response": res.text})
    
    return dsft

# --- 3. 训练主循环 ---

def train_ei():
    # 初始化 WandB 指标定义 
    wandb.init(project="cs336-ei", name=f"ei-db{DB_SIZE}-g{G}-stage2")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 设备分配
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device) # [cite: 188, 194]
    
    vllm_inst = init_vllm(MODEL_PATH, device="cuda:1", seed=42) # [cite: 364]
    
    all_train_questions = load_math_data(TRAIN_DATA_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    global_step = 0
    eval_step = 0

    # Algorithm 2 外部循环 [cite: 425]
    for ei_step in range(N_EI_STEPS):
        print(f"\n--- EI Step {ei_step} ---")
        
        # (1) 采样本轮问题 Db [cite: 432]
        current_db = random.sample(all_train_questions, DB_SIZE)
        
        # (2) 同步模型到 vLLM 并生成正确轨迹 [cite: 433, 434]
        model.eval()
        load_policy_into_vllm(model, vllm_inst)
        dsft_data = collect_expert_data(vllm_inst, current_db, G, r1_zero_reward_fn)
        
        print(f"Generated {len(dsft_data)} correct traces from {DB_SIZE} questions.")
        wandb.log({"train/dsft_size": len(dsft_data), "train_step": global_step})

        if not dsft_data:
            continue

        # (3) 对模型进行 SFT 微调 (Algorithm 1) [cite: 435]
        model.train()
        for epoch in range(SFT_EPOCHS_PER_STEP):
            random.shuffle(dsft_data)
            pbar = tqdm(range(0, len(dsft_data), BATCH_SIZE), desc=f"EI {ei_step} Epoch {epoch}")
            
            for i in pbar:
                batch = dsft_data[i : i + BATCH_SIZE]
                prompts = [item["prompt"] for item in batch]
                responses = [item["response"] for item in batch]
                
                # 数据分词处理 [cite: 247, 248]
                token_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
                input_ids = token_data["input_ids"].to(device)
                labels = token_data["labels"].to(device)
                mask = token_data["response_mask"].to(device)

                # 获取 logprobs 并计算熵 [cite: 286, 287]
                res_probs = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                
                # 记录熵 [cite: 269, 454]
                avg_entropy = (res_probs["token_entropy"] * mask).sum() / mask.sum()

                # 执行更新步 [cite: 325, 326]
                loss, _ = sft_microbatch_train_step(
                    res_probs["log_probs"], mask, GRAD_ACCUM
                )

                if (global_step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # [cite: 449]
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    wandb.log({
                        "train/loss": loss.item() * GRAD_ACCUM,
                        "train/entropy": avg_entropy.item(),
                        "train_step": global_step
                    })
                
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item()*GRAD_ACCUM:.4f}"})

        # (4) 每步 EI 结束后进行验证评估 
        print(f"EI Step {ei_step} complete. Evaluating...")
        load_policy_into_vllm(model, vllm_inst)
        eval_metrics = run_evaluation(
            vllm_inst, 
            VALID_DATA_PATH, 
            PROMPT_PATH
        )
        
        wandb.log({
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/format_rate": eval_metrics.get("format_rate", 0),
            "eval_step": eval_step
        })
        eval_step += 1

    # 保存最终产物 [cite: 437]
    model.save_pretrained("models/expert_iteration_v2")
    tokenizer.save_pretrained("models/expert_iteration_v2")

if __name__ == "__main__":
    train_ei()