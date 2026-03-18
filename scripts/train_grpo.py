import os
import sys
import torch
import json
import wandb
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch

# --- [路径补丁：确保能找到 scripts 下的模块] ---
sys.path.append(os.getcwd())

# 按照 SFT 脚本的成熟做法导入评估函数
try:
    from scripts.evaluate_vllm import run_evaluation
except ModuleNotFoundError:
    import evaluate_vllm as ev
    run_evaluation = ev.run_evaluation

# 导入 GRPO 算子
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.sft import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# --- 核心配置 ---
LR = 1e-5
EPOCHS_PER_ROLLOUT = 3
MODEL_PATH = "models/grpo_final_results/step_50"
SAVE_DIR = "models/grpo_final_results"
TEST_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/test.jsonl"
PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"

ROLLOUT_BATCH_SIZE = 256
GROUP_SIZE = 8
GRAD_ACCUM = 128 
CLIPRANGE = 0.2
MAX_TOKENS = 1024

os.makedirs(SAVE_DIR, exist_ok=True)

def load_policy_into_vllm_instance(policy, llm):
    """参考 SFT 脚本，将 PyTorch 模型权重同步到 vLLM"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def train_grpo():
    # 初始化 WandB，明确定义指标
    wandb.init(project="cs336-grpo", name=f"grpo-FINAL-lr{LR}-ep{EPOCHS_PER_ROLLOUT}")
    wandb.define_metric("step")
    wandb.define_metric("eval/*", step_metric="step")
    wandb.define_metric("train/*", step_metric="step")

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device)
    
    # GPU 1 负责推理采样与评估
    print("Initializing vLLM for rollout and evaluation on cuda:1...")
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    
    with world_size_patch, profiling_patch:
        vllm_inst = LLM(
            model=MODEL_PATH, device="cuda:1", dtype="bfloat16", 
            gpu_memory_utilization=0.5, # 留出空间给评估时的 KV Cache
            enable_prefix_caching=True, enforce_eager=True
        )
    
    with open("/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl", "r") as f:
        questions = [json.loads(l) for l in f]

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    best_accuracy = 0.0  # 初始化历史最高准确率

    for step in range(200):
        # --- [评估逻辑：每 5 步跑一次] ---
        if step % 5 == 0:
            print(f"\n--- Step {step}: Synchronizing weights and evaluating ---")
            load_policy_into_vllm_instance(model, vllm_inst)
            eval_results = run_evaluation(
                vllm_instance=vllm_inst,
                test_data_path=TEST_DATA_PATH,
                prompt_template_path=PROMPT_PATH
            )
            current_acc = eval_results['accuracy']
            print(f"Step {step} Accuracy: {current_acc:.2%} (Format: {eval_results['format_rate']:.2%})")
            
            # --- [核心改动：择优保存] ---
            if current_acc > best_accuracy:
                best_accuracy = current_acc
                best_model_path = os.path.join(SAVE_DIR, "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                print(f"🌟 发现更好的模型! 准确率: {current_acc:.2%}, 已保存至 {best_model_path}")

            wandb.log({
                "eval/accuracy": current_acc,
                "eval/format_rate": eval_results["format_rate"],
                "step": step
            })

        # (1) 采样阶段
        sampled_qs = random.sample(questions, ROLLOUT_BATCH_SIZE // GROUP_SIZE)
        prompts = [q["prompt"] for q in sampled_qs for _ in range(GROUP_SIZE)]
        gts = [q["answer"] for q in sampled_qs for _ in range(GROUP_SIZE)]
        
        sampling_params = SamplingParams(temperature=1.0, max_tokens=MAX_TOKENS, stop=["</answer>"], include_stop_str_in_output=True)
        outputs = vllm_inst.generate(prompts, sampling_params)
        responses = [out.outputs[0].text for out in outputs]

        advantages, _, meta = compute_group_normalized_rewards(
            r1_zero_reward_fn, responses, gts, GROUP_SIZE, normalize_by_std=True
        )
        
        # (2) 计算 Old Logprobs
        token_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids, labels = token_data["input_ids"].to(device), token_data["labels"].to(device)
        mask = token_data["response_mask"].to(device)
        
        model.eval()
        all_old_log_probs = []
        with torch.no_grad():
            for i in range(0, ROLLOUT_BATCH_SIZE, 4):
                all_old_log_probs.append(get_response_log_probs(model, input_ids[i:i+4], labels[i:i+4])["log_probs"])
        old_log_probs = torch.cat(all_old_log_probs, dim=0)

        # (3) 离策训练更新
        model.train()
        for epoch in range(EPOCHS_PER_ROLLOUT):
            micro_size = ROLLOUT_BATCH_SIZE // GRAD_ACCUM
            for i in range(0, ROLLOUT_BATCH_SIZE, micro_size):
                m_idx = slice(i, i + micro_size)
                current_res = get_response_log_probs(model, input_ids[m_idx], labels[m_idx])
                loss, _ = grpo_microbatch_train_step(
                    current_res["log_probs"], mask[m_idx], GRAD_ACCUM, "grpo_clip",
                    advantages=advantages[m_idx].to(device), old_log_probs=old_log_probs[m_idx], cliprange=CLIPRANGE
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # 同步权重以便下一轮采样（如果是 step % 5 == 0，这一步评估前已经同步过了，再同步一次也无妨）
        load_policy_into_vllm_instance(model, vllm_inst)
        
        # 记录训练奖励
        wandb.log({"train/reward": meta["mean"], "step": step})
        print(f"Step {step} finished. Mean Reward: {meta['mean']:.4f}")

    # 最终保存
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"🎉 Training complete. Final model saved at {SAVE_DIR}")

if __name__ == "__main__":
    train_grpo()