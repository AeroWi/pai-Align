import json
import os
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def main():
    # --- 1. 配置路径与参数 
    MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/SFT_model" # 确保这是你备份的路径
    TRAIN_DATA = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl"
    OUTPUT_FILE = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/ei_rollout_n8.jsonl"
    
    N_SAMPLES = 8  # 每道题生成8个结果
    TEMPERATURE = 0.8  # 增加随机性以进行“探索”
    MAX_TOKENS = 1024

    # --- 2. 加载数据 ---
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    
    prompts = [item["prompt"] for item in items]
    ground_truths = [item["answer"] for item in items]

    # --- 3. 初始化 vLLM (利用双卡 4090 的吞吐量) ---
    # 注意：采样对显存要求较高，gpu_memory_utilization 设为 0.9 以获取最大并发
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1, # 1.5B模型单卡即可，双卡可以开两个进程或让vLLM自动管理
        gpu_memory_utilization=0.9,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        n=N_SAMPLES, # 关键：一次性生成N个
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # --- 4. 执行批量采样 ---
    print(f"正在为 {len(prompts)} 道题生成 {N_SAMPLES} 路采样...")
    outputs = llm.generate(prompts, sampling_params)

    # --- 5. 评分并保存 ---
    expert_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, output in enumerate(tqdm(outputs, desc="筛选专家样本")):
            prompt = prompts[i]
            gold = ground_truths[i]
            
            # 遍历这 N 个生成的 response
            for res in output.outputs:
                response_text = res.text
                
                # 使用评分函数判定
                reward_data = r1_zero_reward_fn(response_text, gold)
                
                if reward_data["reward"] == 1.0:
                    # 只有对的才存下来，作为“专家数据”
                    json.dump({
                        "prompt": prompt,
                        "response": response_text,
                        "answer": gold
                    }, f, ensure_ascii=False)
                    f.write("\n")
                    expert_count += 1

    print(f"\n--- Rollout 完成 ---")
    print(f"总计生成样本: {len(prompts) * N_SAMPLES}")
    print(f"筛选出的专家样本数: {expert_count}")
    print(f"数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()