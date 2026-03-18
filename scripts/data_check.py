import json
from tqdm import tqdm
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def audit_sft_data(file_path):
    correct = 0
    total = 0
    errors = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="审核数据中"):
            item = json.loads(line)
            total += 1
            
            # 使用你的评分函数进行校验
            reward_data = r1_zero_reward_fn(item["response"], item["answer"])
            
            if reward_data["reward"] == 1.0:
                correct += 1
            else:
                errors.append({
                    "prompt": item["prompt"],
                    "response": item["response"],
                    "gold": item["answer"]
                })

    print(f"\n--- 审核结果 ---")
    print(f"总样本数: {total}")
    print(f"评分器认可数: {correct}")
    print(f"准确率: {correct/total:.2%}")
    
    if errors:
        print("\n--- 错误样本示例 ---")
        print(f"Prompt: {errors[0]['prompt']}")
        print(f"Response: {errors[0]['response']}")
        print(f"Expected: {errors[0]['gold']}")

if __name__ == "__main__":
    audit_sft_data("/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl")