import os
import json
from vllm import LLM, SamplingParams
# 导入作业提供的评分函数
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def run_evaluation():
    # --- 1. 配置本地路径 ---
    MODEL_PATH = "/root/autodl-tmp/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/test.jsonl"
    PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
    OUTPUT_FILE = "/root/autodl-tmp/assignment5-alignment/results/baseline_results.jsonl"

    # --- 2. 加载提示词模板 ---
    with open(PROMPT_PATH, "r") as f:
        r1_zero_template = f.read()

    # --- 3. 初始化 vLLM ---
    # Qwen2.5-Math-1.5B 较小，单张显卡通常可直接加载
    llm = LLM(model=MODEL_PATH)

    # 设置采样参数：T=1.0, P=1.0, 遇到 </answer> 停止
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    # --- 4. 加载数据并格式化 ---
    prompts = []
    gold_answers = []
    
    with open(DATA_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            # 注意：GSM8K 的字段通常是 'question' 和 'answer'
            prompt = r1_zero_template.format(question=item["question"])
            prompts.append(prompt)
            
            # 2. 提取标准答案数字并【存入列表】
            # 原来的代码漏掉了下面这一行 append
            clean_answer = item["answer"].split("####")[-1].strip()
            gold_answers.append(clean_answer) # <--- 必须补上这一行

    # --- 5. 批量生成回复 ---
    print(f"正在对 {len(prompts)} 个示例进行推理...")
    outputs = llm.generate(prompts, sampling_params)

    # --- 6. 评分并保存结果 ---
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            gold_answer = gold_answers[i]
            
            # 调用奖励函数计算得分
            # 返回字典包含 'format_reward' 和 'answer_reward'
            scores = r1_zero_reward_fn(generated_text, gold_answer)
            
            result = {
                "question": prompts[i],
                "model_output": generated_text,
                "gold_answer": gold_answer,
                "scores": scores
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"评估完成！结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    run_evaluation()