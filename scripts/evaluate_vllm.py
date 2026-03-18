import json
import torch
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

def run_evaluation(vllm_instance, test_data_path, prompt_template_path):
    """
    核心评估函数：接收 vLLM 实例，跑完测试集并返回准确率。
    """
    # 1. 加载提示词模板
    with open(prompt_template_path, "r") as f:
        template = f.read()

    # 2. 加载测试数据并提取纯数字答案
    prompts = []
    gold_answers = []
    with open(test_data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # 包装成 R1 Zero 格式：Assistant: <think>
            prompts.append(template.format(question=item["question"]))
            # 提取 GSM8K 标准答案数字
            gold_answers.append(item["answer"].split("####")[-1].strip())
            
    # --- 在这里添加临时切片，仅测试前 () 条 ---
    prompts = prompts[:]
    gold_answers = gold_answers[:]
    print(f"DEBUG: 临时缩减评估规模，当前评估 {len(prompts)} 条数据")

    # 3. 配置推理参数 (严格按照作业要求)
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 4. 批量推理
    print(f"正在评估 {len(prompts)} 条测试数据...")
    outputs = vllm_instance.generate(prompts, sampling_params)

    # 5. 评分逻辑
    correct = 0
    format_correct = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = gold_answers[i]
        
        # 调用作业提供的 R1 奖励函数
        scores = r1_zero_reward_fn(generated_text, gold_answer)
        
        # 统计格式正确率 (可选)
        if scores.get("format", scores.get("format_reward", 0)) > 0:
            format_correct += 1
            
        # 统计答案正确率
        if scores.get("answer", scores.get("answer_reward", 0)) > 0:
            correct += 1

    accuracy = correct / len(prompts)
    format_rate = format_correct / len(prompts)
    
    return {
        "accuracy": accuracy,
        "format_rate": format_rate,
        "total": len(prompts)
    }

if __name__ == "__main__":
    # 如果你想独立运行测试，可以在这里写简单的初始化逻辑
    print("这是一个评估模块，请通过 train_sft.py 调用或补充初始化逻辑后运行。")