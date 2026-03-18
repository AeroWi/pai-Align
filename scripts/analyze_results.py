import json
from collections import Counter

def analyze():
    file_path = "results/baseline_results.jsonl"
    stats = Counter()
    total = 0
    
    # 用来存放格式错误的例子，方便你写 3.2 (b) 的定性分析
    format_error_examples = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                data = json.loads(line)
                # 获取奖励分数
                scores = data.get("scores", {})
                fmt = scores.get("format_reward", 0.0)
                ans = scores.get("answer_reward", 0.0)
                
                if fmt == 1.0 and ans == 1.0:
                    stats["both_correct"] += 1
                elif fmt == 1.0 and ans == 0.0:
                    stats["format_only"] += 1
                else:
                    stats["format_failed"] += 1
                    # 收集前10个格式错误的例子
                    if len(format_error_examples) < 10:
                        format_error_examples.append(data.get("model_output", ""))

        # --- 输出 3.2 (c) 要求的量化结果 ---
        print("\n" + "="*40)
        print(f"评估数据集总量: {total}")
        print("-" * 40)
        print(f"Category 1 (完全正确): {stats['both_correct']} ({stats['both_correct']/total:.2%})")
        print(f"Category 2 (格式对答案错): {stats['format_only']} ({stats['format_only']/total:.2%})")
        print(f"Category 3 (格式错误/0分): {stats['format_failed']} ({stats['format_failed']/total:.2%})")
        print("="*40 + "\n")

        # --- 输出 3.2 (b) 要求的定性例子 ---
        print(f"以下是前 {len(format_error_examples)} 个格式错误的输出片段（供分析原因）：")
        for i, out in enumerate(format_error_examples):
            # 只展示前 150 个字符，防止刷屏
            snippet = out.replace("\n", " ")[:150]
            print(f"[{i+1}] {snippet}...")

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}，请确认评估脚本是否运行成功。")

if __name__ == "__main__":
    analyze()