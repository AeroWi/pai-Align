import json
import os
from tqdm import tqdm

def transform_gsm8k_to_sft_format():
    INPUT_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/train.jsonl"
    OUTPUT_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k/train_filtered.jsonl"
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    transformed_data = []
    
    print(f"正在重构 GSM8K 为 SFT 格式...")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            raw_answer = item["answer"]
            
            if "####" in raw_answer:
                # 核心逻辑：将原始数据拆分为“思考过程”和“数字答案”
                reasoning, gold_num = raw_answer.split("####")
                reasoning = reasoning.strip()
                gold_num = gold_num.strip()
                
                # 【关键】包装成评分器认出来的 XML 格式
                # 严格按照评分器的要求：</think> 和 <answer> 之间用一个空格，且内部不换行
                formatted_response = f"<think>\n{reasoning}\n</think> <answer>{gold_num}</answer>"
                
                # 构造 SFT 脚本需要的 prompt/response 结构
                new_item = {
                    "prompt": item["question"],
                    "response": formatted_response,
                    "answer": gold_num
                }
                transformed_data.append(new_item)

    # 写入文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in transformed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"\n重构完成！保留样本: {len(transformed_data)}")
    print(f"现在你可以用这个文件进行 SFT 训练了。")

if __name__ == "__main__":
    transform_gsm8k_to_sft_format()