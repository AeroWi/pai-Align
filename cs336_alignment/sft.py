import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional # 注意首字母大写
from transformers import PreTrainedModel, PreTrainedTokenizer

# --- 4.2 SFT Helper Methods ---

def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizer):
    """
    Problem: Prompt and output tokenization (2 points)
    拼接 Prompt 和 Output，并构造 response_mask。
    """
    # 逻辑见上一步，这里我们可以直接使用之前的实现
    input_ids_list = []
    response_mask_list = []
    
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p, add_special_tokens=True)
        o_ids = tokenizer.encode(o, add_special_tokens=False)
        full_ids = p_ids + o_ids
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        input_ids_list.append(torch.tensor(full_ids))
        response_mask_list.append(torch.tensor(mask))

    # Padding 到 batch 最大长度
    from torch.nn.utils.rnn import pad_sequence
    batch_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_mask = pad_sequence(response_mask_list, batch_first=True, padding_value=0)

    # 按照作业要求：Slice & Shift
    # input_ids: 去掉最后一个 token
    # labels: 实际上是 input_ids 错位一位
    return {
        "input_ids": batch_input_ids[:, :-1],
        "labels": batch_input_ids[:, 1:],
        "response_mask": batch_mask[:, 1:]
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Problem: Per-token entropy (1 point)
    使用数值稳定的 logsumexp 计算每个位置的熵。
    """
    # H(p) = log(sum(exp(x_i))) - (sum(x_i * exp(x_i)) / sum(exp(x_i)))
    # 简化写法：利用 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Problem: Response log-probs (and entropy) (2 points)
    获取回答部分的对数概率 log p(y|x)。
    """
    logits = model(input_ids).logits # Shape: (B, L, V)
    
    # 使用 log_softmax 获取所有 log_probs
    log_probs_all = F.log_softmax(logits, dim=-1)
    
    # 根据 labels 选出对应的 log_probs
    # 使用 gather 函数从 vocab 维度选取 label 对应的概率
    per_token_log_probs = torch.gather(log_probs_all, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    
    res = {"log_probs": per_token_log_probs}
    if return_token_entropy:
        res["token_entropy"] = compute_entropy(logits)
    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float, # 将其移到 dim 之前
    dim: int | None = None,
) -> torch.Tensor:
    """
    Problem: Masked normalize
    """
    masked_tensor = tensor * mask
    sum_val = torch.sum(masked_tensor, dim=dim)
    return sum_val / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    微批次训练步：计算平均负对数似然损失并反向传播。
    """
    # 获取当前的 Batch Size
    batch_size = policy_log_probs.shape[0]
    
    # 1. 计算负对数似然 (NLL)
    loss_per_token = -policy_log_probs
    
    # 2. 计算被 mask 覆盖的总损失
    # 注意：这里只对 response 部分求和
    total_masked_loss = torch.sum(loss_per_token * response_mask)
    
    # 3. 关键修正：两级归一化
    # 第一级：除以 batch_size 得到单样本平均损失
    # 第二级：除以 gradient_accumulation_steps 用于梯度累积
    # 同时乘以题目给出的 normalize_constant (默认为 1.0)
    loss = total_masked_loss / (batch_size * gradient_accumulation_steps * normalize_constant)
    
    # 4. 执行反向传播
    loss.backward()
    
    return loss, {"loss": loss.detach()}