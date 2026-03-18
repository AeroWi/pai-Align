import torch
from typing import Literal, Any

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """计算遮罩后的均值，并保持与测试用例一致的 NaN 处理逻辑。"""
    # 1. 确保 tensor 和 mask 形状一致且是浮点运算
    masked_tensor = tensor * mask
    
    # 2. 分情况处理维度
    if dim is not None:
        # 求和
        sum_tensor = masked_tensor.sum(dim=dim)
        # 计个数（mask 为 True 的个数）
        count_tensor = mask.sum(dim=dim)
        
        # 关键修改：不要加 1e-8！
        # 直接相除，这样 count 为 0 的地方会自动产生 NaN，符合测试预期
        return sum_tensor / count_tensor
    else:
        # 全局平均
        count = mask.sum()
        # 如果一个有效的 token 都没有，直接返回 NaN
        if count == 0:
            return torch.tensor(float('nan'), device=tensor.device)
        return masked_tensor.sum() / count

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """计算组内归一化的优势（Advantages）。"""
    raw_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        res = reward_fn(resp, gt)
        raw_rewards.append(res["reward"])
    
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    # 变形为 (num_groups, group_size)
    rewards_grouped = raw_rewards.view(-1, group_size)
    
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std = rewards_grouped.std(dim=1, keepdim=True)
        advantages = (rewards_grouped - mean) / (std + advantage_eps)
    else:
        advantages = rewards_grouped - mean
        
    return advantages.view(-1), raw_rewards, {"mean": mean.mean().item()}

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """计算原始策略梯度损失：-A * log_p"""
    # 核心：将 (batch_size,) 或 (batch_size, 1) 广播至 (batch_size, seq_len)
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 GRPO 剪切损失。"""
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
        
    # 重要性采样比率 r_t
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * advantages
    
    # 注意：这里是取负，因为我们要最大化期望回报，而 PyTorch 是进行梯度下降
    loss = -torch.min(surr1, surr2)
    
    # 记录剪切分值
    is_clipped = (surr1 < surr2).float() if advantages.mean() > 0 else (surr1 > surr2).float()
    return loss, {"clip_fraction": is_clipped.mean()}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """损失分发器。"""
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"Invalid loss_type: {loss_type}")

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """单个微批次更新。"""
    loss_per_token, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )
    
    # 1. 对序列维度取均值 (只计入回答 Token)
    loss_per_example = masked_mean(loss_per_token, response_mask, dim=1)
    
    # 2. 对批次维度取均值并除以梯度累积步数
    loss = loss_per_example.mean() / gradient_accumulation_steps
    
    # 3. 反向传播
    loss.backward()
    
    return loss, metadata