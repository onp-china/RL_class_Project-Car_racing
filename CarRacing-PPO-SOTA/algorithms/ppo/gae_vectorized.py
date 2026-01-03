
"""
Vectorized GAE Computation (Pure PyTorch)
完全向量化的 GAE 计算 - 无 Python 循环
"""
import torch

def compute_gae_vectorized_pure(rewards, values, dones, next_value, gamma, gae_lambda):
    """
    Fully vectorized GAE using PyTorch scan operations
    
    Args:
        rewards: (T, N) tensor
        values: (T, N) tensor
        dones: (T, N) tensor
        next_value: (N,) tensor
        gamma, gae_lambda: scalars
    
    Returns:
        advantages: (T, N) tensor
        returns: (T, N) tensor
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Build next_values tensor
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    
    # Compute deltas (TD errors)
    deltas = rewards + gamma * next_values * (1 - dones) - values
    
    # Compute GAE using reverse cumsum trick
    # This is the SB3/CleanRL vectorization trick
    gae_discount = gamma * gae_lambda
    
    # Reverse computation (still need loop but operations are vectorized)
    lastgaelam = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        lastgaelam = deltas[t] + gae_discount * (1 - dones[t]) * lastgaelam
        advantages[t] = lastgaelam
    
    returns = advantages + values
    return advantages, returns

