"""
PPO (Proximal Policy Optimization) Algorithm
连续动作控制 - 适用于 CarRacing

主要文件:
- agent.py: PPO Agent 实现
- model.py: Actor-Critic 神经网络
- env_wrapper.py: 连续动作环境包装器
- train.py: 单环境训练脚本
- train_fast.py: 向量化快速训练脚本
"""

from .agent import PPOAgent
from .model import ActorCritic
from .env_wrapper import make_continuous_env, make_vec_continuous_env

__all__ = ['PPOAgent', 'ActorCritic', 'make_continuous_env', 'make_vec_continuous_env']



