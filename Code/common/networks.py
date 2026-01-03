"""
神经网络架构模块 - 所有算法共享
包含CNN编码器、Actor-Critic网络、Q网络等
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ConvEncoder(nn.Module):
    """
    CNN特征提取器（Nature DQN架构）
    适用于84x84灰度图像 + 4帧堆叠
    所有算法共用
    """
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # Input: (batch, 4, 84, 84)
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),                                # -> 3136
        )
    
    def forward(self, x):
        return self.features(x)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络，共享特征提取器
    适用于连续动作空间（PPO、A2C、REINFORCE）
    """
    def __init__(self, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享特征提取器
        self.features = ConvEncoder()
        
        # 共享隐藏层
        self.shared_fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        
        # Actor head (连续动作)
        self.actor_mean = nn.Linear(512, action_dim)
        # 学习标准差（对数形式，保证正数）
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head
        self.critic = nn.Linear(512, 1)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization - SB3风格"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Actor输出层使用小初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)
        
        # SOTA Improvement: Gas bias = 0.5
        # 鼓励从一开始就踩油门，避免"不动"的问题
        with torch.no_grad():
            self.actor_mean.bias[1] = 0.5
        
        # Critic输出层
        nn.init.orthogonal_(self.critic.weight, gain=1)
        nn.init.constant_(self.critic.bias, 0)
    
    def forward(self, x):
        """前向传播，返回action分布和value"""
        features = self.features(x)
        shared = self.shared_fc(features)
        
        # Actor
        action_mean = torch.tanh(self.actor_mean(shared))  # 限制在[-1, 1]
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        # Critic
        value = self.critic(shared)
        
        return action_mean, action_std, value
    
    def get_action(self, x, deterministic=False):
        """采样动作"""
        action_mean, action_std, value = self.forward(x)
        
        if deterministic:
            return action_mean, None, value
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # 对所有动作维度求和
        
        return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        """评估给定动作的log_prob和value"""
        action_mean, action_std, value = self.forward(x)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class QNetwork(nn.Module):
    """
    Q网络架构
    适用于离散动作空间（Double-DQN、N-step SARSA）
    """
    def __init__(self, action_dim):
        super(QNetwork, self).__init__()
        
        # 特征提取器
        self.features = ConvEncoder()
        
        # Q值头
        self.q_head = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播，返回Q值"""
        features = self.features(x)
        q_values = self.q_head(features)
        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN架构（可选）
    将Q值分解为状态价值V和优势函数A
    """
    def __init__(self, action_dim):
        super(DuelingQNetwork, self).__init__()
        
        # 特征提取器
        self.features = ConvEncoder()
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播，返回Q值"""
        features = self.features(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DDPGActor(nn.Module):
    """
    DDPG Actor网络 - 输出确定性动作
    适用于连续动作空间（DDPG）
    """
    def __init__(self, action_dim):
        super(DDPGActor, self).__init__()
        
        # 特征提取器（复用ConvEncoder）
        self.features = ConvEncoder()
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU()
        )
        
        # 动作输出层
        self.action_out = nn.Linear(512, action_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Actor输出层使用小初始化（更稳定）
        nn.init.orthogonal_(self.action_out.weight, gain=0.01)
        nn.init.constant_(self.action_out.bias, 0)
    
    def forward(self, state):
        """
        前向传播，返回确定性动作
        
        针对CarRacing-v3的3维动作：
        - action[0]: steering [-1, 1] → tanh
        - action[1]: gas [0, 1] → sigmoid
        - action[2]: brake [0, 1] → sigmoid
        """
        features = self.features(state)
        x = self.fc(features)
        action = self.action_out(x)
        
        # 分别对不同动作维度应用激活函数
        steering = torch.tanh(action[:, 0:1])
        gas = torch.sigmoid(action[:, 1:2])
        brake = torch.sigmoid(action[:, 2:3])
        
        return torch.cat([steering, gas, brake], dim=1)


class DDPGCritic(nn.Module):
    """
    DDPG Critic网络（Q网络）
    输入：state + action，输出：Q值
    """
    def __init__(self, action_dim):
        super(DDPGCritic, self).__init__()
        
        # 状态特征提取器
        self.features = ConvEncoder()
        
        # 状态特征 + 动作 → Q值
        self.fc = nn.Sequential(
            nn.Linear(3136 + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        """
        前向传播，返回Q值
        
        Args:
            state: (batch, 4, 84, 84)
            action: (batch, action_dim)
        
        Returns:
            q_value: (batch, 1)
        """
        state_features = self.features(state)
        # 拼接状态特征和动作
        x = torch.cat([state_features, action], dim=1)
        q_value = self.fc(x)
        return q_value

