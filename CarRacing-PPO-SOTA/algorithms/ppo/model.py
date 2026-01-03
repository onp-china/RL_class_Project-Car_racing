
"""
PPO Networks for Continuous Control - Optimized
优化版 PPO 网络：正交初始化、共享骨干网、无界均值输出
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal Initialization for PPO
    正交初始化：对 RL 训练稳定性至关重要
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNBase(nn.Module):
    """
    Shared CNN Backbone for feature extraction
    Input: (N, 4, 96, 96) or (N, 96, 96, 4)
    Output: Flat Feature Vector
    """
    def __init__(self, input_channels=4):
        super().__init__()
        # 3-layer NatureCNN structure
        self.conv1 = layer_init(nn.Conv2d(input_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.flatten_size = 64 * 8 * 8  # 4096

    def forward(self, x):
        # Handle input transpose if necessary
        # PyTorch expects (N, C, H, W)
        if x.dim() == 4 and x.shape[-1] == 4 and x.shape[1] != 4:
            x = x.permute(0, 3, 1, 2)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        return x

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network with Shared Backbone
    共享 CNN 特征提取器
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Shared CNN Backbone
        self.cnn = CNNBase(input_channels=state_dim[0])
        
        # Critic Head (Value)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.cnn.flatten_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0)
        )
        
        # Actor Head (Policy)
        self.actor_fc = nn.Sequential(
            layer_init(nn.Linear(self.cnn.flatten_size, 512)),
            nn.ReLU(),
        )
        
        # Action Mean (std=0.01 for initial exploration)
        # 输出无界均值，不加 Tanh/Sigmoid，避免边界梯度消失
        self.actor_mean = layer_init(nn.Linear(512, action_dim), std=0.01)
        
        # CRITICAL FIX: Initialize gas/brake bias to positive values
        # This encourages the agent to press gas from the start
        with torch.no_grad():
            self.actor_mean.bias[1] = 0.5  # Gas channel: bias = 0.5
            self.actor_mean.bias[2] = 0.0  # Brake channel: keep at 0
        
        # Log standard deviation (learnable parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def get_value(self, state):
        features = self.cnn(state)
        return self.critic(features)
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy
        """
        features = self.cnn(state)
        
        # Calculate distribution parameters
        x = self.actor_fc(features)
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if deterministic:
            action = action_mean
        else:
            action = probs.sample()
            
        # Log prob of the RAW action (before clipping)
        # This is mathematically correct for PPO update
        log_prob = probs.log_prob(action).sum(1)
        
        return action, log_prob

    def evaluate(self, state, action):
        """
        Evaluate state-action pairs for PPO update
        """
        features = self.cnn(state)
        
        # Value
        value = self.critic(features)
        
        # Policy distribution
        x = self.actor_fc(features)
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        probs = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        return log_prob, entropy, value
