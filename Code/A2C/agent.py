import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.cuda.amp import autocast, GradScaler


class RolloutBuffer:
    """
    高性能Rollout Buffer - 参考PPO实现
    用于A2C的n-step轨迹存储
    """
    def __init__(self, buffer_size, num_envs, obs_shape, action_dim):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # 预分配内存
        self.observations = np.zeros((buffer_size, num_envs, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        # GAE相关
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        self.pos = 0
        self.full = False
    
    def add(self, obs, action, reward, done, log_prob, value):
        """添加一步数据"""
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(self, last_values, gamma=0.99, gae_lambda=0.95):
        """向量化计算GAE优势函数和returns"""
        advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        last_gae_lam = 0
        
        # 从后向前计算GAE
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        self.advantages = advantages
        self.returns = advantages + self.values
    
    def get(self):
        """获取所有数据并打平"""
        assert self.full or self.pos > 0, "Buffer is empty"
        
        # Flatten
        obs_flat = self.observations[:self.pos].reshape(-1, *self.obs_shape)
        actions_flat = self.actions[:self.pos].reshape(-1, self.action_dim)
        log_probs_flat = self.log_probs[:self.pos].reshape(-1)
        advantages_flat = self.advantages[:self.pos].reshape(-1)
        returns_flat = self.returns[:self.pos].reshape(-1)
        
        # 标准化advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        return {
            'observations': obs_flat,
            'actions': actions_flat,
            'old_log_probs': log_probs_flat,
            'advantages': advantages_flat,
            'returns': returns_flat
        }
    
    def reset(self):
        """重置buffer"""
        self.pos = 0
        self.full = False


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络，共享特征提取器（与PPO完全相同）
    """
    def __init__(self, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享特征提取器（Nature DQN架构）
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
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
        with torch.no_grad():
            self.actor_mean.bias[1] = 0.5
        
        # Critic输出层
        nn.init.orthogonal_(self.critic.weight, gain=1)
        nn.init.constant_(self.critic.bias, 0)
    
    def forward(self, x):
        """前向传播，返回action分布和value"""
        features = self.features(x)
        shared = self.shared_fc(features)
        
        # Actor - 分别处理不同动作维度
        action_raw = self.actor_mean(shared)
        # steering: [-1, 1] → tanh
        steering = torch.tanh(action_raw[:, 0:1])
        # gas: [0, 1] → sigmoid
        gas = torch.sigmoid(action_raw[:, 1:2])
        # brake: [0, 1] → sigmoid
        brake = torch.sigmoid(action_raw[:, 2:3])
        action_mean = torch.cat([steering, gas, brake], dim=1)
        
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        # Critic
        value = self.critic(shared)
        
        return action_mean, action_std, value
    
    def get_action(self, x, deterministic=False):
        """采样动作"""
        action_mean, action_std, value = self.forward(x)
        
        if deterministic:
            # 确定性策略：直接返回action_mean（已经通过正确的激活函数）
            action = action_mean
            # 额外clip确保在范围内
            action[:, 0:1] = torch.clamp(action[:, 0:1], -1.0, 1.0)  # steering
            action[:, 1:2] = torch.clamp(action[:, 1:2], 0.0, 1.0)   # gas
            action[:, 2:3] = torch.clamp(action[:, 2:3], 0.0, 1.0)   # brake
            return action, None, value
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        # Clip到正确的范围（CarRacing-v3的动作空间）
        action[:, 0:1] = torch.clamp(action[:, 0:1], -1.0, 1.0)  # steering: [-1, 1]
        action[:, 1:2] = torch.clamp(action[:, 1:2], 0.0, 1.0)   # gas: [0, 1]
        action[:, 2:3] = torch.clamp(action[:, 2:3], 0.0, 1.0)   # brake: [0, 1]
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        """评估给定动作的log_prob和value"""
        action_mean, action_std, value = self.forward(x)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class A2CAgent:
    """
    A2C智能体 - 基于PPO架构，但只更新一次
    - 混合精度训练
    - 梯度裁剪
    - AdamW优化器
    - 无Clipping（与PPO的主要区别）
    """
    def __init__(
        self,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        
        # 网络
        self.network = ActorCriticNetwork(action_dim).to(device)
        
        # 优化器 - AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-5
        )
        
        # 混合精度训练
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            try:
                from torch.amp import GradScaler as AmpGradScaler
                self.scaler = AmpGradScaler('cuda')
            except (ImportError, AttributeError):
                self.scaler = GradScaler()
    
    def process_state(self, state):
        """状态预处理：归一化到[0,1]"""
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32) / 255.0
        return state
    
    def get_action(self, states, deterministic=False):
        """获取动作（支持批处理）"""
        states = self.process_state(states)
        states_tensor = torch.from_numpy(states).to(self.device)
        
        with torch.no_grad():
            actions, log_probs, values = self.network.get_action(states_tensor, deterministic)
        
        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy() if log_probs is not None else None
        values_np = values.cpu().numpy().flatten()
        
        return actions_np, log_probs_np, values_np
    
    def get_value(self, states):
        """获取状态价值"""
        states = self.process_state(states)
        states_tensor = torch.from_numpy(states).to(self.device)
        
        with torch.no_grad():
            _, _, values = self.network.forward(states_tensor)
        
        return values.cpu().numpy().flatten()
    
    def update(self, rollout_buffer):
        """
        使用rollout buffer中的数据更新网络
        A2C只更新一次（与PPO的主要区别）
        """
        # 获取所有数据
        data = rollout_buffer.get()
        
        # 提取数据
        obs_batch = torch.from_numpy(data['observations']).to(self.device)
        actions_batch = torch.from_numpy(data['actions']).to(self.device)
        advantages_batch = torch.from_numpy(data['advantages']).to(self.device)
        returns_batch = torch.from_numpy(data['returns']).to(self.device)
        
        # 混合精度训练
        if self.use_amp:
            try:
                from torch.amp import autocast as amp_autocast
                with amp_autocast('cuda'):
                    loss, stats = self._compute_loss(
                        obs_batch, actions_batch, advantages_batch, returns_batch
                    )
            except (ImportError, AttributeError):
                with autocast():
                    loss, stats = self._compute_loss(
                        obs_batch, actions_batch, advantages_batch, returns_batch
                    )
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, stats = self._compute_loss(
                obs_batch, actions_batch, advantages_batch, returns_batch
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return stats
    
    def _compute_loss(self, obs, actions, advantages, returns):
        """计算A2C loss - 无clipping"""
        # 评估当前策略
        log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
        values = values.flatten()
        
        # Policy loss (无clipping，与PPO的主要区别)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss (MSE)
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy loss (鼓励探索)
        entropy_loss = -entropy.mean()
        
        # 总损失
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        return loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
