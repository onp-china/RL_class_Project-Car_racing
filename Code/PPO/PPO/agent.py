import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
try:
    from torch.cuda.amp import autocast, GradScaler
except (ImportError, AssertionError):
    # Dummy implementation for CPU/MPS
    from contextlib import contextmanager
    @contextmanager
    def autocast(enabled=False):
        yield
    class GradScaler:
        def __init__(self, *args, **kwargs): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


class RolloutBuffer:
    """
    高性能Rollout Buffer - 参考Stable Baselines3
    使用预分配的NumPy数组，避免动态append开销
    """
    def __init__(self, buffer_size, num_envs, obs_shape, action_dim):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # 预分配内存（使用float32节省空间）
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
        """
        向量化计算GAE优势函数和returns
        参考SB3实现
        """
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
    
    def get(self, batch_size=None):
        """
        获取所有数据并打平为(buffer_size * num_envs, ...)
        可选返回mini-batch索引
        """
        assert self.full or self.pos > 0, "Buffer is empty"
        
        # 打平所有维度
        indices = np.arange(self.pos * self.num_envs)
        
        # Flatten
        obs_flat = self.observations[:self.pos].reshape(-1, *self.obs_shape)
        actions_flat = self.actions[:self.pos].reshape(-1, self.action_dim)
        log_probs_flat = self.log_probs[:self.pos].reshape(-1)
        advantages_flat = self.advantages[:self.pos].reshape(-1)
        returns_flat = self.returns[:self.pos].reshape(-1)
        
        # 标准化advantages（提升训练稳定性）
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        
        return {
            'observations': obs_flat,
            'actions': actions_flat,
            'old_log_probs': log_probs_flat,
            'advantages': advantages_flat,
            'returns': returns_flat,
            'indices': indices
        }
    
    def reset(self):
        """重置buffer"""
        self.pos = 0
        self.full = False


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络，共享特征提取器
    适用于84x84灰度图像 + 4帧堆叠
    """
    def __init__(self, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # 共享特征提取器（Nature DQN架构）
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


class PPOAgent:
    """
    PPO智能体 - 包含所有优化特性
    - 混合精度训练
    - 梯度裁剪
    - AdamW优化器
    - Clipped surrogate objective
    """
    def __init__(
        self,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=256,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # 网络
        self.network = ActorCriticNetwork(action_dim)
        if str(device) != 'cpu':
            self.network = self.network.to(device)
        
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
            # PyTorch 2.0+ 新API
            try:
                from torch.amp import GradScaler as AmpGradScaler
                self.scaler = AmpGradScaler('cuda')
            except (ImportError, AttributeError):
                # 兼容旧版本PyTorch
                self.scaler = GradScaler()
    
    def process_state(self, state):
        """状态预处理：归一化到[0,1]"""
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32) / 255.0
        return state
    
    def get_action(self, states, deterministic=False):
        """
        获取动作（支持批处理）
        返回：actions, log_probs, values (都是numpy数组)
        """
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
        进行n_epochs次更新，每次使用mini-batch
        """
        # 获取所有数据
        data = rollout_buffer.get()
        
        total_samples = len(data['indices'])
        
        # 统计信息
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        # 多个epoch
        for epoch in range(self.n_epochs):
            # 打乱数据
            indices = np.random.permutation(total_samples)
            
            # Mini-batch训练
            for start_idx in range(0, total_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # 提取batch数据
                obs_batch = torch.from_numpy(data['observations'][batch_indices]).to(self.device)
                actions_batch = torch.from_numpy(data['actions'][batch_indices]).to(self.device)
                old_log_probs_batch = torch.from_numpy(data['old_log_probs'][batch_indices]).to(self.device)
                advantages_batch = torch.from_numpy(data['advantages'][batch_indices]).to(self.device)
                returns_batch = torch.from_numpy(data['returns'][batch_indices]).to(self.device)
                
                # 混合精度训练
                if self.use_amp:
                    # PyTorch 2.0+ 新API
                    try:
                        from torch.amp import autocast as amp_autocast
                        with amp_autocast('cuda'):
                            loss, stats = self._compute_loss(
                                obs_batch, actions_batch, old_log_probs_batch,
                                advantages_batch, returns_batch
                            )
                    except (ImportError, AttributeError):
                        # 兼容旧版本PyTorch
                        with autocast():
                            loss, stats = self._compute_loss(
                                obs_batch, actions_batch, old_log_probs_batch,
                                advantages_batch, returns_batch
                            )
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss, stats = self._compute_loss(
                        obs_batch, actions_batch, old_log_probs_batch,
                        advantages_batch, returns_batch
                    )
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # 记录统计
                policy_losses.append(stats['policy_loss'])
                value_losses.append(stats['value_loss'])
                entropy_losses.append(stats['entropy_loss'])
                clip_fractions.append(stats['clip_fraction'])
        
        # 返回训练统计
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'clip_fraction': np.mean(clip_fractions)
        }
    
    def _compute_loss(self, obs, actions, old_log_probs, advantages, returns):
        """计算PPO loss"""
        # 评估当前策略
        log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
        values = values.flatten()
        
        # Policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss (MSE)
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy loss (鼓励探索)
        entropy_loss = -entropy.mean()
        
        # 总损失
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        
        # 统计clip fraction（衡量更新幅度）
        with torch.no_grad():
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
        
        return loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'clip_fraction': clip_fraction
        }

