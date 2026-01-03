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


class EpisodeBuffer:
    """
    存储完整episode轨迹的缓冲区（针对REINFORCE）
    支持并行环境，每个环境独立追踪episode
    """
    def __init__(self, num_envs, obs_shape, action_dim):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # 每个环境一个缓冲区
        self.observations = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]
        self.rewards = [[] for _ in range(num_envs)]
    
    def add(self, obs, actions, rewards, dones):
        """
        添加一步数据到各个环境的buffer中
        
        Args:
            obs: (num_envs, *obs_shape)
            actions: (num_envs, action_dim)
            rewards: (num_envs,)
            dones: (num_envs,)
        """
        for i in range(self.num_envs):
            self.observations[i].append(obs[i])
            self.actions[i].append(actions[i])
            self.rewards[i].append(rewards[i])
    
    def get_completed_episodes(self, dones):
        """
        获取已完成的episodes，并清空对应buffer
        
        Args:
            dones: (num_envs,) boolean array
        
        Returns:
            List of (observations, actions, rewards) tuples
        """
        completed = []
        for i in range(self.num_envs):
            if dones[i] and len(self.observations[i]) > 0:
                completed.append((
                    np.array(self.observations[i], dtype=np.float32),
                    np.array(self.actions[i], dtype=np.float32),
                    np.array(self.rewards[i], dtype=np.float32)
                ))
                # 清空该环境的buffer
                self.observations[i] = []
                self.actions[i] = []
                self.rewards[i] = []
        
        return completed


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
        
        # Critic head (REINFORCE也可以用baseline减少方差)
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
        action_mean = torch.tanh(self.actor_mean(shared))
        action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
        
        # Critic
        value = self.critic(shared)
        
        return action_mean, action_std, value
    
    def get_action(self, x, deterministic=False):
        """采样动作"""
        action_mean, action_std, value = self.forward(x)
        
        if deterministic:
            # 确定性策略：直接返回action_mean（已经是tanh输出，在[-1,1]范围内）
            # 确保输出是正确格式
            action = action_mean
            # 额外clip确保在范围内（虽然tanh已经限制）
            action = torch.clamp(action, -1.0, 1.0)
            return action, None, value
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        # Clip到[-1, 1]范围（CarRacing-v3的动作空间）
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, x, actions):
        """评估给定动作的log_prob和value"""
        action_mean, action_std, value = self.forward(x)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value, entropy


class REINFORCEAgent:
    """
    REINFORCE智能体 - 基于PPO架构，但使用蒙特卡洛回报
    - 只在episode结束时更新
    - 使用完整episode的回报
    - 可选value baseline减少方差
    """
    def __init__(
        self,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        use_baseline=True,
        max_grad_norm=0.5,
        device='cuda'
    ):
        self.device = device
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.use_baseline = use_baseline
        self.max_grad_norm = max_grad_norm
        
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
        # 确保动作在[-1, 1]范围内（虽然tanh已经限制，但为了安全）
        actions_np = np.clip(actions_np, -1.0, 1.0)
        log_probs_np = log_probs.cpu().numpy() if log_probs is not None else None
        values_np = values.cpu().numpy().flatten()
        
        return actions_np, log_probs_np, values_np
    
    def update_trajectory(self, observations, actions, rewards):
        """
        使用完整轨迹更新网络（REINFORCE的核心）
        
        Args:
            observations: (T, 4, 84, 84) 已归一化的观察序列
            actions: (T, action_dim) 动作序列
            rewards: (T,) 奖励序列
        """
        # 计算蒙特卡洛回报（G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}）
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = np.array(returns, dtype=np.float32)
        
        # 改进的归一化：只归一化advantages，不归一化returns本身
        # 这样可以保持returns的原始尺度，同时减少方差
        # 如果使用baseline，advantages会在loss计算时归一化
        
        # 转为tensor
        obs_tensor = torch.from_numpy(self.process_state(observations)).to(self.device)
        actions_tensor = torch.from_numpy(actions).to(self.device)
        returns_tensor = torch.from_numpy(returns).to(self.device).float()
        
        # 混合精度训练
        if self.use_amp:
            try:
                from torch.amp import autocast as amp_autocast
                with amp_autocast('cuda'):
                    loss, stats = self._compute_loss(
                        obs_tensor, actions_tensor, returns_tensor
                    )
            except (ImportError, AttributeError):
                with autocast():
                    loss, stats = self._compute_loss(
                        obs_tensor, actions_tensor, returns_tensor
                    )
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, stats = self._compute_loss(
                obs_tensor, actions_tensor, returns_tensor
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return stats
    
    def _compute_loss(self, obs, actions, returns):
        """计算REINFORCE loss"""
        # 重新评估actions以获得带梯度的log_probs
        log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
        values = values.flatten()
        
        # REINFORCE policy loss
        if self.use_baseline:
            # 使用value function作为baseline减少方差
            advantages = returns - values.detach()
            # 归一化advantages以减少方差（关键改进）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = -(log_probs * advantages).mean()
            
            # Value loss（使用MSE，但可以添加Huber loss提高稳定性）
            value_loss = nn.MSELoss()(values, returns)
        else:
            # 纯REINFORCE（无baseline）
            # 归一化returns以减少方差
            returns_normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
            policy_loss = -(log_probs * returns_normalized).mean()
            value_loss = torch.tensor(0.0)
        
        # Entropy loss (鼓励探索)
        entropy_loss = -entropy.mean()
        
        # 总损失
        if self.use_baseline:
            loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
        else:
            loss = policy_loss + self.ent_coef * entropy_loss
        
        return loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0,
            'entropy_loss': entropy_loss.item(),
            'mean_return': returns.mean().item()
        }
