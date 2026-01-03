import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.networks import QNetwork

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DoubleDQNAgent:
    def __init__(
        self,
        action_dim,
        lr=3e-4,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        update_interval=1000,
        device='cuda'
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_interval = update_interval
        self.device = device
        
        # Networks - 使用common的QNetwork
        self.q_net = QNetwork(action_dim).to(device)
        self.target_q_net = QNetwork(action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.q_net.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-5
        )
        
        self.buffer = ReplayBuffer(buffer_size)
        self.update_count = 0
        
        # Mixed Precision
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            try:
                from torch.amp import GradScaler as AmpGradScaler
                self.scaler = AmpGradScaler('cuda')
            except (ImportError, AttributeError):
                self.scaler = torch.cuda.amp.GradScaler()

    def process_state(self, state):
        # Handle Batch Input: (N, 4, 84, 84)
        # Handle Single Input: (4, 84, 84)
        
        # 优化: 避免在 CPU 上进行浮点转换，直接转为 tensor 后在 GPU 上处理
        if isinstance(state, np.ndarray):
            # non_blocking=True 允许异步传输
            state = torch.from_numpy(state).to(self.device, non_blocking=True)
            
        return state.float() / 255.0

    def select_action(self, state, epsilon=0.0):
        """兼容接口：供test_policy调用"""
        return self.get_action(state, epsilon)

    def get_action(self, state, epsilon):
        # Handle batch
        if isinstance(state, np.ndarray) and state.ndim == 4:
            num_envs = state.shape[0]
            
            # 优化: 使用GPU优化的process_state
            state_tensor = self.process_state(state)
            
            # 使用混合精度推理
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16) if self.device.type == 'cuda' else torch.amp.autocast("cpu"):
                    q_values = self.q_net(state_tensor)
                    greedy_actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int32)
            
            random_mask = np.random.random(num_envs) < epsilon
            random_actions = np.random.randint(0, self.action_dim, size=num_envs, dtype=np.int32)
            
            # Use explicit indexing to avoid type promotion issues with np.where
            final_actions = greedy_actions.copy().astype(np.int32)
            final_actions[random_mask] = random_actions[random_mask]
            return final_actions
        else:
            if random.random() < epsilon:
                return random.randint(0, self.action_dim - 1)
            
            if isinstance(state, np.ndarray) and state.ndim == 3:
                state = state[None, ...]  # Add batch dim
            
            state_tensor = self.process_state(state)
            
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16) if self.device.type == 'cuda' else torch.amp.autocast("cpu"):
                    q_values = self.q_net(state_tensor)
                    return q_values.argmax().item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        # 优化: 批量处理，减少循环
        states_list = [self.process_state(s) for s in states]
        next_states_list = [self.process_state(ns) for ns in next_states]
        
        # Stack to batch tensor
        states_tensor = torch.stack(states_list)
        next_states_tensor = torch.stack(next_states_list)
        
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
        dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(1)

        # 混合精度前向传播
        with torch.amp.autocast("cuda", dtype=torch.float16) if self.device.type == 'cuda' else torch.amp.autocast("cpu"):
            current_q = self.q_net(states_tensor).gather(1, actions_tensor)

            # Double DQN Target:
            # a_max = argmax(Q_net(s'))
            # Target = r + gamma * Q_target_net(s', a_max)
            with torch.no_grad():
                next_actions = self.q_net(next_states_tensor).argmax(1).unsqueeze(1)
                max_next_q = self.target_q_net(next_states_tensor).gather(1, next_actions)
                target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

            loss = nn.MSELoss()(current_q, target_q.float())

        self.optimizer.zero_grad()
        
        # Scaler scaling
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.update_count += 1
        if self.update_count % self.update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        return {'q_loss': loss.item()}
