import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.networks import DDPGActor, DDPGCritic

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(
        self,
        action_dim,
        lr_actor=1e-4,
        lr_critic=1e-3,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        tau=0.005,
        device='cuda'
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        # Networks
        self.actor = DDPGActor(action_dim).to(device)
        self.actor_target = DDPGActor(action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = DDPGCritic(action_dim).to(device)
        self.critic_target = DDPGCritic(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        self.buffer = ReplayBuffer(buffer_size)
        
        # Mixed Precision
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            try:
                from torch.amp import GradScaler as AmpGradScaler
                self.scaler = AmpGradScaler('cuda')
            except (ImportError, AttributeError):
                self.scaler = torch.cuda.amp.GradScaler()

    def process_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.astype(np.float32) / 255.0
        return state

    def get_action(self, state, add_noise=True, noise_scale=0.1):
        # Handle batch input
        if isinstance(state, np.ndarray) and state.ndim == 4:
            num_envs = state.shape[0]
            state = self.process_state(state)
            state_tensor = torch.from_numpy(state).to(self.device)
            
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy()
            self.actor.train()
            
            if add_noise:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action += noise
                
            # Clip actions
            # CarRacing-v3 actions: [steer, gas, brake]
            # steer: [-1, 1], gas: [0, 1], brake: [0, 1]
            action[:, 0] = np.clip(action[:, 0], -1, 1)
            action[:, 1] = np.clip(action[:, 1], 0, 1)
            action[:, 2] = np.clip(action[:, 2], 0, 1)
            
            return action
        
        # Single input
        else:
            if isinstance(state, np.ndarray) and state.ndim == 3:
                state = state[None, ...]
                
            state = self.process_state(state)
            state_tensor = torch.from_numpy(state).to(self.device)
            
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).cpu().numpy()[0]
            self.actor.train()
            
            if add_noise:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action += noise
            
            action[0] = np.clip(action[0], -1, 1)
            action[1] = np.clip(action[1], 0, 1)
            action[2] = np.clip(action[2], 0, 1)
            
            return action

    # Alias for test_policy compatibility
    def select_action(self, state, epsilon=0.0): 
        # For DDPG, epsilon=0 usually means no noise
        return self.get_action(state, add_noise=False)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        state = self.process_state(np.array(state))
        next_state = self.process_state(np.array(next_state))
        
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
        reward = torch.tensor(np.array(reward), device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(np.array(done), device=self.device, dtype=torch.float32).unsqueeze(1)

        # Mixed Precision Training
        if self.use_amp:
            try:
                from torch.amp import autocast
                autocast_ctx = autocast('cuda')
            except (ImportError, AttributeError):
                autocast_ctx = torch.cuda.amp.autocast()
            
            # Update Critic
            with autocast_ctx:
                with torch.no_grad():
                    next_action = self.actor_target(next_state)
                    target_Q = self.critic_target(next_state, next_action)
                    target_Q = reward + (1 - done) * self.gamma * target_Q
                
                current_Q = self.critic(state, action)
                critic_loss = nn.MSELoss()(current_Q, target_Q)
            
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.step(self.critic_optimizer)
            
            # Update Actor
            with autocast_ctx:
                actor_loss = -self.critic(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
            
        else:
            # Update Critic
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                target_Q = self.critic_target(next_state, next_action)
                target_Q = reward + (1 - done) * self.gamma * target_Q
            
            current_Q = self.critic(state, action)
            critic_loss = nn.MSELoss()(current_Q, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Update Actor
            actor_loss = -self.critic(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # Soft Update Targets
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}

