
"""
PPO (Proximal Policy Optimization) Agent
Continuous action control for CarRacing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from .model import ActorCritic

class RolloutBuffer:
    """
    Buffer for storing trajectories for PPO
    Unlike DQN's Replay Buffer, this is for on-policy learning
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def store(self, state, action, reward, value, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get(self):
        """Get all stored data as numpy arrays"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.log_probs),
            np.array(self.dones)
        )
    
    def __len__(self):
        return len(self.states)

class PPOAgent:
    """
    PPO Agent for continuous control
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor-Critic Network
        self.ac = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
    
    def get_action(self, state, deterministic=False):
        """
        Get action from policy
        
        Args:
            state: observation (numpy array)
            deterministic: if True, return mean action (for evaluation)
        
        Returns:
            action (numpy), value, log_prob (for training)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.ac.get_action(state_tensor, deterministic)
            value = self.ac.get_value(state_tensor)
        
        action_np = action.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0, 0]
        log_prob_np = log_prob.cpu().numpy()[0] if log_prob is not None else None
        
        return action_np, value_np, log_prob_np
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: list of rewards
            values: list of value estimates
            dones: list of done flags
            next_value: value estimate of the next state
        
        Returns:
            advantages, returns
        """
        advantages = []
        gae = 0
        
        # Compute advantages in reverse order
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            
            # GAE: A = delta + gamma * lambda * A
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_state=None):
        """
        Update policy using PPO
        
        Args:
            next_state: the state after the last step in buffer (for bootstrapping)
        
        Returns:
            dict of losses
        """
        if len(self.buffer) == 0:
            return None
        
        # Get data from buffer
        states, actions, rewards, values, old_log_probs, dones = self.buffer.get()
        
        # Compute next value for GAE
        if next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_value = self.ac.get_value(next_state_tensor).cpu().numpy()[0, 0]
        else:
            next_value = 0.0
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            # Mini-batch updates
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                log_probs, entropy, values_pred = self.ac.evaluate(
                    states_tensor[batch_indices],
                    actions_tensor[batch_indices]
                )
                
                # Policy loss (PPO clipped objective)
                # log_probs: (Batch,)
                # old_log_probs: (Batch,)
                # advantages: (Batch,)
                
                # Match dimensions: use flatten() to ensure 1D
                log_probs = log_probs.flatten()
                old_log_probs_batch = old_log_probs_tensor[batch_indices].flatten()
                advantages_batch = advantages_tensor[batch_indices].flatten()
                
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                # values_pred: (Batch, 1) -> flatten -> (Batch,)
                # returns: (Batch,)
                values_pred = values_pred.flatten()
                returns_batch = returns_tensor[batch_indices].flatten()
                value_loss = F.mse_loss(values_pred, returns_batch)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_count += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / update_count,
            'value_loss': total_value_loss / update_count,
            'entropy': total_entropy / update_count
        }
    
    def save(self, filename):
        torch.save(self.ac.state_dict(), filename)
    
    def load(self, filename):
        self.ac.load_state_dict(torch.load(filename, map_location=self.device))


