
"""
Improved PPO Training with Grass Penalty
改进版 PPO 训练 - 添加草地惩罚机制
"""
import os
import argparse
import time
import numpy as np
import torch
import sys
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils_env_sota import make_vec_sota_env
from algorithms.ppo.agent import PPOAgent
from gae_vectorized import compute_gae_vectorized_pure

def train(args):
    print("=" * 70)
    print("SOTA PPO Training - Track Geometry Based")
    print("Strategy: Progress Projection + Centerline + Stability")
    print(f"Envs: {args.num_envs} (AsyncVectorEnv), Frame Skip: {args.frame_skip}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Environment with SOTA Reward Shaping
    envs = make_vec_sota_env(
        num_envs=args.num_envs,
        frame_stack=4,
        frame_skip=args.frame_skip
    )
    
    # Agent
    agent = PPOAgent(
        state_dim=(4, 96, 96),
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.update_epochs,
        batch_size=args.batch_size
    )
    
    # Pre-allocate GPU buffers
    obs_buf = torch.zeros((args.num_steps, args.num_envs, 4, 96, 96), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs, 3), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    
    # Initialize
    os.makedirs("saved_models", exist_ok=True)
    obs = envs.reset()[0]
    obs = np.transpose(obs, (0, 3, 1, 2)).astype(np.float32)
    
    global_step = 0
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    episode_rewards = []
    current_ep_rewards = np.zeros(args.num_envs)
    
    pbar = tqdm(range(1, num_updates + 1), desc="Training")
    
    for update in pbar:
        # Rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            
            obs_gpu = torch.from_numpy(obs).to(device)
            obs_buf[step] = obs_gpu
            
            with torch.no_grad():
                action, log_prob = agent.ac.get_action(obs_gpu, deterministic=False)
                value = agent.ac.get_value(obs_gpu)
            
            actions_buf[step] = action
            logprobs_buf[step] = log_prob
            values_buf[step] = value.flatten()
            
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = envs.step(action_np)
            
            obs = np.transpose(next_obs, (0, 3, 1, 2)).astype(np.float32)
            
            rewards_buf[step] = torch.from_numpy(reward).to(device)
            dones = np.logical_or(terminated, truncated).astype(np.float32)
            dones_buf[step] = torch.from_numpy(dones).to(device)
            
            current_ep_rewards += reward
            for i in range(args.num_envs):
                if dones[i]:
                    episode_rewards.append(current_ep_rewards[i])
                    current_ep_rewards[i] = 0
            
            if len(episode_rewards) > 0 and len(episode_rewards) % 20 == 0:
                avg = np.mean(episode_rewards[-100:])
                pbar.set_postfix({"Avg": f"{avg:.1f}", "Eps": len(episode_rewards)})
        
        # GAE
        with torch.no_grad():
            next_obs_gpu = torch.from_numpy(obs).to(device)
            next_value = agent.ac.get_value(next_obs_gpu).flatten()
        
        advantages, returns = compute_gae_vectorized_pure(
            rewards_buf, values_buf, dones_buf, next_value,
            args.gamma, args.gae_lambda
        )
        
        # Flatten
        b_obs = obs_buf.reshape(-1, 4, 96, 96)
        b_actions = actions_buf.reshape(-1, 3)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Update
        total_batch = args.num_envs * args.num_steps
        inds = torch.randperm(total_batch, device=device)
        
        for epoch in range(args.update_epochs):
            for start in range(0, total_batch, args.batch_size):
                end = min(start + args.batch_size, total_batch)
                mb_inds = inds[start:end]
                
                newlogprob, entropy, newvalue = agent.ac.evaluate(
                    b_obs[mb_inds],
                    b_actions[mb_inds]
                )
                
                newlogprob = newlogprob.flatten()
                entropy = entropy.flatten()
                newvalue = newvalue.flatten()
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_advantages = b_advantages[mb_inds]
                
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                
                loss = pg_loss - args.entropy_coef * entropy_loss + v_loss * 0.5
                
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.ac.parameters(), 0.5)
                agent.optimizer.step()
        
        # Save
        if update % 20 == 0 and len(episode_rewards) > 0:
            agent.save(f"saved_models/ppo_sota_ep{len(episode_rewards)}.pth")

    envs.close()
    
    avg_final = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\nTraining Complete! Episodes: {len(episode_rewards)}, Final Avg: {avg_final:.1f}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Improved PPO with Grass Penalty")
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=512)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    
    train(args)

