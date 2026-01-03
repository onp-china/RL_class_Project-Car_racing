
"""
Fast PPO Training with Vectorized Environments
最快速的 PPO 训练 - 使用并行环境
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from .env_wrapper import make_vec_continuous_env, make_continuous_env
from .agent import PPOAgent

def train(args):
    print("=" * 70)
    print("🚀 FAST PPO Training - Vectorized Environments")
    print("=" * 70)
    print(f"Algorithm: PPO (Continuous Control)")
    print(f"Parallel Environments: {args.num_envs}")
    print(f"Frame Skip: {args.frame_skip}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Rollout Steps per Env: {args.rollout_steps}")
    print(f"Total Samples per Update: {args.num_envs * args.rollout_steps}")
    print(f"Expected Speedup: ~{args.num_envs}x faster")
    print("=" * 70 + "\n")
    
    # Create vectorized environments
    vec_env = make_vec_continuous_env(
        num_envs=args.num_envs,
        frame_stack=4,
        frame_skip=args.frame_skip
    )
    
    # Create single env for evaluation
    eval_env = make_continuous_env(
        render_mode=None,
        frame_stack=4,
        frame_skip=args.frame_skip
    )
    
    # Create agent
    state_dim = (4, 96, 96)
    action_dim = 3  # [steer, gas, brake]
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size
    )
    
    # Training loop
    os.makedirs("saved_models", exist_ok=True)
    
    episode_rewards = []
    total_steps = 0
    episode_count = 0
    
    # Reset all environments
    states, infos = vec_env.reset()
    current_episode_rewards = np.zeros(args.num_envs)
    
    # Create progress bar
    pbar = tqdm(total=args.max_episodes, desc="🚀 Fast PPO Training", unit="ep")
    
    while episode_count < args.max_episodes:
        # Collect rollout from all environments
        for step in range(args.rollout_steps):
            # Get actions for all environments
            actions = []
            values = []
            log_probs = []
            
            for i in range(args.num_envs):
                action, value, log_prob = agent.get_action(states[i], deterministic=False)
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)
            
            actions = np.array(actions)
            
            # Execute actions in all environments
            next_states, rewards, terminateds, truncateds, infos = vec_env.step(actions)
            
            current_episode_rewards += rewards
            total_steps += args.num_envs
            
            # Store transitions for each environment
            for i in range(args.num_envs):
                agent.buffer.store(
                    states[i],
                    actions[i],
                    rewards[i],
                    values[i],
                    log_probs[i],
                    terminateds[i] or truncateds[i]
                )
                
                # Episode finished
                if terminateds[i] or truncateds[i]:
                    episode_count += 1
                    episode_rewards.append(current_episode_rewards[i])
                    
                    avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'Reward': f'{current_episode_rewards[i]:.1f}',
                        'Avg100': f'{avg_reward:.1f}',
                        'TotalSteps': total_steps
                    })
                    
                    # Print detailed info occasionally
                    if episode_count % 10 == 0:
                        tqdm.write(f"Episode {episode_count:4d}: Reward={current_episode_rewards[i]:6.1f}, Avg100={avg_reward:6.1f}")
                    
                    current_episode_rewards[i] = 0
                    
                    # Save model
                    if episode_count % args.save_freq == 0:
                        filename = f"saved_models/ppo_fast_carracing_ep{episode_count}.pth"
                        agent.save(filename)
                        tqdm.write(f"✓ Saved: {filename}")
                    
                    # Evaluation
                    if args.eval_freq > 0 and episode_count % args.eval_freq == 0:
                        eval_reward = evaluate_agent(agent, eval_env)
                        tqdm.write(f"📊 Eval (greedy): {eval_reward:.2f}")
            
            states = next_states
            
            if episode_count >= args.max_episodes:
                break
        
        # PPO Update after collecting enough samples
        if len(agent.buffer) > 0:
            # Bootstrap with next states (if not done)
            # For simplicity, we use None here (can be improved)
            losses = agent.update(next_state=None)
            
            if losses and args.verbose and episode_count > 0:
                tqdm.write(f"  Update: PolicyLoss={losses['policy_loss']:.4f}, "
                          f"ValueLoss={losses['value_loss']:.4f}, Entropy={losses['entropy']:.4f}")
    
    pbar.close()
    vec_env.close()
    eval_env.close()
    
    final_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print("\n" + "=" * 70)
    print(f"✅ Training Complete! Final Avg Reward (last 100): {final_avg:.2f}")
    print("=" * 70)

def evaluate_agent(agent, env, num_episodes=3):
    """Evaluate agent with deterministic policy"""
    total_reward = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = truncated = False
        
        while not (done or truncated):
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
    
    return total_reward / num_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🚀 Fast PPO Training (Vectorized)")
    
    # Vectorization
    parser.add_argument("--num_envs", type=int, default=4, help="Parallel environments (4-8 recommended)")
    parser.add_argument("--frame_skip", type=int, default=2, help="Action repeat for speed")
    
    # Training
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--rollout_steps", type=int, default=512, help="Steps per env per rollout (reduced for vectorized)")
    
    # PPO Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Monitoring
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    train(args)


