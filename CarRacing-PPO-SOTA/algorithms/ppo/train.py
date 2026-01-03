
"""
PPO Training Script for CarRacing-v3
Continuous action control with frame skip for acceleration
"""
import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from .env_wrapper import make_continuous_env
from .agent import PPOAgent

def train(args):
    print("=" * 70)
    print("🚀 PPO Training - Continuous Control")
    print("=" * 70)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Frame Stack: 4")
    print(f"Frame Skip: {args.frame_skip} (for acceleration)")
    print(f"Rollout Steps: {args.rollout_steps}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 70 + "\n")
    
    # Create environment
    env = make_continuous_env(
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
    
    # Create progress bar
    pbar = tqdm(total=args.max_episodes, desc="Training PPO", unit="episode")
    
    for episode in range(1, args.max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        step_count = 0
        
        while not (done or truncated):
            # Collect rollout
            for _ in range(args.rollout_steps):
                # Get action from policy
                action, value, log_prob = agent.get_action(state, deterministic=False)
                
                # Execute action
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition
                agent.buffer.store(state, action, reward, value, log_prob, done or truncated)
                
                state = next_state
                episode_reward += reward
                total_steps += 1
                step_count += 1
                
                if done or truncated:
                    break
            
            # Update policy
            if len(agent.buffer) > 0:
                # Use next_state for bootstrapping if episode is not done
                next_state_for_update = state if not (done or truncated) else None
                losses = agent.update(next_state=next_state_for_update)
                
                if losses and args.verbose:
                    print(f"  Step {step_count}: PolicyLoss={losses['policy_loss']:.4f}, "
                          f"ValueLoss={losses['value_loss']:.4f}, Entropy={losses['entropy']:.4f}")
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Reward': f'{episode_reward:.1f}',
            'Avg100': f'{avg_reward:.1f}',
            'Steps': step_count,
            'TotalSteps': total_steps
        })
        
        # Print detailed info occasionally
        if episode % 10 == 0:
            tqdm.write(f"Episode {episode:4d}: Reward={episode_reward:7.2f}, Avg100={avg_reward:7.2f}, Steps={step_count:4d}")
        
        # Save model periodically
        if episode % args.save_freq == 0:
            filename = f"saved_models/ppo_carracing_ep{episode}.pth"
            agent.save(filename)
            tqdm.write(f"✓ Model saved to {filename}")
        
        # Evaluation
        if args.eval_freq > 0 and episode % args.eval_freq == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=3)
            tqdm.write(f"📊 Evaluation Reward (greedy): {eval_reward:.2f}")
    
    pbar.close()
    env.close()
    
    final_avg = np.mean(episode_rewards[-100:])
    print("\n" + "=" * 70)
    print(f"✅ Training Complete! Final Avg Reward (last 100): {final_avg:.2f}")
    print("=" * 70)

def evaluate_agent(agent, env, num_episodes=3):
    """Evaluate agent with deterministic policy (no exploration)"""
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
    parser = argparse.ArgumentParser(description="PPO Training for CarRacing-v3")
    
    # Environment
    parser.add_argument("--frame_skip", type=int, default=2, help="Action repeat (2-4 for speed)")
    
    # Training
    parser.add_argument("--max_episodes", type=int, default=1000)
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps per rollout")
    
    # PPO Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ppo_epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size")
    
    # Monitoring
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--verbose", action="store_true", help="Print detailed loss info")
    
    args = parser.parse_args()
    train(args)


