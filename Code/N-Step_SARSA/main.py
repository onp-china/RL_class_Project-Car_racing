import os
# Fix OMP Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import numpy as np
import argparse
import torch
from tqdm import tqdm
from gymnasium.vector import SyncVectorEnv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.wrappers import make_env
from common.utils import save_model, plot_training_results, save_training_data_csv, create_training_progress_bar, init_device
from agent import NStepSarsaAgent

def init_parameters():
    parser = argparse.ArgumentParser(description="N-Step SARSA for CarRacing-v3")
    
    # Environment
    parser.add_argument("--env_name", type=str, default="CarRacing-v3")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_episode_freq", type=int, default=50)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    
    # Algorithm
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_interval", type=int, default=1000)
    
    # Epsilon Greedy
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_steps", type=int, default=500000)
    
    return parser.parse_args()

def main():
    args = init_parameters()
    
    print("=" * 60)
    print("N-Step SARSA Training for CarRacing-v3")
    print("=" * 60)
    
    if args.max_episodes is not None:
        estimated_steps = args.max_episodes * 1000
        if estimated_steps > args.total_timesteps:
            args.total_timesteps = estimated_steps
            print(f"Adjusted total_timesteps to {estimated_steps} based on max_episodes")

    # Set seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Initialize Device
    device = init_device()
    
    # Create Environments (使用common的make_env)
    print(f"\nCreating {args.num_envs} environments...")
    envs = SyncVectorEnv([
        make_env(args.env_name, seed=args.seed + i if args.seed else None, continuous=False, max_steps=args.max_steps)
        for i in range(args.num_envs)
    ])
    
    # Create Agent
    action_dim = envs.single_action_space.n
    agent = NStepSarsaAgent(
        action_dim=action_dim,
        lr=args.lr,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        update_interval=args.update_interval,
        device=device
    )
    
    # Logging setup
    current_path = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_path, "plots")
    models_dir = os.path.join(current_path, "models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    data_file = os.path.join(plots_dir, "training_data.txt")
    if not os.path.exists(data_file):
        with open(data_file, 'w') as f:
            f.write("# Episode,Reward,Timestep\n")
            
    # Training Loop
    obs, _ = envs.reset()
    
    # Get initial actions for SARSA (s, a)
    epsilon = args.epsilon_start
    actions = agent.get_action(obs, epsilon)
    # Fix: gymnasium 1.0.0 requires Python native int
    if isinstance(actions, np.ndarray):
        actions = [int(a) for a in actions]
    
    total_steps = 0
    episode_rewards = []
    episode_timesteps = []
    loss_history = {
        'q_loss': []
    }
    current_ep_rewards = np.zeros(args.num_envs)
    
    pbar = create_training_progress_bar(args.total_timesteps, "N-Step SARSA")
    
    while total_steps < args.total_timesteps:
        if args.max_episodes is not None and len(episode_rewards) >= args.max_episodes:
            break
            
        # Epsilon decay
        epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                  max(0, (args.epsilon_decay_steps - total_steps) / args.epsilon_decay_steps)
        
        # Environment step (s, a) -> (s', r)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = terminations | truncations
        
        # Get next actions (a') for (s', a')
        next_actions = agent.get_action(next_obs, epsilon)
        # Fix: gymnasium 1.0.0 requires Python native int
        if isinstance(next_actions, np.ndarray):
            next_actions = [int(a) for a in next_actions]
        
        # Store transitions and manual reward accumulation
        current_ep_rewards += rewards
        
        for i in range(args.num_envs):
            # Store (s, a, r, s', a', done)
            agent.buffer.push(obs[i], actions[i], rewards[i], next_obs[i], next_actions[i], dones[i])
            
            if dones[i]:
                ep_reward = current_ep_rewards[i]
                episode_rewards.append(ep_reward)
                episode_timesteps.append(total_steps)
                current_ep_rewards[i] = 0
                
                # Logging
                with open(data_file, 'a') as f:
                    f.write(f"{len(episode_rewards)},{ep_reward:.2f},{total_steps}\n")
                    
                if len(episode_rewards) % args.save_episode_freq == 0:
                    save_model(agent.q_net, models_dir, f"sarsa_checkpoint_ep{len(episode_rewards)}")
        
        # Update agent
        update_stats = agent.update()
        
        # 记录损失
        if update_stats:
            loss_history['q_loss'].append(update_stats.get('q_loss', 0))
        
        obs = next_obs
        actions = next_actions
        
        step_increment = args.num_envs
        total_steps += step_increment
        pbar.update(step_increment)
        
        # Update progress bar status
        if total_steps % 1000 == 0:
            avg_rew = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards) if episode_rewards else 0
            pbar.set_postfix({
                'Eps': len(episode_rewards),
                'AvgRew': f"{avg_rew:.1f}",
                'Epsilon': f"{epsilon:.3f}"
            })
            
    pbar.close()
    
    # Final save and plotting
    save_model(agent.q_net, models_dir, "sarsa_final")
    
    # 绘制奖励曲线和保存CSV
    if episode_rewards:
        # 计算移动平均
        avg_reward_20 = []
        for i in range(len(episode_rewards)):
            start = max(0, i - 19)
            avg_reward_20.append(np.mean(episode_rewards[start:i+1]))
        
        # 保存CSV - 调整loss数组长度以匹配episode数量
        num_episodes = len(episode_rewards)
        
        def adjust_loss_length(loss_array, target_length):
            if not loss_array:
                return [0] * target_length
            if len(loss_array) == target_length:
                return loss_array
            elif len(loss_array) > target_length:
                # 分组平均
                chunk_size = len(loss_array) / target_length
                return [np.mean(loss_array[int(i*chunk_size):int((i+1)*chunk_size)]) for i in range(target_length)]
            else:
                # 用最后一个值填充
                return loss_array + [loss_array[-1]] * (target_length - len(loss_array))
        
        data_dict = {
            'episode': list(range(1, num_episodes + 1)),
            'reward': episode_rewards,
            'timestep': episode_timesteps,
            'avg_reward_20': avg_reward_20,
            'q_loss': adjust_loss_length(loss_history['q_loss'], num_episodes)
        }
        save_training_data_csv(data_dict, plots_dir, "sarsa")
        
        # 绘制4子图
        plot_training_results(
            episode_rewards, 
            plots_dir, 
            "N-Step SARSA",
            timesteps=episode_timesteps,
            losses=loss_history,
            loss_names=['q_loss']
        )
    
    print("\nTraining Completed!")
    envs.close()

if __name__ == "__main__":
    main()
