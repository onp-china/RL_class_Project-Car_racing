import os
# 修复OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gymnasium as gym
import numpy as np
import argparse
import torch
import sys
import time
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv

# Add project root to path
# 添加 Code 目录到路径，以便导入 common 模块
code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
from common.wrappers import make_env, SOTARewardWrapper, GrayScaleObservation, CropObservation, FrameStack
from common.utils import save_model, plot_training_results, save_training_data_csv, create_training_progress_bar
from agent import PPOAgent, RolloutBuffer


def init_parameters():
    """初始化命令行参数"""
    parser = argparse.ArgumentParser(description="PPO for CarRacing-v3 (Optimized)")
    
    # 环境参数
    parser.add_argument("--env_name", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000, help="Total training timesteps")
    
    # PPO参数
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per rollout (SB3 default)")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size (SB3 default, 适合2-3M参数的CNN网络)")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (根据2-3M参数的CNN网络调整，略低于SB3默认值)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    
    # 其他
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save_freq", type=int, default=100000, help="Save model every N timesteps (deprecated, use save_episode_freq)")
    parser.add_argument("--save_episode_freq", type=int, default=50, help="Save model every N episodes")
    parser.add_argument("--max_episodes", type=int, default=None, help="Maximum number of episodes to train (overrides total_timesteps if set)")
    
    args = parser.parse_args()
    return args


def main():
    args = init_parameters()
    
    print("=" * 60)
    print("PPO Training for CarRacing-v3")
    print("=" * 60)
    
    # 如果设置了max_episodes，自动调整total_timesteps以防止提前停止
    if args.max_episodes is not None:
        # 假设每个episode平均1000步（保守估计），确保timesteps足够大
        estimated_steps = args.max_episodes * 1000
        if estimated_steps > args.total_timesteps:
            print(f"  [Info] Max episodes set to {args.max_episodes:,}")
            print(f"  [Info] Adjusting total_timesteps from {args.total_timesteps:,} to {estimated_steps:,} to ensure training continues.")
            args.total_timesteps = estimated_steps

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # 设备设置
    print("\n[1/5] Initializing device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 启用 cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        print("  cuDNN benchmark enabled")
    else:
        print("[WARN] Using CPU")
    
    # 创建向量化环境
    print(f"\n[2/5] Creating {args.num_envs} parallel environments...")
    start_env = time.time()
    envs = AsyncVectorEnv([
        make_env(args.env_name, seed=args.seed + i if args.seed else None) 
        for i in range(args.num_envs)
    ])
    print(f"[OK] Environments created in {time.time() - start_env:.1f}s")
    
    # 创建智能体
    print(f"\n[3/5] Creating PPO agent...")
    action_dim = envs.single_action_space.shape[0]
    agent = PPOAgent(
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device
    )
    print("[OK] Agent created")
    
    # 创建Rollout Buffer
    print(f"\n[4/5] Creating rollout buffer...")
    obs_shape = (4, 84, 84)  # 4帧堆叠的84x84灰度图
    rollout_buffer = RolloutBuffer(
        buffer_size=args.n_steps,
        num_envs=args.num_envs,
        obs_shape=obs_shape,
        action_dim=action_dim
    )
    print(f"[OK] Buffer created (size: {args.n_steps} × {args.num_envs} = {args.n_steps * args.num_envs:,} transitions)")
    
    # 训练统计
    episode_rewards = []
    episode_timesteps = []
    current_episode_rewards = np.zeros(args.num_envs)  # 手动追踪每个环境的累积reward
    loss_history = {
        'policy_loss': [],
        'value_loss': []
    }
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # 创建plots目录和实时数据文件
    plots_dir = os.path.join(current_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    data_file = os.path.join(plots_dir, "training_data.txt")
    
    # 初始化数据文件（如果不存在）
    if not os.path.exists(data_file):
        with open(data_file, 'w') as f:
            f.write("# Episode,Reward,Timestep\n")
    
    # 训练循环
    total_updates = args.total_timesteps // (args.n_steps * args.num_envs)
    print(f"\n[5/5] Starting training...")
    print(f"  Total timesteps: {args.total_timesteps:,}")
    if args.max_episodes is not None:
        print(f"  Max episodes: {args.max_episodes:,}")
    print(f"  Updates needed: {total_updates}")
    print(f"  Steps per update: {args.n_steps}")
    print("=" * 60)
    
    global_step = 0
    num_updates = 0
    start_time = time.time()
    
    print("\nResetting environments...")
    obs, _ = envs.reset()
    print("[OK] Ready to train\n")
    
    # 使用统一进度条
    pbar = create_training_progress_bar(args.total_timesteps, "PPO")
    
    # 检查停止条件
    max_episodes_reached = False
    
    while global_step < args.total_timesteps:
        # 检查是否达到最大episode数
        if args.max_episodes is not None and len(episode_rewards) >= args.max_episodes:
            max_episodes_reached = True
            break
        
        # Rollout阶段：收集n_steps数据
        for step in range(args.n_steps):
            # 再次检查（在rollout过程中可能完成更多episodes）
            if args.max_episodes is not None and len(episode_rewards) >= args.max_episodes:
                max_episodes_reached = True
                break
            
            actions, log_probs, values = agent.get_action(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            dones = terminations | truncations
            
            # 手动累积episode reward
            current_episode_rewards += rewards
            
            # 检查是否有episode完成
            for i in range(args.num_envs):
                if dones[i]:
                    ep_reward = current_episode_rewards[i]
                    episode_rewards.append(ep_reward)
                    episode_timesteps.append(global_step)
                    current_episode_rewards[i] = 0
                    
                    # 实时保存数据到文件
                    with open(data_file, 'a') as f:
                        f.write(f"{len(episode_rewards)},{ep_reward:.2f},{global_step}\n")
                        f.flush()
                    
                    # 每N个episode保存一次模型
                    if len(episode_rewards) % args.save_episode_freq == 0:
                        save_path = save_model(agent.network, os.path.join(current_path, "models"), f"ppo_checkpoint_ep{len(episode_rewards)}")
                        print(f"\n[OK] Checkpoint saved (Episode {len(episode_rewards)}): {os.path.basename(save_path)}")
            
            # 存入buffer
            rollout_buffer.add(obs, actions, rewards, dones, log_probs, values)
            
            obs = next_obs
            global_step += args.num_envs
            
            # 每步都更新进度条（就像图像识别训练那样）
            pbar.update(args.num_envs)
            
            # 每50步更新一次状态显示（减少刷新频率，但保持流畅）
            if step % 50 == 0:
                elapsed = time.time() - start_time
                fps = global_step / elapsed if elapsed > 0 else 0
                status = f"Rollout {num_updates+1}, FPS: {fps:.0f}"
                if episode_rewards:
                    avg_r = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
                    max_eps_info = f"/{args.max_episodes}" if args.max_episodes else ""
                    status = f"Eps: {len(episode_rewards)}{max_eps_info}, AvgR: {avg_r:.0f}, FPS: {fps:.0f}"
                pbar.set_postfix_str(status)
            
            # 记录episode信息（保留作为验证，但主要记录已在上面完成）
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # 仅用于调试验证
                        pass
        
        if max_episodes_reached:
            break
        
        # 计算GAE和returns
        last_values = agent.get_value(obs)
        rollout_buffer.compute_returns_and_advantages(last_values, args.gamma, args.gae_lambda)
        
        # 更新策略
        train_stats = agent.update(rollout_buffer)
        
        # 记录损失
        if train_stats:
            loss_history['policy_loss'].append(train_stats.get('policy_loss', 0))
            loss_history['value_loss'].append(train_stats.get('value_loss', 0))
        
        # 重置buffer
        rollout_buffer.reset()
        num_updates += 1
        
        # 更新状态显示
        elapsed = time.time() - start_time
        fps = global_step / elapsed
        if episode_rewards:
            avg_reward = np.mean(episode_rewards[-20:]) if len(episode_rewards) >= 20 else np.mean(episode_rewards)
            last_reward = episode_rewards[-1]
            max_eps_info = f"/{args.max_episodes}" if args.max_episodes else ""
            pbar.set_postfix_str(f"Eps: {len(episode_rewards)}{max_eps_info}, LastR: {last_reward:.0f}, AvgR: {avg_reward:.0f}, FPS: {fps:.0f}")
        else:
            pbar.set_postfix_str(f"FPS: {fps:.0f}, Collecting...")
        
        # 检查是否完成
        if global_step >= args.total_timesteps:
            break
            
    pbar.close()
    
    # 训练完成
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("[OK] Training completed!")
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print(f"  Average FPS: {global_step/elapsed_time:.0f}")
    print(f"  Total episodes: {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Final avg reward: {np.mean(episode_rewards[-20:]):.1f}")
    
    # 保存最终模型
    final_path = save_model(agent.network, os.path.join(current_path, "models"), "ppo_final")
    print(f"  Final model: {os.path.basename(final_path)}")
    
    # 绘制奖励曲线和保存CSV
    if episode_rewards:
        try:
            plot_path = os.path.join(current_path, "plots")
            
            # 计算移动平均
            avg_reward_20 = []
            for i in range(len(episode_rewards)):
                start = max(0, i - 19)
                avg_reward_20.append(np.mean(episode_rewards[start:i+1]))
            
            # 保存CSV - 调整loss数组长度以匹配episode数量
            num_episodes = len(episode_rewards)
            
            # 如果loss历史记录比episode多，取平均；如果少，用最后一个值填充
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
                'policy_loss': adjust_loss_length(loss_history['policy_loss'], num_episodes),
                'value_loss': adjust_loss_length(loss_history['value_loss'], num_episodes)
            }
            save_training_data_csv(data_dict, plot_path, "ppo")
            
            # 绘制4子图
            plot_training_results(
                episode_rewards, 
                plot_path, 
                "PPO",
                timesteps=episode_timesteps,
                losses=loss_history,
                loss_names=['policy_loss', 'value_loss']
            )
            
            print(f"  [OK] Training data saved to: {data_file}")
            
            # 打印统计信息
            if len(episode_rewards) >= 3:
                print(f"\n  Training Statistics:")
                print(f"    Total episodes: {len(episode_rewards)}")
                print(f"    Average reward: {np.mean(episode_rewards):.2f}")
                print(f"    Best reward: {np.max(episode_rewards):.2f}")
                print(f"    Worst reward: {np.min(episode_rewards):.2f}")
                if len(episode_rewards) >= 20:
                    print(f"    Last 20 avg: {np.mean(episode_rewards[-20:]):.2f}")
        except Exception as e:
            print(f"  [WARN] Failed to generate plot/CSV: {e}")
            print(f"  [OK] Training data still saved to: {data_file}")
    else:
        print("  [WARN] No episodes completed, skipping plot")
    
    print("=" * 60)
    
    envs.close()
    
    # 测试阶段
    print("\n[OK] Starting test phase...")
    try:
        test_env = gym.make(args.env_name, continuous=True, render_mode="human")
    except:
        test_env = gym.make(args.env_name, continuous=True)
    
    # Apply same wrappers as training
    test_env = SOTARewardWrapper(test_env, frame_skip=4)
    test_env = GrayScaleObservation(test_env)
    test_env = CropObservation(test_env, crop_bottom=12)
    test_env = FrameStack(test_env, num_stack=4)
    
    for i in range(3):
        state, _ = test_env.reset()
        state = np.array(state)
        ep_reward = 0
        done = False
        
        while not done:
            # 使用确定性策略测试
            action, _, _ = agent.get_action(state[np.newaxis, ...], deterministic=True)
            action = action[0]
            
            next_state, reward, term, trunc, _ = test_env.step(action)
            state = np.array(next_state)
            ep_reward += reward
            done = term or trunc
        
        print(f"  Test Episode {i+1}: Reward = {ep_reward:.2f}")
    
    test_env.close()
    print("\n[OK] All done!")


if __name__ == "__main__":
    main()
