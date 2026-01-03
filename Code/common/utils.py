"""
工具函数模块 - 所有算法共享
包含模型保存、奖励曲线绘制、设备初始化等功能
"""

import os
import time
import torch
import numpy as np
import matplotlib
# 设置非交互式后端（Windows上很重要）
matplotlib.use('Agg')  # 在导入pyplot之前设置
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm


def init_device():
    """
    初始化设备并打印信息
    
    Returns:
        torch.device对象
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n[设备初始化]")
    if torch.cuda.is_available():
        print(f"[OK] 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # 启用 cuDNN benchmark
        torch.backends.cudnn.benchmark = True
        print("  cuDNN benchmark已启用")
    else:
        print("[WARN] 使用CPU")
    
    return device


def save_model(model, path, name):
    """
    保存模型检查点
    
    Args:
        model: 要保存的模型（nn.Module）
        path: 保存路径
        name: 模型名称前缀
    
    Returns:
        完整的保存路径
    """
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(path, f"{name}_{timestamp}.pth")
    torch.save(model.state_dict(), filepath)
    return filepath


def plot_and_save_rewards(rewards, path, name_prefix, timesteps=None):
    """
    绘制并保存奖励曲线（向后兼容接口）
    
    Args:
        rewards: 奖励列表
        path: 保存路径
        name_prefix: 文件名前缀（如"ppo", "ddqn"等）
        timesteps: 可选的时间步列表
    
    Note: 此函数保留用于向后兼容，新代码建议使用plot_training_results()
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 转换为numpy数组以便处理
        rewards = np.array(rewards)
        
        if len(rewards) < 3:
            print(f"  [WARN] Only {len(rewards)} episodes collected, too few to generate meaningful plot")
            return
        
        # 检查timesteps长度
        if timesteps is not None:
            timesteps = np.array(timesteps)
            if len(timesteps) != len(rewards):
                print(f"  [WARN] Warning: timesteps length ({len(timesteps)}) != rewards length ({len(rewards)}), using episode numbers instead")
                timesteps = None
        
        window_size = min(20, len(rewards))
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(rewards[start:i+1]))
        moving_avg = np.array(moving_avg)
        
        # 创建更大的图像，提高清晰度
        plt.figure(figsize=(14, 7))
        
        if timesteps is not None and len(timesteps) == len(rewards):
            plt.plot(timesteps, rewards, alpha=0.3, color='gray', label='Episode Reward', linewidth=0.5)
            plt.plot(timesteps, moving_avg, color='b', linewidth=2.5, label=f'Moving Average ({window_size})')
            plt.xlabel('Timesteps', fontsize=12)
        else:
            episodes = np.arange(1, len(rewards) + 1)
            plt.plot(episodes, rewards, alpha=0.3, color='gray', label='Episode Reward', linewidth=0.5)
            plt.plot(episodes, moving_avg, color='b', linewidth=2.5, label=f'Moving Average ({window_size})')
            plt.xlabel('Episode', fontsize=12)
        
        plt.ylabel('Reward', fontsize=12)
        plt.title(f'{name_prefix.upper()} Training on CarRacing-v3', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 添加统计信息到图上
        if len(rewards) >= 20:
            final_avg = np.mean(rewards[-20:])
            best_reward = np.max(rewards)
            textstr = f'Final Avg (20): {final_avg:.1f}\nBest: {best_reward:.1f}'
            plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(path, f"{name_prefix}_rewards_{timestamp}.png")
        
        # 保存高质量图像
        plt.savefig(plot_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()  # 确保关闭图形
        
        # 保存数据（带时间戳的备份）
        data_file = os.path.join(path, f"{name_prefix}_reward_data_{timestamp}.txt")
        with open(data_file, 'w') as f:
            f.write("# Episode, Reward, Timestep\n")
            for i, r in enumerate(rewards):
                t = timesteps[i] if timesteps is not None and i < len(timesteps) else i + 1
                f.write(f"{i+1}, {r:.2f}, {t}\n")
        
        print(f"  [OK] Plot saved: {os.path.basename(plot_file)}")
        
    except Exception as e:
        print(f"  [WARN] Failed to generate plot: {e}")
        import traceback
        traceback.print_exc()
        # 确保即使出错也关闭图形
        try:
            plt.close('all')
        except:
            pass


def base_argparse():
    """
    创建基础参数解析器（所有算法共享的参数）
    
    Returns:
        argparse.ArgumentParser对象
    """
    import argparse
    parser = argparse.ArgumentParser()
    
    # 环境参数
    parser.add_argument("--env_name", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save_freq", type=int, default=100000, help="Save model every N timesteps/episodes")
    
    return parser


def test_policy(agent, env_name, continuous, num_episodes=3, render=True):
    """
    统一的策略测试接口
    
    Args:
        agent: 智能体对象（需要有get_action方法）
        env_name: 环境名称
        continuous: True表示连续动作空间
        num_episodes: 测试的episode数量
        render: 是否渲染
    """
    from .wrappers import GrayScaleObservation, CropObservation, FrameStack, SOTARewardWrapper
    
    print("\n[OK] 开始测试阶段...")
    try:
        if render:
            test_env = gym.make(env_name, continuous=continuous, render_mode="human")
        else:
            test_env = gym.make(env_name, continuous=continuous)
    except:
        test_env = gym.make(env_name, continuous=continuous)
    
    # SOTA Improvements: Frame Skip (Reward Shaping doesn't affect test actions, but Frame Skip does)
    test_env = SOTARewardWrapper(test_env, frame_skip=4)
    test_env = GrayScaleObservation(test_env)
    test_env = CropObservation(test_env, crop_bottom=12)
    test_env = FrameStack(test_env, num_stack=4)
    
    for i in range(num_episodes):
        state, _ = test_env.reset()
        state = np.array(state)
        ep_reward = 0
        done = False
        
        while not done:
            # 使用确定性策略测试
            if hasattr(agent, 'get_action'):
                # PPO/A2C/REINFORCE风格
                action, _, _ = agent.get_action(state[np.newaxis, ...], deterministic=True)
                # 确保action是1D数组
                if action.ndim > 1:
                    action = action[0]
                # 确保action是numpy数组且形状正确
                action = np.asarray(action).flatten()
            else:
                # DQN风格
                action = agent.select_action(state, epsilon=0.0)
            
            next_state, reward, term, trunc, _ = test_env.step(action)
            state = np.array(next_state)
            ep_reward += reward
            done = term or trunc
        
        print(f"  测试Episode {i+1}: Reward = {ep_reward:.2f}")
    
    test_env.close()
    print("[OK] 测试完成!\n")


def create_training_progress_bar(total_steps, algorithm_name):
    """
    创建统一格式的tqdm进度条
    
    Args:
        total_steps: 总训练步数
        algorithm_name: 算法名称（如"PPO", "DQN"等）
    
    Returns:
        tqdm对象
    """
    return tqdm(
        total=total_steps,
        desc=f"{algorithm_name} Training",
        unit="step",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
        dynamic_ncols=True,
        mininterval=0.5  # 每0.5秒更新一次
    )


def save_training_data_csv(data_dict, save_path, filename_prefix):
    """
    保存详细训练数据到CSV
    
    Args:
        data_dict: 数据字典，包含:
            - 'episode': episode编号列表
            - 'reward': 奖励列表
            - 'timestep': 时间步列表
            - 'avg_reward_20': 20期移动平均（可选）
            - 其他损失数据（如'policy_loss', 'value_loss'等）
        save_path: 保存路径
        filename_prefix: 文件名前缀（如"ppo", "ddqn"等）
    """
    try:
        import pandas as pd
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        df = pd.DataFrame(data_dict)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(save_path, f"{filename_prefix}_data_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"  [OK] CSV saved: {os.path.basename(csv_file)}")
        
    except ImportError:
        print("  [WARN] pandas not installed, CSV save skipped")
    except Exception as e:
        print(f"  [WARN] Failed to save CSV: {e}")


def plot_training_results(rewards, plots_dir, algorithm_name, 
                          timesteps=None, losses=None, loss_names=None):
    """
    绘制统一风格的4子图训练结果
    
    Args:
        rewards: 奖励列表
        plots_dir: 保存路径
        algorithm_name: 算法名称（如"PPO", "DQN"等）
        timesteps: 可选的时间步列表
        losses: 损失数据字典，例如 {"policy_loss": [...], "value_loss": [...]}
        loss_names: 损失名称列表（用于图例）
    """
    try:
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        rewards = np.array(rewards)
        if len(rewards) < 3:
            print(f"  [WARN] Only {len(rewards)} episodes collected, too few to plot")
            return
        
        # 创建4子图布局（14x14英寸）
        fig = plt.figure(figsize=(14, 14))
        
        # ===== 左上：Episode Rewards =====
        ax1 = plt.subplot(2, 2, 1)
        episodes = np.arange(1, len(rewards) + 1)
        
        # 计算50期移动平均
        window_size = min(50, len(rewards))
        moving_avg = []
        for i in range(len(rewards)):
            start = max(0, i - window_size + 1)
            moving_avg.append(np.mean(rewards[start:i+1]))
        moving_avg = np.array(moving_avg)
        
        # 绘制原始奖励和移动平均
        ax1.plot(episodes, rewards, alpha=0.2, color='lightblue', linewidth=0.5, label='Raw Reward')
        ax1.plot(episodes, moving_avg, color='b', linewidth=2.5, label=f'{window_size}-Episode Moving Avg')
        ax1.axhline(y=900, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Score (900)')
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Reward', fontsize=11)
        ax1.set_title('Episode Rewards', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='lower right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # ===== 右上：Training Losses =====
        ax2 = plt.subplot(2, 2, 2)
        if losses and loss_names:
            # 自适应绘制不同算法的损失
            colors = ['darkred', 'orange', 'purple', 'brown']
            for idx, loss_name in enumerate(loss_names):
                if loss_name in losses and len(losses[loss_name]) > 0:
                    loss_data = np.array(losses[loss_name])
                    # 损失通常按更新步数记录，需要映射到episode
                    loss_episodes = np.linspace(1, len(rewards), len(loss_data))
                    ax2.plot(loss_episodes, loss_data, 
                            color=colors[idx % len(colors)], 
                            linewidth=1.5, 
                            label=loss_name.replace('_', ' ').title(),
                            alpha=0.8)
            
            ax2.set_xlabel('Training Progress', fontsize=11)
            ax2.set_ylabel('Loss', fontsize=11)
            ax2.set_title('Training Losses', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, 'No Loss Data Available', 
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Training Losses', fontsize=13, fontweight='bold')
        
        # ===== 左下：Reward Distribution =====
        ax3 = plt.subplot(2, 2, 3)
        bins = min(50, len(rewards) // 10) if len(rewards) > 50 else 30
        ax3.hist(rewards, bins=bins, color='lightblue', edgecolor='black', alpha=0.7)
        mean_reward = np.mean(rewards)
        ax3.axvline(x=mean_reward, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_reward:.1f}')
        
        ax3.set_xlabel('Reward', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Reward Distribution', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # ===== 右下：Learning Progress =====
        ax4 = plt.subplot(2, 2, 4)
        # 计算累积平均奖励（学习进度）
        cumulative_avg = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
        
        if timesteps is not None and len(timesteps) == len(rewards):
            timesteps_arr = np.array(timesteps)
            ax4.fill_between(timesteps_arr, cumulative_avg, alpha=0.4, color='purple')
            ax4.plot(timesteps_arr, cumulative_avg, color='darkviolet', linewidth=2.5)
            ax4.set_xlabel('Training Steps', fontsize=11)
        else:
            ax4.fill_between(episodes, cumulative_avg, alpha=0.4, color='purple')
            ax4.plot(episodes, cumulative_avg, color='darkviolet', linewidth=2.5)
            ax4.set_xlabel('Episode', fontsize=11)
        
        ax4.set_ylabel('Avg Reward (Last N Episodes)', fontsize=11)
        ax4.set_title('Learning Progress', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 总标题
        fig.suptitle(f'CarRacing {algorithm_name.upper()} Training Results (SOTA Reward Shaping)', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # 保存图像
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(plots_dir, f"{algorithm_name.lower()}_training_results_{timestamp}.png")
        plt.savefig(plot_file, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  [OK] Training results plot saved: {os.path.basename(plot_file)}")
        
    except Exception as e:
        print(f"  [WARN] Failed to generate training results plot: {e}")
        import traceback
        traceback.print_exc()
        try:
            plt.close('all')
        except:
            pass
