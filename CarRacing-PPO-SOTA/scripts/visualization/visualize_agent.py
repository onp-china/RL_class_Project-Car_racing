
"""
Visualize Trained PPO Agent
加载训练好的 PPO 模型并在 CarRacing 环境中运行
支持 PID 控制器平滑动作，减少抖动
"""
import argparse
import time
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils_env_sota import make_sota_env
from algorithms.ppo.agent import PPOAgent

class SimplePID:
    """简单的 PID 控制器，用于平滑转向角度"""
    def __init__(self, kp=0.8, ki=0.0, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0
    
    def update(self, target, current):
        """target: 模型输出, current: 上一帧的实际输出"""
        error = target - current
        
        # P 项
        p_term = self.kp * error
        
        # I 项（带限幅）
        self.integral += error
        self.integral = np.clip(self.integral, -2.0, 2.0)
        i_term = self.ki * self.integral
        
        # D 项
        d_term = self.kd * (error - self.prev_error)
        
        # PID 输出
        output = current + p_term + i_term - d_term
        
        self.prev_error = error
        self.prev_output = output
        
        return np.clip(output, -1.0, 1.0)

def visualize(args):
    # 自动计算决策间隔，以匹配训练时的 0.08s (50Hz/4)
    base_training_dt = (1.0 / 50.0) * args.frame_skip  # 通常是 0.08s
    
    if args.decision_freq is None:
        # 自动计算: new_fps * 0.08s
        decision_interval = int(args.physics_fps * base_training_dt)
        if decision_interval < 1: 
            decision_interval = 1
        print(f"Auto-calculated decision_freq: {decision_interval} (match ~{base_training_dt:.3f}s at {args.physics_fps}Hz)")
    else:
        decision_interval = args.decision_freq

    print("=" * 60)
    print(f"Visualizing PPO Agent: {args.model}")
    print(f"Episodes: {args.episodes}, Frame Skip: 1 (Manual Control)")
    print(f"Physics FPS: {args.physics_fps}")
    print(f"Decision Freq: Every {decision_interval} frames ({args.physics_fps/decision_interval:.1f} Hz)")
    
    if args.use_pid:
        print(f"PID Smoothing: Enabled (Kp={args.pid_kp}, Ki={args.pid_ki}, Kd={args.pid_kd})")
        if args.physics_fps > 60:
            print("提示: 在高 FPS 下，PID 的 Kp 可能需要降低，因为每秒更新次数变多了。")
    print("=" * 60)
    
    # 1. 创建环境 (Human Render Mode)
    # 使用 SOTA 环境包装器，强制 frame_skip=1 以便进行高频 PID 控制
    env = make_sota_env(render_mode="human", frame_stack=4, frame_skip=1, physics_fps=args.physics_fps)
    
    # 2. 创建 Agent
    agent = PPOAgent(
        state_dim=(4, 96, 96),
        action_dim=3,
        lr=3e-4 # 加载模型时 lr 不重要
    )
    
    # 3. 加载权重
    print(f"Loading model from {args.model}...")
    try:
        agent.load(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. 创建 PID 控制器（如果启用）
    if args.use_pid:
        pid = SimplePID(kp=args.pid_kp, ki=args.pid_ki, kd=args.pid_kd)
        current_action = np.array([0.0, 0.0, 0.0]) # 当前实际执行的动作
    else:
        pid = None
        current_action = None

    # 5. 运行 Loop - 连续跑多个赛道
    total_rewards = []
    # 注意：decision_interval 已经在上面计算好了
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        # SOTA 环境返回的是 (H, W, C) 格式，需要转换为 (C, H, W)
        if len(state.shape) == 3 and state.shape[2] == 4:
            state = np.transpose(state, (2, 0, 1)).astype(np.float32)
        
        # 重置 PID
        if pid:
            pid.reset()
            current_action = np.array([0.0, 0.0, 0.0])
        
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        target_action = np.array([0.0, 0.0, 0.0])
        
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes} Started...")
        print(f"决策频率: 每 {decision_interval} 帧决策一次")
        print(f"{'='*60}")
        
        while not (done or truncated):
            # 策略：可调决策频率，高频执行
            # 每 decision_interval 帧调用一次模型获取新目标
            if steps % decision_interval == 0:
                target_action, _, _ = agent.get_action(state, deterministic=True)
            
            # 每一帧都进行 PID 插值/平滑
            if pid:
                # 转向：使用 PID 追踪目标值
                current_action[0] = pid.update(target_action[0], current_action[0])
                
                # 油门/刹车：使用简单的线性插值 (Lerp) 平滑
                # alpha 越大响应越快，越小越平滑
                alpha = 0.25 
                current_action[1] = (1 - alpha) * current_action[1] + alpha * target_action[1]
                current_action[2] = (1 - alpha) * current_action[2] + alpha * target_action[2]
                
                # 确保在有效范围内
                current_action[0] = np.clip(current_action[0], -1.0, 1.0)
                current_action[1] = np.clip(current_action[1], 0.0, 1.0)
                current_action[2] = np.clip(current_action[2], 0.0, 1.0)
                
                action_to_exec = current_action.copy()
            else:
                action_to_exec = target_action

            # 执行一步 (frame_skip=1)
            next_state, reward, done, truncated, _ = env.step(action_to_exec)
            episode_reward += reward
            steps += 1
            
            # 转换下一个状态的格式
            if len(next_state.shape) == 3 and next_state.shape[2] == 4:
                state = np.transpose(next_state, (2, 0, 1)).astype(np.float32)
            else:
                state = next_state
            
            # 显示实时信息
            if steps % 100 == 0:
                print(f"\rStep: {steps:4d} | Reward: {episode_reward:7.1f} | "
                      f"Target: [{target_action[0]:5.2f}] | "
                      f"Actual: [{action_to_exec[0]:5.2f}]", end="", flush=True)
            
        total_rewards.append(episode_reward)
        print(f"\nEpisode {episode + 1} Finished!")
        print(f"  Total Steps: {steps}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Average Reward (so far): {np.mean(total_rewards):.2f}")
        
        # 休息一下再开下一把
        if episode < args.episodes - 1:
            print("\n准备下一个赛道...")
            time.sleep(2.0)

    env.close()
    
    # 打印总结
    print(f"\n{'='*60}")
    print("所有赛道完成！")
    print(f"{'='*60}")
    print(f"总回合数: {len(total_rewards)}")
    print(f"平均奖励: {np.mean(total_rewards):.2f}")
    print(f"最高奖励: {np.max(total_rewards):.2f}")
    print(f"最低奖励: {np.min(total_rewards):.2f}")
    print(f"标准差: {np.std(total_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练好的 PPO 模型")
    parser.add_argument("--model", type=str, default="saved_models/ppo_sota_ep5120.pth", 
                       help="Path to .pth model file")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of test episodes (连续跑多个赛道)")
    parser.add_argument("--frame_skip", type=int, default=4, 
                       help="Frame skip (必须匹配训练时的设置!)")
    parser.add_argument("--physics_fps", type=int, default=50,
                       help="物理引擎 FPS (默认: 50, 推荐尝试 180)")
    parser.add_argument("--decision_freq", type=int, default=None,
                       help="模型决策频率 (每N帧决策一次, None=自动根据FPS计算)")
    parser.add_argument("--use_pid", action="store_true",
                       help="启用 PID 控制器平滑动作")
    parser.add_argument("--pid_kp", type=float, default=0.8,
                       help="PID 比例系数 (默认: 0.8)")
    parser.add_argument("--pid_ki", type=float, default=0.0,
                       help="PID 积分系数 (默认: 0.0)")
    parser.add_argument("--pid_kd", type=float, default=0.2,
                       help="PID 微分系数 (默认: 0.2)")
    args = parser.parse_args()
    
    visualize(args)

