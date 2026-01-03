"""
PID 参数自动优化器
目标：寻找一组 PID 参数，使 PPO 模型在保持高分的同时，动作最平滑（减少蛇形走位）。
"""
import argparse
import numpy as np
import torch
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils_env_sota import make_sota_env
from algorithms.ppo.agent import PPOAgent
import multiprocessing
from copy import deepcopy

# --- 1. 定义 PID 控制器 ---
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        
    def update(self, target, current):
        error = target - current
        self.integral += error
        self.integral = np.clip(self.integral, -1.0, 1.0) # 积分限幅
        
        p = self.kp * error
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error)
        
        output = current + p + i - d
        self.prev_error = error
        return np.clip(output, -1.0, 1.0)

# --- 2. 评估函数 ---
def evaluate_params(args_tuple):
    """
    运行一个 Episode，评估特定 PID 参数的表现
    使用高频 PID 控制：模型低频决策，PID 高频执行
    返回: (混合评分, 原始奖励, 平滑度惩罚)
    """
    pid_params, model_path, decision_interval, physics_fps, max_steps = args_tuple
    kp, ki, kd = pid_params
    
    # 初始化环境 (无渲染模式，frame_skip=1 以便高频控制)
    env = make_sota_env(render_mode=None, frame_stack=4, frame_skip=1, physics_fps=physics_fps)
    
    # 初始化 Agent
    agent = PPOAgent(state_dim=(4, 96, 96), action_dim=3)
    try:
        agent.load(model_path)
    except:
        return -9999, 0, 0

    # 初始化 PID
    pid = PIDController(kp, ki, kd)
    
    state, _ = env.reset()
    if len(state.shape) == 3 and state.shape[2] == 4:
        state = np.transpose(state, (2, 0, 1)).astype(np.float32)
        
    total_reward = 0
    total_jerk = 0 # 记录抖动（动作变化率）
    current_action = np.array([0.0, 0.0, 0.0]) # 当前实际执行的动作
    target_action = np.array([0.0, 0.0, 0.0]) # 模型输出的目标动作
    prev_action = np.array([0.0, 0.0, 0.0])
    
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps: # 限制最大步数防止死循环
        # 可调决策频率：每 decision_interval 帧才调用一次模型
        if steps % decision_interval == 0:
            target_action, _, _ = agent.get_action(state, deterministic=True)
        
        # 高频执行：每一帧都用 PID 追踪目标值
        # 转向：PID 平滑
        current_action[0] = pid.update(target_action[0], current_action[0])
        
        # 油门/刹车：线性插值平滑
        alpha = 0.25
        current_action[1] = (1 - alpha) * current_action[1] + alpha * target_action[1]
        current_action[2] = (1 - alpha) * current_action[2] + alpha * target_action[2]
        
        # 限幅
        current_action[0] = np.clip(current_action[0], -1.0, 1.0)
        current_action[1] = np.clip(current_action[1], 0.0, 1.0)
        current_action[2] = np.clip(current_action[2], 0.0, 1.0)
        
        # 记录抖动 (本次动作 - 上次动作 的绝对值)
        if steps > 0:
            jerk = abs(current_action[0] - prev_action[0])
            total_jerk += jerk
        
        prev_action = current_action.copy()
        
        # 执行 (frame_skip=1)
        next_state, reward, done, truncated, _ = env.step(current_action)
        
        total_reward += reward
        steps += 1
        
        if len(next_state.shape) == 3 and next_state.shape[2] == 4:
            state = np.transpose(next_state, (2, 0, 1)).astype(np.float32)
        else:
            state = next_state
            
    env.close()
    
    # --- 核心：计算混合评分 ---
    # 我们希望 Reward 高，Jerk 低
    # Jerk Penalty 系数可以调整，越大越强调平滑
    jerk_penalty = total_jerk * 5.0 
    score = total_reward - jerk_penalty
    
    return score, total_reward, total_jerk

# --- 3. 随机搜索主循环 ---
def optimize(args):
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
    
    print(f"开始 PID 参数搜索... (Model: {args.model})")
    print(f"物理 FPS: {args.physics_fps}")
    print(f"控制模式：高频 PID (模型每 {decision_interval} 帧决策，PID 每帧执行)")
    print(f"目标：最大化 (Reward - 5.0 * Jerk)")
    
    if args.physics_fps > 100:
        kp_max = 0.3
        print(f"注意：由于物理 FPS 提高到 {args.physics_fps}Hz (超高频)，Kp 搜索范围调整为 [0.01, {kp_max}]")
    elif args.physics_fps > 60:
        kp_max = 0.5
        print(f"注意：由于物理 FPS 提高到 {args.physics_fps}Hz，Kp 搜索范围调整为 [0.05, {kp_max}]")
    else:
        kp_max = 0.8
        print(f"注意：由于控制频率提高，Kp 搜索范围调整为 [0.05, {kp_max}]")
    
    best_score = -float('inf')
    best_params = (0.15, 0.0, 0.05) if args.physics_fps > 100 else (0.3, 0.0, 0.1) # 默认初始值
    best_reward = 0
    best_jerk = 0
    
    # 搜索配置 - 支持快速模式
    if args.fast:
        num_trials = 15  # 快速模式：15次试验
        max_steps = 600  # 快速模式：600步（120Hz下约5秒）
        print(f"[快速模式] 试验次数: {num_trials}, 每episode最大步数: {max_steps}")
    else:
        num_trials = args.num_trials
        max_steps = args.max_steps
        print(f"[标准模式] 试验次数: {num_trials}, 每episode最大步数: {max_steps}")
    
    for i in range(num_trials):
        # 随机生成参数
        # Kp: 响应速度，范围根据 FPS 调整
        # Ki: 积分，通常很小 [0.0, 0.05]
        # Kd: 阻尼，范围 [0.0, 0.3]
        if i == 0:
            # 第一次尝试默认值
            if args.physics_fps > 100:
                kp, ki, kd = 0.15, 0.0, 0.05
            elif args.physics_fps > 60:
                kp, ki, kd = 0.2, 0.0, 0.1
            else:
                kp, ki, kd = 0.3, 0.0, 0.1
        elif i == 1:
            # 第二次尝试更保守的值
            if args.physics_fps > 100:
                kp, ki, kd = 0.10, 0.0, 0.08
            elif args.physics_fps > 60:
                kp, ki, kd = 0.15, 0.0, 0.15
            else:
                kp, ki, kd = 0.2, 0.0, 0.15
        elif i == 2:
            # 第三次尝试更激进的值
            if args.physics_fps > 100:
                kp, ki, kd = 0.20, 0.0, 0.02
            elif args.physics_fps > 60:
                kp, ki, kd = 0.3, 0.0, 0.05
            else:
                kp, ki, kd = 0.5, 0.0, 0.05
        else:
            kp_min = 0.01 if args.physics_fps > 100 else 0.05
            kp = np.random.uniform(kp_min, kp_max)
            ki = np.random.uniform(0.0, 0.03)
            kd = np.random.uniform(0.0, 0.2 if args.physics_fps > 100 else 0.3)
            
        params = (kp, ki, kd)
        
        # 评估 (运行 1 次，为了速度)
        s, r, j = evaluate_params((params, args.model, decision_interval, args.physics_fps, max_steps))
        
        print(f"Trial {i+1}/{num_trials}: Kp={kp:.2f}, Ki={ki:.3f}, Kd={kd:.2f} | "
              f"Score: {s:.1f} (R: {r:.1f}, J: {j:.1f})")
        
        if s > best_score:
            best_score = s
            best_params = params
            best_reward = r
            best_jerk = j
            print(f"  >>> 发现新最佳参数！ <<<")

    print("\n" + "="*60)
    print("搜索完成！最佳参数组合：")
    print(f"Kp (比例): {best_params[0]:.4f}")
    print(f"Ki (积分): {best_params[1]:.4f}")
    print(f"Kd (微分): {best_params[2]:.4f}")
    print("-" * 30)
    print(f"预期平均奖励: {best_reward:.1f}")
    print(f"预期平滑度代价 (越低越好): {best_jerk:.1f}")
    print("="*60)
    
    print("\n您可以直接使用以下命令进行可视化：")
    print(f"python visualize_agent.py --episodes 5 --use_pid --physics_fps {args.physics_fps} --pid_kp {best_params[0]:.4f} --pid_ki {best_params[1]:.4f} --pid_kd {best_params[2]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="saved_models/ppo_sota_ep5120.pth")
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--physics_fps", type=int, default=50,
                       help="物理引擎 FPS (默认: 50, 推荐尝试 120)")
    parser.add_argument("--decision_freq", type=int, default=None,
                       help="模型决策频率 (每N帧决策一次, None=自动根据FPS计算)")
    parser.add_argument("--num_trials", type=int, default=30,
                       help="参数搜索试验次数 (默认: 30, 快速模式: 15)")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="每个episode最大步数 (默认: 1000, 快速模式: 600)")
    parser.add_argument("--fast", action="store_true",
                       help="快速模式: 减少试验次数(15)和步数(600)以加速优化")
    args = parser.parse_args()
    
    optimize(args)
