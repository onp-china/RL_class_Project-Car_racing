"""
多车比赛游戏运行脚本
支持 Agent vs Agent 和 Human vs Agent 两种模式
"""
import os
import sys
import argparse
import collections
import numpy as np
import torch
import pygame
from typing import Optional, Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_car_racing import MultiCarRacing


class FrameStacker:
    """
    手动实现 FrameStack，用于需要4帧历史的模型（DQN, SARSA, Double-DQN）
    """
    def __init__(self, n_stack=4, shape=(96, 96, 3)):
        self.n_stack = n_stack
        self.buffer = collections.deque(maxlen=n_stack)
        # 初始化时填满全黑帧
        for _ in range(n_stack):
            self.buffer.append(np.zeros(shape, dtype=np.uint8))
    
    def update(self, obs):
        """接收新的一帧画面"""
        self.buffer.append(obs.copy() if hasattr(obs, 'copy') else obs)
    
    def get_stack(self):
        """
        返回拼接好的4帧画面
        形状: (96, 96, 3) -> (96, 96, 12)
        """
        return np.concatenate(list(self.buffer), axis=-1)
    
    def reset(self):
        """重置帧历史"""
        self.buffer.clear()
        for _ in range(self.n_stack):
            self.buffer.append(np.zeros((96, 96, 3), dtype=np.uint8))


def discrete_to_continuous(action):
    """
    将离散动作转换为连续动作 [steer, gas, brake]
    0: nothing -> [0, 0, 0]
    1: left -> [-1, 0, 0]
    2: right -> [+1, 0, 0]
    3: gas -> [0, 1, 0]
    4: brake -> [0, 0, 0.8]
    """
    if action == 0:
        return np.array([0.0, 0.0, 0.0])
    elif action == 1:
        return np.array([-1.0, 0.0, 0.0])
    elif action == 2:
        return np.array([+1.0, 0.0, 0.0])
    elif action == 3:
        return np.array([0.0, 1.0, 0.0])
    elif action == 4:
        return np.array([0.0, 0.0, 0.8])
    else:
        return np.array([0.0, 0.0, 0.0])


def load_agent(algorithm: str, model_path: str, action_dim: int, lr: float, gamma: float, device: torch.device):
    """
    动态加载不同算法的 Agent
    
    Args:
        algorithm: 算法名称 ("DQN", "SARSA", "Double-DQN", "A2C", "REINFORCE")
        model_path: 模型文件路径
        action_dim: 动作维度
        lr: 学习率（用于初始化，实际不使用）
        gamma: 折扣因子（用于初始化，实际不使用）
        device: 设备
    
    Returns:
        Agent 对象
    """
    algorithm_map = {
        "DQN": ("DQN", "DQNAgent"),
        "SARSA": ("N-Step_SARSA", "NStepSarsaAgent"),
        "Double-DQN": ("Double_DQN", "DoubleDQNAgent"),
        "A2C": ("A2C", "A2CAgent"),
        "REINFORCE": ("REINFORCE", "REINFORCEAgent"),
        "PPO": ("PPO/PPO", "PPOAgent"),
    }
    
    if algorithm not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(algorithm_map.keys())}")
    
    folder, class_name = algorithm_map[algorithm]
    
    # 构建路径：优先尝试 Code/ 目录
    # 假设脚本在 Code/../multi_car_game.py (即根目录)
    # 那么 Code 目录在 ./Code
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(root_dir, "Code", folder, "agent")
    
    # 检查是否存在
    if not os.path.exists(module_path + ".py"):
         # 尝试旧的数字前缀结构 (fallback)
         old_algorithm_map = {
            "DQN": "1. DQN",
            "SARSA": "2. SARSA",
            "Double-DQN": "3. Double-DQN",
            "A2C": "4. A2C",
            "REINFORCE": "5. REINFORCE",
         }
         if algorithm in old_algorithm_map:
             folder = old_algorithm_map[algorithm]
             # 尝试在根目录或 Code 目录下查找
             paths_to_try = [
                 os.path.join(root_dir, folder, "agent"),
                 os.path.join(root_dir, "Code", folder, "agent")
             ]
             for p in paths_to_try:
                 if os.path.exists(p + ".py"):
                     module_path = p
                     break

    # 动态导入
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_module", f"{module_path}.py")
    if spec is None:
         # 尝试直接从当前目录查找（如果结构不同）
         print(f"Warning: Could not find module at {module_path}.py, trying relative import...")
         module_path = os.path.join(os.path.dirname(__file__), folder.split(". ")[1], "agent")
         spec = importlib.util.spec_from_file_location("agent_module", f"{module_path}.py")

    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    
    AgentClass = getattr(agent_module, class_name)
    
    print(f"Initializing {algorithm} agent on device {device}...")
    
    # 创建 Agent 实例
    if algorithm in ["DQN", "SARSA", "Double-DQN"]:
        # 这些算法需要 buffer_size, batch_size 等参数
        # 注意：这里假设这些 Agent 的 __init__ 接受位置参数
        # 最好检查一下 Code/DQN/agent.py 等
        # 暂时保持原样，如果不报错
        agent = AgentClass(action_dim, lr, 10000, 64, gamma, 1000, device)
    else:
        # A2C, REINFORCE, PPO
        # 使用关键字参数以避免位置错误
        agent = AgentClass(action_dim, lr=lr, gamma=gamma, device=device)
    
    # 加载模型权重
    if model_path and os.path.exists(model_path):
        if algorithm in ["DQN", "SARSA", "Double-DQN"]:
            agent.q_net.load_state_dict(torch.load(model_path, map_location=device))
            agent.q_net.eval()
        elif algorithm == "A2C":
            agent.network.load_state_dict(torch.load(model_path, map_location=device))
            agent.network.eval()
        elif algorithm == "REINFORCE":
            agent.network.load_state_dict(torch.load(model_path, map_location=device))
            agent.network.eval()
        print(f"✓ 成功加载模型: {model_path}")
    else:
        print(f"⚠ 模型文件不存在: {model_path}，使用随机初始化的模型")
    
    return agent, algorithm


def get_action_from_agent(agent, algorithm: str, obs: np.ndarray, use_framestack: bool, frame_stacker: Optional[FrameStacker] = None, epsilon: float = 0.0):
    """
    从 Agent 获取动作
    """
    if use_framestack:
        # 更新帧历史
        frame_stacker.update(obs)
        # 获取堆叠的观察
        stacked_obs = frame_stacker.get_stack()
        
        if algorithm in ["DQN", "SARSA", "Double-DQN"]:
            # 离散动作模型
            action = agent.get_action(stacked_obs, epsilon)
            return action
        else:
            # 连续动作模型（不应该使用 FrameStack，但为了兼容性）
            return agent.get_action(stacked_obs)
    else:
        # 单帧模型
        if algorithm in ["A2C", "REINFORCE"]:
            return agent.get_action(obs)
        else:
            # 离散动作模型但不用 FrameStack（不应该发生）
            return agent.get_action(obs, epsilon)


def get_human_action(keys: pygame.key.ScancodeWrapper) -> np.ndarray:
    """
    从键盘输入获取人类玩家的动作
    """
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # 转向
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        action[0] = -1.0
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        action[0] = 1.0
    
    # 油门
    if keys[pygame.K_UP] or keys[pygame.K_w]:
        action[1] = 1.0
    
    # 刹车
    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        action[2] = 1.0
    
    return action


def find_latest_model(model_dir: str, pattern: str = "*final*.pth") -> Optional[str]:
    """查找最新的模型文件"""
    if not os.path.exists(model_dir):
        return None
    
    import glob
    model_files = glob.glob(os.path.join(model_dir, pattern))
    if not model_files:
        # 尝试查找 checkpoint
        model_files = glob.glob(os.path.join(model_dir, "*checkpoint*.pth"))
    
    if not model_files:
        return None
    
    # 返回最新的文件（按修改时间）
    return max(model_files, key=os.path.getmtime)


def run_race(
    mode: str = "agent_vs_agent",
    car0_config: Optional[Dict] = None,
    car1_config: Optional[Dict] = None,
    num_episodes: int = 5,
    max_steps: int = 1000,
    render: bool = True,
):
    """
    运行多车比赛
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if device.type == "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    print(f"使用设备: {device}")
    print(f"Device type: {device.type}, str: {str(device)}")
    print(f"Torch version: {torch.__version__}")
    
    # 创建环境
    env = MultiCarRacing(
        num_agents=2,
        continuous=True,  # 统一使用连续动作空间
        render_mode="human" if render else None
    )
    
    # 加载模型
    agents = [None, None]
    algorithms = [None, None]
    use_framestack = [False, False]
    frame_stackers = [None, None]
    
    def setup_agent(idx, config):
        if not config: return
        
        algo = config.get("algorithm", "A2C")
        model_path = config.get("model_path")
        
        if not model_path:
            # 自动查找
            algo_folder_map = {
                "A2C": "A2C",
                "REINFORCE": "REINFORCE",
                "DQN": "DQN",
                "SARSA": "N-Step_SARSA",
                "Double-DQN": "Double_DQN",
                "PPO": "PPO/PPO"
            }
            if algo in algo_folder_map:
                model_dir = os.path.join(
                    os.path.dirname(__file__),
                    "Code",
                    algo_folder_map[algo],
                    "models"
                )
                if not os.path.exists(model_dir):
                     # Try without Code
                     model_dir = os.path.join(
                        os.path.dirname(__file__),
                        algo_folder_map[algo],
                        "models"
                    )
                model_path = find_latest_model(model_dir)
        
        # 确定 Action Dim
        if algo in ["DQN", "SARSA", "Double-DQN"]:
            action_dim = 5
        else:
            action_dim = 3
            
        agents[idx], algorithms[idx] = load_agent(
            algo,
            model_path,
            action_dim,
            config.get("lr", 0.0003),
            config.get("gamma", 0.99),
            device
        )
        use_framestack[idx] = (algo in ["DQN", "SARSA", "Double-DQN"])
        if use_framestack[idx]:
            frame_stackers[idx] = FrameStacker()
        print(f"车辆{idx}: {algo} 模型")

    # 车辆0配置
    if mode == "human_vs_agent":
        print("车辆0: 人类玩家（使用 WASD 或方向键控制）")
        agents[0] = None
    else:
        setup_agent(0, car0_config)
    
    # 车辆1配置
    setup_agent(1, car1_config)
    
    # 记录统计信息
    total_rewards = [[], []]
    
    print(f"\n开始比赛，共 {num_episodes} 回合...")
    print("=" * 50)
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        
        # 重置帧历史
        for i in range(2):
            if frame_stackers[i] is not None:
                frame_stackers[i].reset()
                # 初始化时填入第一帧
                frame_stackers[i].update(observations[i])
        
        episode_rewards = [0.0, 0.0]
        
        running = True
        step_count = 0
        
        while running and step_count < max_steps:
            # 处理 pygame 事件（用于人类输入和窗口关闭）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # 获取动作
            actions = []
            for i in range(2):
                if agents[i] is None:
                    # 人类玩家
                    keys = pygame.key.get_pressed()
                    action = get_human_action(keys)
                else:
                    # AI Agent
                    action = get_action_from_agent(
                        agents[i],
                        algorithms[i],
                        observations[i],
                        use_framestack[i],
                        frame_stackers[i],
                        epsilon=0.0  # 测试时使用贪婪策略
                    )
                    
                    # 如果是离散动作（整数），转换为连续动作
                    if isinstance(action, (int, np.integer)):
                        action = discrete_to_continuous(action)
                        
                actions.append(action)
            
            # 执行动作
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # 更新帧历史
            for i in range(2):
                if frame_stackers[i] is not None:
                    frame_stackers[i].update(observations[i])
            
            # 累计奖励
            for i in range(2):
                episode_rewards[i] += rewards[i]
            
            # 检查是否有车完成比赛
            lap_finished = any(info.get("lap_finished", False) for info in infos)
            if lap_finished:
                print("\n🏁 比赛结束！有车辆完成比赛。")
                break
                
            # 检查是否所有车都终止（例如出界或完成）
            if all(terminations) or all(truncations):
                print("\n🛑 比赛结束！所有车辆终止。")
                break
            
            step_count += 1
        
        # 记录本轮奖励
        print(f"回合 {episode+1}/{num_episodes}:")
        for i in range(2):
            total_rewards[i].append(episode_rewards[i])
            player_name = "人类" if agents[i] is None else f"Agent({algorithms[i]})"
            lap_status = "完成" if infos[i].get("lap_finished", False) else "未完成"
            print(f"  {player_name}: {episode_rewards[i]:.2f} ({lap_status})")
    
    # 打印统计结果
    print("\n" + "=" * 50)
    print("=== 比赛结果统计 ===")
    print("=" * 50)
    
    for i in range(2):
        player_name = "人类" if agents[i] is None else f"Agent({algorithms[i]})"
        rewards_list = total_rewards[i]
        if rewards_list:
            avg_reward = np.mean(rewards_list)
            std_reward = np.std(rewards_list)
            max_reward = np.max(rewards_list)
            min_reward = np.min(rewards_list)
            
            print(f"\n{player_name}:")
            print(f"  平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
            print(f"  最高奖励: {max_reward:.2f}")
            print(f"  最低奖励: {min_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多车比赛游戏")
    parser.add_argument("--mode", type=str, default="agent_vs_agent", 
                       choices=["agent_vs_agent", "human_vs_agent"],
                       help="比赛模式")
    parser.add_argument("--car0_algorithm", type=str, default="A2C",
                       choices=["DQN", "SARSA", "Double-DQN", "A2C", "REINFORCE"],
                       help="车辆0的算法")
    parser.add_argument("--car0_model", type=str, default=None,
                       help="车辆0的模型路径（None表示自动查找）")
    parser.add_argument("--car1_algorithm", type=str, default="REINFORCE",
                       choices=["DQN", "SARSA", "Double-DQN", "A2C", "REINFORCE"],
                       help="车辆1的算法")
    parser.add_argument("--car1_model", type=str, default=None,
                       help="车辆1的模型路径（None表示自动查找）")
    parser.add_argument("--episodes", type=int, default=5,
                       help="比赛回合数")
    parser.add_argument("--max_steps", type=int, default=1000,
                       help="每回合最大步数")
    parser.add_argument("--no_render", action="store_true",
                       help="不渲染（用于快速测试）")
    
    args = parser.parse_args()
    
    # 构建配置
    car0_config = None
    car1_config = None
    
    if args.mode == "human_vs_agent":
        # 人类 vs Agent
        car1_config = {
            "algorithm": args.car1_algorithm,
            "model_path": args.car1_model,
            "lr": 0.0003,
            "gamma": 0.99
        }
    else:
        # Agent vs Agent
        car0_config = {
            "algorithm": args.car0_algorithm,
            "model_path": args.car0_model,
            "lr": 0.0003,
            "gamma": 0.99
        }
        car1_config = {
            "algorithm": args.car1_algorithm,
            "model_path": args.car1_model,
            "lr": 0.0003,
            "gamma": 0.99
        }
    
    run_race(
        mode=args.mode,
        car0_config=car0_config,
        car1_config=car1_config,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=not args.no_render
    )
