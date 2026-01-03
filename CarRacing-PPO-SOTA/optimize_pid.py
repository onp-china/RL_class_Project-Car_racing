"""
PID 优化脚本包装器 - 从根目录调用
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入并运行实际脚本
from scripts.optimization.optimize_pid import optimize
import argparse

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

