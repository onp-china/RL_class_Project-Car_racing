"""
可视化脚本包装器 - 从根目录调用
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入并运行实际脚本
from scripts.visualization.visualize_agent import visualize
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化训练好的 PPO 模型")
    parser.add_argument("--model", type=str, default="saved_models/ppo_sota_ep5120.pth", 
                       help="Path to .pth model file")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of test episodes (连续跑多个赛道)")
    parser.add_argument("--frame_skip", type=int, default=4, 
                       help="Frame skip (必须匹配训练时的设置!)")
    parser.add_argument("--physics_fps", type=int, default=50,
                       help="物理引擎 FPS (默认: 50, 推荐尝试 120)")
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
