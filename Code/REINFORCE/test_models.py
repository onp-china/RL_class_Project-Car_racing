import os
import sys
import argparse
import numpy as np
import torch
import glob

# 修复OpenMP库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import test_policy, init_device

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import REINFORCEAgent


def find_models(models_dir):
    """查找models目录中的所有模型文件"""
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)  # 按修改时间排序，最新的在前
    return model_files


def extract_episode_number(filename):
    """从文件名中提取episode编号"""
    basename = os.path.basename(filename)
    if "checkpoint_ep" in basename:
        try:
            # 提取 "ep" 后面的数字
            start = basename.find("ep") + 2
            end = basename.find("_", start)
            if end == -1:
                end = basename.find(".", start)
            return int(basename[start:end])
        except:
            return 0
    elif "final" in basename:
        return 999999  # final模型排在最后
    return 0


def load_model(agent, model_path):
    """加载模型权重"""
    try:
        state_dict = torch.load(model_path, map_location=agent.device)
        agent.network.load_state_dict(state_dict)
        agent.network.eval()  # 设置为评估模式
        return True
    except Exception as e:
        print(f"  [ERROR] 加载模型失败: {e}")
        return False


def test_single_model(agent, model_path, env_name, num_episodes=3, render=True):
    """测试单个模型"""
    model_name = os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    if not load_model(agent, model_path):
        print(f"  [SKIP] 跳过模型 {model_name}")
        return False
    
    # 测试前检查（和训练脚本一样）
    print("\n[测试前检查] 验证模型输出...")
    test_state = np.zeros((1, 4, 84, 84), dtype=np.float32)
    test_action, _, _ = agent.get_action(test_state, deterministic=True)
    print(f"  测试动作输出: shape={test_action.shape}, mean={np.mean(test_action):.4f}, std={np.std(test_action):.4f}")
    print(f"  动作范围: min={np.min(test_action):.4f}, max={np.max(test_action):.4f}")
    
    # 使用和训练脚本相同的测试方法
    test_policy(agent, env_name, continuous=True, num_episodes=num_episodes, render=render)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="测试REINFORCE模型")
    
    # 模型选择
    parser.add_argument("--model_path", type=str, default=None, 
                       help="指定要测试的模型路径（如果不指定，则测试所有模型）")
    parser.add_argument("--models_dir", type=str, default=None,
                       help="模型目录路径（默认: Code/REINFORCE/models）")
    parser.add_argument("--test_final_only", action="store_true",
                       help="只测试final模型")
    parser.add_argument("--test_latest", action="store_true",
                       help="只测试最新的模型")
    
    # 测试参数
    parser.add_argument("--env_name", type=str, default="CarRacing-v3", help="环境名称")
    parser.add_argument("--num_episodes", type=int, default=3, help="每个模型测试的episode数量")
    parser.add_argument("--render", action="store_true", default=True, help="是否渲染")
    parser.add_argument("--no_render", dest="render", action="store_false", help="不渲染")
    
    # REINFORCE参数（需要和训练时一致）
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--use_baseline", type=bool, default=True, help="Use value baseline")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    
    args = parser.parse_args()
    
    # 确定模型目录
    if args.models_dir is None:
        current_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_path, "models")
    else:
        models_dir = args.models_dir
    
    if not os.path.exists(models_dir):
        print(f"[ERROR] 模型目录不存在: {models_dir}")
        return
    
    # 初始化设备
    device = init_device()
    
    # 创建智能体（需要知道action_dim，CarRacing-v3是3维连续动作）
    action_dim = 3  # [steering, gas, brake]
    agent = REINFORCEAgent(
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        use_baseline=args.use_baseline,
        max_grad_norm=args.max_grad_norm,
        device=device
    )
    
    # 确定要测试的模型
    if args.model_path:
        # 测试指定模型
        if not os.path.exists(args.model_path):
            print(f"[ERROR] 模型文件不存在: {args.model_path}")
            return
        model_files = [args.model_path]
    elif args.test_final_only:
        # 只测试final模型
        model_files = [f for f in find_models(models_dir) if "final" in os.path.basename(f)]
        if not model_files:
            print("[ERROR] 未找到final模型")
            return
        print(f"[INFO] 找到 {len(model_files)} 个final模型")
    elif args.test_latest:
        # 只测试最新的模型
        model_files = find_models(models_dir)
        if not model_files:
            print("[ERROR] 未找到任何模型")
            return
        model_files = [model_files[0]]  # 最新的一个
        print(f"[INFO] 测试最新模型: {os.path.basename(model_files[0])}")
    else:
        # 测试所有模型
        model_files = find_models(models_dir)
        if not model_files:
            print("[ERROR] 未找到任何模型")
            return
        # 按episode编号排序
        model_files.sort(key=extract_episode_number)
        print(f"[INFO] 找到 {len(model_files)} 个模型，将依次测试")
    
    # 测试所有选定的模型
    print(f"\n{'='*60}")
    print(f"开始测试 {len(model_files)} 个模型")
    print(f"{'='*60}")
    
    success_count = 0
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}]")
        if test_single_model(agent, model_path, args.env_name, args.num_episodes, args.render):
            success_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print(f"测试完成!")
    print(f"  成功测试: {success_count}/{len(model_files)} 个模型")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

