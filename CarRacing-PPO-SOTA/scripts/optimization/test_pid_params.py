"""
快速测试多组 PID 参数，用于减少蛇形跑动
"""
import subprocess
import sys
import os

def test_params(params_list, physics_fps=120, decision_freq=1, episodes=3):
    """
    测试多组 PID 参数
    
    Args:
        params_list: [(kp, ki, kd, description), ...] 参数列表
        physics_fps: 物理 FPS
        decision_freq: 决策频率
        episodes: 每个参数测试的回合数
    """
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)  # 切换到项目根目录
    
    print("=" * 80)
    print(f"PID 参数对比测试 - 120Hz 超高频控制")
    print(f"物理 FPS: {physics_fps} Hz, 决策频率: {decision_freq} Hz")
    print(f"每个参数组合测试 {episodes} 个回合")
    print("=" * 80)
    
    results = []
    
    for idx, (kp, ki, kd, desc) in enumerate(params_list, 1):
        print(f"\n{'='*80}")
        print(f"测试 {idx}/{len(params_list)}: {desc}")
        print(f"参数: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
        print(f"{'='*80}")
        
        cmd = [
            sys.executable,
            "visualize_agent.py",
            "--episodes", str(episodes),
            "--use_pid",
            "--physics_fps", str(physics_fps),
            "--decision_freq", str(decision_freq),
            "--pid_kp", str(kp),
            "--pid_ki", str(ki),
            "--pid_kd", str(kd)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            # 解析输出，提取统计信息
            output = result.stdout + result.stderr
            print(output)
            
            # 尝试提取平均奖励（如果输出格式一致）
            import re
            avg_match = re.search(r'平均奖励[：:]\s*([\d.]+)', output)
            max_match = re.search(r'最高奖励[：:]\s*([\d.]+)', output)
            min_match = re.search(r'最低奖励[：:]\s*([\d.]+)', output)
            std_match = re.search(r'标准差[：:]\s*([\d.]+)', output)
            
            if avg_match:
                avg_reward = float(avg_match.group(1))
                max_reward = float(max_match.group(1)) if max_match else 0
                min_reward = float(min_match.group(1)) if min_match else 0
                std_reward = float(std_match.group(1)) if std_match else 0
                
                results.append({
                    'desc': desc,
                    'kp': kp,
                    'ki': ki,
                    'kd': kd,
                    'avg': avg_reward,
                    'max': max_reward,
                    'min': min_reward,
                    'std': std_reward
                })
                
                print(f"\n[结果] 平均奖励: {avg_reward:.2f}, 最高: {max_reward:.2f}, 最低: {min_reward:.2f}, 标准差: {std_reward:.2f}")
            
        except Exception as e:
            print(f"测试失败: {e}")
            continue
        
        print("\n" + "-"*80)
        input("按 Enter 继续下一个测试...")
    
    # 打印总结
    if results:
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print(f"{'序号':<4} {'描述':<25} {'Kp':<8} {'Ki':<8} {'Kd':<8} {'平均奖励':<10} {'最高':<10} {'最低':<10} {'标准差':<10}")
        print("-"*80)
        
        for idx, r in enumerate(results, 1):
            print(f"{idx:<4} {r['desc']:<25} {r['kp']:<8.4f} {r['ki']:<8.4f} {r['kd']:<8.4f} "
                  f"{r['avg']:<10.2f} {r['max']:<10.2f} {r['min']:<10.2f} {r['std']:<10.2f}")
        
        # 找出最佳参数
        best_avg = max(results, key=lambda x: x['avg'])
        best_stable = min(results, key=lambda x: x['std'])
        
        print("\n" + "="*80)
        print("推荐参数")
        print("="*80)
        print(f"最高平均奖励: {best_avg['desc']}")
        print(f"  Kp={best_avg['kp']:.4f}, Ki={best_avg['ki']:.4f}, Kd={best_avg['kd']:.4f}")
        print(f"  平均奖励: {best_avg['avg']:.2f}, 标准差: {best_avg['std']:.2f}")
        print(f"\n最稳定: {best_stable['desc']}")
        print(f"  Kp={best_stable['kp']:.4f}, Ki={best_stable['ki']:.4f}, Kd={best_stable['kd']:.4f}")
        print(f"  平均奖励: {best_stable['avg']:.2f}, 标准差: {best_stable['std']:.2f}")

if __name__ == "__main__":
    # 定义测试参数组合
    # 格式: (Kp, Ki, Kd, 描述)
    test_params_list = [
        # 当前最佳参数（基准）
        (0.2338, 0.0127, 0.1293, "当前最佳（基准）"),
        
        # 方案1: 降低Kp，提高Kd（更平滑，减少蛇形）
        (0.18, 0.01, 0.18, "降低Kp+提高Kd（平滑）"),
        (0.20, 0.01, 0.20, "平衡Kp+Kd（平衡）"),
        
        # 方案2: 进一步降低Kp
        (0.15, 0.01, 0.15, "保守Kp+Kd（更平滑）"),
        (0.16, 0.012, 0.16, "微调保守参数"),
        
        # 方案3: 提高Kd（增加阻尼）
        (0.22, 0.012, 0.20, "提高Kd（增加阻尼）"),
        (0.20, 0.01, 0.22, "更高Kd（强阻尼）"),
        
        # 方案4: 降低Kp，保持Kd
        (0.18, 0.012, 0.13, "降低Kp保持Kd"),
        (0.19, 0.01, 0.13, "微降Kp保持Kd"),
    ]
    
    # 可以自定义参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            # 快速测试：只测试3组关键参数
            test_params_list = [
                (0.2338, 0.0127, 0.1293, "当前最佳（基准）"),
                (0.18, 0.01, 0.18, "降低Kp+提高Kd"),
                (0.20, 0.01, 0.20, "平衡参数"),
            ]
    
    test_params(test_params_list, physics_fps=120, decision_freq=1, episodes=3)

