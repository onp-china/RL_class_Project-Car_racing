"""
PID 参数测试脚本包装器
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入并运行实际脚本
from scripts.optimization.test_pid_params import test_params

if __name__ == "__main__":
    # 定义测试参数组合
    test_params_list = [
        (0.2338, 0.0127, 0.1293, "当前最佳（基准）"),
        (0.18, 0.01, 0.18, "降低Kp+提高Kd（平滑）"),
        (0.20, 0.01, 0.20, "平衡Kp+Kd（平衡）"),
        (0.15, 0.01, 0.15, "保守Kp+Kd（更平滑）"),
        (0.16, 0.012, 0.16, "微调保守参数"),
        (0.22, 0.012, 0.20, "提高Kd（增加阻尼）"),
        (0.20, 0.01, 0.22, "更高Kd（强阻尼）"),
        (0.18, 0.012, 0.13, "降低Kp保持Kd"),
        (0.19, 0.01, 0.13, "微降Kp保持Kd"),
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        test_params_list = [
            (0.2338, 0.0127, 0.1293, "当前最佳（基准）"),
            (0.18, 0.01, 0.18, "降低Kp+提高Kd"),
            (0.20, 0.01, 0.20, "平衡参数"),
        ]
    
    test_params(test_params_list, physics_fps=120, decision_freq=1, episodes=3)

