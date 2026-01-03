
"""
120Hz 超高频控制演示脚本
物理频率: 120Hz
决策频率: 120Hz (每帧预测)
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("启动 120Hz 超高频控制演示")
    print("=" * 60)
    print("配置:")
    print("  - 物理 FPS: 120 Hz")
    print("  - 决策频率: 120 Hz (每帧)")
    print("  - PID 参数: Kp=0.15, Ki=0.02, Kd=0.05 (初始推荐值)")
    print("-" * 60)
    
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    os.chdir(project_root)  # 切换到项目根目录
    
    cmd = [
        sys.executable,  # 使用当前 Python 解释器
        "visualize_agent.py",
        "--episodes", "3",
        "--use_pid",
        "--physics_fps", "120",
        "--decision_freq", "1",
        "--pid_kp", "0.15",
        "--pid_ki", "0.02",
        "--pid_kd", "0.05"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    # 使用 subprocess 更可靠
    subprocess.run(cmd)

if __name__ == "__main__":
    main()

