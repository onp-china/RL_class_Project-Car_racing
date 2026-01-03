"""
验证物理频率和渲染频率是否真的达到设定值
"""
import time
import gymnasium as gym
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from utils_env_sota import make_sota_env

def verify_fps(target_fps=180, test_duration=5.0):
    """
    验证环境是否真的以目标 FPS 运行
    
    Args:
        target_fps: 目标 FPS
        test_duration: 测试持续时间（秒）
    """
    print("=" * 60)
    print(f"验证物理和渲染频率: 目标 {target_fps} Hz")
    print("=" * 60)
    
    # 创建环境
    env = make_sota_env(render_mode=None, frame_stack=4, frame_skip=1, physics_fps=target_fps)
    
    # 检查 fps 参数
    if hasattr(env, 'unwrapped'):
        actual_fps_param = getattr(env.unwrapped, 'fps', None)
        print(f"环境 fps 参数: {actual_fps_param}")
        
        # 检查 Box2D 物理引擎的 dt
        if hasattr(env.unwrapped, 'world'):
            world = env.unwrapped.world
            if hasattr(world, 'dt'):
                print(f"Box2D dt 参数: {world.dt:.6f} (物理频率: {1.0/world.dt:.1f} Hz)")
        
        # 检查是否有其他控制物理频率的属性
        if hasattr(env.unwrapped, 'dt'):
            print(f"环境 dt 参数: {env.unwrapped.dt}")
        if hasattr(env.unwrapped, 'timeStep'):
            print(f"环境 timeStep 参数: {env.unwrapped.timeStep}")
    
    state, _ = env.reset()
    action = np.array([0.0, 0.0, 0.0])
    
    # 测试：运行指定时间，统计实际步数
    start_time = time.time()
    step_count = 0
    decision_count = 0
    
    while time.time() - start_time < test_duration:
        # 每帧决策（decision_freq=1）
        decision_count += 1
        state, reward, done, truncated, _ = env.step(action)
        step_count += 1
        
        if done or truncated:
            state, _ = env.reset()
    
    elapsed_time = time.time() - start_time
    actual_fps = step_count / elapsed_time
    actual_decision_fps = decision_count / elapsed_time
    
    print(f"\n[测试结果]")
    print(f"测试时长: {elapsed_time:.2f} 秒")
    print(f"总步数: {step_count}")
    print(f"实际物理 FPS: {actual_fps:.1f} Hz")
    print(f"实际决策频率: {actual_decision_fps:.1f} Hz")
    print(f"目标 FPS: {target_fps} Hz")
    print(f"误差: {abs(actual_fps - target_fps):.1f} Hz ({abs(actual_fps - target_fps)/target_fps*100:.1f}%)")
    
    # 判断是否达到目标
    if abs(actual_fps - target_fps) / target_fps < 0.1:  # 允许 10% 误差
        print(f"\n[OK] 物理频率验证通过！")
    else:
        print(f"\n[FAIL] 物理频率未达到目标！")
        print(f"   可能原因：")
        print(f"   1. fps 参数只控制渲染，不控制物理")
        print(f"   2. Box2D dt 参数需要单独设置")
        print(f"   3. 系统性能限制")
    
    # 检查决策频率
    if decision_count == step_count:
        print(f"\n[OK] 每帧决策验证通过！每秒预测 {actual_decision_fps:.1f} 次")
    else:
        print(f"\n[INFO] 决策频率: {actual_decision_fps:.1f} Hz (步数: {step_count}, 决策: {decision_count})")
    
    env.close()
    
    return actual_fps, actual_decision_fps

if __name__ == "__main__":
    # 测试 180Hz
    print("\n" + "=" * 60)
    print("测试 1: 180Hz 物理频率")
    print("=" * 60)
    fps_180, decision_180 = verify_fps(target_fps=180, test_duration=3.0)
    
    print("\n" + "=" * 60)
    print("测试 2: 50Hz 物理频率（对比）")
    print("=" * 60)
    fps_50, decision_50 = verify_fps(target_fps=50, test_duration=3.0)
    
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"50Hz 实际物理 FPS: {fps_50:.1f} Hz")
    print(f"180Hz 实际物理 FPS: {fps_180:.1f} Hz")
    print(f"物理频率比例: {fps_180/fps_50:.2f}x")
    print(f"\n50Hz 决策频率: {decision_50:.1f} Hz")
    print(f"180Hz 决策频率: {decision_180:.1f} Hz")
    print(f"决策频率比例: {decision_180/decision_50:.2f}x")
    
    if decision_180 >= 170:  # 允许一些误差
        print(f"\n[OK] 180Hz 下每帧决策验证通过！每秒预测约 {decision_180:.0f} 次")
    else:
        print(f"\n[FAIL] 180Hz 下决策频率未达到预期！")

