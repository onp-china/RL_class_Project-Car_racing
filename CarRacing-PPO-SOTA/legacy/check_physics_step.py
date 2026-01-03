"""
检查物理更新频率：通过测量实际执行速度
分析为什么实际FPS只有122Hz左右
"""
import time
import gymnasium as gym
import numpy as np
from utils_env_sota import make_sota_env

def measure_actual_step_time(physics_fps=50, num_steps=1000):
    """测量实际每步执行时间"""
    print(f"\n测量 physics_fps={physics_fps} 时的实际执行时间:")
    print("-" * 60)
    
    env = make_sota_env(render_mode=None, frame_stack=4, frame_skip=1, physics_fps=physics_fps)
    state, _ = env.reset()
    action = np.array([0.0, 0.0, 0.0])
    
    times = []
    for i in range(num_steps):
        t0 = time.perf_counter()
        state, reward, done, truncated, _ = env.step(action)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # 转换为毫秒
        
        if done or truncated:
            state, _ = env.reset()
    
    env.close()
    
    avg_time_ms = np.mean(times)
    min_time_ms = np.min(times)
    max_time_ms = np.max(times)
    std_time_ms = np.std(times)
    actual_fps = 1000.0 / avg_time_ms
    
    print(f"执行 {num_steps} 步的统计:")
    print(f"  平均时间: {avg_time_ms:.3f} ms")
    print(f"  最小时间: {min_time_ms:.3f} ms")
    print(f"  最大时间: {max_time_ms:.3f} ms")
    print(f"  标准差: {std_time_ms:.3f} ms")
    print(f"  实际 FPS: {actual_fps:.1f} Hz")
    print(f"  目标 FPS: {physics_fps} Hz")
    print(f"  瓶颈: {'CPU/GPU性能' if actual_fps < physics_fps else '参数设置'}")
    
    return actual_fps, avg_time_ms

if __name__ == "__main__":
    print("=" * 60)
    print("物理执行时间分析")
    print("=" * 60)
    
    fps_50, time_50 = measure_actual_step_time(physics_fps=50, num_steps=500)
    fps_180, time_180 = measure_actual_step_time(physics_fps=180, num_steps=500)
    
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print(f"50Hz 设置: 实际 {fps_50:.1f} Hz, 每步 {time_50:.3f} ms")
    print(f"180Hz 设置: 实际 {fps_180:.1f} Hz, 每步 {time_180:.3f} ms")
    print(f"\n关键发现:")
    print(f"  1. 实际FPS受系统性能限制，约 {fps_50:.0f} Hz")
    print(f"  2. fps参数可能只控制渲染，不控制物理执行频率")
    print(f"  3. 物理频率 = 1 / 每步执行时间 = {fps_50:.1f} Hz")
    print(f"\n如果要达到180Hz物理频率:")
    print(f"  需要每步执行时间 < {1000/180:.2f} ms")
    print(f"  当前实际: {time_180:.3f} ms")
    print(f"  性能差距: {time_180 - 1000/180:.3f} ms ({((time_180 - 1000/180)/(1000/180)*100):.1f}%)")

