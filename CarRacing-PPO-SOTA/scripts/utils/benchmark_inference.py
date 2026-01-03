"""
模型推理速度基准测试
测量在不同配置下的推理 FPS，这对 PID 设计至关重要
"""
import argparse
import time
import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from algorithms.ppo.agent import PPOAgent

def benchmark_inference(args):
    """基准测试推理速度"""
    print("=" * 60)
    print("模型推理速度基准测试")
    print("=" * 60)
    
    # 创建 Agent
    agent = PPOAgent(
        state_dim=(4, 96, 96),
        action_dim=3,
        lr=3e-4
    )
    
    # 加载模型
    print(f"Loading model: {args.model}")
    agent.load(args.model)
    print("Model loaded!")
    
    # 准备测试数据（模拟真实状态）
    dummy_state = np.random.rand(4, 96, 96).astype(np.float32)
    
    # 预热（让 GPU/CUDA 初始化）
    print("\n[预热] 运行 10 次推理...")
    for _ in range(10):
        _ = agent.get_action(dummy_state, deterministic=True)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测试不同配置
    configs = [
        {"name": "FP32 (原始)", "use_fp16": False, "use_compile": False},
        {"name": "FP16", "use_fp16": True, "use_compile": False},
        {"name": "FP16 + Compiled", "use_fp16": True, "use_compile": True},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n[测试] {config['name']}")
        
        # 重新加载并应用配置
        agent.load(args.model)
        try:
            agent.enable_inference_mode(
                use_fp16=config['use_fp16'],
                use_compile=config['use_compile']
            )
        except AttributeError:
            # 如果方法不存在，手动设置
            agent.ac.eval()
            if config['use_fp16'] and agent.device.type == 'cuda':
                agent.ac.half()
            if config['use_compile'] and hasattr(torch, "compile"):
                agent.ac = torch.compile(agent.ac, mode="reduce-overhead")
        
        # 准备输入（如果是 FP16，需要转换为 half）
        use_fp16 = config['use_fp16'] and agent.device.type == 'cuda'
        if use_fp16:
            state_tensor = torch.FloatTensor(dummy_state).unsqueeze(0).to(agent.device).half()
        else:
            state_tensor = None  # 使用 get_action 方法
        
        # 预热
        for _ in range(10):
            if use_fp16:
                with torch.no_grad():
                    _ = agent.ac.get_action(state_tensor, deterministic=True)
                    _ = agent.ac.get_value(state_tensor)
            else:
                _ = agent.get_action(dummy_state, deterministic=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 正式测试
        num_runs = args.num_runs
        times = []
        
        start_time = time.time()
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t0 = time.time()
            
            if use_fp16:
                with torch.no_grad():
                    _ = agent.ac.get_action(state_tensor, deterministic=True)
                    _ = agent.ac.get_value(state_tensor)
            else:
                _ = agent.get_action(dummy_state, deterministic=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            t1 = time.time()
            times.append((t1 - t0) * 1000)  # 转换为毫秒
        
        total_time = time.time() - start_time
        
        # 统计
        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        min_time_ms = np.min(times)
        max_time_ms = np.max(times)
        fps = 1000.0 / avg_time_ms  # 每秒推理次数
        
        results.append({
            'config': config['name'],
            'avg_ms': avg_time_ms,
            'std_ms': std_time_ms,
            'min_ms': min_time_ms,
            'max_ms': max_time_ms,
            'fps': fps
        })
        
        print(f"  平均时间: {avg_time_ms:.2f} ms")
        print(f"  标准差: {std_time_ms:.2f} ms")
        print(f"  最小时间: {min_time_ms:.2f} ms")
        print(f"  最大时间: {max_time_ms:.2f} ms")
        print(f"  推理 FPS: {fps:.1f} Hz")
        print(f"  总时间: {total_time:.2f} s ({num_runs} 次)")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"{'配置':<20} {'平均时间(ms)':<15} {'推理FPS':<15} {'推荐决策频率'}")
    print("-" * 60)
    
    for r in results:
        # 推荐决策频率：推理 FPS 的 1/2（留有余量）
        if r['fps'] > 0:
            max_decision_freq = r['fps'] / 2
            recommended_freq_50hz = max(1, int(50 / max_decision_freq)) if max_decision_freq > 0 else 1
            recommended_freq_180hz = max(1, int(180 / max_decision_freq)) if max_decision_freq > 0 else 1
        else:
            recommended_freq_50hz = 999
            recommended_freq_180hz = 999
        
        print(f"{r['config']:<20} {r['avg_ms']:>10.2f} ms    {r['fps']:>10.1f} Hz    "
              f"50Hz:每{recommended_freq_50hz}帧, 180Hz:每{recommended_freq_180hz}帧")
    
    # 关键结论
    best_result = max(results, key=lambda x: x['fps'])
    print(f"\n[最佳配置] {best_result['config']}")
    print(f"  推理速度: {best_result['fps']:.1f} Hz")
    print(f"  单次推理: {best_result['avg_ms']:.2f} ms")
    
    # 分析不同物理 FPS 下的限制
    print(f"\n[决策频率限制分析]")
    max_decision_freq = best_result['fps'] / 2  # 保守估计，留 50% 余量
    
    for physics_fps in [50, 100, 120, 180]:
        if max_decision_freq > 0:
            min_interval = max(1, int(physics_fps / max_decision_freq))
        else:
            min_interval = 999
        
        actual_decision_freq = physics_fps / min_interval if min_interval > 0 else 0
        
        print(f"  物理 FPS {physics_fps:3d} Hz: "
              f"最多 {int(max_decision_freq):3d} Hz 决策频率 "
              f"(每 {min_interval:2d} 帧决策一次, 实际 {actual_decision_freq:.1f} Hz)")
    
    # PID 设计建议
    print(f"\n[PID 设计建议]")
    print(f"基于推理速度 {best_result['fps']:.1f} Hz:")
    
    for physics_fps in [50, 180]:
        if max_decision_freq > 0:
            decision_interval = max(1, int(physics_fps / max_decision_freq))
        else:
            decision_interval = 999
        
        actual_freq = physics_fps / decision_interval if decision_interval > 0 else 0
        
        print(f"\n  物理 FPS {physics_fps} Hz:")
        print(f"    - 推荐决策频率: 每 {decision_interval} 帧决策一次")
        print(f"    - 实际决策频率: {actual_freq:.1f} Hz")
        print(f"    - PID 插值次数: {decision_interval - 1} 次")
        print(f"    - PID 作用: {'重要' if decision_interval > 2 else '较小'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型推理速度基准测试")
    parser.add_argument("--model", type=str, default="saved_models/ppo_sota_ep5120.pth",
                       help="模型文件路径")
    parser.add_argument("--num_runs", type=int, default=1000,
                       help="测试运行次数（越多越准确）")
    args = parser.parse_args()
    
    benchmark_inference(args)

