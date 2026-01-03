# CarRacing-PPO-SOTA 项目结构说明

## 📂 目录结构

### 🚀 核心脚本
- **`run_120hz_demo.py`**: **【入口】** 一键运行 120Hz 超高频控制演示（使用最佳参数）。
- **`visualize_agent.py`**: 可视化脚本，支持加载模型、PID 控制、调整物理/决策频率。
- **`optimize_pid.py`**: PID 参数自动优化器，用于搜索最佳 Kp/Ki/Kd 参数。
- **`test_pid_params.py`**: 批量测试多组 PID 参数，生成对比报告。
- **`train_improved.py`**: PPO 模型训练脚本。

### 🛠️ 工具模块
- **`utils_env_sota.py`**: **【核心】** SOTA 环境包装器，包含奖励重塑、Box2D 物理参数设置。
- **`benchmark_inference.py`**: 测试模型推理速度和系统性能。
- **`verify_fps.py`**: 验证实际物理 FPS 和决策频率是否达标。

### 📚 文档
- **`BEST_RESULTS.md`**: 记录最佳测试结果、参数配置和性能对比。
- **`PID_OPTIMIZATION_GUIDE.md`**: PID 优化指南和理论说明。
- **`QUICK_START_PPO.md`**: 快速开始指南。

### 📁 子目录
- **`algorithms/`**: PPO 算法实现细节。
- **`saved_models/`**: 存放训练好的模型权重（如 `ppo_sota_ep5120.pth`）。
- **`legacy/`**: 归档的旧版本脚本（如 `*_decision_freq2.py`）。

---

## 🔧 常用命令速查

### 1. 运行 120Hz 演示（推荐）
```bash
python run_120hz_demo.py
```

### 2. 手动运行可视化
```bash
python visualize_agent.py --episodes 3 --use_pid --physics_fps 120 --decision_freq 1 --pid_kp 0.22 --pid_ki 0.012 --pid_kd 0.20
```

### 3. 自动优化 PID 参数
```bash
python optimize_pid.py --physics_fps 120 --decision_freq 1 --fast
```

### 4. 批量测试多组参数
```bash
python test_pid_params.py --quick
```

