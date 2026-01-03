# CarRacing PPO SOTA

基于 PPO 算法的 CarRacing 环境强化学习项目，支持 120Hz 超高频控制和 PID 平滑优化。

## 📂 项目结构

```
CarRacing-PPO-SOTA/
├── README.md                    # 本文件
├── requirements_carracing.txt   # 依赖包
├── utils_env_sota.py            # 核心环境包装器
│
├── scripts/                     # 脚本目录
│   ├── visualization/          # 可视化脚本
│   │   ├── visualize_agent.py  # 主可视化脚本
│   │   └── run_120hz_demo.py   # 120Hz 演示
│   ├── optimization/           # 优化脚本
│   │   ├── optimize_pid.py     # PID 参数优化
│   │   └── test_pid_params.py  # 批量参数测试
│   ├── training/               # 训练脚本
│   │   └── train_improved.py   # PPO 训练
│   └── utils/                  # 工具脚本
│       ├── benchmark_inference.py  # 推理速度测试
│       └── verify_fps.py          # FPS 验证
│
├── docs/                       # 文档目录
│   ├── BEST_RESULTS.md         # 最佳测试结果
│   ├── PID_OPTIMIZATION_GUIDE.md  # PID 优化指南
│   ├── QUICK_START_PPO.md      # 快速开始
│   └── README_STRUCTURE.md     # 结构说明
│
├── algorithms/                 # 算法实现
│   └── ppo/                    # PPO 算法
├── saved_models/               # 模型权重
└── legacy/                     # 旧版本脚本（归档）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements_carracing.txt
```

### 2. 运行 120Hz 演示（推荐）

```bash
python run_120hz_demo.py
```

### 3. 手动运行可视化

```bash
python visualize_agent.py --episodes 3 --use_pid --physics_fps 120 --decision_freq 1 --pid_kp 0.22 --pid_ki 0.012 --pid_kd 0.20
```

## 🔧 常用命令

### 可视化相关
```bash
# 基本可视化
python visualize_agent.py --episodes 5 --use_pid

# 120Hz 超高频控制
python visualize_agent.py --episodes 5 --use_pid --physics_fps 120 --decision_freq 1 --pid_kp 0.22 --pid_ki 0.012 --pid_kd 0.20
```

### PID 优化
```bash
# 快速优化（推荐）
python optimize_pid.py --physics_fps 120 --decision_freq 1 --fast

# 完整优化
python optimize_pid.py --physics_fps 120 --decision_freq 1

# 批量测试参数
python test_pid_params.py --quick
```

### 训练
```bash
python scripts/training/train_improved.py
```

## 📊 最佳参数配置

基于测试结果，推荐使用以下参数：

**120Hz 超高频控制（最佳性能）**：
- Kp=0.22, Ki=0.012, Kd=0.20
- 平均奖励: 5740.39
- 最高奖励: 5806.77
- 标准差: 54.82

**最稳定配置**：
- Kp=0.20, Ki=0.01, Kd=0.20
- 标准差: 24.63（最稳定）

详细结果请查看 `docs/BEST_RESULTS.md`

## 📚 文档

- **`docs/PID_CONTROL_OPTIMIZATION.md`**: ⭐ **PID 控制优化完整记录**（推荐阅读）
- **`docs/BEST_RESULTS.md`**: 详细测试结果和参数对比
- **`docs/PID_OPTIMIZATION_GUIDE.md`**: PID 优化理论和实践
- **`docs/QUICK_START_PPO.md`**: 快速开始指南
- **`docs/ORGANIZATION.md`**: 项目文件结构说明

## 🎯 核心特性

1. **120Hz 超高频控制**: 物理、渲染、决策频率统一为 120Hz
2. **PID 平滑优化**: 自动搜索最佳 PID 参数，减少蛇形走位
3. **SOTA 奖励重塑**: 基于赛道几何的奖励机制
4. **快速优化模式**: 支持快速测试和完整优化两种模式

## 📝 注意事项

- 所有根目录的脚本都是包装器，实际脚本在 `scripts/` 目录下
- `utils_env_sota.py` 保留在根目录，因为被多个脚本引用
- 模型文件存放在 `saved_models/` 目录
