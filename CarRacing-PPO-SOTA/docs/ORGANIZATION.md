# 文件整理说明

## 📂 整理后的目录结构

### 根目录（保持简洁）
- **`visualize_agent.py`**: 可视化脚本包装器（实际脚本在 `scripts/visualization/`）
- **`optimize_pid.py`**: PID 优化脚本包装器（实际脚本在 `scripts/optimization/`）
- **`test_pid_params.py`**: 参数测试脚本包装器
- **`run_120hz_demo.py`**: 120Hz 演示脚本包装器
- **`utils_env_sota.py`**: 核心环境包装器（保留在根目录，被多个脚本引用）
- **`README.md`**: 主文档
- **`requirements_carracing.txt`**: 依赖包

### scripts/ 目录（按功能分类）

#### scripts/visualization/
- `visualize_agent.py`: 主可视化脚本
- `run_120hz_demo.py`: 120Hz 演示脚本

#### scripts/optimization/
- `optimize_pid.py`: PID 参数自动优化器
- `test_pid_params.py`: 批量测试多组 PID 参数

#### scripts/training/
- `train_improved.py`: PPO 模型训练脚本

#### scripts/utils/
- `benchmark_inference.py`: 模型推理速度测试
- `verify_fps.py`: FPS 验证工具

### docs/ 目录（所有文档）
- `BEST_RESULTS.md`: 最佳测试结果记录
- `PID_OPTIMIZATION_GUIDE.md`: PID 优化指南
- `QUICK_START_PPO.md`: 快速开始指南
- `README_STRUCTURE.md`: 结构说明（旧版）
- `ORGANIZATION.md`: 本文件

### legacy/ 目录（归档的旧版本）
- `visualize_agent_decision_freq2.py`: 旧版可视化脚本
- `optimize_pid_decision_freq2.py`: 旧版优化脚本
- `check_box2d_dt.py`: 调试脚本
- `check_physics_step.py`: 调试脚本

## 🔧 使用方式

### 从根目录运行（推荐）
所有常用脚本都可以从根目录直接运行，包装器会自动处理路径：

```bash
# 可视化
python visualize_agent.py --episodes 3 --use_pid

# PID 优化
python optimize_pid.py --physics_fps 120 --decision_freq 1 --fast

# 参数测试
python test_pid_params.py --quick

# 120Hz 演示
python run_120hz_demo.py
```

### 直接运行 scripts/ 下的脚本
也可以直接运行 scripts/ 目录下的脚本，它们已经配置好路径：

```bash
python scripts/visualization/visualize_agent.py --episodes 3
```

## 📝 整理原则

1. **保持向后兼容**: 根目录保留包装脚本，确保原有命令仍然可用
2. **清晰的分类**: 按功能将脚本分类到不同目录
3. **文档集中**: 所有文档统一放在 `docs/` 目录
4. **核心文件保留**: `utils_env_sota.py` 保留在根目录，因为被广泛引用
5. **旧版本归档**: 不再使用的脚本移至 `legacy/` 目录

## ✅ 整理完成

所有文件已按功能分类整理，import 路径已修复，可以正常使用。

