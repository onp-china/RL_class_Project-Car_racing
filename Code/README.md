# 强化学习算法实现集合

本项目包含多种强化学习算法在 CarRacing-v3 环境下的实现。

## 📦 包含的算法

- **PPO** (Proximal Policy Optimization) - 策略优化算法
- **A2C** (Advantage Actor-Critic) - Actor-Critic算法
- **DDPG** (Deep Deterministic Policy Gradient) - 连续动作空间算法
- **Double DQN** - 深度Q网络改进版
- **N-Step SARSA** - 时序差分学习算法
- **REINFORCE** - 策略梯度算法

## 🏗️ 项目结构

```
Code/
├── common/              # 共享模块
│   ├── networks.py     # 网络架构定义
│   ├── utils.py        # 工具函数
│   └── wrappers.py     # 环境包装器
├── A2C/                # A2C算法实现
├── DDPG/               # DDPG算法实现
├── Double_DQN/         # Double DQN算法实现
├── N-Step_SARSA/       # N-Step SARSA算法实现
├── PPO/                # PPO算法实现
├── REINFORCE/          # REINFORCE算法实现
├── requirements.txt    # 依赖包列表
└── README.md           # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行算法

每个算法目录下都有独立的实现，进入对应目录运行：

```bash
# 例如运行PPO
cd PPO/PPO
python main.py

# 例如运行A2C
cd A2C
python main.py
```

## 📝 环境说明

所有算法都在 **CarRacing-v3** 环境下训练，这是一个连续控制的赛车游戏环境。

## 🔧 共享模块

- **common/networks.py**: 包含CNN编码器、Actor-Critic网络等共享网络架构
- **common/utils.py**: 包含模型保存、绘图、设备初始化等工具函数
- **common/wrappers.py**: 包含环境预处理包装器（灰度化、裁剪、帧堆叠等）

## 📊 输出文件

每个算法训练后会生成：
- `models/`: 保存的模型文件
- `plots/`: 训练曲线图和训练数据

## ⚙️ 系统要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（可选，用于GPU加速）

## 📚 更多信息

每个算法目录下都有详细的 README.md 文件，包含该算法的具体使用方法、参数说明和实现细节。

