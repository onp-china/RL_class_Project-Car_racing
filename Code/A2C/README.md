# A2C算法实现

基于PPO架构的A2C（Advantage Actor-Critic）算法实现，用于CarRacing-v3环境。

## 算法特点

### 与PPO的区别
- **单次更新**：A2C在收集完rollout后只更新一次，不像PPO会进行多轮（n_epochs）更新
- **无策略裁剪**：不使用PPO的clipping机制，直接优化策略梯度
- **更小的rollout**：使用256步rollout（PPO使用2048步）

### 与PPO的相似
- **网络架构**：使用相同的Actor-Critic网络（Nature DQN特征提取器 + 共享FC层）
- **GAE优势估计**：使用相同的GAE（Generalized Advantage Estimation）
- **连续动作**：使用Normal分布输出连续动作
- **混合精度训练**：支持AMP加速
- **进度条逻辑**：与PPO完全一致的实时进度显示

## 代码架构

```
A2C/
├── agent.py          # A2C智能体和网络定义
├── main.py           # 训练脚本（基于PPO框架）
└── README.md         # 本文件
```

### 复用的Common模块
- `common.wrappers`: 环境预处理（灰度化、裁剪、帧堆叠）
- `common.utils`: 工具函数（保存模型、绘图、测试策略）

## 核心组件

### 1. ActorCriticNetwork (agent.py)
```python
- 特征提取器: Nature DQN架构（3层CNN）
- 共享FC层: 512维隐藏层
- Actor head: 输出动作均值和学习的log_std
- Critic head: 输出状态价值
```

### 2. RolloutBuffer (agent.py)
```python
- 高性能预分配内存
- 支持GAE计算
- 自动标准化advantages
```

### 3. A2CAgent (agent.py)
```python
- 混合精度训练（AMP）
- AdamW优化器（weight decay=0.01）
- 梯度裁剪（max_norm=0.5）
- 单次更新策略
```

## 训练参数

### 默认参数
- **n_steps**: 256（比PPO小，更频繁更新）
- **num_envs**: 8（并行环境数）
- **lr**: 3e-4（学习率）
- **gamma**: 0.99（折扣因子）
- **gae_lambda**: 0.95（GAE参数）
- **vf_coef**: 0.5（value loss系数）
- **ent_coef**: 0.01（熵正则化系数）

## 使用方法

### 训练
```bash
cd DRL-main/CarRacing/重构/A2C
python main.py
```

### 自定义参数
```bash
python main.py --num_envs 16 --n_steps 512 --lr 1e-4
```

## 与PPO的性能对比

**A2C优势**：
- 更新更频繁，学习更快
- 计算效率更高（无需多轮epoch）
- 适合需要快速反应的环境

**A2C劣势**：
- 样本效率较低（不重复使用数据）
- 可能不如PPO稳定（无clipping保护）
- 对超参数更敏感

## 文件说明

### agent.py
- `RolloutBuffer`: n-step轨迹存储，支持GAE
- `ActorCriticNetwork`: 共享特征的Actor-Critic网络
- `A2CAgent`: A2C训练逻辑（单次更新，无clipping）

### main.py
- 完全基于PPO的main.py框架
- 使用common模块的wrappers和utils
- 进度条逻辑与PPO一致（实时更新，流畅显示）
- 自动保存模型和奖励曲线

## 算法原理

A2C使用Actor-Critic架构：
1. **Actor**：学习策略π(a|s)，输出动作
2. **Critic**：学习价值函数V(s)，评估状态
3. **优势函数**：A(s,a) = Q(s,a) - V(s)，用GAE估计
4. **策略梯度**：∇J = E[∇log π(a|s) * A(s,a)]

**Loss函数**：
```
L = L_policy + 0.5 * L_value - 0.01 * H(π)
  = -E[log π * A] + 0.5 * MSE(V, R) - 0.01 * Entropy
```

## 注意事项

1. **进度条**：与PPO一致，每步更新，流畅显示FPS和奖励
2. **更新频率**：由于n_steps较小（256），更新更频繁
3. **内存占用**：比PPO小（rollout buffer更小）
4. **收敛速度**：通常比PPO快，但可能不够稳定
