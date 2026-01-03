# REINFORCE算法实现

基于PPO架构的REINFORCE（Monte Carlo Policy Gradient）算法实现，用于CarRacing-v3环境。

## 算法特点

### 核心特性
- **蒙特卡洛回报**：使用完整episode的回报进行更新
- **Episode-based**：只在episode结束时更新策略
- **可选Baseline**：支持使用value function作为baseline减少方差
- **并行环境**：支持多个环境并行收集轨迹

### 与PPO/A2C的区别
- **更新时机**：只在episode结束时更新（不是固定步数）
- **回报计算**：使用蒙特卡洛回报G_t，不是TD或GAE
- **无bootstrapping**：不依赖value估计来计算回报
- **高方差**：因为使用完整轨迹，方差通常较高

### 与PPO的相似
- **网络架构**：使用相同的Actor-Critic网络
- **连续动作**：使用Normal分布输出连续动作
- **混合精度训练**：支持AMP加速
- **进度条逻辑**：与PPO完全一致的实时进度显示

## 代码架构

```
REINFORCE/
├── agent.py          # REINFORCE智能体和网络定义
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
- Critic head: 可选的baseline（减少方差）
```

### 2. EpisodeBuffer (agent.py)
```python
- 存储完整episode轨迹
- 支持并行环境（每个环境独立追踪）
- 自动检测完成的episodes
```

### 3. REINFORCEAgent (agent.py)
```python
- 蒙特卡洛回报计算
- 可选value baseline
- 混合精度训练（AMP）
- 梯度裁剪（max_norm=0.5）
```

## 训练参数

### 默认参数
- **num_envs**: 8（并行环境数）
- **lr**: 3e-4（学习率）
- **gamma**: 0.99（折扣因子）
- **ent_coef**: 0.01（熵正则化系数）
- **use_baseline**: True（使用value baseline减少方差）

## 使用方法

### 训练
```bash
cd DRL-main/CarRacing/重构/REINFORCE
python main.py
```

### 自定义参数
```bash
python main.py --num_envs 16 --gamma 0.95 --use_baseline False
```

## 算法原理

REINFORCE是最基础的策略梯度算法：

### 1. 蒙特卡洛回报
```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^{T-t}*r_T
```

### 2. 策略梯度（无baseline）
```
∇J = E[∇log π(a_t|s_t) * G_t]
```

### 3. 策略梯度（with baseline）
```
∇J = E[∇log π(a_t|s_t) * (G_t - V(s_t))]
其中 V(s_t) 是baseline，减少方差但不影响期望
```

### 4. Loss函数
```
L = L_policy + 0.5 * L_value - 0.01 * H(π)
  = -E[log π * (G - V)] + 0.5 * MSE(V, G) - 0.01 * Entropy
```

## 优势与劣势

### 优势
1. **理论简单**：最原始的策略梯度算法
2. **无偏估计**：使用完整轨迹，不引入bias
3. **适合episodic任务**：自然地利用episode结构

### 劣势
1. **高方差**：完整轨迹的随机性导致方差很大
2. **样本效率低**：每个样本只用一次
3. **学习慢**：需要完整episode才能更新
4. **不适合长episode**：episode太长会导致训练困难

## 性能优化

### 1. Baseline减少方差
```python
use_baseline=True  # 使用V(s)作为baseline
```

### 2. 回报标准化
```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

### 3. 并行环境
```python
num_envs=8  # 同时收集多个轨迹，增加样本多样性
```

### 4. 熵正则化
```python
ent_coef=0.01  # 鼓励探索，避免过早收敛
```

## 文件说明

### agent.py
- `EpisodeBuffer`: 存储完整episode轨迹（支持并行环境）
- `ActorCriticNetwork`: 共享特征的Actor-Critic网络
- `REINFORCEAgent`: REINFORCE训练逻辑（蒙特卡洛更新）

### main.py
- 基于PPO的main.py框架
- 使用common模块的wrappers和utils
- 进度条逻辑与PPO一致（实时更新）
- 自动检测完成的episodes并更新

## 与其他算法的对比

| 特性 | REINFORCE | A2C | PPO |
|------|-----------|-----|-----|
| 更新频率 | Episode结束 | 每n步 | 每n步 |
| 回报估计 | Monte Carlo | GAE | GAE |
| 样本效率 | 低 | 中 | 高 |
| 方差 | 高 | 中 | 低 |
| 复杂度 | 简单 | 中等 | 复杂 |
| 稳定性 | 较差 | 中等 | 好 |

## 注意事项

1. **进度条**：与PPO一致，每步更新，显示FPS和奖励
2. **更新次数**：显示实际执行的策略更新次数（=完成的episodes数）
3. **适用场景**：更适合episode较短的环境
4. **调试建议**：如果训练不稳定，尝试启用baseline或减小学习率

## 调试技巧

1. **检查episode长度**：
   ```python
   # 如果episode太长，考虑使用A2C或PPO
   print(f"Avg episode length: {np.mean([len(ep) for ep in episodes])}")
   ```

2. **监控更新频率**：
   ```python
   # 显示在postfix中
   pbar.set_postfix_str(f"..., Updates: {num_updates}")
   ```

3. **方差分析**：
   ```python
   # 如果loss波动很大，考虑：
   # - 启用baseline (use_baseline=True)
   # - 增加并行环境数
   # - 调整gamma
   ```
