# PPO (Proximal Policy Optimization)

> 连续动作控制算法 - CarRacing 最推荐方案 ⭐⭐⭐⭐⭐

## 📖 算法简介

PPO 是目前工业界最常用的强化学习算法之一，由 OpenAI 在 2017 年提出。它是 Policy Gradient 算法的改进版本，通过限制策略更新幅度来保证训练稳定性。

### 优势

- ✅ **连续动作空间**：直接输出平滑的方向盘/油门/刹车控制
- ✅ **训练稳定**：不容易崩溃，适合新手
- ✅ **样本效率高**：相比 DQN 更快收敛
- ✅ **工业标准**：OpenAI、DeepMind 等都在使用

---

## 🚀 快速开始

### 单环境训练

```bash
python -m algorithms.ppo.train --max_episodes 500
```

### 向量化快速训练（推荐）

```bash
python -m algorithms.ppo.train_fast --num_envs 8 --frame_skip 3 --max_episodes 500
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `agent.py` | PPO Agent 实现（RolloutBuffer, 训练逻辑） |
| `model.py` | Actor-Critic 神经网络（CNN + 策略网络 + 价值网络） |
| `env_wrapper.py` | 连续动作环境包装器（帧堆叠、Frame Skip） |
| `train.py` | 单环境训练脚本 |
| `train_fast.py` | 向量化快速训练脚本 ⭐ |

---

## ⚙️ 参数说明

### 训练参数

```bash
python -m algorithms.ppo.train_fast \
    --num_envs 8 \              # 并行环境数量（4-8 推荐）
    --frame_skip 3 \            # 帧跳跃（2-4 推荐）
    --max_episodes 500 \        # 总训练回合数
    --rollout_steps 512 \       # 每次采集步数
    --batch_size 64 \           # Mini-batch 大小
    --ppo_epochs 10 \           # 每次更新的训练轮数
    --lr 3e-4 \                 # 学习率
    --gamma 0.99 \              # 折扣因子
    --gae_lambda 0.95 \         # GAE lambda
    --clip_epsilon 0.2 \        # PPO 裁剪范围
    --save_freq 50 \            # 保存频率
    --eval_freq 100             # 评估频率
```

### 参数调优建议

| 参数 | 默认值 | 调优建议 |
|------|-------|---------|
| `lr` | 3e-4 | 如果不收敛降到 1e-4 |
| `clip_epsilon` | 0.2 | 训练不稳定可改为 0.1 |
| `ppo_epochs` | 10 | 样本少时可增加到 15-20 |
| `rollout_steps` | 512 | 内存够可增加到 1024-2048 |

---

## 📊 训练效果

### 典型训练曲线

```
Episode  100: Reward= 180.23, Avg100= 145.32
Episode  200: Reward= 425.67, Avg100= 298.45
Episode  300: Reward= 556.89, Avg100= 412.78
Episode  500: Reward= 678.90, Avg100= 567.89
```

### 性能基准

| 训练回合 | Avg100 分数 | 训练时间（8 envs） |
|---------|------------|------------------|
| 100 | 150-250 | ~10 分钟 |
| 300 | 400-500 | ~30 分钟 |
| 500 | 550-650 | ~50 分钟 |
| 1000 | 700-800+ | ~1.5-2 小时 |

---

## 🎯 使用示例

### 1. 训练新模型

```bash
# 标准配置（平衡速度与性能）
python -m algorithms.ppo.train_fast --num_envs 8 --frame_skip 3 --max_episodes 500

# 快速测试（200 episodes）
python -m algorithms.ppo.train_fast --num_envs 8 --frame_skip 4 --max_episodes 200

# 高质量训练
python -m algorithms.ppo.train_fast --num_envs 6 --frame_skip 2 --max_episodes 1000
```

### 2. 加载并测试模型

```python
from algorithms.ppo import PPOAgent, make_continuous_env

# 创建环境和智能体
env = make_continuous_env(render_mode="human")
agent = PPOAgent(state_dim=(4, 96, 96), action_dim=3)

# 加载模型
agent.load("../../saved_models/ppo/ppo_fast_carracing_ep500.pth")

# 测试
state, _ = env.reset()
for _ in range(1000):
    action, _, _ = agent.get_action(state, deterministic=True)
    state, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        break
```

### 3. 继续训练

```bash
# 从检查点继续（需要添加 --resume 参数）
python -m algorithms.ppo.train_fast --num_envs 8 --resume saved_models/ppo/ppo_fast_carracing_ep500.pth
```

---

## 🔧 自定义修改

### 修改网络结构

编辑 `model.py` 中的 `CNNBase`:

```python
class CNNBase(nn.Module):
    def __init__(self, input_channels=4):
        super().__init__()
        # 修改这里的网络层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
```

### 修改奖励函数

编辑 `env_wrapper.py` 中的 `step` 方法：

```python
def step(self, action):
    # ... 执行动作 ...
    
    # 自定义奖励塑造
    shaped_reward = total_reward
    if is_on_grass:
        shaped_reward -= 1.0  # 惩罚跑到草地
    if speed > threshold:
        shaped_reward += 0.1  # 奖励高速
    
    return self.frames.copy(), shaped_reward, terminated, truncated, info
```

---

## 📈 性能优化

### 加速技巧

1. **增加并行环境**
   ```bash
   --num_envs 12  # 如果 CPU 核心够
   ```

2. **调大 Frame Skip**
   ```bash
   --frame_skip 4  # 快 2 倍但精度稍降
   ```

3. **更大的 Batch Size**
   ```bash
   --batch_size 128  # 充分利用 GPU
   ```

4. **组合使用**
   ```bash
   python -m algorithms.ppo.train_fast --num_envs 12 --frame_skip 4 --batch_size 128
   # 预期加速：15-20x ⚡⚡⚡
   ```

---

## 🐛 常见问题

### Q: 训练不收敛，Reward 一直是负数？

**A:** 尝试：
1. 降低学习率：`--lr 1e-4`
2. 增加训练时间：至少 300+ episodes
3. 检查环境是否正常：运行 `human_play.py` 测试

### Q: 训练速度太慢？

**A:** 
1. 使用 `train_fast.py` 而不是 `train.py`
2. 增加 `--num_envs` 和 `--frame_skip`
3. 确保使用了 GPU：检查输出中的 "Device: cuda"

### Q: 内存不足？

**A:** 
1. 减少 `--num_envs`
2. 减少 `--rollout_steps`
3. 减少 `--batch_size`

---

## 📚 参考资料

- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [详细教程](../../docs/PPO_GUIDE.md)

---

**开始训练你的 PPO Agent！** 🚀

```bash
python -m algorithms.ppo.train_fast --num_envs 8 --max_episodes 500
```



