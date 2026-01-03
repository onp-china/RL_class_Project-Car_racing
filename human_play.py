import gymnasium as gym
import pygame
import numpy as np


def play_game():
    # 创建环境，render_mode="human" 允许我们看到画面
    env = gym.make("CarRacing-v3", render_mode="human")

    print("🚗 人类驾驶模式启动！")
    print("🎮 操作说明：")
    print("   ⬆️ : 油门")
    print("   ⬇️ : 刹车")
    print("   ⬅️ : 左转")
    print("   ➡️ : 右转")
    print("   Esc : 退出")

    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    quit_game = False

    while not quit_game:
        # 1. 获取键盘输入
        # 动作格式: [方向盘(-1~1), 油门(0~1), 刹车(0~1)]
        action = np.array([0.0, 0.0, 0.0])

        # 必须先调用 render 或者手动处理事件，pygame 才能获取键盘状态
        env.render()

        # 获取按键状态
        keys = pygame.key.get_pressed()

        # 退出逻辑
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_game = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                quit_game = True

        if quit_game:
            break

        # 🎮 键盘映射逻辑
        if keys[pygame.K_LEFT]:
            action[0] = -1.0  # 左转满舵
        elif keys[pygame.K_RIGHT]:
            action[0] = +1.0  # 右转满舵

        if keys[pygame.K_UP]:
            action[1] = 1.0  # 油门到底

        if keys[pygame.K_DOWN]:
            action[2] = 0.8  # 刹车

        # 2. 环境执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # 3. 实时显示分数 (每20帧打印一次，防止刷屏)
        if steps % 20 == 0:
            print(f"\r当前得分: {total_reward:.2f}", end="")

        # 4. 游戏结束逻辑
        if terminated or truncated:
            print(f"\n🏁 游戏结束！最终得分: {total_reward:.2f}")
            # 重置环境，开始新的一局
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            print("🔄 新回合开始...")

    env.close()
    print("已退出游戏。")


if __name__ == "__main__":
    play_game()