from stable_baselines3 import DQN
from pacman_env import PacmanEnv
import time

# ✅ 创建带可视化的环境
env = PacmanEnv(visual=True)

# ✅ 加载训练好的模型（请确保路径正确）
model = DQN.load("pacman_dqn_agent/pacman_dqn_final")

# ✅ 正确解包 obs
obs, _ = env.reset()

# ✅ 游戏循环
for step in range(2000):
    # 推理动作
    action, _ = model.predict(obs, deterministic=True)

    # 执行动作并获取环境反馈
    obs, reward, terminated, truncated, _ = env.step(action)

    # 可视化
    env.render()

    # ✅ 打印当前步数与奖励
    print(f"Step {step} | Reward: {reward}")

    # 可选：控制播放速度
    time.sleep(0.03)

    # 若游戏结束则重置
    if terminated or truncated:
        print("Game over. Resetting...\n")
        obs, _ = env.reset()

# 清理资源
env.close()
