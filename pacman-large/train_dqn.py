import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from pacman_env import PacmanEnv  # ✅ 请确保 pacman_env.py 是兼容 gymnasium 的


# ✅ 自定义回调：每 log_freq 步输出累计 reward
class LogProgressCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.cumulative_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", 0.0)

        # 修复：如果 reward 是 ndarray，转 float
        if isinstance(reward, (np.ndarray, list)):
            reward = float(np.sum(reward))

        self.cumulative_reward += reward

        # 每 log_freq 步输出一次累计奖励
        if self.num_timesteps % self.log_freq == 0:
            print(f"[Step {self.num_timesteps}] Accumulated reward: {self.cumulative_reward:.2f}")
            self.cumulative_reward = 0.0

        return True


# ✅ 创建环境
env = PacmanEnv(visual=False)

# ✅ Gym 接口兼容性检查（只检查一次）
check_env(env, warn=True)

# ✅ 模型保存路径
model_save_path = "./pacman_dqn_agent"
os.makedirs(model_save_path, exist_ok=True)

# ✅ 定期保存模型
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_save_path,
    name_prefix="dqn_pacman_checkpoint"
)

# ✅ 自定义打印回调
log_callback = LogProgressCallback(log_freq=1000)

# ✅ 创建 DQN 模型
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
    # tensorboard_log="./pacman_tensorboard/"
)

# ✅ 开始训练
total_timesteps = 20000
model.learn(
    total_timesteps=total_timesteps,
    callback=[checkpoint_callback, log_callback]
)

# ✅ 保存最终模型
model.save(os.path.join(model_save_path, "pacman_dqn_final"))

# ✅ 清理资源
env.close()
print("训练完成 ✅，模型已保存！")
