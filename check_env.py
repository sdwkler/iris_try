import gymnasium

# 尝试创建 ALE Atari 环境
env = gymnasium.make("ALE/MontezumaRevenge-v5", render_mode="rgb_array")

print("✅ 成功创建 ALE/MontezumaRevenge-v5 环境！")
print("观察空间:", env.observation_space)
print("动作空间:", env.action_space)

# 取一帧观察，看看是否正常
obs, _ = env.reset()
print("初始观测形状:", obs.shape)

env.close()