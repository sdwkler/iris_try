import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image


class TetrisObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=64):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.size, self.size, 3),
            dtype=np.uint8
        )

    def observation(self, obs):
        # 处理观测：调整大小并确保RGB格式
        img = Image.fromarray(obs).convert('RGB')
        img = img.resize((self.size, self.size), Image.LANCZOS)
        return np.array(img)


class RewardScalingWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class MaxEpisodeStepsWrapper(gym.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_steps = 0

    def reset(self, **kwargs):
        self.current_steps = 0
        return super().reset(** kwargs)

    def step(self, action):
        self.current_steps += 1
        obs, reward, terminated, truncated, info = super().step(action)
        if self.current_steps >= self.max_steps:
            truncated = True  # gymnasium使用truncated表示超时
            info['truncated'] = True
        return obs, reward, terminated, truncated, info
