import numpy as np
from pathlib import Path
from PIL import Image

class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add_step(self, observation: np.ndarray, action: int, reward: float, done: bool):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_sequence(self, start_idx: int, end_idx: int) -> dict:
        return {
            'observations': np.array(self.observations[start_idx:end_idx]),
            'actions': np.array(self.actions[start_idx:end_idx-1]),  # 动作比观测少1步
            'rewards': np.array(self.rewards[start_idx:end_idx-1]),
            'dones': np.array(self.dones[start_idx:end_idx-1])
        }

    def save(self, save_dir: Path):
        save_dir.mkdir(exist_ok=True)
        # 保存观测图像
        obs_dir = save_dir / 'observations'
        obs_dir.mkdir(exist_ok=True)
        for i, obs in enumerate(self.observations):
            Image.fromarray(obs).save(obs_dir / f'{i}.png')
        # 保存动作/奖励/终止标志
        np.save(save_dir / 'actions.npy', self.actions)
        np.save(save_dir / 'rewards.npy', self.rewards)
        np.save(save_dir / 'dones.npy', self.dones)

    def __len__(self) -> int:
        return len(self.observations)
