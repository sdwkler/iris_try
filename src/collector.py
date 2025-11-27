import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, device='cpu'):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_buf = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.logps = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)

    def insert(self, step, obs, action, reward, done, logp, value):
        self.obs_buf[step] = obs
        self.actions[step] = action
        self.rewards[step] = reward
        self.dones[step] = done
        self.logps[step] = logp
        self.values[step] = value

    def get(self):
        return self.obs_buf, self.actions, self.rewards, self.dones, self.logps, self.values