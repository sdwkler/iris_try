import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, device='cpu'):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.device = device
        
        # 初始化缓冲区
        self.obs_buf = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.logps = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        
        # 松弛计数：记录实际插入的step数（而非总数据数）
        self.inserted_steps = 0  # 关键：改为统计已插入的step数（0~num_steps）
        self.is_full = False     # 标记是否完全填满

    def insert(self, step, obs, action, reward, done, logp, value):
        """插入数据（允许step不连续，适配环境死亡重置）"""
        if step >= self.num_steps:
            print(f"[WARNING] Step {step} exceeds buffer size {self.num_steps}, skip insert")
            return
        
        # 校验核心维度（放宽非关键校验）
        if obs.shape == (self.num_envs,) + self.obs_shape and action.shape == (self.num_envs,):
            self.obs_buf[step] = obs
            self.actions[step] = action
            self.rewards[step] = reward
            self.dones[step] = done
            self.logps[step] = logp
            self.values[step] = value
            self.inserted_steps = max(self.inserted_steps, step + 1)  # 更新已插入的最大step
            self.is_full = (self.inserted_steps == self.num_steps)
        else:
            print(f"[WARNING] Data shape mismatch at step {step}, skip insert")
            print(f"Obs shape: {obs.shape}, expected: {(self.num_envs,) + self.obs_shape}")
            print(f"Action shape: {action.shape}, expected: {(self.num_envs,)}")

    def get(self):
        """返回实际插入的有效数据（而非全部缓冲区）"""
        # 只返回已插入的step范围（0~inserted_steps）
        valid_obs = self.obs_buf[:self.inserted_steps]
        valid_actions = self.actions[:self.inserted_steps]
        valid_rewards = self.rewards[:self.inserted_steps]
        valid_dones = self.dones[:self.inserted_steps]
        valid_logps = self.logps[:self.inserted_steps]
        valid_values = self.values[:self.inserted_steps]
        print(f"[Montezuma Info] Buffer used: {self.inserted_steps}/{self.num_steps} steps, {self.inserted_steps*self.num_envs}/{self.num_steps*self.num_envs} samples")

        # 检查数值有效性（保留关键校验）
        assert not np.isnan(valid_obs).any(), "Obs contains NaN (Montezuma buffer)"
        assert not np.isinf(valid_obs).any(), "Obs contains Inf (Montezuma buffer)"
        return valid_obs, valid_actions, valid_rewards, valid_dones, valid_logps, valid_values

    def reset(self):
        """重置缓冲区（保留未填满的数据标记）"""
        self.obs_buf.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.dones.fill(0)
        self.logps.fill(0)
        self.values.fill(0)
        self.inserted_steps = 0
        self.is_full = False
