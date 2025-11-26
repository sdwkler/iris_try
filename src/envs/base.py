from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class Env(ABC):
    """环境基类，统一接口"""
    @property
    @abstractmethod
    def num_actions(self) -> int:
        """返回动作空间大小"""
        pass

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """返回环境数量（用于并行）"""
        pass

    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置环境，返回初始观测"""
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> tuple:
        """执行动作，返回(next_obs, reward, done, info)"""
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """渲染环境，返回图像"""
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭环境"""
        pass


class SingleProcessEnv(Env):
    """单进程环境包装器"""
    def __init__(self, env_fn):
        self.env = env_fn()
        # 适配gymnasium的Discrete动作空间
        assert isinstance(self.env.action_space, gym.spaces.Discrete)
        self._num_actions = self.env.action_space.n

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def num_envs(self) -> int:
        return 1

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        return np.array([obs])  # 增加批次维度

    def step(self, actions: np.ndarray) -> tuple:
        action = actions[0]  # 取第一个元素（因为单环境）
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return (
            np.array([obs]),
            np.array([reward]),
            np.array([done]),
            [info]
        )

    def render(self) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()


class MultiProcessEnv(Env):
    """多进程环境包装器，使用subprocess实现并行"""
    def __init__(self, env_fn, num_envs: int, should_wait_num_envs_ratio: float = 1.0):
        self.num_envs = num_envs
        self.envs = [env_fn() for _ in range(num_envs)]
        # 适配gymnasium的Discrete动作空间
        assert isinstance(self.envs[0].action_space, gym.spaces.Discrete)
        self._num_actions = self.envs[0].action_space.n
        self.should_wait_num_envs = max(1, int(should_wait_num_envs_ratio * num_envs))

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def reset(self) -> np.ndarray:
        obs_list = []
        for env in self.envs:
            obs, _ = env.reset()
            obs_list.append(obs)
        return np.array(obs_list)

    def step(self, actions: np.ndarray) -> tuple:
        results = []
        for env, action in zip(self.envs, actions):
            results.append(env.step(action))
        
        obs, rewards, terminated, truncated, infos = zip(*results)
        dones = [t or tr for t, tr in zip(terminated, truncated)]
        return (
            np.array(obs),
            np.array(rewards),
            np.array(dones),
            infos
        )

    def render(self) -> np.ndarray:
        # 只渲染第一个环境
        return self.envs[0].render()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @property
    def num_envs(self):
        return self._num_envs

    @num_envs.setter  # 添加 setter 允许赋值
    def num_envs(self, value):
        self._num_envs = value
