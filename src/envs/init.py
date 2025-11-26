from gymnasium import make as gym_make
from gymnasium.wrappers import AtariPreprocessing,FrameStack
from .wrappers import TetrisObservationWrapper, RewardScalingWrapper, MaxEpisodeStepsWrapper


def make_tetris(size=64, max_episode_steps=10000, reward_scale=1.0):
    # 使用gymnasium的Atari Tetris环境
    env = gym_make('ALE/Tetris-v5', render_mode="rgb_array")
    # 添加Atari预处理（帧堆叠等）
    env = AtariPreprocessing(
        env,
        screen_size=size,
        grayscale_obs=False,  # 保留RGB以便后续处理
        frame_skip=1,
        noop_max=30
    )
    env = TetrisObservationWrapper(env, size=size)
    env = FrameStack(env, num_stack=4)  # 新增帧堆叠，输入维度变为 (4, size, size)
    env = RewardScalingWrapper(env, scale=reward_scale)
    env = MaxEpisodeStepsWrapper(env, max_steps=max_episode_steps)
    return env


# 暴露环境相关组件
__all__ = [
    'make_tetris',
    'TetrisObservationWrapper',
    'RewardScalingWrapper',
    'MaxEpisodeStepsWrapper'
]
