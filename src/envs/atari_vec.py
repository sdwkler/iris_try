import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import cv2

def make_atari_env(env_id, seed=None, idx=0, image_size=(84,84), frame_stack=4, grayscale=True, atari_frame_skip=4, render_mode=None):
    def _thunk():
        make_kwargs = {}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        # disable env-level frameskip
        make_kwargs["frameskip"] = 1
        env = gym.make(env_id, **make_kwargs)
        env = AtariPreprocessing(env, frame_skip=atari_frame_skip, grayscale_obs=grayscale, scale_obs=False)
        env = FrameStack(env, num_stack=frame_stack)
        return AtariResizeWrapper(env, image_size)
    return _thunk

class AtariResizeWrapper(gym.Wrapper):
    def __init__(self, env, image_size=(84,84)):
        super().__init__(env)
        self.image_size = image_size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._resize(obs), info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        return self._resize(obs), rew, term, trunc, info

    def _resize(self, obs):
        arr = np.array(obs)
        # Ensure shape (stack, H, W) or (H, W) -> convert to (stack, H, W, 1)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[0] > 1:
            arr = arr[..., None] if arr.ndim == 3 and arr.shape[2] != 1 else arr
        if arr.ndim == 3 and arr.shape[2] != 1 and arr.shape[0] != 1:
            # ambiguous shapes; try to ensure last dim is channel
            if arr.shape[0] == 4:
                # assume (stack,H,W)
                arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[2] == 1:
            pass
        # final: arr (stack,H,W,1) or (H,W,1)
        if arr.ndim == 3:
            arr = arr[..., None]
        resized = []
        for f in arr:
            resized.append(cv2.resize(f, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST))
        return np.stack(resized)