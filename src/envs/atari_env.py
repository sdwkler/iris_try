import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import numpy as np
import cv2

class AtariEnv:
    """
    Lightweight wrapper around gymnasium Atari envs.
    Returns stacked frames as a numpy array with shape (stack, H, W, C) or (stack, H, W) if grayscale.
    We also resize frames to image_size.
    """
    def __init__(self, env_id="ALE/Pong-v5", frame_stack=4, image_size=(84,84), render_mode=None, seed=None, grayscale=False):
        # Create env with optional render mode
        # Note: some gymnasium versions accept render_mode param in make
        make_kwargs = {}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        env = gym.make(env_id, **make_kwargs)
        # AtariPreprocessing: can convert to grayscale if desired
        env = AtariPreprocessing(env, grayscale_obs=grayscale, scale_obs=False)
        env = FrameStack(env, num_stack=frame_stack)
        self.env = env
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.grayscale = grayscale
        if seed is not None:
            # gymnasium reset seed on first reset usually
            self.env.reset(seed=seed)

    def _resize_stack(self, arr):
        # arr: np.array with shape (stack, H, W) or (stack, H, W, C) or LazyFrames
        arr = np.array(arr)  # convert LazyFrames to ndarray
        if arr.ndim == 3:
            # (stack, H, W) -> make channels=1
            arr = arr[..., None]
        resized = []
        for f in arr:
            # f shape H,W,C or H,W,1
            resized_f = cv2.resize(f, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            resized.append(resized_f)
        return np.stack(resized)  # (stack, H_new, W_new, C)

    def reset(self):
        obs, info = self.env.reset()
        return self._resize_stack(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self._resize_stack(obs), float(reward), bool(done), info

    @property
    def action_space_n(self):
        return self.env.action_space.n
