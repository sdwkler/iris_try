import os
import numpy as np
from collections import deque
from tqdm import trange
import torch
from src.envs.tetris_env import TetrisEnv
from src.envs.atari_env import AtariEnv
from src.agent import VisualDQNAgent
from src.utils import save_plot

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        env_cfg = cfg.get("env", {})
        env_id = env_cfg.get("id", None)
        if env_id:
            # create Atari env
            self.env = AtariEnv(env_id=env_id,
                                frame_stack=env_cfg.get("frame_stack", 4),
                                image_size=tuple(env_cfg.get("image_size", [84,84])),
                                render_mode=env_cfg.get("render_mode", None),
                                seed=cfg.get("seed", None),
                                grayscale=env_cfg.get("grayscale", False))
            self.is_atari = True
        else:
            # fallback to Tetris
            self.env = TetrisEnv(height=env_cfg.get("height",20),
                                 width=env_cfg.get("width",10),
                                 seed=cfg.get("seed", None),
                                 render_frame=True,
                                 image_size=tuple(env_cfg.get("image_size",[84,84])),
                                 channels=env_cfg.get("frame_channels",3))
            self.is_atari = False

        coll_cfg = cfg.get("collector", {})
        self.seq_len = coll_cfg.get("frame_stack", 4)
        # observation shape for agent: (seq_len, C, H, W)
        if self.is_atari:
            C = 1 if env_cfg.get("grayscale", False) else 3
            H, W = env_cfg.get("image_size", [84,84])
        else:
            C = env_cfg.get("frame_channels", 3)
            H, W = env_cfg.get("image_size", [84,84])
        obs_shape = (self.seq_len, C, H, W)
        self.agent = VisualDQNAgent(cfg=cfg.get("agent", {}), obs_shape=obs_shape, action_n=self.env.action_space_n)
        self.output_dir = cfg.get("output_dir", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_rewards = []
        self.losses = []

    def _preprocess_single_frame(self, frame):
        arr = np.array(frame)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = np.transpose(arr, (2,0,1))  # C,H,W
        return arr

    def _stack_init(self, frame):
        buf = deque(maxlen=self.seq_len)
        arr = np.array(frame)
        if arr.ndim == 4:
            # (stack, H, W, C) or (stack, H, W, 1)
            frames = arr
        elif arr.ndim == 3 and arr.shape[0] == self.seq_len:
            frames = arr
        else:
            single = self._preprocess_single_frame(arr)
            for _ in range(self.seq_len):
                buf.append(single)
            return buf

        if frames.shape[0] >= self.seq_len:
            frames = frames[-self.seq_len:]
        else:
            pad_cnt = self.seq_len - frames.shape[0]
            first = frames[0:1]
            frames = np.concatenate([np.repeat(first, pad_cnt, axis=0), frames], axis=0)

        for f in frames:
            buf.append(self._preprocess_single_frame(f))
        return buf

    def run_episode(self, train=True, max_steps=1000):
        frame = self.env.reset()
        buf = self._stack_init(frame)
        total_reward = 0.0
        for step in range(max_steps):
            state = np.stack(buf)  # (seq_len, C, H, W)
            action = self.agent.select_action(state)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward

            arr = np.array(next_obs)
            if arr.ndim == 4:
                frames = arr
                if frames.shape[0] >= self.seq_len:
                    frames = frames[-self.seq_len:]
                else:
                    pad_cnt = self.seq_len - frames.shape[0]
                    first = frames[0:1]
                    frames = np.concatenate([np.repeat(first, pad_cnt, axis=0), frames], axis=0)
                new_buf = deque(maxlen=self.seq_len)
                for f in frames:
                    new_buf.append(self._preprocess_single_frame(f))
                next_state = np.stack(new_buf)
                buf = deque(new_buf, maxlen=self.seq_len)
            else:
                next_frame_proc = self._preprocess_single_frame(arr)
                buf.append(next_frame_proc)
                next_state = np.stack(buf)

            if train:
                self.agent.push(state, action, reward, next_state, float(done))
                loss = self.agent.optimize()
                if loss is not None:
                    self.losses.append(loss)
            if done:
                break
        return total_reward

    def train(self):
        episodes = self.cfg.get("training", {}).get("episodes", 2000)
        save_every = self.cfg.get("save_every", 200)
        print("Starting training for {} episodes".format(episodes))
        for ep in trange(episodes):
            r = self.run_episode(train=True, max_steps=self.cfg.get("training", {}).get("max_steps_per_episode", 1000))
            self.episode_rewards.append(r)
            if (ep + 1) % save_every == 0:
                mpath = os.path.join(self.output_dir, f"model_ep{ep+1}.pth")
                self.agent.save(mpath)
                print(f"\nSaved model to {mpath}")
        final_path = os.path.join(self.output_dir, "model_final.pth")
        self.agent.save(final_path)
        print("Training complete. Saved final model to", final_path)
        save_plot(list(range(len(self.episode_rewards))), self.episode_rewards, os.path.join(self.output_dir, "rewards.png"), title="Episode Rewards")
        if self.losses:
            save_plot(list(range(len(self.losses))), self.losses, os.path.join(self.output_dir, "losses.png"), title="Losses")

    def evaluate(self, episodes=10):
        print(f"Evaluating for {episodes} episodes")
        rewards = []
        for _ in range(episodes):
            r = self.run_episode(train=False, max_steps=self.cfg.get("training", {}).get("max_steps_per_episode", 1000))
            rewards.append(r)
        print("Eval rewards:", rewards)
        print("Average reward:", np.mean(rewards))
        return rewards
