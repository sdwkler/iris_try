#!/usr/bin/env python3
"""
Watch a trained agent play (real-time window or save to mp4).

Usage examples:
  python tests/watch_agent.py --config config/config.yaml --model outputs/model_final.pth --episodes 3 --render
  python tests/watch_agent.py --config config/config.yaml --model outputs/model_final.pth --episodes 3 --save out.mp4

Notes:
- Works with the VisualDQNAgent / Visual agent interface used in this repo:
  agent = VisualDQNAgent(cfg=..., obs_shape=(seq_len,C,H,W), action_n=env.action_space_n)
  and supports agent.load(path).
- If using a different agent implementation (A2C/PPO), ensure agent.select_action(state) and .load(path) exist.
"""
import argparse
import yaml
import time
import numpy as np
import cv2

try:
    import imageio
except Exception:
    imageio = None
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from collections import deque
from src.envs.atari_env import AtariEnv
from src.envs.tetris_env import TetrisEnv
from src.agent import VisualDQNAgent

def build_agent_and_env(cfg):
    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("id", None)
    if env_id:
        env = AtariEnv(env_id=env_id,
                       frame_stack=env_cfg.get("frame_stack", 4),
                       image_size=tuple(env_cfg.get("image_size", [84,84])),
                       grayscale=env_cfg.get("grayscale", False))
    else:
        env = TetrisEnv(height=env_cfg.get("height",20),
                        width=env_cfg.get("width",10),
                        render_frame=True,
                        image_size=tuple(env_cfg.get("image_size",[84,84])),
                        channels=env_cfg.get("frame_channels",3))
    coll_cfg = cfg.get("collector", {})
    seq_len = coll_cfg.get("frame_stack", 4)
    C = 1 if env_cfg.get("grayscale", False) else env_cfg.get("frame_channels", 3) if not env_id else (1 if env_cfg.get("grayscale", False) else 3)
    H, W = env_cfg.get("image_size", [84,84])
    obs_shape = (seq_len, C, H, W)
    agent = VisualDQNAgent(cfg=cfg.get("agent", {}), obs_shape=obs_shape, action_n=env.action_space_n)
    return env, agent, seq_len, C, H, W

def preprocess_frame_for_display(frame):
    # frame: H,W,C or H,W,1 or grayscale H,W
    arr = np.array(frame)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        disp = cv2.cvtColor((arr).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        disp = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return disp

def stack_init_from_obs(obs, seq_len):
    arr = np.array(obs)
    # obs could be (stack,H,W,C) or (H,W,C)
    if arr.ndim == 4:
        frames = arr
        # ensure length seq_len
        if frames.shape[0] >= seq_len:
            frames = frames[-seq_len:]
        else:
            pad_cnt = seq_len - frames.shape[0]
            first = frames[0:1]
            frames = np.concatenate([np.repeat(first, pad_cnt, axis=0), frames], axis=0)
        # convert to C,H,W normalized floats in [0,1]
        buf = deque(maxlen=seq_len)
        for f in frames:
            f_proc = f.astype(np.float32) / 255.0
            if f_proc.ndim == 2:
                f_proc = f_proc[..., None]
            f_proc = np.transpose(f_proc, (2,0,1))
            buf.append(f_proc)
        return buf
    else:
        # single frame H,W,C or H,W
        f = arr
        f_proc = f.astype(np.float32) / 255.0
        if f_proc.ndim == 2:
            f_proc = f_proc[..., None]
        f_proc = np.transpose(f_proc, (2,0,1))
        buf = deque(maxlen=seq_len)
        for _ in range(seq_len):
            buf.append(f_proc)
        return buf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model .pth")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Show real-time window (cv2.imshow)")
    parser.add_argument("--save", type=str, default=None, help="Save video to this mp4 path (requires imageio)")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    env, agent, seq_len, C, H, W = build_agent_and_env(cfg)
    print("Env and agent created. Loading model:", args.model)
    agent.load(args.model)
    print("Model loaded.")

    writer = None
    if args.save:
        if imageio is None:
            raise RuntimeError("imageio not installed. Install with: pip install imageio")
        writer = imageio.get_writer(args.save, fps=args.fps)

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            buf = stack_init_from_obs(obs, seq_len)
            done = False
            total_r = 0.0
            while not done:
                state = np.stack(buf)  # seq_len, C, H, W
                action = agent.select_action(state)
                next_obs, reward, done, _ = env.step(action)
                total_r += reward
                # update buffer
                arr = np.array(next_obs)
                if arr.ndim == 4:
                    frames = arr
                    if frames.shape[0] >= seq_len:
                        frames = frames[-seq_len:]
                    else:
                        pad_cnt = seq_len - frames.shape[0]
                        first = frames[0:1]
                        frames = np.concatenate([np.repeat(first, pad_cnt, axis=0), frames], axis=0)
                    # take last frame for display
                    last_frame = frames[-1]
                    # update buffer with last frame processed
                    buf = deque(maxlen=seq_len)
                    for f in frames:
                        f_proc = f.astype(np.float32)/255.0
                        if f_proc.ndim == 2:
                            f_proc = f_proc[..., None]
                        f_proc = np.transpose(f_proc, (2,0,1))
                        buf.append(f_proc)
                else:
                    last_frame = arr
                    f_proc = arr.astype(np.float32)/255.0
                    if f_proc.ndim == 2:
                        f_proc = f_proc[..., None]
                    f_proc = np.transpose(f_proc, (2,0,1))
                    buf.append(f_proc)

                vis = preprocess_frame_for_display(last_frame)
                if args.render:
                    cv2.imshow("agent", vis)
                    # waitKey returns -1 if no key pressed; quit on 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt()

                if writer is not None:
                    writer.append_data(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

                # small sleep to slow down if needed
                time.sleep(0.01)

            print(f"Episode {ep+1} finished. Reward: {total_r}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if writer is not None:
            writer.close()
        if args.render:
            cv2.destroyAllWindows()
        try:
            env.env.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()