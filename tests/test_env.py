#!/usr/bin/env python3
"""
Quick environment smoke tests.
Run: python tests/test_env.py
"""
import yaml
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from src.envs.atari_env import AtariEnv
from src.envs.tetris_env import TetrisEnv

def test_atari(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("env", {})
    env_id = env_cfg.get("id", None)
    if env_id:
        env = AtariEnv(env_id=env_id,
                       frame_stack=env_cfg.get("frame_stack", 4),
                       image_size=tuple(env_cfg.get("image_size", [84,84])),
                       grayscale=env_cfg.get("grayscale", False))
        obs = env.reset()
        print("Atari reset obs shape:", getattr(obs, "shape", None))
        step_obs, r, done, info = env.step(0)
        print("Atari step obs shape:", getattr(step_obs, "shape", None), "reward:", r, "done:", done)
    else:
        print("No Atari env id found in config; skipping Atari test.")

def test_tetris():
    env = TetrisEnv()
    obs = env.reset()
    print("Tetris reset obs shape (image):", getattr(obs, "shape", None))
    step_obs, r, done, _ = env.step(5)
    print("Tetris step obs shape:", getattr(step_obs, "shape", None), "reward:", r, "done:", done)

if __name__ == "__main__":
    print("Running env smoke tests...")
    test_atari()
    test_tetris()
    print("Done.")