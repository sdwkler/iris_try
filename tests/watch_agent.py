#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import torch
import cv2
from src.trainer import Trainer
from src.utils import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg, device='cpu')
    ckpt = torch.load(args.model, map_location='cpu')
    trainer.encoder.load_state_dict(ckpt.get('encoder', {}))
    trainer.policy.load_state_dict(ckpt.get('policy', {}))
    env = trainer.venv
    for ep in range(args.episodes):
        obs = env.reset()
        done = [False]*trainer.num_envs
        total = 0.0
        while not all(done):
            arr = obs[0]
            if arr.ndim == 4:
                last = arr[-1]
            elif arr.ndim == 3 and arr.shape[0] == trainer.frame_stack:
                last = arr[-1]
            else:
                last = arr
            img = (last).astype(np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim ==2 or img.shape[2]==1 else img
            cv2.imshow('agent', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            ft = torch.FloatTensor(img).permute(2,0,1).unsqueeze(0).float()/255.0
            z = trainer.encoder(ft)
            zpool = torch.mean(z, dim=[2,3]).unsqueeze(0)
            seq = torch.stack([zpool.squeeze(0) for _ in range(cfg.get("transformer", {}).get("seq_len",4))]).unsqueeze(0)
            logits, _ = trainer.policy(seq)
            action = torch.argmax(logits, dim=-1).item()
            obs, rew, done, info = env.step([action])
            total += rew[0]
        print("Episode reward:", total)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()