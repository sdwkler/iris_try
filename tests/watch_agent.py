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
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    with open(args.config,"r") as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg, device='cuda', mode='eval')
    ckpt = torch.load(args.model, map_location='cuda')
    trainer.encoder.load_state_dict(ckpt.get('encoder', {}))
    trainer.policy.load_state_dict(ckpt.get('policy', {}))
    env = trainer.venv
    for ep in range(args.episodes):
        obs = env.reset()
        obs = obs[0]
        done = [False]*trainer.num_envs
        total = 0.0
        while not all(done):
            arr = obs
            if arr.ndim == 4:
                last = arr[-1]
            elif arr.ndim == 3 and arr.shape[0] == trainer.frame_stack:
                last = arr[-1]
            else:
                last = arr
            img = (last).astype(np.uint8)
            last = last.squeeze()[-1]
            if last.dtype == np.float32:
                last = (last * 255).astype(np.uint8)
            else:
                last = last.astype(np.uint8)  # 防止其他数据类型
            img_rgb = last
            cv2.imshow('agent', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
            ft = torch.FloatTensor(img_rgb).permute(2,0,1).unsqueeze(0).float()/255.0
            ft = ft.to(trainer.device)
            z = trainer.encoder(ft)
            # 1. 全局池化后保留 (batch_size, D)，无需额外增加维度

            zpool = torch.mean(z, dim=[2, 3])  # 假设z的原始形状为(1, D, h, w)，则结果为(1, D)

            seq_len = cfg.get("transformer", {}).get("seq_len", 4)
            # 重复seq_len次，形状变为(seq_len, batch_size, D)，再转置为(batch_size, seq_len, D)
            seq = torch.stack([zpool for _ in range(seq_len)]).permute(1, 0, 2)
            logits, _ = trainer.policy(seq)
            action = torch.argmax(logits, dim=-1).item()
            # 将原来的4个变量改为5个
            obs, rew, terminated, truncated, info = env.step([action])
            done = terminated or truncated  # 用逻辑或合并两种终止情况
            total += rew[0]
        print("Episode reward:", total)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()