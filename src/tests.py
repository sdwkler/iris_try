#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import torch
import cv2
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Show OpenCV window")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    trainer = Trainer(cfg, device='cuda', mode='eval')

    # 只加载policy（DINO extractor为Frozen，不需要加载）
    ckpt = torch.load(args.model, map_location='cuda')
    if 'policy' in ckpt:
        trainer.policy.load_state_dict(ckpt['policy'])
    else:
        raise KeyError("Cannot find 'policy' in checkpoint.")

    env = trainer.venv
    seq_len = cfg.get("transformer", {}).get("seq_len", 4)
    episodes = args.episodes if args.episodes > 0 else 1

    for ep in range(episodes):
        obs, info = env.reset()
        # 适应vectorEnv和单环境
        if isinstance(obs, dict):
            obs = obs.get('observation', obs)
        done = [False] * trainer.num_envs
        total = 0.0
        per_env_seqs = None

        # 初始化per_env_seqs
        zpool_init = trainer.encode_frame(obs)     # (num_envs, T, D)
        init_z = zpool_init[:, -1, :]              # (num_envs, D)
        per_env_seqs = np.repeat(init_z[:, None, :], seq_len, axis=1)  # (num_envs, seq_len, D)

        while not all(done):
            # 当前序列输入policy
            obs_seq_tensor = torch.FloatTensor(per_env_seqs).to(trainer.device)
            with torch.no_grad():
                logits, _ = trainer.policy(obs_seq_tensor)
                actions = torch.argmax(logits, dim=-1).cpu().numpy()

            # 展示窗口（只展示主环境第一个栈帧，如需可调整）
            if args.render:
                arr = obs[0] if isinstance(obs, np.ndarray) and obs.ndim >= 1 else obs
                # 支持frame stack格式，取最后一帧
                if arr.ndim == 4:
                    last = arr[-1]
                elif arr.ndim == 3 and arr.shape[0] == trainer.frame_stack:
                    last = arr[-1]
                else:
                    last = arr
                img = np.ascontiguousarray(last)
                # 灰度转RGB
                if img.ndim == 2:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 1:
                    img_rgb = cv2.cvtColor(img.squeeze(-1), cv2.COLOR_GRAY2BGR)
                else:
                    img_rgb = img[..., :3]
                cv2.imshow('agent', img_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

            # Step
            next_obs, rew, terminated, truncated, info = env.step(actions)
            dones = np.logical_or(terminated, truncated)
            total += np.array(rew).sum()
            
            # 对新观测做特征编码，更新轨迹序列（和训练一致）
            if isinstance(next_obs, dict):
                next_obs = next_obs.get('observation', next_obs)
            zpool_next = trainer.encode_frame(next_obs)
            new_z = zpool_next[:, -1, :]
            per_env_seqs = np.concatenate([
                per_env_seqs[:, 1:, :],
                new_z[:, None, :]
            ], axis=1)
            obs = next_obs
            done = dones
        print(f"Episode {ep+1} reward:", total)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()