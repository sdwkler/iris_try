#!/usr/bin/env python3
import argparse
import yaml
import torch
import cv2
import numpy as np
from src.trainer import Trainer
from src.utils import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True)  # 包含 policy 权重的 checkpoint
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    # --- 1. 加载配置 ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- 2. 初始化 Trainer（评估模式，加载 VQ-VAE + Policy）---
    trainer = Trainer(cfg, device='cuda', mode='eval')  # 注意 mode='eval'

    # --- 3. 加载训练好的模型权重（包含 policy 的 state_dict）---
    # 假设你的 checkpoint 是通过 torch.save({'policy': policy.state_dict()}) 保存的
    ckpt = torch.load(args.model, map_location='cuda')
    policy_state_dict = ckpt.get('policy', {})
    trainer.policy.load_state_dict(policy_state_dict)

    print(f"[INFO] 已加载模型权重 from {args.model}")

    # --- 4. 获取环境 ---
    env = trainer.venv

    # --- 5. 开始评估循环 ---
    for ep in range(args.episodes):
        obs, infos = env.reset()
        if isinstance(obs, dict):
            obs = obs.get('observation', obs)  # 兼容 wrapper
        done = [False] * trainer.num_envs
        total_reward = 0.0

        while not all(done):
            # 取第一个环境的观测（用于显示和推理）
            current_obs = obs[0]  # (T, H, W, C) 或类似，取决于你的 FrameStack
            # 如果是 Atari 原始帧（经过 FrameStack 和 Resize），通常是 (4, 84, 84) 或 (4, 84, 84, 1)
            # 我们直接显示最后一帧用于观察
            if current_obs.ndim == 4:  # (T, H, W, C)
                last_frame = current_obs[-1]  # 取最后一帧：(H, W, C)
            elif current_obs.ndim == 3:  # (H, W, C)
                last_frame = current_obs
            else:
                raise ValueError(f"Unexpected observation shape: {current_obs.shape}")

            # 转为 uint8，确保可以显示
            if last_frame.dtype == np.float32:
                last_frame = (last_frame * 255).clip(0, 255).astype(np.uint8)
            elif last_frame.dtype == np.uint8:
                pass
            else:
                last_frame = last_frame.astype(np.uint8)

            # 显示当前帧（用于观察环境状态）
            if last_frame.ndim == 3 and last_frame.shape[2] == 1:
                last_frame = last_frame.squeeze(-1)  # 去掉 channel dim 如果是单通道
            cv2.imshow('Atari Agent (VQ-VAE)', last_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户退出.")
                cv2.destroyAllWindows()
                return

            # --- 使用 Trainer.encode_frame() 获取潜在向量序列（替代你原来手动处理的流程）---
            latent_seq = trainer.encode_frame([last_frame])  # 输入是 (1, T, H, W, C)，返回 (1, T, D)

            # latent_seq shape: (num_envs=1, seq_len=T, D)
            seq = torch.FloatTensor(latent_seq).to(trainer.device)  # (1, T, D)

            # --- 通过 policy 得到动作 ---
            logits, _ = trainer.policy(seq)  # 输入是 (1, T, D)
            action = torch.argmax(logits, dim=-1).item()  # 选最高概率的动作

            # --- 执行动作 ---
            obs, rew, terminated, truncated, infos = env.step([action])
            done = np.logical_or(terminated, truncated)  # Atari 环境常用 terminated + truncated
            total_reward += rew[0]

        print(f"[Episode {ep + 1}] 总奖励: {total_reward:.1f}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()