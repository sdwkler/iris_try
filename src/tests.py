#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import torch
import cv2
import os
from gymnasium.wrappers import RecordVideo
from src.trainer import Trainer
from src.utils import ensure_dir

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试训练好的模型并录制视频")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default="outputs/model_final.pth", help="模型 checkpoint 路径")
    parser.add_argument("--episodes", type=int, default=3, help="测试的episode数量")
    parser.add_argument("--video-dir", type=str, default="outputs/videos", help="视频保存目录")
    args = parser.parse_args()

    # 创建视频保存目录
    ensure_dir(args.video_dir)

    # 加载配置文件
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # 初始化训练器（包含环境和模型）
    trainer = Trainer(cfg, device='cuda')  # 使用CPU推理，如需GPU可改为'cuda'

    # 加载模型权重
    ckpt = torch.load(args.model, map_location=trainer.device)
    trainer.encoder.load_state_dict(ckpt.get('encoder', {}))
    trainer.policy.load_state_dict(ckpt.get('policy', {}))

    # 关闭梯度计算（推理模式）
    trainer.encoder.eval()
    trainer.policy.eval()
    torch.set_grad_enabled(False)

    # 获取原始环境（用于包装录像功能）
    env = trainer.venv.envs[0]  # 取第一个环境

    # 记录每个episode的总奖励
    total_rewards = []

    for ep in range(args.episodes):
        # 第一个episode录制视频
        if ep == 0:
            video_path = os.path.join(args.video_dir, f"episode_{ep}")
            ensure_dir(video_path)
            env = RecordVideo(
                env, 
                video_folder=video_path,
                episode_trigger=lambda x: True,  # 总是录制第一个episode
                name_prefix="test"
            )

        # 重置环境
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # 处理观测数据（获取最后一帧用于显示和推理）
            arr = np.array(obs)
            if arr.ndim == 4:  # 帧堆叠格式 (T, H, W, C)
                last_frame = arr[-1]
            elif arr.ndim == 3 and arr.shape[0] == trainer.frame_stack:  # (T, H, W)
                last_frame = arr[-1]
            else:
                last_frame = arr

            # 转换为RGB用于显示
            img = last_frame.astype(np.uint8)
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_rgb = img

            # 显示游戏画面
            cv2.imshow('Agent Play', img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                env.close()
                cv2.destroyAllWindows()
                return

            # 准备模型输入
            frame_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, C, H, W)
            frame_tensor = frame_tensor.to(trainer.device)

            # 编码图像并生成动作
            z = trainer.encoder(frame_tensor)  # 编码器输出
            z_pool = torch.mean(z, dim=[2, 3])  # 全局平均池化 (1, D)

            # 构建Transformer序列（重复当前特征以匹配序列长度）
            seq_len = cfg.get("transformer", {}).get("seq_len", 4)
            seq = torch.stack([z_pool.squeeze(0) for _ in range(seq_len)]).unsqueeze(0)  # (1, seq_len, D)

            # 策略网络输出动作
            logits, _ = trainer.policy(seq)
            action = torch.argmax(logits, dim=-1).item()

            # 执行动作
            obs, rew, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += rew

        # 保存总奖励
        total_rewards.append(total_reward)
        print(f"Episode {ep + 1}/{args.episodes} - Total Reward: {total_reward:.2f}")

        # 释放录像环境（仅第一个episode）
        if ep == 0:
            env.close()  # 确保视频文件正确保存
            # 恢复原始环境引用
            env = trainer.venv.envs[0]

    # 打印所有episode结果
    print("\n测试结果:")
    for i, rew in enumerate(total_rewards):
        print(f"Episode {i + 1}: {rew:.2f}")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

    # 清理窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
