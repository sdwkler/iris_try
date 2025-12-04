#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import torch
import cv2
import time
from src.trainer import Trainer

def resize_frame(frame, target_size=(400, 600)):
    """调整帧大小，保证可视化窗口比例正常"""
    h, w = frame.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 填充黑边，使窗口大小固定
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    return canvas

def draw_info_on_frame(frame, ep, step, reward, lives, total_reward, idle_count):
    """在帧上绘制监控信息（命数、步数、奖励、不动计数）"""
    # 字体配置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # 白色
    thickness = 2
    bg_color = (0, 0, 0)     # 黑色背景
    
    # 绘制文本（带背景框）
    info_texts = [
        f"Episode: {ep+1}",
        f"Step: {step}",
        f"Current Reward: {reward:.2f}",
        f"Total Reward: {total_reward:.2f}",
        f"Lives Remaining: {lives:.1f}/5",
        f"Continuous Idle: {idle_count:.0f}"
    ]
    
    y_offset = 30
    for text in info_texts:
        # 绘制背景框
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (10, y_offset-text_h-5), (10+text_w+10, y_offset+5), bg_color, -1)
        # 绘制文本
        cv2.putText(frame, text, (15, y_offset), font, font_scale, color, thickness)
        y_offset += 30
    
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Show OpenCV window with visualization")
    parser.add_argument("--fps", type=int, default=30, help="Render FPS (controls playback speed)")
    parser.add_argument("--apply_idle_penalty", action="store_true", help="Apply idle penalty during evaluation (same as training)")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 初始化Trainer（eval模式）
    trainer = Trainer(cfg, device='cuda' if torch.cuda.is_available() else 'cpu', mode='eval')
    
    # 加载预训练策略
    ckpt = torch.load(args.model, map_location=trainer.device, weights_only=False)
    if 'policy' in ckpt:
        trainer.policy.load_state_dict(ckpt['policy'])
        print(f"[INFO] Loaded policy from {args.model}")
    else:
        raise KeyError("Cannot find 'policy' key in checkpoint file!")
    
    # 加载RND（如果配置开启）
    if trainer.rnd is not None and 'rnd' in ckpt:
        trainer.rnd.load_state_dict(ckpt['rnd'])
        trainer.rnd.eval()
        print(f"[INFO] Loaded RND from {args.model}")
    
    # 设置策略为评估模式
    trainer.policy.eval()
    
    # 获取环境和核心配置
    env = trainer.venv
    seq_len = cfg.get("transformer", {}).get("seq_len", 4)
    episodes = max(args.episodes, 1)
    frame_delay = 1000 // args.fps  # 每帧延迟（毫秒）
    
    # 不动惩罚配置（和训练对齐）
    idle_threshold = cfg.get("training", {}).get("idle_threshold", 10)
    idle_penalty = cfg.get("training", {}).get("idle_penalty", -0.1)
    rnd_scale = cfg.get("rnd", {}).get("rnd_scale", 2.0)

    # 存储评估结果
    episode_rewards = []
    episode_ext_rewards = []  # 单独记录外在奖励
    episode_int_rewards = []  # 单独记录内在奖励
    episode_steps = []
    episode_lives_used = []

    # 初始化可视化窗口
    if args.render:
        cv2.namedWindow("Montezuma Agent Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Montezuma Agent Visualization", 400, 600)

    try:
        for ep in range(episodes):
            # 重置环境
            obs, info = env.reset()
            if isinstance(obs, dict):
                obs = obs.get('observation', obs)
            
            # 初始化命数和不动计数器（和训练逻辑对齐）
            lives = np.ones(trainer.num_envs, dtype=int) * 5
            if "lives" in info:
                lives = np.array(info["lives"]) if isinstance(info["lives"], list) else info["lives"]
            continuous_idle_counts = np.zeros(trainer.num_envs, dtype=int)
            
            # 初始化序列（和训练逻辑完全一致）
            zpool_init = trainer.encode_frame(obs)
            per_env_seqs = zpool_init[:, -seq_len:, :].clone()  # (num_envs, seq_len, D) - torch tensor
            total_reward = 0.0
            total_ext_reward = 0.0
            total_int_reward = 0.0
            step = 0
            done = np.zeros(trainer.num_envs, dtype=bool)

            while not all(done):
                step += 1
                # 1. 准备输入（和训练逻辑对齐）
                obs_seq_np = per_env_seqs.cpu().numpy().astype(np.float32)
                obs_seq_tensor = torch.FloatTensor(obs_seq_np).to(trainer.device)
                
                # 2. 策略推理（无梯度，和训练的get_action逻辑对齐）
                with torch.no_grad():
                    logits, _ = trainer.policy(obs_seq_tensor)
                    actions_filtered = torch.argmax(logits, dim=-1).cpu().numpy()  # 过滤后的动作索引
                
                # 3. 映射到原始动作空间（核心：和训练的动作过滤对齐）
                actions = np.array([trainer.action_mapping[act] for act in actions_filtered])

                # 4. 执行动作
                next_obs, ext_rew, terminated, truncated, info = env.step(actions)
                dones = np.logical_or(terminated, truncated)
                
                # 5. 不动惩罚计算（和训练逻辑完全一致）
                step_idle_penalty = np.zeros(trainer.num_envs, dtype=np.float32)
                if args.apply_idle_penalty:
                    idle_mask = (actions == 0)
                    continuous_idle_counts[idle_mask] += 1
                    continuous_idle_counts[~idle_mask] = 0
                    
                    # 计算不动惩罚
                    step_idle_penalty = np.where(
                        continuous_idle_counts >= idle_threshold,
                        idle_penalty,
                        0.0
                    )

                # 6. RND内在奖励计算（和训练逻辑对齐）
                step_int_rew = np.zeros(trainer.num_envs, dtype=np.float32)
                if trainer.rnd is not None:
                    zpool_next = trainer.encode_frame(next_obs)
                    current_z = zpool_next[:, -1, :]
                    current_z_tensor = current_z.to(trainer.device)
                    
                    with torch.no_grad():
                        intrinsic_rewards = trainer.rnd.compute_intrinsic(current_z_tensor)
                    step_int_rew = rnd_scale * intrinsic_rewards.detach().cpu().numpy()

                # 7. 总奖励计算（和训练逻辑完全一致）
                step_total_rew = ext_rew + step_idle_penalty + step_int_rew

                # 8. 更新统计信息
                avg_ext_rew = np.mean(ext_rew)
                avg_int_rew = np.mean(step_int_rew)
                avg_total_rew = np.mean(step_total_rew)
                avg_idle_penalty = np.mean(step_idle_penalty)
                
                total_reward += avg_total_rew
                total_ext_reward += avg_ext_rew
                total_int_reward += avg_int_rew

                # 9. 更新命数
                if "lives" in info:
                    current_lives = np.array(info["lives"]) if isinstance(info["lives"], list) else info["lives"]
                    lives = current_lives

                # 10. 可视化渲染
                if args.render:
                    # 处理观测帧（取第一个环境的最后一帧）
                    frame = obs[0]  # (frame_stack, H, W)
                    # 提取最后一帧
                    if frame.ndim == 3 and frame.shape[0] == trainer.frame_stack:
                        display_frame = frame[-1]
                    else:
                        display_frame = frame
                    
                    # 格式转换（灰度→RGB）
                    if display_frame.ndim == 2:
                        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                    elif display_frame.shape[-1] == 1:
                        display_rgb = cv2.cvtColor(display_frame.squeeze(-1), cv2.COLOR_GRAY2BGR)
                    else:
                        display_rgb = display_frame[..., :3]
                    
                    # 调整大小+绘制信息
                    resized_frame = resize_frame(display_rgb)
                    avg_lives = np.mean(lives)
                    avg_idle_count = np.mean(continuous_idle_counts)
                    final_frame = draw_info_on_frame(
                        resized_frame, ep, step, avg_total_rew, 
                        avg_lives, total_reward, avg_idle_count
                    )
                    
                    # 显示帧
                    cv2.imshow("Montezuma Agent Visualization", final_frame)
                    
                    # 按键控制
                    key = cv2.waitKey(frame_delay) & 0xFF
                    if key == ord('q'):  # 退出
                        raise KeyboardInterrupt
                    elif key == ord('p'):  # 暂停
                        cv2.waitKey(-1)  # 按任意键继续

                # 11. 更新时序序列（和训练逻辑完全一致）
                if isinstance(next_obs, dict):
                    next_obs = next_obs.get('observation', next_obs)
                zpool_next = trainer.encode_frame(next_obs)
                new_z = zpool_next[:, -1, :].unsqueeze(1)  # (num_envs, 1, D)
                per_env_seqs = torch.cat([
                    per_env_seqs[:, 1:, :],
                    new_z
                ], dim=1)  # 保持tensor类型，和训练一致
                
                # 12. 更新观测和结束状态
                obs = next_obs
                done = dones

            # 统计本回合结果
            avg_lives_used = 5 - np.mean(lives)
            episode_rewards.append(total_reward)
            episode_ext_rewards.append(total_ext_reward)
            episode_int_rewards.append(total_int_reward)
            episode_steps.append(step)
            episode_lives_used.append(avg_lives_used)
            
            # 打印回合总结
            print(f"\n=== Episode {ep+1} Summary ===")
            print(f"Total Reward: {total_reward:.2f} (Ext: {total_ext_reward:.2f}, Int: {total_int_reward:.2f})")
            print(f"Total Steps: {step}")
            print(f"Average Lives Used: {avg_lives_used:.1f}/5")
            print(f"Remaining Lives: {np.mean(lives):.1f}/5")

        # 打印最终评估报告
        print("\n=== Final Evaluation Report ===")
        print(f"Number of Episodes: {episodes}")
        print(f"Average Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Extrinsic Reward: {np.mean(episode_ext_rewards):.2f} ± {np.std(episode_ext_rewards):.2f}")
        print(f"Average Intrinsic Reward: {np.mean(episode_int_rewards):.2f} ± {np.std(episode_int_rewards):.2f}")
        print(f"Average Steps: {np.mean(episode_steps):.0f} ± {np.std(episode_steps):.0f}")
        print(f"Average Lives Used: {np.mean(episode_lives_used):.1f} ± {np.std(episode_lives_used):.1f}")

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")
    finally:
        # 清理资源
        if args.render:
            cv2.destroyAllWindows()
        env.close()
        print("[INFO] Environment closed successfully")

if __name__ == "__main__":
    main()
