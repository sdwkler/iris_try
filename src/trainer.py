import os
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

from src.utils import ensure_dir
from src.envs.atari_vec import make_atari_env
from src.models.dino import DINOv2FeatureExtractor
from src.models.rnd import RND
from src.models.transformer_policy import TransformerPolicy
from src.collector import RolloutBuffer
from src.agent.ppo_agent import PPOAgent

class Trainer:
    def __init__(self, cfg, device='cpu', mode='train'):
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.get("output_dir", "outputs")
        ensure_dir(self.output_dir)
        env_cfg = cfg.get("env", {})
        self.env_id = env_cfg.get("id")
        if mode == 'train':
            self.num_envs = env_cfg.get("num_envs", 8)
        else:
            self.num_envs = env_cfg.get("eval_num_envs", 1)
        self.frame_stack = env_cfg.get("frame_stack", 4)
        self.image_size = tuple(env_cfg.get("image_size", [224, 224]))
        self.grayscale = env_cfg.get("grayscale", False)
        self.atari_frame_skip = env_cfg.get("atari_frame_skip", 4)
        env_fns = [make_atari_env(self.env_id, image_size=self.image_size, frame_stack=self.frame_stack,
                                  grayscale=self.grayscale, atari_frame_skip=self.atari_frame_skip) for _ in range(self.num_envs)]
        self.venv = gym.vector.SyncVectorEnv(env_fns)

        # DINOv2 extractor
        vcfg = cfg.get("visual", {})
        self.dino_extractor = DINOv2FeatureExtractor(
            model_name=vcfg.get("dino_model", "vit_base_patch14_dinov2.lvd142m"),
            input_size=tuple(vcfg.get("image_size", [224, 224]))
        ).to(device)
        self.dino_extractor.eval()
        input_dim = self.dino_extractor.model.num_features

        tcfg = cfg.get("transformer", {})
        self.policy = TransformerPolicy(
            input_dim=input_dim,
            action_n=self.venv.single_action_space.n,
            d_model=tcfg.get("d_model", 256),
            n_heads=tcfg.get("n_heads", 4),
            num_layers=tcfg.get("num_layers", 3),
            mlp_dim=tcfg.get("mlp_dim", 512),
            dropout=tcfg.get("dropout", 0.1),
            seq_len=tcfg.get("seq_len", 4)
        ).to(device)

        if cfg.get("rnd", {}).get("enabled", True):
            self.rnd = RND(input_dim, device=device,
                    predictor_hidden=cfg.get("rnd",{}).get("predictor_hidden",256),
                    target_hidden=cfg.get("rnd",{}).get("target_hidden",256),
                    lr=cfg.get("rnd",{}).get("rnd_lr",1e-4))
        else:
            self.rnd = None
        print("RND enabled:", self.rnd is not None)
        agent_cfg = cfg.get("agent", {})
        ppo_cfg = {
            "lr": agent_cfg.get("lr", 2.5e-4),
            "clip_eps": agent_cfg.get("clip_eps", 0.2),
            "gamma": agent_cfg.get("gamma", 0.99),
            "gae_lambda": agent_cfg.get("gae_lambda", 0.95),
            "value_coef": agent_cfg.get("value_coef", 0.5),
            "entropy_coef": agent_cfg.get("entropy_coef", 0.01),
            "max_grad_norm": agent_cfg.get("max_grad_norm", 0.5),
            "epochs": agent_cfg.get("epochs", 4),
            "minibatch_size": agent_cfg.get("minibatch_size", 64)
        }
        self.agent = PPOAgent(self.dino_extractor, self.policy, ppo_cfg, device=device)
        self.steps_per_env = cfg.get("training", {}).get("steps_per_env", 128)
        self.batch_size = self.steps_per_env * self.num_envs
        self.rollout = RolloutBuffer(self.steps_per_env, self.num_envs, obs_shape=(tcfg.get("seq_len", 4), input_dim))
        self.total_updates = cfg.get("training", {}).get("total_updates", 1000)
        self.prev_lives = None

    def encode_frame(self, frame):
        """
        用DINOv2批量提取特征，frame: (num_envs,T,H,W,C) 或 (T,H,W,C)
        输出: (num_envs, T, D)
        """
        # 转为 numpy 数组
        if isinstance(frame, list):
            frame = np.stack([np.array(f, dtype=np.uint8) for f in frame])
        else:
            frame = np.array(frame, dtype=np.uint8)

        if frame.ndim == 4:
            frame = frame[np.newaxis, ...]
        num_envs, T, H, W, C = frame.shape

        batch = frame.reshape(-1, H, W, C)
        imgs = [self.dino_extractor.preprocess(f) for f in batch]
        imgs = torch.stack(imgs).to(self.device)

        with torch.no_grad():
            feats = self.dino_extractor(imgs)  # (B, D)

        feats = feats.reshape(num_envs, T, -1).cpu().numpy()
        return feats

    # 在 trainer.py 的 collect_rollout 方法中，替换 reward 相关逻辑
    def collect_rollout(self):
        seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)
        obs, info = self.venv.reset()
        if isinstance(obs, dict):
            obs = obs.get('observation', obs)
        zpool_init = self.encode_frame(obs)  # (num_envs, T, D)
        init_z = zpool_init[:, -1, :]  # 取最后一帧的特征 (num_envs, D)
        per_env_seqs = np.repeat(init_z[:, None, :], seq_len, axis=1)  # (num_envs, seq_len, D)

        input_dim = per_env_seqs.shape[2]
        self.rollout = RolloutBuffer(self.steps_per_env, self.num_envs, obs_shape=(seq_len, input_dim))

        for step in range(self.steps_per_env):
            obs_seq_tensor = torch.FloatTensor(per_env_seqs).to(self.device)
            actions, logps, values = self.agent.get_action_and_value(obs_seq_tensor)
            next_obs, ext_rewards, terms, truncs, infos = self.venv.step(actions)  # 外部奖励
            dones = np.logical_or(terms, truncs)

            # 关键修改：计算 RND 内在奖励并融合
            if self.rnd is not None:
                # 对当前 next_obs 提取特征（用于计算 RND 奖励）
                zpool_next = self.encode_frame(next_obs)  # (num_envs, T, D)
                current_z = zpool_next[:, -1, :]  # 取最后一帧的特征 (num_envs, D)
                # 转换为 tensor 并计算内在奖励
                current_z_tensor = torch.FloatTensor(current_z).to(self.device)
                intrinsic_rewards = self.rnd.compute_intrinsic(current_z_tensor)  # (num_envs,)
                # 融合外部奖励和内在奖励（按配置的 rnd_scale 加权）
                total_rewards = ext_rewards + self.cfg["rnd"]["rnd_scale"] * intrinsic_rewards.detach().cpu().numpy()
            else:
                total_rewards = ext_rewards  # 无 RND 时直接用外部奖励

            # 更新序列（与之前逻辑一致）
            zpool_next = self.encode_frame(next_obs)
            new_z = zpool_next[:, -1, :]
            per_env_seqs = np.concatenate([
                per_env_seqs[:, 1:, :],  # 移除最旧帧
                new_z[:, None, :]        # 追加新帧
            ], axis=1)

            # 插入融合后的总奖励
            self.rollout.insert(
                step,
                obs=per_env_seqs.copy(),
                action=actions,
                reward=total_rewards,  # 这里改为融合后的总奖励
                done=dones,
                logp=logps,
                value=values
            )
            obs = next_obs

        # 剩余代码保持不变...
        final_obs_tensor = torch.FloatTensor(per_env_seqs).to(self.device)
        with torch.no_grad():
            _, final_values = self.policy(final_obs_tensor)
        final_values = final_values.cpu().numpy()
        return self.rollout.get(), final_values


    def train(self):
        total_updates = self.total_updates
        for update in trange(total_updates, desc="Training", unit="update"):
            rollout_get, last_values = self.collect_rollout()
            obs_roll, actions, rewards, dones, logps, values = rollout_get
            advantages, returns = self.agent.compute_gae(rewards, values, dones, last_values)
            T, N = actions.shape
            seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)
            input_dim = obs_roll.shape[-1]
            obs_flat = obs_roll.reshape(T*N, seq_len, input_dim)
            actions_flat = actions.reshape(T*N)
            old_logps_flat = logps.reshape(T*N)
            returns_flat = returns.reshape(T*N)
            advs_flat = advantages.reshape(T*N)
            if self.rnd is not None:
                last_frame_latents = obs_roll[:, :, -1, :]
                last_frame_latents_flat = last_frame_latents.reshape(T*N, -1)
                self.rnd.update(torch.FloatTensor(last_frame_latents_flat).to(self.device))
            self.agent.ppo_update(obs_flat, actions_flat, old_logps_flat, returns_flat, advs_flat)
            if (update+1) % self.cfg.get("training", {}).get("save_every", 500) == 0:
                torch.save({'policy': self.policy.state_dict()}, os.path.join(self.output_dir, f"checkpoint_{update+1}.pth"))
                print(f"Saved checkpoint at update {update+1}")
        torch.save({'policy': self.policy.state_dict()}, os.path.join(self.output_dir, "model_final.pth"))
        print("Training finished.")