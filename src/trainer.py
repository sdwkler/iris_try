import os
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym

from src.utils import ensure_dir
from src.envs.atari_vec import make_atari_env
from src.models.vqvae import Encoder, Decoder, VectorQuantizerEMA
from src.models.rnd import RND
from src.models.transformer_policy import TransformerPolicy
from src.collector import RolloutBuffer
from src.agent.ppo_agent import PPOAgent

class Trainer:
    def __init__(self, cfg, device='cpu',mode='train'):
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.get("output_dir","outputs")
        ensure_dir(self.output_dir)
        env_cfg = cfg.get("env", {})
        self.env_id = env_cfg.get("id")
        if mode == 'train':
            self.num_envs = env_cfg.get("num_envs", 8)
        else:
            self.num_envs = env_cfg.get("eval_num_envs", 1)
        self.frame_stack = env_cfg.get("frame_stack", 4)
        self.image_size = tuple(env_cfg.get("image_size", [84,84]))
        self.grayscale = env_cfg.get("grayscale", True)
        self.atari_frame_skip = env_cfg.get("atari_frame_skip", 4)
        env_fns = [make_atari_env(self.env_id, image_size=self.image_size, frame_stack=self.frame_stack, grayscale=self.grayscale, atari_frame_skip=self.atari_frame_skip) for _ in range(self.num_envs)]
        self.venv = gym.vector.SyncVectorEnv(env_fns)
        # build visual / vq
        vcfg = cfg.get("visual", {})
        vqcfg = vcfg.get("vq", {})
        in_ch = vcfg.get("channels", 1)
        self.encoder = Encoder(in_ch=in_ch, hidden=vcfg.get("encoder",{}).get("hidden",128), z_dim=vqcfg.get("embedding_dim",64)).to(device)
        self.decoder = Decoder(out_ch=in_ch, hidden=vcfg.get("decoder",{}).get("hidden",128), z_dim=vqcfg.get("embedding_dim",64)).to(device)
        self.vq = VectorQuantizerEMA(num_embeddings=vqcfg.get("num_embeddings",512), embedding_dim=vqcfg.get("embedding_dim",64), commitment_cost=vqcfg.get("commitment_cost",0.25), decay=vqcfg.get("decay",0.99)).to(device)
        # transformer policy
        tcfg = cfg.get("transformer", {})
        input_dim = vqcfg.get("embedding_dim",64)
        self.policy = TransformerPolicy(input_dim=input_dim, action_n=self.venv.single_action_space.n,
                                        d_model=tcfg.get("d_model",256), n_heads=tcfg.get("n_heads",4),
                                        num_layers=tcfg.get("num_layers",3), mlp_dim=tcfg.get("mlp_dim",512),
                                        dropout=tcfg.get("dropout",0.1), seq_len=tcfg.get("seq_len",4)).to(device)
        # RND
        if cfg.get("rnd", {}).get("enabled", True):
            self.rnd = RND(input_dim, device=device, predictor_hidden=cfg.get("rnd",{}).get("predictor_hidden",256), target_hidden=cfg.get("rnd",{}).get("target_hidden",256), lr=cfg.get("rnd",{}).get("rnd_lr",1e-4))
        else:
            self.rnd = None
        # PPO agent
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
        self.agent = PPOAgent(self.encoder, self.policy, ppo_cfg, device=device)
        self.steps_per_env = cfg.get("training", {}).get("steps_per_env", 128)
        self.batch_size = self.steps_per_env * self.num_envs
        self.rollout = RolloutBuffer(self.steps_per_env, self.num_envs, obs_shape=(self.frame_stack, self.image_size[0], self.image_size[1]))
        self.total_updates = cfg.get("training", {}).get("total_updates", 1000)

    def frame_to_tensor(self, frame):
        """
        处理多环境/多帧输入，返回标准化张量
        输入支持：(num_envs, T, H, W, C) 或单环境 (T, H, W, C)
        返回：(num_envs, T, C, H, W) 张量
        """
        # 转换为numpy数组（处理列表输入）
        if isinstance(frame, list):
            frame = np.stack([np.array(f, dtype=np.float32) for f in frame])
        else:
            frame = np.array(frame, dtype=np.float32)
        
        # 补全单环境输入的维度（增加num_envs维度）
        if frame.ndim == 4:  # 单环境: (T, H, W, C) → 补为 (1, T, H, W, C)
            frame = frame[np.newaxis, ...]
        
        # 检查最终维度是否符合预期
        if frame.ndim != 5:
            raise ValueError(f"输入帧必须为5维 (num_envs, T, H, W, C)，实际为 {frame.shape}")
        
        # 标准化：0-255 → 0-1（匹配预处理逻辑）
        frame /= 255.0
        
        # 维度重排：(num_envs, T, H, W, C) → (num_envs, T, C, H, W)（适配PyTorch卷积输入）
        frame = np.transpose(frame, (0, 1, 4, 2, 3))
        
        # 转换为tensor并移到设备
        return torch.from_numpy(frame).to(self.device)

    def encode_frame(self, frame):
        """
        批量编码多环境多帧，返回每个环境的帧序列潜向量
        输入：多环境多帧数组 (num_envs, T, H, W, C)
        返回：(num_envs, T, D) numpy数组（D为编码器输出维度）
        """
        # 转换为tensor：(num_envs, T, C, H, W)
        ft = self.frame_to_tensor(frame)
        num_envs, T = ft.shape[0], ft.shape[1]
        
        # 批量编码所有帧（避免循环，提升效率）
        # 重塑为 (num_envs*T, C, H, W) → 一次性编码
        ft_flat = ft.reshape(-1, *ft.shape[2:])  # (num_envs*T, C, H, W)
        
        with torch.no_grad():
            z_flat = self.encoder(ft_flat)  # (num_envs*T, D, h, w)
            # 全局平均池化 → (num_envs*T, D)
            zpool_flat = torch.mean(z_flat, dim=[2, 3])
        
        # 恢复维度：(num_envs*T, D) → (num_envs, T, D)
        zpool = zpool_flat.reshape(num_envs, T, -1)
        
        # 返回numpy数组（方便后续处理）
        return zpool.cpu().numpy()



    def pretrain_vq(self):
        print("Collecting frames and pretraining VQ-VAE...")
        pre_cfg = self.cfg.get("visual", {}).get("pretrain", {})
        epochs = pre_cfg.get("epochs", 10)
        batch_size = pre_cfg.get("batch_size", 64)
        lr = pre_cfg.get("lr", 1e-3)
        frames = []
        steps = 5000
        obs = self.venv.reset()
        for _ in range(steps):
            actions = [self.venv.single_action_space.sample() for _ in range(self.num_envs)]
            obs, rs, ds, infos = self.venv.step(actions)
            for e in obs:
                arr = np.array(e)
                # take last stacked frame if needed
                if arr.ndim == 4:
                    f = arr[-1]
                elif arr.ndim == 3 and arr.shape[0] == self.frame_stack:
                    f = arr[-1]
                else:
                    f = arr
                frames.append(f)
            if len(frames) >= 20000:
                break
        frames = np.stack(frames[:20000])
        if frames.ndim == 3:
            frames = frames[..., None]
        frames = frames.astype('float32')/255.0
        frames_t = torch.FloatTensor(frames).permute(0,3,1,2)
        ds = TensorDataset(frames_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr)
        for ep in range(epochs):
            epoch_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                z_e = self.encoder(xb)
                quantized, qloss, _ = self.vq(z_e)
                xr = self.decoder(quantized)
                recon_loss = ((xr - xb)**2).mean()
                loss = recon_loss + qloss
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            print(f"Pretrain epoch {ep+1}/{epochs} loss {epoch_loss/len(loader):.6f}")
        torch.save({'encoder': self.encoder.state_dict(), 'decoder': self.decoder.state_dict(), 'vq': self.vq.state_dict()}, os.path.join(self.output_dir, "vq_pretrained.pth"))
        print("VQ-VAE pretraining finished and saved.")

    def collect_rollout(self):
        seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)
        obs, info = self.venv.reset()  # 初始化环境观测
        
        # ========== 优化1：批量初始化所有环境的序列 ==========
        # 批量编码初始观测：(num_envs, T, H, W, C) → (num_envs, T, D)
        # 处理可能的字典格式
        if isinstance(obs, dict):
            obs = obs.get('observation', obs)
        # 批量编码所有环境的初始观测
        zpool_init = self.encode_frame(obs)  # (num_envs, T, D)
        # 取每个环境最后一帧的潜向量，重复seq_len次初始化序列
        init_z = zpool_init[:, -1, :]  # (num_envs, D)
        per_env_seqs = np.repeat(init_z[:, None, :], seq_len, axis=1)  # (num_envs, seq_len, D)
        
        # 重置rollout缓冲区
        self.rollout = RolloutBuffer(
            self.steps_per_env, 
            self.num_envs, 
            obs_shape=(seq_len, self.encoder.net[-1].out_channels)
        )
        
        # 收集rollout数据
        for step in range(self.steps_per_env):
            # 准备当前步骤的观测序列（转为tensor输入策略网络）
            obs_seq_tensor = torch.FloatTensor(per_env_seqs).to(self.device)
            
            # 获取动作、对数概率和价值
            actions, logps, values = self.agent.get_action_and_value(obs_seq_tensor)
            
            # 执行环境步骤
            next_obs, rewards, terms, truncs, infos = self.venv.step(actions)
            dones = np.logical_or(terms, truncs)
            
            # ========== 优化2：批量处理next_obs并编码 ==========
            # 统一处理next_obs格式（字典/数组）
            if isinstance(next_obs, dict):
                next_obs = next_obs.get('observation', next_obs)
            # 批量编码所有环境的next_obs → (num_envs, T, D)
            zpool_next = self.encode_frame(next_obs)  # 一次性编码所有环境，无循环
            new_z = zpool_next[:, -1, :]  # 所有环境的最新帧潜向量 (num_envs, D)
            
            # ========== 优化3：向量化滑动窗口更新序列 ==========
            # 替换append/pop：(num_envs, seq_len, D) → 滑动窗口保留后seq_len-1个，追加新z
            per_env_seqs = np.concatenate([
                per_env_seqs[:, 1:, :],  # 移除最旧的一列 (num_envs, seq_len-1, D)
                new_z[:, None, :]        # 追加新向量 (num_envs, 1, D)
            ], axis=1)
            
            # ========== 插入缓冲区（无修改） ==========
            self.rollout.insert(
                step,
                obs=per_env_seqs.copy(),  # 当前序列作为观测
                action=actions,
                reward=rewards,
                done=dones,
                logp=logps,
                value=values
            )
            
            # 更新当前观测
            obs = next_obs
        
        # 获取最终价值
        final_obs_tensor = torch.FloatTensor(per_env_seqs).to(self.device)
        with torch.no_grad():
            _, final_values = self.policy(final_obs_tensor)
        final_values = final_values.cpu().numpy()
        return self.rollout.get(), final_values




    def train(self):
        total_updates = self.total_updates
        # 使用 trange 替换 range，添加进度条
        for update in trange(total_updates, desc="Training", unit="update"):
            rpllout_get, last_values = self.collect_rollout()
            obs_roll, actions, rewards, dones, logps, values = rpllout_get
            advantages, returns = self.agent.compute_gae(rewards, values, dones, last_values)
            T, N = actions.shape
            seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)
            obs_flat = obs_roll.reshape(T*N, seq_len, -1)
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
                torch.save({'encoder': self.encoder.state_dict(), 'policy': self.policy.state_dict()}, os.path.join(self.output_dir, f"checkpoint_{update+1}.pth"))
                print(f"Saved checkpoint at update {update+1}")
        torch.save({'encoder': self.encoder.state_dict(), 'policy': self.policy.state_dict()}, os.path.join(self.output_dir, f"model_final.pth"))
        print("Training finished.")
