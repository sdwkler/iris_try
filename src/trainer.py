import os

import numpy as np

import tqdm

from tqdm import trange

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

import gymnasium as gym

import random  # 用于生成随机种子



from src.utils import ensure_dir

from src.envs.atari_vec import make_atari_env

from src.models.dino import DINOv2FeatureExtractor  # 保留但未使用

from src.models.rnd import RND

# 注意：导入新版的 TransformerPolicy/PPOAgent/RolloutBuffer

from src.models.transformer_policy import TransformerPolicy

from src.collector import RolloutBuffer

from src.agent.ppo_agent import PPOAgent

from src.models.vqvae import SimpleVQVAE



class Trainer:

    def __init__(self, cfg, device='cpu', mode='train'):

        self.cfg = cfg

        self.device = torch.device(device)  # 统一为 torch.device 类型

        self.output_dir = cfg.get("output_dir", "outputs")

        ensure_dir(self.output_dir)



        # ---- 环境配置 ----

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



        # =============================

        # 环境种子设置（保留，但简化逻辑）

        # =============================

        def make_random_seed_env():

            def _thunk():

                seed = random.randint(0, 2**32 - 1)

                env = make_atari_env(

                    self.env_id,

                    image_size=self.image_size,

                    frame_stack=self.frame_stack,

                    grayscale=self.grayscale,

                    atari_frame_skip=self.atari_frame_skip

                )

                env.reset(seed=seed)

                return env

            return _thunk



        env_fns = [make_random_seed_env() for _ in range(self.num_envs)]

        self.venv = gym.vector.SyncVectorEnv(env_fns)



        # ---- VQ-VAE ----

        vqvae_cfg = cfg.get("vqvae", {}).get("model", {})

        checkpoint_path = cfg.get("vqvae", {}).get("checkpoint_path", None)



        self.vqvae = SimpleVQVAE(

            in_channel=1 if self.grayscale else 3,

            channel=vqvae_cfg.get("channel", 128),

            n_res_block=vqvae_cfg.get("n_res_block", 2),

            n_res_channel=vqvae_cfg.get("n_res_channel", 64),

            embed_dim=vqvae_cfg.get("embed_dim", 128),

            n_embed=vqvae_cfg.get("n_embed", 512),

        ).to(self.device)



        if checkpoint_path and os.path.isfile(checkpoint_path):

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            model_state_dict = checkpoint.get('model_state_dict', checkpoint)

            self.vqvae.load_state_dict(model_state_dict)

            print(f"[INFO] 已加载 VQ-VAE 预训练权重 from {checkpoint_path}")

        else:

            print("[WARNING] 未提供 VQ-VAE 预训练权重，将使用随机初始化的 VQ-VAE")



        self.vqvae.eval()



        # ---- Transformer Policy（适配新版接口）----

        tcfg = cfg.get("transformer", {})

        self.policy = TransformerPolicy(

            input_dim=vqvae_cfg.get("embed_dim", 128),

            action_n=self.venv.single_action_space.n,

            d_model=tcfg.get("d_model", 256),

            n_heads=tcfg.get("n_heads", 4),

            num_layers=tcfg.get("num_layers", 3),

            mlp_dim=tcfg.get("mlp_dim", 512),

            dropout=tcfg.get("dropout", 0.1),

            seq_len=tcfg.get("seq_len", 4),

            # 新增：探索参数（从配置读取）

            exploration_noise=cfg.get("exploration", {}).get("noise", 0.1),

            action_perturb_prob=cfg.get("exploration", {}).get("perturb_prob", 0.1)

        ).to(self.device)



        # ---- RND ----

        if cfg.get("rnd", {}).get("enabled", True):

            self.rnd = RND(

                input_dim=vqvae_cfg.get("embed_dim", 128),

                device=self.device,

                predictor_hidden=cfg.get("rnd", {}).get("predictor_hidden", 256),

                target_hidden=cfg.get("rnd", {}).get("target_hidden", 256),

                lr=cfg.get("rnd", {}).get("rnd_lr", 1e-4)

            )

        else:

            self.rnd = None

        print("RND enabled:", self.rnd is not None)



        # ---- PPO Agent（适配新版接口）----

        # src/trainer.py 中构建 ppo_cfg 的部分

        agent_cfg = cfg.get("agent", {})

        ppo_cfg = {

            # ✅ 强制转换为数值类型，即使配置是字符串

            "lr": float(agent_cfg.get("lr", 2.5e-4)),

            "clip_eps": float(agent_cfg.get("clip_eps", 0.2)),

            "gamma": float(agent_cfg.get("gamma", 0.99)),

            "gae_lambda": float(agent_cfg.get("gae_lambda", 0.95)),

            "value_coef": float(agent_cfg.get("value_coef", 0.5)),

            "entropy_coef": float(agent_cfg.get("entropy_coef", 0.03)),

            "max_grad_norm": float(agent_cfg.get("max_grad_norm", 0.5)),

            "epochs": int(agent_cfg.get("epochs", 4)),

            "minibatch_size": int(agent_cfg.get("minibatch_size", 64)),

            # ✅ 关键：强制转换 weight_decay 为 float

            "weight_decay": float(agent_cfg.get("weight_decay", 1e-4)),

            "max_train_steps": int(agent_cfg.get("max_train_steps", 100000))

        }

        self.agent = PPOAgent(self.policy, ppo_cfg, device=self.device)

        # ---- Rollout & Training Config ----

        self.steps_per_env = cfg.get("training", {}).get("steps_per_env", 128)

        self.batch_size = self.steps_per_env * self.num_envs



        # 初始化缓冲区（只初始化一次，避免重复创建）

        seq_len = tcfg.get("seq_len", 4)

        input_dim = vqvae_cfg.get("embed_dim", 128)

        self.rollout = RolloutBuffer(

            self.steps_per_env,

            self.num_envs,

            obs_shape=(seq_len, input_dim),

            device=self.device  # 新增：传递设备信息

        )

        self.total_updates = cfg.get("training", {}).get("total_updates", 1000)

        self.prev_lives = None



        # 随机探索配置（简化：新版策略已内置探索，这里保留兜底）

        self.random_action_steps = 0

        self.use_random_actions_for_n_steps = cfg.get("training", {}).get("use_random_actions_for_n_steps", 1000)



    def encode_frame(self, obs):

        """修复VQ-VAE编码逻辑，确保维度正确"""

        # obs shape: [num_envs, frame_stack, H, W]

        num_envs, T, H, W = obs.shape

        # 调整维度：[num_envs*T, C, H, W]（C=1 for grayscale）

        batch_frames = torch.from_numpy(obs).float()  # [N, T, H, W]

        batch_frames = batch_frames.reshape(-1, H, W)  # [N*T, H, W]

        batch_frames = batch_frames.unsqueeze(1)  # [N*T, 1, H, W]（添加通道维）

        batch_frames = batch_frames.to(self.device)



        with torch.no_grad():

            z = self.vqvae.encoder(batch_frames)  # VQ-VAE编码器输出

            z_q, _, _ = self.vqvae.quantize(z)    # 量化后的潜在向量

            z_q_pooled = F.adaptive_avg_pool2d(z_q, 1)  # [N*T, D, 1, 1]

            z_q_pooled = z_q_pooled.squeeze(-1).squeeze(-1)  # [N*T, D]

        

        # 恢复时序维度：[num_envs, T, D]

        z_q_pooled = z_q_pooled.reshape(num_envs, T, -1)

        return z_q_pooled



    def collect_rollout(self):

        """适配蒙特祖马复仇：

        1. 区分单条命死亡（复活继续）和5条命耗光（重置局）

        2. 兼容SyncVectorEnv（无indices参数）的部分环境重置

        3. 允许缓冲区未填满，但尽量收集完整数据

        """

        seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)

        num_envs = self.num_envs



        # 重置环境（SyncVectorEnv 整体重置）

        obs, info = self.venv.reset()

        if isinstance(obs, dict):

            obs = obs.get('observation', obs)



        # 初始化剩余命数字典（跟踪每个环境的命数）

        self.prev_lives = np.ones(num_envs, dtype=int) * 5  # 初始5条命

        if "lives" in info:

            self.prev_lives = np.array(info["lives"]) if isinstance(info["lives"], list) else info["lives"]



        # 初始编码

        zpool_init = self.encode_frame(obs)

        per_env_seqs = zpool_init[:, -seq_len:, :].clone()  # [N, seq_len, D]



        # 重置缓冲区

        self.rollout.reset()



        # 收集数据（允许提前终止，但尽量收集到steps_per_env）

        for step in range(self.steps_per_env):

            # 蒙特祖马特判：如果所有环境都耗光5条命，提前终止收集

            if step > 0 and np.all(self.rollout.dones[step-1]):

                print(f"[Montezuma Warning] All envs lost all lives at step {step}, stop collecting")

                break



            obs_seq_np = per_env_seqs.cpu().numpy().astype(np.float32)



            # 随机/策略动作逻辑（保留）

            if self.random_action_steps < self.use_random_actions_for_n_steps:

                n_actions = self.venv.single_action_space.n

                actions = np.random.randint(0, n_actions, size=(num_envs,))

                logps = np.log(1.0 / n_actions) * np.ones(num_envs)

                with torch.no_grad():

                    obs_seq_tensor = torch.FloatTensor(obs_seq_np).to(self.device)

                    _, values = self.policy(obs_seq_tensor)

                values = values.cpu().numpy()

                self.random_action_steps += 1

            else:

                if self.random_action_steps == self.use_random_actions_for_n_steps:

                    print(f"[INFO] Switch to policy actions after {self.use_random_actions_for_n_steps} random steps")

                    self.random_action_steps += 1

                actions, logps, values = self.agent.get_action_and_value(obs_seq_np)



            # 执行动作（蒙特祖马易死亡，捕获异常）

            try:

                next_obs, ext_rewards, terms, truncs, infos = self.venv.step(actions)

            except Exception as e:

                print(f"[Montezuma Error] Env step failed at step {step}: {e}")

                break



            # ===================== 核心修改：5条命机制 =====================

            # 1. 获取当前剩余命数

            current_lives = np.ones(num_envs, dtype=int) * 5

            if "lives" in infos:

                current_lives = np.array(infos["lives"]) if isinstance(infos["lives"], list) else infos["lives"]

            

            # 2. 区分：

            # - terms（terminated）：5条命耗光/局结束 → 真正的done

            # - truncs（truncated）：步数超限 → 截断

            # - 单条命死亡：current_lives < prev_lives 但 terms=False → 复活继续

            single_life_death_indices = np.where((current_lives < self.prev_lives) & (~terms))[0]

            all_lives_lost_indices = np.where(terms)[0]



            # 3. 打印命数日志

            if len(single_life_death_indices) > 0:

                remaining_lives = current_lives[single_life_death_indices]

            # 4. 仅将5条命耗光视为done（单条命死亡不触发done）

            dones = terms  # 核心：只有耗光5条命才是真正的done

            self.prev_lives = current_lives.copy()  # 更新前序命数



            # ===================== 奖励计算 =====================

            # 稀疏奖励增强（蒙特祖马关键：放大有效奖励）

            total_rewards = ext_rewards

            if self.rnd is not None:

                zpool_next = self.encode_frame(next_obs)

                current_z = zpool_next[:, -1, :]

                current_z_tensor = current_z.to(self.device)

                intrinsic_rewards = self.rnd.compute_intrinsic(current_z_tensor)

                # 蒙特祖马：RND奖励权重翻倍，缓解稀疏奖励

                total_rewards = ext_rewards + self.cfg["rnd"].get("rnd_scale", 2.0) * intrinsic_rewards.detach().cpu().numpy()



            # ===================== 时序序列更新 =====================

            # 更新时序序列

            zpool_next = self.encode_frame(next_obs)

            new_z = zpool_next[:, -1, :].unsqueeze(1)

            per_env_seqs = torch.cat([per_env_seqs[:, 1:, :], new_z], dim=1)



            # 插入数据（即使部分环境单条命死亡，仍保留数据）

            self.rollout.insert(

                step,

                obs=obs_seq_np,

                action=actions,

                reward=total_rewards,

                done=dones,

                logp=logps,

                value=values

            )



            # ===================== SyncVectorEnv 重置逻辑 =====================

            # 仅当5条命耗光（terms=True）时才重置环境

            if dones.any():

                done_indices = np.where(dones)[0]

                

                # 1. 整体重置所有环境（SyncVectorEnv 只能这样做）

                all_reset_obs, all_reset_info = self.venv.reset()

                if isinstance(all_reset_obs, dict):

                    all_reset_obs = all_reset_obs.get('observation', all_reset_obs)

                

                # 2. 重新编码所有环境的新状态

                reset_zpool = self.encode_frame(all_reset_obs)

                

                # 3. 仅更新耗光5条命的环境的时序序列（非死亡环境保持原状态）

                per_env_seqs[done_indices] = reset_zpool[done_indices, -seq_len:, :].clone()

                

                # 4. 更新next_obs和命数（重置后的环境恢复5条命）

                next_obs = all_reset_obs

                if "lives" in all_reset_info:

                    self.prev_lives[done_indices] = np.array(all_reset_info["lives"])[done_indices]

                else:

                    self.prev_lives[done_indices] = 5  # 默认重置后恢复5条命



            obs = next_obs



        # ===================== 最终价值计算 =====================

        # 计算最终value（适配未填满的缓冲区）

        final_obs_seq_np = per_env_seqs.cpu().numpy().astype(np.float32)

        with torch.no_grad():

            final_obs_tensor = torch.FloatTensor(final_obs_seq_np).to(self.device)

            _, final_values = self.policy(final_obs_tensor)

        final_values = final_values.cpu().numpy()



        # ===================== 返回数据 =====================

        # 返回有效数据（即使缓冲区未填满）

        rollout_data = self.rollout.get()

        obs_roll = rollout_data[0] if len(rollout_data) > 0 else np.array([])

        # 打印缓冲区使用情况（监控数据收集完整性）

        valid_steps = obs_roll.shape[0] if obs_roll.size > 0 else step + 1

        total_samples = valid_steps * self.num_envs

        print(f"[Montezuma Info] Buffer used: {valid_steps}/{self.steps_per_env} steps, {total_samples}/{self.steps_per_env * self.num_envs} samples")

        

        return rollout_data, final_values



    def train(self):

        """适配蒙特祖马：稀疏奖励+不完整缓冲区的训练逻辑"""

        total_updates = self.cfg["training"]["total_updates"]



        with trange(total_updates, desc="Montezuma Training", unit="update") as pbar:

            for update in pbar:

                # 收集数据（允许未填满）

                rollout_data, last_values = self.collect_rollout()

                obs_roll, actions, rewards, dones, logps, values = rollout_data



                # 蒙特祖马特判：如果没有有效数据，跳过本次更新

                if obs_roll.size == 0:

                    print(f"[Montezuma Warning] No valid data at update {update}, skip")

                    continue



                # 计算GAE（适配非完整序列）

                advantages, returns = self.agent.compute_gae(rewards, values, dones, last_values)



                # PPO更新（新版agent已适配扁平化，直接传原始数据）

                loss = self.agent.ppo_update(obs_roll, actions, logps, returns, advantages)



                # 更新RND（保留）

                if self.rnd is not None:

                    last_frame_latents = obs_roll[:, :, -1, :]

                    last_frame_latents_flat = last_frame_latents.reshape(-1, last_frame_latents.shape[-1])

                    self.rnd.update(torch.FloatTensor(last_frame_latents_flat).to(self.device))



                # 保存检查点（保留）

                if (update + 1) % self.cfg.get("training", {}).get("save_every", 500) == 0:

                    checkpoint_path = os.path.join(self.output_dir, f"montezuma_checkpoint_{update + 1}.pth")

                    save_dict = {

                        'policy': self.policy.state_dict(),

                        'agent': self.agent.opt.state_dict(),

                        'update': update + 1,

                        'avg_reward': np.mean(rewards)

                    }

                    # 仅当RND存在且有state_dict方法时保存

                    if self.rnd is not None and hasattr(self.rnd, 'state_dict'):

                        save_dict['rnd'] = self.rnd.state_dict()

                    torch.save(save_dict, checkpoint_path)

                    print(f"✅ Saved Montezuma checkpoint to {checkpoint_path}")



                # 进度条（突出蒙特祖马的稀疏奖励）

                avg_ext_reward = np.mean(rewards)

                

                # ✅ 修复：正确计算RND内在奖励均值

                avg_int_reward = 0.0

                if self.rnd and hasattr(self.rnd, 'intrinsic_rewards') and self.rnd.intrinsic_rewards is not None:

                    avg_int_reward = torch.mean(self.rnd.intrinsic_rewards).item()



                # 新增：计算平均剩余命数

                avg_remaining_lives = 0.0

                if hasattr(self, 'prev_lives') and self.prev_lives is not None:

                    avg_remaining_lives = np.mean(self.prev_lives)



                pbar.set_postfix({

                    "ext_rew": f"{avg_ext_reward:.3f}",  # 蒙特祖马奖励稀疏，保留3位小数

                    "int_rew": f"{avg_int_reward:.3f}",

                    "loss": f"{loss:.3f}",

                    "valid_steps": f"{obs_roll.shape[0]}/{self.cfg['training']['steps_per_env']}",

                    "avg_lives": f"{avg_remaining_lives:.1f}/5",  # 显示平均剩余命数

                    "upd": update + 1

                })



    def evaluate(self, num_episodes=5):

        """评估蒙特祖马智能体性能（适配5条命机制）"""

        self.policy.eval()

        total_rewards = []

        total_lives_used = []

        total_steps = []



        for ep in range(num_episodes):

            obs, info = self.venv.reset()

            if isinstance(obs, dict):

                obs = obs.get('observation', obs)

            

            lives = np.ones(self.num_envs, dtype=int) * 5

            if "lives" in info:

                lives = np.array(info["lives"]) if isinstance(info["lives"], list) else info["lives"]

            

            ep_reward = np.zeros(self.num_envs)

            ep_steps = 0

            done = np.zeros(self.num_envs, dtype=bool)



            while not np.all(done):

                # 编码观测

                zpool = self.encode_frame(obs)

                seq_len = self.cfg.get("transformer", {}).get("seq_len", 4)

                per_env_seqs = zpool[:, -seq_len:, :].clone()

                obs_seq_np = per_env_seqs.cpu().numpy().astype(np.float32)



                # 策略动作（无随机探索）

                with torch.no_grad():

                    obs_seq_tensor = torch.FloatTensor(obs_seq_np).to(self.device)

                    action_logits, _ = self.policy(obs_seq_tensor)

                    actions = torch.argmax(action_logits, dim=-1).cpu().numpy()



                # 执行动作

                next_obs, rewards, terms, truncs, infos = self.venv.step(actions)

                done = np.logical_or(terms, truncs)



                # 更新命数和奖励

                current_lives = lives

                if "lives" in infos:

                    current_lives = np.array(infos["lives"]) if isinstance(infos["lives"], list) else infos["lives"]

                

                ep_reward += rewards

                ep_steps += 1

                lives = current_lives

                obs = next_obs



            # 统计结果

            total_rewards.append(ep_reward.mean())

            total_lives_used.append(5 - lives.mean())

            total_steps.append(ep_steps)



            print(f"Episode {ep+1}: Avg Reward={ep_reward.mean():.2f}, Avg Lives Used={5 - lives.mean():.1f}, Steps={ep_steps}")



        # 打印最终评估结果

        print("\n=== Montezuma Evaluation Results ===")

        print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")

        print(f"Average Lives Used: {np.mean(total_lives_used):.1f} ± {np.std(total_lives_used):.1f}")

        print(f"Average Steps per Episode: {np.mean(total_steps):.0f} ± {np.std(total_steps):.0f}")



        self.policy.train()

        return {

            "avg_reward": np.mean(total_rewards),

            "avg_lives_used": np.mean(total_lives_used),

            "avg_steps": np.mean(total_steps)

        }

