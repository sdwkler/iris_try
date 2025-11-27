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
    def __init__(self, cfg, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.get("output_dir","outputs")
        ensure_dir(self.output_dir)
        env_cfg = cfg.get("env", {})
        self.env_id = env_cfg.get("id")
        self.num_envs = env_cfg.get("num_envs", 8)
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
        Robustly convert a numpy `frame` to torch tensor shape (1, C, H, W), float32 in [0,1].
        Accepts:
          - frame: (H, W) grayscale
          - frame: (H, W, C) color
          - frame: (stack, H, W) stacked grayscale frames -> take last frame
          - frame: (stack, H, W, C) stacked color frames -> take last frame
        Returns: tensor on self.device
        """
        arr = np.array(frame)
        # If it's a stack (first dim equals frame_stack), take last frame
        if arr.ndim == 3 and arr.shape[0] == self.frame_stack:
            # shape (stack, H, W) -> pick last
            arr = arr[-1]
        if arr.ndim == 4 and arr.shape[0] == self.frame_stack:
            # (stack, H, W, C)
            arr = arr[-1]
        # Now arr should be (H,W) or (H,W,C)
        if arr.ndim == 2:
            # grayscale -> (H,W) -> (H,W,1)
            arr = arr[..., None]
        # Ensure channels last
        if arr.ndim != 3:
            raise ValueError(f"Unexpected frame ndim after normalization: {arr.shape}")
        H, W, C = arr.shape
        # If channel-first already (rare), try to detect (C,H,W)
        # But we assume arr is (H,W,C) here.
        arr = arr.astype('float32') / 255.0
        # Convert to (C,H,W)
        arr = np.transpose(arr, (2,0,1)).copy()
        t = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1,C,H,W)
        return t

    def encode_frame(self, frame):
        """
        Return pooled latent vector for one frame.
        frame: numpy array (various shapes) -> converted with frame_to_tensor
        returns: 1D numpy array of size embedding_dim
        """
        ft = self.frame_to_tensor(frame)  # (1,C,H,W)
        with torch.no_grad():
            z = self.encoder(ft)  # (1, D, h, w)
            zpool = torch.mean(z, dim=[2,3]).squeeze(0)  # (D,)
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
        per_env_seqs = [ [] for _ in range(self.num_envs) ]
        obs = self.venv.reset()
        # initialize per-env sequences (z vectors)
        for i,e in enumerate(obs):
            zpool = self.encode_frame(e)  # uses frame_to_tensor inside
            for _ in range(seq_len):
                per_env_seqs[i].append(zpool)
        obs_roll, actions_roll, rewards_roll, dones_roll, logps_roll, values_roll = [], [], [], [], [], []
        for step in range(self.steps_per_env):
            # build batch obs of shape (N, seq_len, D)
            obs_batch = np.stack([np.stack(per_env_seqs[i][-seq_len:]) for i in range(self.num_envs)])
            obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
            logits, values = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample().cpu().numpy()
            logps = dist.log_prob(torch.tensor(actions)).cpu().numpy()
            values_np = values.cpu().numpy()
            next_obs, rews, terms, infos = self.venv.step(actions.tolist())
            # compute intrinsic reward
            if self.rnd is not None:
                zlist = []
                for e in next_obs:
                    zpool = self.encode_frame(e)
                    zlist.append(zpool)
                zbatch = torch.FloatTensor(np.stack(zlist)).to(self.device)
                intrinsic = self.rnd.compute_intrinsic(zbatch).cpu().numpy() * self.cfg.get("rnd",{}).get("rnd_scale",1.0)
            else:
                intrinsic = np.zeros_like(rews)
            total_rew = rews + intrinsic
            # update per_env_seqs with new latents
            for i in range(self.num_envs):
                zpool = self.encode_frame(next_obs[i])
                per_env_seqs[i].append(zpool)
            obs_roll.append(obs_batch)
            actions_roll.append(actions)
            rewards_roll.append(total_rew)
            dones_roll.append(np.array(terms, dtype=np.float32))
            logps_roll.append(logps)
            values_roll.append(values_np)
        obs_roll = np.stack(obs_roll)
        actions_roll = np.stack(actions_roll)
        rewards_roll = np.stack(rewards_roll)
        dones_roll = np.stack(dones_roll)
        logps_roll = np.stack(logps_roll)
        values_roll = np.stack(values_roll)
        last_obs_batch = np.stack([np.stack(per_env_seqs[i][-seq_len:]) for i in range(self.num_envs)])
        last_obs_t = torch.FloatTensor(last_obs_batch).to(self.device)
        _, last_values = self.policy(last_obs_t)
        last_values = last_values.cpu().numpy()
        return obs_roll, actions_roll, rewards_roll, dones_roll, logps_roll, values_roll, last_values

    def train(self):
        total_updates = self.total_updates
        for update in range(total_updates):
            obs_roll, actions, rewards, dones, logps, values, last_values = self.collect_rollout()
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