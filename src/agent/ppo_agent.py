import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import GradScaler, autocast  # 修正导入路径

class PPOAgent:
    def __init__(self, encoder, policy, cfg, device='cpu'):
        self.device = device
        self.encoder = encoder.to(device)
        self.policy = policy.to(device)
        params = list(self.encoder.parameters()) + list(self.policy.parameters())
        self.opt = optim.Adam(params, lr=cfg.get("lr", 2.5e-4))
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.gamma = cfg.get("gamma", 0.99)
        self.lmbda = cfg.get("gae_lambda", 0.95)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.epochs = cfg.get("epochs", 4)
        self.minibatch_size = cfg.get("minibatch_size", 64)
        
        # 修正 GradScaler 初始化（显式指定设备类型和正确参数）
        self.scaler = GradScaler(device=device) if device == 'cuda' else None

    def get_action_and_value(self, obs_seq):
        # 修正 autocast 上下文管理器
        with torch.no_grad(), autocast(device_type=self.device, enabled=self.scaler is not None):
            logits, value = self.policy(obs_seq)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
        return action.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

    def compute_gae(self, rewards, values, dones, last_value):
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam
        returns = advantages + values
        return advantages, returns

    def ppo_update(self, obs, actions, old_logps, returns, advantages):
        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_logps_t = torch.FloatTensor(old_logps).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advs = torch.FloatTensor(advantages).to(self.device)
        dataset_size = obs_t.shape[0]
        
        for epoch in range(self.epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start+self.minibatch_size]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logps = old_logps_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_advs = advs[mb_idx]
                
                # 修正 autocast 上下文管理器
                with autocast(device_type=self.device, enabled=self.scaler is not None):
                    logits, values = self.policy(mb_obs)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    mb_logps = dist.log_prob(mb_actions)
                    ratio = torch.exp(mb_logps - mb_old_logps)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (mb_returns - values).pow(2).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 修正梯度缩放逻辑
                self.opt.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)  # 为梯度裁剪准备
                else:
                    loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy.parameters()),
                    self.max_grad_norm
                )
                
                # 优化器步骤
                if self.scaler is not None:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()
