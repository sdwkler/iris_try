import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
class PPOAgent:
    def __init__(self, policy, cfg, device='cpu'):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.policy.train()  # 确保训练模式
        
        # 优化器（调整学习率策略+权重衰减）
        params = list(self.policy.parameters())
        # src/agent/ppo_agent.py 中 __init__ 方法
        self.opt = optim.AdamW(  # 使用AdamW更稳定
            params, 
            lr=cfg.get("lr", 2.5e-4),
            weight_decay=cfg.get("weight_decay", 1e-4),  # 新增默认值 1e-4
            eps=1e-5
        )

        
        # 学习率调度器（防止学习率过高）
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, 
            T_max=cfg.get("max_train_steps", 10000),
            eta_min=1e-6
        )

        # PPO超参数
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.gamma = cfg.get("gamma", 0.99)
        self.lmbda = cfg.get("gae_lambda", 0.95)
        self.value_coef = cfg.get("value_coef", 0.5)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)
        self.max_grad_norm = cfg.get("max_grad_norm", 0.5)
        self.epochs = cfg.get("epochs", 4)
        self.minibatch_size = cfg.get("minibatch_size", 64)

        # 混合精度训练（修正设备判断）
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        
        # 记录梯度和损失（调试用）
        self.loss_history = []
        self.grad_norm_history = []

    def get_action_and_value(self, obs_seq):
        """获取动作、对数概率、值函数（修复梯度上下文）"""
        # obs_seq: [batch_size, seq_len, feature_dim]
        obs_seq_t = torch.FloatTensor(obs_seq).to(self.device)
        
        # 训练时保留梯度上下文（修复原代码no_grad导致的梯度中断）
        if self.policy.training:
            action, logp, value = self.policy.get_action(obs_seq_t, training=True)
        else:
            with torch.no_grad(), autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                action, logp, value = self.policy.get_action(obs_seq_t, training=False)
        return (
            action.detach().cpu().numpy(),
            logp.detach().cpu().numpy(),
            value.detach().cpu().numpy()
        )

    def compute_gae(self, rewards, values, dones, last_value):
        """修复GAE计算（处理维度和数值稳定性）"""
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)
        
        # 数值稳定：限制rewards范围
        rewards = np.clip(rewards, -10.0, 10.0)
        
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.lmbda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        # 归一化优势函数（关键：解决梯度爆炸）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def ppo_update(self, obs, actions, old_logps, returns, advantages):
        """修复PPO更新逻辑（确保梯度回传）"""
        # 数据形状调整：(T, N, ...) -> (T*N, ...)
        T, N = actions.shape
        obs_flat = obs.reshape(T*N, *obs.shape[2:])
        actions_flat = actions.reshape(T*N)
        old_logps_flat = old_logps.reshape(T*N)
        returns_flat = returns.reshape(T*N)
        advantages_flat = advantages.reshape(T*N)

        # 转换为tensor
        obs_t = torch.FloatTensor(obs_flat).to(self.device)
        actions_t = torch.LongTensor(actions_flat).to(self.device)
        old_logps_t = torch.FloatTensor(old_logps_flat).to(self.device)
        returns_t = torch.FloatTensor(returns_flat).to(self.device)
        advs_t = torch.FloatTensor(advantages_flat).to(self.device)

        dataset_size = obs_t.shape[0]
        total_loss = 0.0

        for epoch in range(self.epochs):
            # 打乱索引
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            
            for start in range(0, dataset_size, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_old_logps = old_logps_t[mb_idx]
                mb_returns = returns_t[mb_idx]
                mb_advs = advs_t[mb_idx]

                # 混合精度前向
                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    # 前向传播（policy.get_action会调用forward）
                    logits, values = self.policy(mb_obs)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    
                    # 计算新的logp（确保数值稳定）
                    mb_logps = dist.log_prob(mb_actions)
                    mb_logps = torch.clamp(mb_logps, -10.0, 0.0)  # 避免log(0)
                    
                    # PPO损失计算
                    ratio = torch.exp(mb_logps - mb_old_logps)
                    ratio = torch.clamp(ratio, 1e-8, 1e8)  # 数值稳定
                    
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # 值函数损失（clip避免过拟合）
                    value_clipped = mb_returns - torch.clamp(values - mb_returns, -self.clip_eps, self.clip_eps)
                    value_loss = 0.5 * torch.max(
                        (values - mb_returns).pow(2),
                        (value_clipped - mb_returns).pow(2)
                    ).mean()
                    
                    # 熵奖励（鼓励探索）
                    entropy = dist.entropy().mean()
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # 梯度回传
                self.opt.zero_grad(set_to_none=True)  # 更高效的梯度清零
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    # 梯度裁剪
                    self.scaler.unscale_(self.opt)
                    grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    # 优化器步进
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt.step()
                
                total_loss += loss.item()
                self.grad_norm_history.append(grad_norm.item())

            # 学习率调度
            self.lr_scheduler.step()

        # 记录损失
        avg_loss = total_loss / (self.epochs * (dataset_size // self.minibatch_size))
        self.loss_history.append(avg_loss)
        return avg_loss
