import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

# 调整导入顺序，避免循环依赖
from models.actor_critic import ActorCritic  # 直接导入ActorCritic
from agent import Agent
from episode import Episode


class WorldModelEnv:
    def __init__(self, agent: Agent, horizon: int, gamma: float, lambda_: float):
        self.agent = agent
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ = lambda_
        self.reset()

    def reset(self, batch_size: Optional[int] = None, burnin_observations: Optional[torch.Tensor] = None) -> None:
        self.batch_size = batch_size if batch_size is not None else 1
        self.burnin_observations = burnin_observations
        self.hidden_states = None
        self.returns = torch.zeros(self.batch_size, device=self.agent.device)
        self.advantages = torch.zeros(self.batch_size, device=self.agent.device)

    def imagine(self, actor_critic: ActorCritic) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """明确指定actor_critic的类型为ActorCritic，避免类型模糊"""
        assert self.burnin_observations is not None, "Burn-in observations must be provided"
        
        # 编码初始观测
        with torch.no_grad():
            obs_tokens = self.agent.tokenizer.encode(self.burnin_observations, should_preprocess=True)
        
        # 初始化世界模型状态
        self.hidden_states = self.agent.world_model.initial_state(self.batch_size)
        self.hidden_states = self.agent.world_model.burn_in(obs_tokens, self.hidden_states)
        
        imagined_trajectory = []
        for t in range(self.horizon):
            # Actor-Critic 决策（使用正确的观测输入）
            # 根据actor_critic配置选择原始观测或tokenized观测
            if actor_critic.use_original_obs:
                # 使用原始观测（从burnin中取当前步）
                current_obs = self.burnin_observations[:, t:t+1]  # [B, 1, C, H, W]
            else:
                # 使用解码后的token观测
                current_obs = self.agent.tokenizer.decode(obs_tokens, should_postprocess=True).unsqueeze(1)
            
            policy_output = actor_critic(current_obs)
            action = torch.distributions.Categorical(logits=policy_output.logits_actions[:, -1]).sample()
            
            # 世界模型预测下一个状态和奖励
            with torch.no_grad():
                next_tokens, reward_pred, done_pred, self.hidden_states = self.agent.world_model.step(
                    action, self.hidden_states
                )
            
            # 解码观测（用于可视化）
            next_obs = self.agent.tokenizer.decode(next_tokens, should_postprocess=True)
            
            imagined_trajectory.append({
                'observations': next_obs,
                'actions': action,
                'rewards': reward_pred,
                'dones': done_pred
            })
            
            # 更新优势估计（修正GAE计算逻辑）
            delta = reward_pred + self.gamma * (1 - done_pred.float()) * actor_critic(current_obs).values[:, -1] - actor_critic(current_obs).values[:, -1]
            self.advantages = delta + self.gamma * self.lambda_ * (1 - done_pred.float()) * self.advantages
            self.returns = reward_pred + self.gamma * (1 - done_pred.float()) * self.returns
            
            # 更新当前观测token
            obs_tokens = next_tokens
        
        return imagined_trajectory, self.returns, self.advantages
