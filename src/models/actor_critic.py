import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

@dataclass
class ActorCriticOutput:
    logits_actions: torch.Tensor  # 动作logits [B, T, act_vocab_size]
    values: torch.Tensor          # 状态价值 [B, T]

@dataclass
class LossOutput:
    loss_total: torch.Tensor
    intermediate_losses: dict

class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size: int, use_original_obs: bool = False):
        super().__init__()
        self.act_vocab_size = act_vocab_size
        self.use_original_obs = use_original_obs  # 是否使用原始图像观测
        
        # 视觉特征提取（针对原始图像）
        if use_original_obs:
            self.cnn = nn.Sequential(
                nn.Conv2d(12, 32, kernel_size=8, stride=4),  # 输入4帧堆叠×3通道=12
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            self.fc = nn.Linear(64 * 7 * 7, 512)  # 适配64x64图像
        else:
            # 针对token嵌入的特征提取
            self.fc = nn.Linear(512, 512)  # 假设token嵌入维度为512
        
        # 策略头（输出动作概率）和价值头（输出状态价值）
        self.policy_head = nn.Linear(512, act_vocab_size)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> ActorCriticOutput:
        """
        前向传播计算动作logits和状态价值
        x: 输入观测
            - 若use_original_obs=True: 形状 [B, T, C, H, W]（C=12，4帧×3通道）
            - 若use_original_obs=False: 形状 [B, T, D]（D=512，token嵌入维度）
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # 处理观测特征
        if self.use_original_obs:
            # 图像输入：展平批次和时间维度，通过CNN提取特征
            x = x.flatten(0, 1)  # [B*T, C, H, W]
            x = self.cnn(x)      # [B*T, 特征数]
            x = self.fc(x)       # [B*T, 512]
            x = x.view(batch_size, seq_len, -1)  # [B, T, 512]
        else:
            # Token嵌入输入：直接通过全连接层
            x = self.fc(x)  # [B, T, 512]
        
        # 计算动作logits和状态价值
        logits_actions = self.policy_head(x)  # [B, T, act_vocab_size]
        values = self.value_head(x).squeeze(-1)  # [B, T]
        
        return ActorCriticOutput(
            logits_actions=logits_actions,
            values=values
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor], world_model: nn.Module, tokenizer: nn.Module, 
                    burn_in: int, imagine_horizon: int, gamma: float, lambda_: float, **kwargs) -> LossOutput:
        """计算Actor-Critic损失（PPO损失+价值损失+熵正则）"""
        from envs.world_model_env import WorldModelEnv  # 局部导入避免循环依赖
        
        # 1. 提取预热观测（用于初始化世界模型）
        burnin_obs = batch['observations'][:, :burn_in]  # [B, burn_in, C, H, W]
        
        # 2. 使用世界模型环境生成想象轨迹
        imagination_env = WorldModelEnv(
            actor_critic=self,
            horizon=imagine_horizon,
            gamma=gamma,
            lambda_=lambda_
        )
        imagination_env.reset(
            batch_size=burnin_obs.shape[0],
            burnin_observations=burnin_obs
        )
        imagined_trajectory, returns, advantages = imagination_env.imagine(self)
        
        # 3. 收集轨迹中的动作和价值
        actions = torch.stack([step['actions'] for step in imagined_trajectory], dim=1)  # [B, horizon]
        values = torch.stack([self.forward(step['observations'].unsqueeze(1)).values for step in imagined_trajectory], dim=1)  # [B, horizon]
        
        # 4. 计算PPO策略损失
        logits = self.forward(burnin_obs).logits_actions[:, -1]  # [B, act_vocab_size]
        action_dist = torch.distributions.Categorical(logits=logits)
        log_probs = action_dist.log_prob(actions)  # [B, horizon]
        policy_loss = -(log_probs * advantages.detach()).mean()  # 带优势估计的策略梯度
        
        # 5. 计算价值损失（MSE）
        value_loss = F.mse_loss(values, returns)
        
        # 6. 计算熵损失（鼓励探索）
        entropy = action_dist.entropy().mean()
        entropy_loss = -kwargs.get('entropy_weight', 0.001) * entropy
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        return LossOutput(
            loss_total=total_loss,
            intermediate_losses={
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item()
            }
        )
