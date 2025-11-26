import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class WorldModelOutput:
    obs_pred_logits: torch.Tensor
    reward_pred: torch.Tensor
    done_pred_logits: torch.Tensor


@dataclass
class LossOutput:
    loss_total: torch.Tensor
    intermediate_losses: Dict[str, float]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_pdrop, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(resid_pdrop)
        )
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: DictConfig):
        super().__init__()
        self.config = config
        self.obs_vocab_size = obs_vocab_size
        self.act_vocab_size = act_vocab_size
        self.sequence_length = config.tokens_per_block * config.max_blocks
        
        # 嵌入层
        self.obs_embedding = nn.Embedding(obs_vocab_size, config.embed_dim)
        self.act_embedding = nn.Embedding(act_vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(self.sequence_length, config.embed_dim)
        self.dropout = nn.Dropout(config.embed_pdrop)
        
        # Transformer层
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.attn_pdrop,
                config.resid_pdrop
            ) for _ in range(config.num_layers)
        ])
        
        # 输出头
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.obs_pred_head = nn.Linear(config.embed_dim, obs_vocab_size)
        self.reward_pred_head = nn.Linear(config.embed_dim, 1)
        self.done_pred_head = nn.Linear(config.embed_dim, 1)
        
        # 因果掩码（用于自注意力）
        self.register_buffer("causal_mask", self._create_causal_mask(self.sequence_length))

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """创建因果掩码，确保只能关注过去的token"""
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def initial_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """初始化隐藏状态"""
        return {
            'hidden': torch.zeros(batch_size, 0, self.config.embed_dim, device=self.device),
            'position': 0
        }

    def burn_in(self, obs_tokens: torch.Tensor, hidden_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预热阶段：处理初始观测序列"""
        batch_size, seq_len = obs_tokens.shape
        positions = torch.arange(hidden_state['position'], hidden_state['position'] + seq_len, 
                                device=self.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 嵌入观测
        obs_emb = self.obs_embedding(obs_tokens)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(obs_emb + pos_emb)
        
        # 通过Transformer
        for block in self.blocks:
            x = block(x, mask=self.causal_mask[:seq_len, :seq_len])
        
        # 更新隐藏状态
        return {
            'hidden': x,
            'position': hidden_state['position'] + seq_len
        }

    def step(self, action: torch.Tensor, hidden_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """单步预测：根据动作和当前状态预测下一个观测、奖励和结束标志"""
        batch_size = action.shape[0]
        pos = hidden_state['position']
        
        # 嵌入动作
        act_emb = self.act_embedding(action.unsqueeze(1))  # [B, 1, D]
        pos_emb = self.position_embedding(torch.tensor([pos], device=self.device).unsqueeze(0).repeat(batch_size, 1))
        x = self.dropout(act_emb + pos_emb)
        
        # 与历史状态拼接
        if hidden_state['hidden'].shape[1] > 0:
            x = torch.cat([hidden_state['hidden'], x], dim=1)
        
        # 确保不超过最大序列长度
        current_seq_len = x.shape[1]
        if current_seq_len > self.sequence_length:
            x = x[:, -self.sequence_length:]
        
        # 通过Transformer
        mask = self.causal_mask[:current_seq_len, :current_seq_len] if self.config.attention == 'causal' else None
        for block in self.blocks:
            x = block(x, mask=mask)
        
        # 预测输出
        x = self.ln_f(x)
        last_token = x[:, -1]  # 取最后一个token的输出
        obs_logits = self.obs_pred_head(last_token)
        reward = self.reward_pred_head(last_token).squeeze(-1)
        done_logits = self.done_pred_head(last_token).squeeze(-1)
        
        # 采样下一个观测token
        obs_token = torch.distributions.Categorical(logits=obs_logits).sample()
        
        # 更新隐藏状态
        new_hidden_state = {
            'hidden': x,
            'position': pos + 1
        }
        
        return obs_token, reward, done_logits.sigmoid(), new_hidden_state

    def forward(self, obs_tokens: torch.Tensor, act_tokens: torch.Tensor) -> WorldModelOutput:
        """前向传播，用于训练"""
        batch_size, seq_len = obs_tokens.shape
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)
        
        # 交替嵌入观测和动作（obs0, act0, obs1, act1, ...）
        obs_emb = self.obs_embedding(obs_tokens)  # [B, T, D]
        act_emb = self.act_embedding(act_tokens)  # [B, T-1, D]
        
        # 构建输入序列：obs0, act0, obs1, act1, ..., obsT-1
        x = torch.zeros(batch_size, 2*seq_len - 1, self.config.embed_dim, device=self.device)
        x[:, ::2, :] = obs_emb  # 偶数位置放观测
        x[:, 1::2, :] = act_emb  # 奇数位置放动作
        
        # 添加位置嵌入
        x = x + self.position_embedding(positions[:, :x.shape[1]])
        x = self.dropout(x)
        
        # 通过Transformer
        mask = self.causal_mask[:x.shape[1], :x.shape[1]] if self.config.attention == 'causal' else None
        for block in self.blocks:
            x = block(x, mask=mask)
        
        x = self.ln_f(x)
        
        # 提取预测目标
        obs_pred_logits = self.obs_pred_head(x[:, 2::2, :])  # 预测下一个观测
        reward_pred = self.reward_pred_head(x[:, 1::2, :]).squeeze(-1)  # 预测动作后的奖励
        done_pred_logits = self.done_pred_head(x[:, 1::2, :]).squeeze(-1)  # 预测动作后的结束标志
        
        return WorldModelOutput(
            obs_pred_logits=obs_pred_logits,
            reward_pred=reward_pred,
            done_pred_logits=done_pred_logits
        )

    def compute_loss(self, batch: Dict[str, torch.Tensor], tokenizer: nn.Module, **kwargs) -> LossOutput:
        """计算世界模型损失"""
        # 对观测进行编码
        obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True)
        act_tokens = batch['actions']
        
        # 前向传播获取预测
        outputs = self(obs_tokens, act_tokens)
        
        # 计算各部分损失
        obs_loss = F.cross_entropy(
            outputs.obs_pred_logits.transpose(1, 2),
            obs_tokens[:, 1:]  # 预测下一个观测
        )
        reward_loss = F.mse_loss(
            outputs.reward_pred,
            batch['rewards'][:, :-1]
        )
        done_loss = F.binary_cross_entropy_with_logits(
            outputs.done_pred_logits,
            batch['dones'][:, :-1].float()
        )
        
        # 总损失
        total_loss = obs_loss + reward_loss + done_loss
        
        return LossOutput(
            loss_total=total_loss,
            intermediate_losses={
                'observation_loss': obs_loss.item(),
                'reward_loss': reward_loss.item(),
                'done_loss': done_loss.item()
            }
        )

    @property
    def device(self):
        return next(self.parameters()).device
