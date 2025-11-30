import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)



    def forward(self, x):

        seq_len = x.size(1)

        return x + self.pe[:seq_len].unsqueeze(0)



class Swish(nn.Module):

    """激活函数增强非线性"""

    def forward(self, x):

        return x * torch.sigmoid(x)



class LayerNorm(nn.Module):

    """自定义LayerNorm增强稳定性"""

    def __init__(self, dim, eps=1e-5):

        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(dim))

        self.beta = nn.Parameter(torch.zeros(dim))



    def forward(self, x):

        mean = x.mean(-1, keepdim=True)

        var = x.var(-1, unbiased=False, keepdim=True)

        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta



class TransformerPolicy(nn.Module):

    def __init__(self, input_dim, action_n, d_model=256, n_heads=4, num_layers=3, mlp_dim=512, dropout=0.1, seq_len=4, 

                 exploration_noise=0.1, action_perturb_prob=0.1):

        super().__init__()

        self.input_dim = input_dim

        self.action_n = action_n

        self.d_model = d_model

        self.seq_len = seq_len

        

        # 探索相关参数

        self.exploration_noise = exploration_noise  # 动作概率噪声

        self.action_perturb_prob = action_perturb_prob  # 随机动作概率

        

        # 输入投影层（增加维度变换和正则化）

        self.input_proj = nn.Sequential(

            nn.Linear(input_dim, d_model * 2),

            Swish(),

            LayerNorm(d_model * 2),

            nn.Dropout(dropout),

            nn.Linear(d_model * 2, d_model),

            Swish(),

            LayerNorm(d_model)

        )

        

        # Transformer编码器（增加层归一化和dropout）

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=d_model, 

            nhead=n_heads, 

            dim_feedforward=mlp_dim, 

            dropout=dropout, 

            activation='gelu',  # 更优的激活函数

            layer_norm_eps=1e-5,

            batch_first=False  # 保持permute逻辑

        )

        self.transformer = nn.TransformerEncoder(

            encoder_layer, 

            num_layers=num_layers,

            norm=LayerNorm(d_model)  # 输出层归一化

        )

        

        self.pos = PositionalEncoding(d_model, max_len=seq_len)

        

        # Actor头（更深的网络+正则化）

        self.actor = nn.Sequential(

            nn.Linear(d_model, mlp_dim),

            Swish(),

            LayerNorm(mlp_dim),

            nn.Dropout(dropout),

            nn.Linear(mlp_dim, mlp_dim // 2),

            Swish(),

            LayerNorm(mlp_dim // 2),

            nn.Linear(mlp_dim // 2, action_n)

        )

        

        # Critic头（分离值函数网络，避免耦合）

        self.critic = nn.Sequential(

            nn.Linear(d_model, mlp_dim),

            Swish(),

            LayerNorm(mlp_dim),

            nn.Dropout(dropout),

            nn.Linear(mlp_dim, mlp_dim // 2),

            Swish(),

            LayerNorm(mlp_dim // 2),

            nn.Linear(mlp_dim // 2, 1)

        )

        

        # 初始化权重（解决梯度消失）

        self.apply(self._init_weights)



    def _init_weights(self, m):

        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)

            if m.bias is not None:

                torch.nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):

            torch.nn.init.constant_(m.weight, 1.0)

            torch.nn.init.constant_(m.bias, 0.0)



    def forward(self, x):

        """前向传播（训练/评估）"""

        # x: [batch_size, seq_len, input_dim]

        h = self.input_proj(x)  # [B, L, D]

        h = self.pos(h)         # [B, L, D]

        h = h.permute(1, 0, 2)  # [L, B, D] (Transformer要求)

        out = self.transformer(h)  # [L, B, D]

        out = out.permute(1, 0, 2)  # [B, L, D]

        last = out[:, -1, :]       # [B, D] (取最后一个时序步)

        

        logits = self.actor(last)  # [B, action_n]

        value = self.critic(last).squeeze(-1)  # [B]

        return logits, value



    def get_action(self, x, training=True):

        """获取动作（带探索机制）"""

        logits, value = self.forward(x)

        probs = F.softmax(logits, dim=-1)

        

        # 训练时添加探索噪声

        if training:

            # 1. 动作概率添加高斯噪声（平滑分布）

            noise = torch.randn_like(probs) * self.exploration_noise

            probs = probs + noise

            probs = torch.clamp(probs, 1e-8, 1.0 - 1e-8)  # 避免0/1概率

            probs = probs / probs.sum(dim=-1, keepdim=True)  # 重新归一化

            

            # 2. 以一定概率随机选择动作（ε-贪心）

            batch_size = x.shape[0]

            rand_mask = torch.rand(batch_size, device=x.device) < self.action_perturb_prob

            if rand_mask.any():

                rand_probs = torch.ones(batch_size, self.action_n, device=x.device) / self.action_n

                probs[rand_mask] = rand_probs[rand_mask]

        

        # 创建分布并采样

        dist = torch.distributions.Categorical(probs=probs)

        action = dist.sample()

        logp = dist.log_prob(action)

        

        return action, logp, value

