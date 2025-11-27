import torch
import torch.nn as nn

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

class TransformerPolicy(nn.Module):
    def __init__(self, input_dim, action_n, d_model=256, n_heads=4, num_layers=3, mlp_dim=512, dropout=0.1, seq_len=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.actor = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, action_n)
        )
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos(h)
        h = h.permute(1,0,2)
        out = self.transformer(h)
        out = out.permute(1,0,2)
        last = out[:, -1, :]
        logits = self.actor(last)
        value = self.critic(last).squeeze(-1)
        return logits, value