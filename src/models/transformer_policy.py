import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerPolicy(nn.Module):
    def __init__(self, d_model=256, n_heads=4, num_layers=3, mlp_dim=512, seq_len=4, action_n=6, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, action_n)
        )

    def forward(self, x):
        # x: (B, seq_len, d_model)
        # transformer expects (seq_len, B, d_model)
        b, seq, d = x.shape
        x = x + self.pos_emb.unsqueeze(0)  # broadcast
        x = x.permute(1,0,2)
        out = self.transformer(x)  # (seq_len, B, d_model)
        out = out.permute(1,0,2)   # (B, seq_len, d_model)
        feat = out[:, -1, :]       # use last token
        q = self.action_head(feat)
        return q