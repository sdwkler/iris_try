import torch
import torch.nn as nn
import torch.nn.functional as F

class RNDModel(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
    def forward(self, x):
        return self.net(x)

class RND:
    def __init__(self, input_dim, device='cpu', predictor_hidden=256, target_hidden=256, lr=1e-4):
        self.device = device
        self.predictor = RNDModel(input_dim, predictor_hidden).to(device)
        self.target = RNDModel(input_dim, target_hidden).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic(self, obs_latent):
        with torch.no_grad():
            t = self.target(obs_latent)
        p = self.predictor(obs_latent)
        loss = F.mse_loss(p, t.detach(), reduction='none').mean(dim=1)
        return loss

    def update(self, obs_latent):
        self.predictor.train()
        t = self.target(obs_latent)
        p = self.predictor(obs_latent)
        loss = F.mse_loss(p, t.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()