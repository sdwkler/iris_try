import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, conv_channels=(32,64,128), fc_dim=256):
        super().__init__()
        layers = []
        prev = in_channels
        for ch, k, s in zip(conv_channels, (8,4,3), (4,2,1)):
            layers.append(nn.Conv2d(prev, ch, kernel_size=k, stride=s))
            layers.append(nn.ReLU())
            prev = ch
        self.conv = nn.Sequential(*layers)
        # compute flattened conv output size in forward by adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(prev, fc_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv(x)
        h = self.pool(h).view(h.size(0), -1)
        h = self.fc(h)
        return h  # (B, fc_dim)