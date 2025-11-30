import torch
from torch import nn
from torch.nn import functional as F


# ======================
# 1. Quantize 模块（核心，保持不变）
# ======================
class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        B, C, H, W = input.shape
        flatten = input.permute(0, 2, 3, 1).contiguous().view(-1, C)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(B, H, W)
        quantize = self.embed_code(embed_ind)
        quantize = quantize.view(B, H, W, C).permute(0, 3, 1, 2)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


# ======================
# 2. ResBlock（增强表达能力）
# ======================
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, x):
        return x + self.conv(x)


# ======================
# 3. Encoder（增强：更多通道 + 1~2 ResBlock）
# ======================
class Encoder(nn.Module):
    def __init__(self, in_channel, channel=128, n_res_block=2, n_res_channel=64, stride=4):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ======================
# 4. Decoder（对称结构，匹配 Encoder）
# ======================
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel=128, n_res_block=2, n_res_channel=64, stride=4):
        super().__init__()
        blocks = [
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))
        blocks += [
            nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ======================
# 5. 完整 VQ-VAE（增强版，推荐使用 ✅）
# ======================
class SimpleVQVAE(nn.Module):
    def __init__(self, in_channel=1, channel=128, n_res_block=2, n_res_channel=64, embed_dim=128, n_embed=512):
        super().__init__()
        self.encoder = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.quantize = Quantize(embed_dim, n_embed)
        self.decoder = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)

    def forward(self, x):
        z = self.encoder(x)
        z_q, diff, _ = self.quantize(z)
        recon = self.decoder(z_q)
        return recon, diff