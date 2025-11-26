import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict  # 新增：导入Dict类型
@dataclass  # 新增：定义LossOutput类
class LossOutput:
    loss_total: torch.Tensor
    intermediate_losses: dict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        h = self.activation(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.activation(h + self.skip(x))


class Tokenizer(nn.Module):
    def __init__(self, vocab_size, image_channels=3, image_size=64, num_res_blocks=2, 
                 channels=128, z_channels=64, beta=0.25, discriminator_channels=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.image_channels = image_channels
        self.image_size = image_size
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, channels, kernel_size=3, padding=1),
            *[ResidualBlock(channels, channels) for _ in range(num_res_blocks)],
            nn.Conv2d(channels, z_channels, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((image_size // 8, image_size // 8))  # 下采样
        )
        
        # 量化层
        self.quant_conv = nn.Conv2d(z_channels, 2 * z_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.beta = beta
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(z_channels, channels, kernel_size=3, padding=1),
            *[ResidualBlock(channels, channels) for _ in range(num_res_blocks)],
            nn.ConvTranspose2d(channels, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 输出[0,1]范围
        )
        
        # 判别器（用于对抗训练）
        self.discriminator = nn.Sequential(
            nn.Conv2d(image_channels, discriminator_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(discriminator_channels, 2 * discriminator_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * discriminator_channels, 4 * discriminator_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4 * discriminator_channels * (image_size // 8) **2, 1)
        )

        # 代码本（词汇表）
        self.codebook = nn.Embedding(vocab_size, z_channels)

    def encode(self, x, should_preprocess=True):
        if should_preprocess:
            x = x.float() / 255.0  # 归一化到[0,1]
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        
        # 离散化（VQ-VAE）
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        codebook = self.codebook.weight  # 使用实例化的codebook
        distances = torch.cdist(z_flat, codebook)
        tokens = distances.argmin(dim=1).reshape(z.shape[0], z.shape[2], z.shape[3])
        
        return tokens

    def decode(self, tokens, should_postprocess=True):
        # 从token获取嵌入
        z = F.embedding(tokens, self.codebook.weight).permute(0, 3, 1, 2)
        z = self.post_quant_conv(z)
        
        x_recon = self.decoder(z)
        
        if should_postprocess:
            x_recon = x_recon.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            x_recon = (x_recon * 255.0).byte()  # 转换回[0,255]
        
        return x_recon

    def encode_decode(self, x, should_preprocess=True, should_postprocess=True):
        tokens = self.encode(x, should_preprocess)
        return self.decode(tokens, should_postprocess)

    def compute_loss(self, batch: Dict[str, torch.Tensor],** kwargs) -> LossOutput:
        # 从批次中提取观测数据
        x = batch['observations']
        x_processed = x.float() / 255.0
        x_processed = x_processed.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # 编码解码过程
        tokens = self.encode(x, should_preprocess=False)  # 已预处理，设为False
        x_recon = self.decode(tokens, should_postprocess=False)
        
        # 重构损失
        recon_loss = F.mse_loss(x_recon, x_processed)
        
        # KL散度损失
        moments = self.quant_conv(self.encoder(x_processed))
        mean, logvar = torch.chunk(moments, 2, dim=1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean**2 - logvar.exp())
        
        # 对抗损失
        real_pred = self.discriminator(x_processed)
        fake_pred = self.discriminator(x_recon.detach())
        disc_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) + \
                   F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
        
        gen_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))
        
        # 计算VQ损失
        z = self.encoder(x_processed)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        codebook = self.codebook.weight
        distances = torch.cdist(z_flat, codebook)
        min_indices = distances.argmin(dim=1)
        z_q = F.embedding(min_indices, codebook).view(z.shape)
        vq_loss = F.mse_loss(z.detach(), z_q) + self.beta * F.mse_loss(z, z_q.detach())
        
        # 总损失（使用配置文件中的权重）
        loss_weights = kwargs.get('loss_weights', {})
        total_loss = (
            loss_weights.get('reconstruction', 1.0) * recon_loss +
            loss_weights.get('kl', 0.0001) * kl_loss +
            loss_weights.get('vq', 1.0) * vq_loss +
            loss_weights.get('discriminator', 0.1) * disc_loss +
            loss_weights.get('generator', 0.1) * gen_loss
        )
        
        return LossOutput(
            loss_total=total_loss,
            intermediate_losses={
                'reconstruction_loss': recon_loss.item(),
                'kl_loss': kl_loss.item(),
                'vq_loss': vq_loss.item(),
                'discriminator_loss': disc_loss.item(),
                'generator_loss': gen_loss.item()
            }
        )