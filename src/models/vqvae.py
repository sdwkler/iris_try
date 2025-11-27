import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch=1, hidden=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, hidden, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(hidden, z_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, hidden=128, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(z_dim, hidden, 3, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 1)
        )
    def forward(self, x):
        return self.net(x)

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', embed.clone())

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.permute(0,2,3,1).contiguous()
        flat_input = flat_input.view(-1, self.embedding_dim)
        emb = self.embedding.t()
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     - 2 * flat_input @ emb.t()
                     + (emb.pow(2).sum(1).unsqueeze(0)))
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).type(flat_input.dtype)
        quantized = encodings @ emb
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim).permute(0,3,1,2).contiguous()
        if self.training:
            n = encodings.sum(0)
            ema_w = torch.matmul(flat_input.t(), encodings)
            self.ema_cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
            self.ema_w.mul_(self.decay).add_(ema_w, alpha=1 - self.decay)
            n = self.ema_cluster_size + self.eps
            self.embedding.copy_(self.ema_w / n.unsqueeze(0))
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_loss = e_latent_loss + self.commitment_cost * F.mse_loss(quantized, inputs.detach())
        quantized = inputs + (quantized - inputs).detach()
        return quantized, q_loss, encoding_indices.view(input_shape[0], input_shape[2], input_shape[3])