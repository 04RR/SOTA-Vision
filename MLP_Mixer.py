import torch
import torch.nn as nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, mlp_dim, dim, dropout=0.0):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, mlp_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        return self.mlp(x)


class Embedding(nn.Module):
    def __init__(self, in_channels, dim, patch_dim):
        super(Embedding, self).__init__()

        self.patch_dim = patch_dim
        self.conv = nn.Conv2d(in_channels, dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):

        assert (
            x.shape[-1] % self.patch_dim == 0
        ), "patch_dim must be divisible by img_dim"

        out = self.conv(x)
        out = rearrange(out, "b c h w -> b (h w) c")

        return out


class MLPMixerBlock(nn.Module):
    def __init__(self, img_dim, token_dim, channel_dim, dim, in_channels, patch_dim):
        super(MLPMixerBlock, self).__init__()

        num_patches = (img_dim // patch_dim) ** 2

        self.norm1 = nn.LayerNorm(dim)
        self.mlp1 = MLP(num_patches, token_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = MLP(dim, channel_dim)

    def forward(self, x):

        x_ = self.norm1(x)
        x_ = torch.transpose(x_, 1, 2)
        x_ = self.mlp1(x_)
        x_ = torch.transpose(x_, 1, 2) + x

        x = self.norm2(x_)
        out = self.mlp2(x) + x_

        return out


class GlobalAveragePooling(nn.Module):
    def __init__(self, dim: int = 1):
        super(GlobalAveragePooling, self).__init__()

        self.dim = dim

    def forward(self, x):

        return x.mean(self.dim)


class MLPMixer(nn.Module):
    def __init__(
        self,
        classes: int,
        blocks: int = 8,
        img_dim: int = 256,
        patch_dim: int = 32,
        in_channels: int = 3,
        dim: int = 512,
        token_dim: int = 256,
        channel_dim: int = 2048,
    ):
        super(MLPMixer, self).__init__()

        try:
            num_patches = (img_dim // patch_dim) ** 2

        except:
            "patch_dim must be divisible by img_dim"

        self.embedding = Embedding(in_channels, dim, patch_dim)

        self.mixer = nn.ModuleList(
            [
                MLPMixerBlock(
                    img_dim, token_dim, channel_dim, dim, in_channels, patch_dim
                )
                for _ in range(blocks)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.gap = GlobalAveragePooling()
        self.mlp_head = nn.Linear(dim, classes)

    def forward(self, x):

        x = self.embedding(x)

        for layer in self.mixer:
            x = layer(x)
        x = self.norm(x)
        x = self.gap(x)
        x = self.mlp_head(x)

        return x

