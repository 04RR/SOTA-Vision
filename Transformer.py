import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(MultiHeadSelfAttention, self).__init__()

        self.heads = heads

        try:
            self.head_dim = dim // heads

        except:
            print("dim must be divisible by number of heads.")

        self.to_key = nn.Linear(dim, self.head_dim * self.heads, bias=False)
        self.to_query = nn.Linear(dim, self.head_dim * self.heads, bias=False)
        self.to_value = nn.Linear(dim, self.head_dim * self.heads, bias=False)

        self.final = nn.Linear(self.heads * self.head_dim, dim, bias=False)

        self.scale = self.head_dim ** (1 / 2)

    def forward(self, x, mask=None):

        key = self.to_key(x)
        query = self.to_query(x)
        value = self.to_value(x)

        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.scale

        if mask:
            energy = energy.masked_fill(mask, -np.inf)

        attention = nn.Softmax()(energy)

        out = torch.einsum("... i j , ... j d -> ... i d", attention, value)
        # out = rearrange(out, "b h t d -> b t (h d)")
        out = self.final(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1, linear_dim=1024):
        super().__init__()

        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads)

        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, linear_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):

        x = self.dropout(self.mhsa(x, mask)) + x
        x = self.norm_1(x)
        x = self.linear(x) + x
        out = self.norm_2(x)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, linear_dim, layers, heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads, dropout, linear_dim) for _ in range(layers)]
        )

    def forward(self, x, mask=None):

        for layer in self.layers:
            x = layer(x, mask)

        return x

