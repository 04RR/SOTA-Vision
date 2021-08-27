import torch.nn as nn
import einops
import torch


class FastFormer(nn.Module):
    def __init__(self, in_dims= 512, token_dim= 512, num_heads= 1):
        super().__init__()

        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.to_value = nn.Linear(in_dims, token_dim * num_heads)
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

        self.alpha = nn.Parameter(torch.randn(token_dim * num_heads))
        self.beta = nn.Parameter(torch.randn(token_dim * num_heads))

        self.scale_factor = token_dim ** -0.5
        self.softmax = nn.Softmax(-1)

    def forward(self, x):

        key = self.to_key(x)
        query = self.to_query(x)
        value = self.to_value(x)

        query_weight = query * self.alpha
        query_weight = self.softmax(query_weight * self.scale_factor)
        global_query = torch.einsum("b n d -> b d", query * query_weight)

        global_query = einops.repeat(
            global_query, "b d -> b repeat d", repeat=key.shape[1]
        )
        key = global_query * key

        key_weight = key * self.beta
        key_weight = self.softmax(key_weight * self.scale_factor)
        global_key = torch.einsum("b n d -> b d", key * key_weight)

        attention = torch.einsum("b d, b n d -> b n d", global_key, value)
        out = self.final(attention + query)

        return out

