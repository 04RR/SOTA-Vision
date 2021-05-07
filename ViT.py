import torch
import torch.nn as nn
from einops import rearrange, repeat
from Transformer import TransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels=3,
        patch_dim=16,
        classes=10,
        dim=512,
        blocks=6,
        heads=4,
        linear_dim=1024,
        classification=True,
    ):
        super(ViT, self).__init__()

        try:
            self.num_tokens = (img_dim // patch_dim) ** 2

        except:
            print(f"patch_dim must be a factor of img_dim")

        self.patch_dim = patch_dim
        self.classification = classification

        self.token_dim = (self.patch_dim ** 2) * in_channels

        self.project = nn.Linear(self.token_dim, dim)
        self.dropout = nn.Dropout(0.1)

        self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, dim))

        self.transformer = TransformerEncoder(dim, linear_dim, blocks, heads)

        self.final = nn.Linear(dim, classes)

    def forward(self, x):

        patches = rearrange(
            x,
            "b c (x_patch x) (y_patch y) -> b (x y) (c x_patch y_patch)",
            x_patch=self.patch_dim,
            y_patch=self.patch_dim,
        )
        batch_size = patches.shape[0]

        patches = self.project(patches)

        cls_token = repeat(
            self.cls_token, "b ... -> (b batch_size) ...", batch_size=batch_size
        )
        patches = torch.cat([cls_token, patches], dim=1) + self.embedding

        out = self.dropout(patches)
        out = self.transformer(out)
        out = self.final(out[:, 0, :]) if self.classification else out[:, 1:, :]

        return out

