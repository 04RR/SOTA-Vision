import torch.nn as nn
from ViT import ViT
from einops import rearrange
import torch


class BottleNeckUnit(nn.Module):
    def __init__(self, in_channels, out_channels, base_width=64, stride=1):
        super(BottleNeckUnit, self).__init__()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        gamma = (base_width // 64) * out_channels

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                gamma * out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(gamma * out_channels),
            nn.Conv2d(
                gamma * out_channels,
                gamma * out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=1,
                bias=False,
                dilation=1,
            ),
            nn.BatchNorm2d(gamma * out_channels),
            nn.Conv2d(
                gamma * out_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):

        x = self.layer(x)
        identity = self.downsample(x)

        x += identity

        return nn.ReLU()(x)


class TransUNetEncoder(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        in_channels,
        classes,
        layers=12,
        heads=4,
        linear_dim=1024,
    ):
        super(TransUNetEncoder, self).__init__()

        # inital conv 3 -> channels
        # bottleneck channels -> channels*2
        # bottleneck channels*2 -> channels*4
        # bottleneck channels*4 -> channels*8
        # vit
        # conv -> channels*8 -> 512

        self.channels = 128
        self.img_dim = img_dim
        self.patch_dim = patch_dim

        self.layer1 = nn.Conv2d(
            in_channels, self.channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.layer2 = BottleNeckUnit(self.channels, self.channels * 2)
        self.layer3 = BottleNeckUnit(self.channels * 2, self.channels * 4)
        self.layer4 = BottleNeckUnit(self.channels * 4, self.channels * 8)
        self.layer5 = ViT(img_dim, in_channels=self.channels * 8, classification=False)
        self.layer6 = nn.Conv2d(self.channels * 8, 512, 3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(512)

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.layer4(x3)
        x = self.layer5(x)
        x = rearrange(
            x,
            "b (x y) c -> b c x y",
            x=self.img_dim // self.patch_dim,
            y=self.img_dim // self.patch_dim,
        )

        x = self.batchnorm(self.layer6(x))

        return nn.ReLU()(x), x1, x2, x3


class TransUNetDecoderUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransUNetDecoderUnit, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_skip=None):

        x = self.upsample(x)

        if x_skip is not None:
            x = torch.cat([x_skip, x], dim=1)

        out = self.layer(x)

        return out


class TransUNet(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        in_channels=3,
        classes=1,
        layers=6,
        heads=8,
        linear_dim=1024,
    ):
        super(TransUNet, self).__init__()

        self.encoder = TransUNetEncoder(
            img_dim, patch_dim, in_channels, classes, layers, heads, linear_dim
        )

        self.decoder1 = TransUNetDecoderUnit(1024, 256)
        self.decoder2 = TransUNetDecoderUnit(512, 128)
        self.decoder3 = TransUNetDecoderUnit(256, 64)
        self.decoder4 = TransUNetDecoderUnit(64, 16)

        self.conv = nn.Conv2d(16, classes, kernel_size=1, bias=False)

    def forward(self, x):

        x, x1, x2, x3 = self.encoder(x)

        out = self.decoder1(x, x3)
        out = self.decoder2(out, x2)
        out = self.decoder3(out, x1)
        out = self.decoder4(out)

        out = self.conv(out)

        return out

