import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel // reduction, in_channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.pool(x)
        x_ = self.conv(x_)
        return x * x_


class RCAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Sigmoid(),
        )

        self.expand = nn.Conv2d(in_channels // 4, in_channels, 1)

        self.ca = ChannelAttention(in_channels)

    def forward(self, x):

        x_ = self.ca(self.layer1(x))
        x_1 = self.layer2(x_)

        b, c, h, w = x_1.shape

        x_1 = x_1.reshape((b, c // 4, h * 2, w * 2))
        out = torch.matmul(self.expand(x_1), x_)

        return out + x


class ResidualGroup(nn.Module):
    def __init__(self, in_channels, out_channels, n=8):
        super().__init__()

        self.layer = nn.ModuleList([RCAB(in_channels) for _ in range(n)])

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):

        x_ = x.clone()

        for layer in self.layer:
            x = layer(x)

        x += x_

        return self.conv(x)


class LAM(nn.Module):
    # Copied from the actual implementation: https://github.com/wwlCape/HAN
    def __init__(self, in_dim= 3, out_dim= 64):
        super().__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.final = nn.Conv2d(out_dim * 3, out_dim, 1, 1)

    def forward(self, x):

        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, N, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma * out + x
        out = out.view(m_batchsize, -1, height, width)
        
        return self.softmax(self.final(out))


class CAM(nn.Module): 
    # Copied from the actual implementation: https://github.com/wwlCape/HAN
    def __init__(self, in_dim):
        super().__init__()

        self.chanel_in = in_dim

        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))

        out = self.gamma * out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x


class HAN(nn.Module):
    def __init__(self, in_channels= 3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 1, 1)

        self.rg1 = ResidualGroup(64, 64)
        self.rg2 = ResidualGroup(64, 64)
        self.rg3 = ResidualGroup(64, 64)

        self.conv2 = nn.Conv2d(64, 64, 1, 1)

        self.cam = CAM(64)
        self.lam = LAM()

        self.up = nn.Upsample(scale_factor=2)
        self.final = nn.Conv2d(64, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):

        x = self.conv1(x)

        x1 = self.rg1(x)
        x2 = self.rg1(x1)
        x3 = self.rg1(x2)

        x_1 = self.conv2(x3)

        x_1 = self.cam(x_1)

        b, c, h, w = x1.shape
        x1, x2, x3 = (
            x1.reshape((b, 1, c, h, w)),
            x2.reshape((b, 1, c, h, w)),
            x3.reshape((b, 1, c, h, w)),
        )

        x_2 = torch.cat([x1, x2, x3], dim=1)

        x += x_1 + self.lam(self.softmax(x_2))

        return self.sigmoid(self.final(self.up(x)))