import torch.nn as nn
import einops

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride:int = 1, padding:int = 0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)

        return x

class YOLOv1(nn.Module):
    '''
        input: (B, 3, 448, 448)
        output: (B, S, S, (B * 5 + classes))
    '''
    def __init__(self, in_channels:int = 3, classes: int = 20, B:int = 2, S:int = 7):
        super(YOLOv1, self).__init__()

        self.S = S
        self.B = B
        self.classes = classes

        self.maxpool = nn.MaxPool2d(2, 2)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv1 = ConvBlock(in_channels, 64, S, 2)

        self.conv2 = ConvBlock(64, 192, 3)

        self.conv3 = ConvBlock(192, 128, 1)
        self.conv4 = ConvBlock(128, 256, 3)
        self.conv5 = ConvBlock(256, 256, 1)
        self.conv6 = ConvBlock(256, 512, 3)

        self.conv_layer7 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(512, 256, 1),
                ConvBlock(256, 512, 3),
            ) for _ in range(4)
        ])
        self.conv8 = ConvBlock(512, 512, 1)
        self.conv9 = ConvBlock(512, 1024, 3) 

        self.conv_layer10 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1024, 512, 1),
                ConvBlock(512, 1024, 3),
            ) for _ in range(2)
        ])
        self.conv11 = ConvBlock(1024, 1024, 3, 1, 1)
        self.conv12 = ConvBlock(1024, 1024, 3, 2, 2)

        self.conv13 = ConvBlock(1024, 1024, 3, 1, 2)
        self.conv14 = ConvBlock(1024, 1024, 3, 1, 2)

        self.linear1 = nn.Linear(self.S * self.S * 1024, 4096)

        self.linear2 = nn.Linear(4096, self.S * self.S * (self.B * 5 + self.classes))

    def forward(self, x):
        x = self.maxpool(self.conv1(x))

        x = self.maxpool(self.conv2(x))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(self.conv6(x))

        for layer in self.conv_layer7:
            x = layer(x)
        
        x = self.conv8(x)
        x = self.maxpool(self.conv9(x))

        for layer in self.conv_layer10:
            x = layer(x)

        x = self.conv11(x)
        x = self.conv12(x)

        x = self.conv13(x)
        x = self.conv14(x)

        x = self.leakyrelu(self.linear1(nn.Flatten()(x)))
        x = self.linear2(x)

        x = einops.rearrange(x, 'b (s a k) -> b k s a', s= self.S, a= self.S, k= self.B * 5 + self.classes)

        return x      
