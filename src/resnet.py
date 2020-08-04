import torch
import torch.nn as nn

from noisy_conv2d import NoisyConv2d


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, num_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(num_channels, num_channels, 1, bias=False),
            nn.InstanceNorm2d(num_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(num_channels, in_channels, 3, padding=1),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.block[-1].weight)
            nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        x = x + self.block(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, num_channels, num_blocks):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResNetBlock(in_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.sigmoid()
        return x


if __name__ == "__main__":
    from operator import mul
    from functools import reduce

    x = torch.rand(1, 3, 32, 32)
    resnet = ResNet(3, 8, 128)

    num_parameters = 0
    for param in resnet.parameters():
        num_parameters += reduce(mul, param.size(), 1)
    y = resnet(x)
    print("Output shape:", y.shape)
    print("Number of parameters:", num_parameters)
