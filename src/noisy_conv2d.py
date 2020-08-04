import torch
import torch.nn as nn


class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.noise_scale = nn.Parameter(torch.Tensor(self.out_channels, 1, 1))
        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.weight, std=5e-2)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            nn.init.zeros_(self.noise_scale)


    def forward(self, input):
        x = super().forward(input)
        noise = torch.randn_like(x)
        output = x + self.noise_scale * noise
        return output
