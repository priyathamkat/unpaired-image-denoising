import torch
import torch.nn as nn

from .invertible import InvertibleModule

cuda_device = torch.device("cuda:0")


class Affine(InvertibleModule):
    def __init__(self, num_channels, net_channels=128):
        super().__init__()
        assert num_channels % 2 == 0
        self.split_size = num_channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(self.split_size, net_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
            nn.Conv2d(net_channels, net_channels, 1, bias=False),
            nn.BatchNorm2d(net_channels),
            nn.ReLU(),
            nn.Conv2d(net_channels, 2 * self.split_size, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.ones(1, self.split_size, 1, 1))
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, log_det):
        x_a, x_b = torch.split(x, self.split_size, 1)
        s, t = torch.split(self.net(x_b), self.split_size, 1)
        s = self.scale * s.tanh()

        y_a = s.exp() * (x_a + t)
        y_b = x_b
        y = torch.cat((y_a, y_b), 1)
        log_det = log_det + s.flatten(1).sum(-1)
        return y, log_det

    @torch.no_grad()
    def invert(self, y, log_det):
        y_a, y_b = torch.split(y, self.split_size, 1)
        s, t = torch.split(self.net(y_b), self.split_size, 1)
        s = self.scale * s.tanh()

        x_a = s.mul(-1).exp() * y_a - t
        x_b = y_b
        x = torch.cat((x_a, x_b), 1)
        log_det = log_det - s.flatten(1).sum(-1)
        return x, log_det


if __name__ == "__main__":
    affine = Affine(4).cuda()
    x = torch.randn(2, 4, 4, 4).to(cuda_device)
    log_det = torch.randn(1).to(cuda_device)
    x_, log_det_ = affine.invert(*affine(x, log_det))
    print(x[0, 0])
    print(x_[0, 0])
    print(torch.allclose(x, x_))
    print(torch.allclose(log_det, log_det_))
