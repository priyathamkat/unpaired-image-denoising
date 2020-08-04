import torch
import torch.nn as nn

from .invertible import InvertibleModule


class Actnorm(InvertibleModule):
    def __init__(self, num_channels):
        super().__init__()
        # parameters are lazily initialized in `forward`
        self.log_s = nn.Parameter(torch.empty(1, num_channels, 1, 1))
        self.b = nn.Parameter(torch.empty(1, num_channels, 1, 1))
        self.initialized = False

    def _reset_parameters(self, x):
        eps = 1e-6
        with torch.no_grad():
            s = torch.var(x, (0, 2, 3), keepdim=True) + eps
            print(s.squeeze())
            log_s = s.rsqrt().log()
            b = -torch.mean(x, (0, 2, 3), keepdim=True)
            
            self.log_s = nn.Parameter(log_s)
            self.b = nn.Parameter(b)
        self.initialized = True

    def forward(self, x, log_det):
        if self.training and not self.initialized:
            self._reset_parameters(x)

        h, w = x.shape[2:]
        y = self.log_s.exp() * (x + self.b)
        log_det = log_det + h * w * self.log_s.sum()
        return y, log_det

    @torch.no_grad()
    def invert(self, y, log_det):
        h, w = y.shape[2:]
        x = self.log_s.mul(-1).exp() * y - self.b
        log_det = log_det - h * w * self.log_s.sum()
        return x, log_det


if __name__ == "__main__":
    actnorm = Actnorm(4)
    x = torch.randn(2, 4, 4, 4)
    log_det = torch.randn(1)
    x_, log_det_ = actnorm.invert(*actnorm(x, log_det))
    print(x[0, 0])
    print(x_[0, 0])
    print(torch.allclose(x, x_))
    print(torch.allclose(log_det, log_det_))
