import torch
import torch.nn as nn
import torch.nn.functional as F

from .invertible import InvertibleModule

cuda_device = torch.device("cuda:0")


class Conv1x1(InvertibleModule):
    def __init__(self, num_channels):
        super().__init__()
        q, _ = torch.qr(torch.randn(num_channels, num_channels))
        q = q.to(cuda_device)
        self.W = nn.Parameter(q)

    def forward(self, x, log_det):
        h, w = x.shape[2:]
        y = F.conv2d(x, self.W.unsqueeze(-1).unsqueeze(-1))
        log_det = log_det + h * w * torch.slogdet(self.W)[1]
        return y, log_det

    @torch.no_grad()
    def invert(self, y, log_det):
        h, w = y.shape[2:]
        W_inv = self.W.inverse()
        x = F.conv2d(y, W_inv.unsqueeze(-1).unsqueeze(-1))
        log_det = log_det - h * w * torch.slogdet(self.W)[1]
        return x, log_det


if __name__ == "__main__":
    conv1x1 = Conv1x1(4).cuda()
    x = torch.randn(2, 4, 4, 4).to(cuda_device)
    log_det = torch.randn(1).to(cuda_device)
    y, log_det = conv1x1(x, 0)
    x_, log_det_ = conv1x1.invert(*conv1x1(x, log_det))
    print(x[0, 0])
    print(x_[0, 0])
    print(torch.allclose(x, x_))
    print(torch.allclose(log_det, log_det_))
