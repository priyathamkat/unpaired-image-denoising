import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import flags

from invertible_layers import *

FLAGS = flags.FLAGS

flags.DEFINE_enum("coupling", "affine", ["affine", "additive"], "coupling to use")

cuda_device = torch.device("cuda:0")


def squeeze(x, block_size=2):
    n, c, h, w = x.size()
    unfolded_x = F.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)


def unsqueeze(x, block_size=2):
    return F.pixel_shuffle(x, block_size)


def flow_step(num_channels):
    return InvertibleSequential(
        Actnorm(num_channels),
        Conv1x1(num_channels),
        Affine(num_channels) #if FLAGS.coupling == "affine" else Additive(num_channels)
    )


class FlowBlock(InvertibleModule):
    def __init__(self, K, num_channels):
        super().__init__()
        # num_channels quadrapules due to squeezing
        self.num_channels = num_channels
        self.flow_steps = InvertibleSequential(
            *[flow_step(4 * self.num_channels) for _ in range(K)]
        )

    def forward(self, x, log_det):
        x = squeeze(x, 2)
        x, log_det = self.flow_steps(x, log_det)
        y, z = torch.split(x, 2 * self.num_channels, 1)
        return y, log_det, z

    @torch.no_grad()
    def invert(self, y, log_det, z):
        # spatial size of x should be divisible by 2
        # enough number of times for `unsqueeze` to
        # work accurately
        y = torch.cat((y, z), 1)
        y, log_det = self.flow_steps.invert(y, log_det)
        x = unsqueeze(y, 2)
        return x, log_det


class Flow(InvertibleModule):
    def __init__(self, K, L, num_channels):
        super().__init__()
        self.K = K
        self.L = L
        self.flow_blocks = nn.ModuleList([FlowBlock(K, num_channels * 2 ** i) for i in range(L - 1)])
        self.flow_steps = InvertibleSequential(
            *[flow_step(num_channels * 2 ** (L + 1)) for _ in range(K)]
        )

    def preprocess(self, x):
        alpha = torch.tensor([0.05]).to(cuda_device)  # pylint: disable=not-callable
        x = (255 * x + torch.rand_like(x)) / 256
        x = alpha + (1 - 2 * alpha) * x
        x = x.log() - (1 - x).log()
        log_det = (1 - 2 * alpha).log() + F.softplus(-x) + F.softplus(x)
        log_det = log_det.flatten(1).sum(-1)
        return x, log_det

    def postprocess(self, x, log_det):
        alpha = torch.tensor([0.05]).to(cuda_device)  # pylint: disable=not-callable
        log_det = log_det - ((1 - 2 * alpha).log() + F.softplus(-x) + F.softplus(x)).flatten(1).sum(-1)
        x = F.softplus(-x).mul(-1).exp()
        x = (x - alpha) / (1 - 2 * alpha)
        return x, log_det


    def forward(self, x):
        # Warning: ensure that `x` has enough elements
        # for `actnorm` layers to be initialized stably.
        x, log_det = self.preprocess(x)
        zs = []
        for flow_block in self.flow_blocks:
            x, log_det, z = flow_block(x, log_det)
            zs.append(z)
        x = squeeze(x, 2)
        z, log_det = self.flow_steps(x, log_det)
        zs.append(z)
        return zs, log_det

    @torch.no_grad()
    def invert(self, zs):
        batch_size = zs[0].shape[0]
        log_det = torch.zeros(batch_size).to(cuda_device)
        z = zs.pop()
        x, log_det = self.flow_steps.invert(z, log_det)
        x = unsqueeze(x, 2)
        for flow_block in reversed(self.flow_blocks):
            z = zs.pop()
            x, log_det = flow_block.invert(x, log_det, z)
        x, log_det = self.postprocess(x, log_det)
        return x, log_det


if __name__ == "__main__":
    flow = Flow(32, 3, 3).cuda()
    x = torch.rand(10, 3, 32, 32).to(cuda_device)
    zs, log_det = flow(x)
    x_, log_det_ = flow.invert(zs)
    print(x[1, 0, :4, :4])
    print(x_[1, 0, :4, :4])
    print(log_det)
    print(log_det_)
    print(torch.allclose(x, x_))
    print(torch.allclose(log_det, -log_det_))
