import matplotlib.pyplot as plt
import torch
import torch.distributions as D
from absl import app, flags
from torchvision.utils import make_grid

from loss import prior
from model import Flow

FLAGS = flags.FLAGS
flags.DEFINE_string("ckpt", "logs", "path to checkpoint file")

cuda_device = torch.device("cuda:0")


def sample(flow, zs_shapes, T=0.7):
    prior = D.normal.Normal(0, T)
    zs = [prior.sample(shape).to(cuda_device) for shape in zs_shapes]
    x, log_det = flow.invert(zs)
    return x, log_det


def main(unused_argv):
    K = 32
    L = 3
    num_channels = 3
    flow = Flow(K, L, num_channels)
    ckpt = torch.load(FLAGS.ckpt)
    flow.load_state_dict(ckpt["flow_state_dict"])
    zs_shapes = ckpt["zs_shapes"]
    image_samples = sample(flow, zs_shapes)
    image_samples = make_grid(image_samples, nrow=8, normalize=True, range=(0, 1), pad_value=10)
    plt.imshow(image_samples)
    plt.show()


if __name__ == "__main__":
    app.run(main)
