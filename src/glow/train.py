from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from data import get_testloader, get_trainloader
from loss import nll_loss
from model import Flow
from sample import sample

FLAGS = flags.FLAGS
flags.DEFINE_float("lr", 1e-3, "learning rate")
flags.DEFINE_integer("num_epochs", 100, "number of epochs")
flags.DEFINE_string("logs", "../glow_logs", "folder to write logs")
flags.DEFINE_integer("test_every", 10, "frequency of testing")
flags.DEFINE_string("resume_from", "../glow_logs/20191208-110750/ckpt-100.pth", "ckpt file to preload")
flags.DEFINE_integer("crop_size", 32, "image size for training")

cuda_device = torch.device("cuda:0")
one = torch.ones(1).to(cuda_device)


def train(unused_argv):
    # image size must be divisible by 2 ** L
    if FLAGS.dataset == "CIFAR":
        K = 32
        L = 3
        num_channels = 3
    elif FLAGS.dataset == "COCO":
        K = 32
        L = 3
        num_channels = 3

    flow = Flow(K, L, num_channels).cuda()
    optimizer = optim.Adam(flow.parameters(), lr=FLAGS.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 100))
    epochs_done = 1
    if FLAGS.resume_from:
        ckpt = torch.load(FLAGS.resume_from)
        flow.load_state_dict(ckpt["flow_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        epochs_done = ckpt["epoch"]
        flow.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = Path(FLAGS.logs) / timestamp
    writer = SummaryWriter(str(log_folder))

    zs_shapes = None
    bpd = None

    for epoch in range(epochs_done, FLAGS.num_epochs + epochs_done):
        trainloader = get_trainloader(FLAGS.crop_size)
        pbar = tqdm(trainloader, desc="epoch %d" % epoch, leave=False)
        for n, image_batch in enumerate(pbar):
            if not isinstance(image_batch, torch.Tensor):
                image_batch = image_batch[0]

            image_batch = image_batch.to(cuda_device)
            batch_size = image_batch.shape[0]

            optimizer.zero_grad()
            zs, log_det = flow(image_batch)
            image_dim = image_batch.numel() / batch_size
            nll = -image_dim * torch.log(1 / 256 * one)
            nll = nll + nll_loss(zs, log_det)
            bpd = nll / (image_dim * torch.log(2 * one))

            pbar.set_postfix(nll=nll.item(), bpd=bpd.item())

            nll.backward()
            optimizer.step()

            if zs_shapes is None:
                zs_shapes = [z.size() for z in zs]

            step = (epoch - 1) * len(trainloader) + n
            writer.add_scalar("nll", nll, step)
            writer.add_scalar("bpd", bpd, step)
            scheduler.step(step)
        pbar.close()
        if epoch % FLAGS.test_every == 0:
            torch.save({
                "epoch": epoch,
                "flow_state_dict": flow.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "zs_shapes": zs_shapes,
                "bpd": bpd,
            }, str(log_folder / ("ckpt-%d.pth" % epoch)))
            image_samples, _ = sample(flow, zs_shapes)
            image_samples = make_grid(image_samples, nrow=10, normalize=True, range=(0, 1), pad_value=10)
            writer.add_image("samples", image_samples, epoch)
    writer.close()


if __name__ == "__main__":
    app.run(train)
