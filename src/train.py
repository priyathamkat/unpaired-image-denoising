from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from data import get_dataloader
from loss import mse_loss, nll_loss
from resnet import ResNet

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/glow")
from glow.model import Flow


FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", "/datasets/Priyatham/datasets/denoising", "path to denoising dataset")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_integer("crop_size", 32, "image size for training")
flags.DEFINE_integer("num_epochs", 100, "total number of training steps")
flags.DEFINE_integer("test_every", 10, "testing frequency")
flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
flags.DEFINE_float("alpha", 1.5e-6, "weight on negative log-likelihood")
flags.DEFINE_string("logs", "../denoise_logs", "folder to write logs")
flags.DEFINE_string("saved_flow_model", "../glow_logs/20191208-110750/ckpt-100.pth", "path to saved flow checkpoint")

cuda_device = torch.device("cuda:0")
one = torch.ones(1).to(cuda_device)


def main(unused_argv):
    root = Path(FLAGS.dataset)
    clean_root = str(root / "clean")
    noisy_root = str(root / "noisy")
    test_root = str(root / "test")

    resnet = ResNet(3, 8, 64).cuda()
    flow = Flow(32, 3, 3).cuda()
    optimizer = optim.Adam(resnet.parameters(), lr=FLAGS.learning_rate)

    ckpt = torch.load(FLAGS.saved_flow_model)
    flow.load_state_dict(ckpt["flow_state_dict"])
    flow.eval()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_folder = Path(FLAGS.logs) / timestamp
    writer = SummaryWriter(str(log_folder))

    for epoch in range(1, FLAGS.num_epochs + 1):
        trainloader = get_dataloader(noisy_root, False, FLAGS.batch_size, crop_size=FLAGS.crop_size)

        pbar = tqdm(trainloader, desc="epoch %d" % epoch, leave=False)
        resnet.train()
        for n, noisy_images in enumerate(pbar):

            noisy_images = noisy_images.to(cuda_device)

            optimizer.zero_grad()
            denoised_images = resnet(noisy_images)
            zs, log_det = flow(denoised_images)
    
            mse = mse_loss(denoised_images, noisy_images)
            batch_size = noisy_images.shape[0]
            nll = noisy_images.numel() * torch.log(256 * one) / batch_size
            nll = nll + nll_loss(zs, log_det)
            loss = mse + FLAGS.alpha * nll

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

            step = (epoch - 1) * len(trainloader) + n
            writer.add_scalar("loss", loss, step)
            writer.add_scalar("mse", mse, step)
            writer.add_scalar("nll", nll, step)
            
        if epoch % FLAGS.test_every == 0:
            resnet.eval()
            noisy_testloader = get_dataloader(test_root, False, 1, train=False)
            clean_testloader = get_dataloader(test_root, False, 1, train=False)
            with torch.no_grad():
                psnr = []
                sample_images = None
                for n, (clean_images, noisy_images) in enumerate(zip(clean_testloader, noisy_testloader)):
                # for n, noisy_images in enumerate(noisy_testloader):
                    clean_images = clean_images.to(cuda_device)
                    noisy_images = noisy_images.to(cuda_device)

                    denoised_images = resnet(noisy_images)
                    mse = F.mse_loss(clean_images, denoised_images)
                    psnr.append(10 * torch.log10(1 / mse))

                    if n < 5:
                        sample_images = torch.cat([clean_images, noisy_images, denoised_images])
                        # sample_images = torch.cat([noisy_images, denoised_images])
                        writer.add_images("sample_image_%d" % n, sample_images, step)
                test_psnr = torch.as_tensor(psnr).mean()

            writer.add_scalar("test_psnr", test_psnr, step)
            # writer.add_images("sample_images_%d" % n, sample_images, step)

            torch.save({
                "epoch": epoch,
                "resnet_state_dict": resnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, str(log_folder / ("ckpt-%d.pth" % step)))

    writer.close()


if __name__ == "__main__":
    app.run(main)
