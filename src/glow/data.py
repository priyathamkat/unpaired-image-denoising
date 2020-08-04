from PIL import Image
import numpy as np
import torchvision.transforms as T
from pathlib import Path
import torch
from absl import flags
from torchvision import datasets as D
from torchvision import transforms as T

FLAGS = flags.FLAGS

flags.DEFINE_enum("dataset", "COCO", ["CIFAR", "COCO"], "dataset to use for training")
flags.DEFINE_integer("batch_size", 50, "batch size")
flags.DEFINE_string("clean_path", "/home/priyatham/Datasets/denoising/clean/", "path to coco")

__all__ = ["get_trainloader", "get_testloader"]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.transform = transform
        root = Path(root)
        self.files = [str(p) for p in root.glob("./*.jpg")]
        self.files.extend([str(p) for p in root.glob("./*.png")])

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = np.asarray(image)

        if image.ndim == 2:
            image = np.stack(3 * [image], axis=-1)

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.files)


def get_trainloader(crop_size):
    if FLAGS.dataset == "CIFAR":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        trainset = D.CIFAR10("/home/priyatham/glow/datasets", train=True, transform=transform, download=False)
    elif FLAGS.dataset == "COCO":
        transform = T.Compose([
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        trainset = ImageDataset(FLAGS.clean_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return trainloader


def get_testloader():
    if FLAGS.dataset == "CIFAR":
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        testset = D.CIFAR10("../datasets", train=False, transform=transform, download=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=3, pin_memory=True)
    return testloader
