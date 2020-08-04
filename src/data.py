from pathlib import Path

import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, add_noise=False):
        super().__init__()
        self.add_noise = add_noise
        self.transform = transform
        root = Path(root)
        self.files = [str(p) for p in root.glob("./*.jpg")]
        self.files.extend([str(p) for p in root.glob("./*.png")])

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        image = np.asarray(image)

        if self.add_noise:
            image = image.astype(np.float32)
            image /= 255.0
            # Always add the same noise to a particular image
            np.random.seed(idx)
            sigma = np.random.uniform(0.0, 0.2)
            noise = np.random.normal(scale=sigma, size=image.shape)
            image += noise
            image = np.clip(image, 0.0, 1.0)
            image *= 255.0
            image = image.astype(np.uint8)

        if image.ndim == 2:
            image = np.stack(3 * [image], axis=-1)

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.files)


def get_dataloader(root, add_noise, batch_size, train=True, crop_size=None):
    if not train:
        assert batch_size == 1
        assert crop_size is None
        transform = T.Compose([
            T.ToTensor(),
        ])
        shuffle = False
    else:
        assert crop_size is not None
        transform = T.Compose([
            T.RandomCrop(crop_size),
            T.ToTensor(),
        ])
        shuffle = True
    dataset = ImageDataset(root, transform, add_noise=add_noise)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=10, pin_memory=True)
