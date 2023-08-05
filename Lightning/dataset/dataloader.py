import torch
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from functools import cached_property
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class a_train_transform:
    def __init__(self):
        self.albumentations_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.4914, 0.4822, 0.4471], std=[0.2469, 0.2433, 0.2615]
                ),
                A.HorizontalFlip(p=0.5),
                A.PadIfNeeded(40, 40, p=1),
                A.RandomCrop(32, 32, p=1),
                A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
                A.CoarseDropout(
                    max_holes=1, max_height=16, max_width=16, fill_value=0, p=1
                ),
                A.CenterCrop(32, 32, p=1),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)["image"]
        return img


class a_test_transform:
    def __init__(self):
        self.albumentations_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.4914, 0.4822, 0.4471], std=[0.2469, 0.2433, 0.2615]
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)["image"]
        return img


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        self.train_ds = datasets.CIFAR10(
            self.data_dir, train=True, download=True, transform=a_train_transform()
        )

        self.val_ds = datasets.CIFAR10(
            self.data_dir, train=False, download=True, transform=a_test_transform()
        )
        self.test_ds = datasets.CIFAR10(
            self.data_dir, train=False, download=True, transform=a_test_transform()
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
