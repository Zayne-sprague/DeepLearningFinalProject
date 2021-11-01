import torch
import torchvision.transforms as T
from torch.utils import data
from torchvision import datasets
from src.paths import DATA_DIR
import os, re

__all__ = ["get_dataloder"]


def get_dataloder(name, batch_size, storage_path):
    trans = T.Compose([T.ToTensor()])
    if name == "CIFAR-100":
        train_set = datasets.CIFAR100(root=storage_path, train=True, download=True,
                                      transform=T.ToTensor())
        test_set = datasets.CIFAR100(root=storage_path, train=False, download=True,
                                      transform=T.ToTensor())
    elif name == "STL10":
        train_set = datasets.STL10(root=storage_path, split="train", download=True,
                                   transform=T.ToTensor())
        test_set = datasets.STL10(root=storage_path, split="test", download=True,
                                      transform=T.ToTensor())

    return data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True), \
           data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)

