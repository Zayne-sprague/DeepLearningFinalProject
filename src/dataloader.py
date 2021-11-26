import torch
import torchvision.transforms as T
import torchvision
from torch.utils import data
from torchvision import datasets
from src.paths import DATA_DIR
import os, re
from tiny_img import download_tinyImg200, load_tiny_image
from sklearn.model_selection import train_test_split

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

    elif name == "TINY":
        # source: https://colab.research.google.com/github/yandexdataschool/mlhep2019/blob/master/notebooks/day-3/seminar_convnets.ipynb#scrollTo=tvz-gycUrYD1
        if not os.path.exists(str(DATA_DIR / 'tiny-imagenet-200')):
            download_tinyImg200(storage_path)
        train_set = datasets.ImageFolder(str(DATA_DIR / 'tiny-imagenet-200/train'), transform=T.ToTensor())
        test_set = load_validation_tinynet(train_set)

    elif name == "CALTECH101":
        # !gdown https://drive.google.com/uc?id=1DX_XeKHn3yXtZ18DD7qc1wf-Jy5lnhD5
        # !unzip -qq '101_ObjectCategories.zip'

        CALTECH_IMG_SIZE = 64
        transform = T.Compose(
            [
                 T.CenterCrop(256),
                 T.Resize((CALTECH_IMG_SIZE, CALTECH_IMG_SIZE)),
                 T.ToTensor(),
                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        totalset = torchvision.datasets.ImageFolder(str(DATA_DIR / '101_ObjectCategories/'), transform=transform)

        X, y = zip(*totalset)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3,
                                                          stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                        test_size=0.5,
                                                        stratify=y_val)

        train_set, validset, test_set = list(zip(X_train, y_train)), list(zip(X_val, y_val)), list(zip(X_test, y_test))

    return data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True), \
           data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)


def load_validation_tinynet(train_set: datasets.ImageFolder) -> datasets.ImageFolder:
    # TinyImage net is NOT annotated the same way for train and val folders.
    # Train has each class separated into subfolders, however, val has each class annotated in a separate text file
    # Without this helper function, loading the validation folder as an imagefolder will cause all examples to have
    # a target class of 0

    # Set up some nice path constants
    tinynet_path = DATA_DIR / 'tiny-imagenet-200'
    validation_path = tinynet_path / 'val'
    validation_file = validation_path / 'val_annotations.txt'

    # Read from the validation annotation file
    val_file = open(str(validation_file), 'r')
    val_lines = val_file.readlines()

    # These will be our new targets
    targets = []

    # Read line by line (each line is one example, in order)
    for line in val_lines:
        # Split the line by the files delimiters (tabs)
        target_class = line.split("\t")[1]

        # The train set will have a mapping of "class name" to "idx".
        # IMPORTANT: use whats in the trainset not the order given from tinynets wnids.txt (they will be out of sync)
        target_idx = train_set.class_to_idx[target_class]
        targets.append(target_idx)

    # Load the validation image folder
    validation_set = datasets.ImageFolder(str(DATA_DIR / 'tiny-imagenet-200/val'), transform=T.ToTensor())

    # The image folder will sort all the files alphebetically, this means val_01, val_02, and val_10 will be
    # [val_01, val_10, val_02]
    # we this is not the order seen in the validation annotation file.  So we have to reorder these arrays
    # using a sorter that looks at the numerical value in the file name.
    def sorter(item):
        return int(re.sub('\D', '', item))

    # samples & imgs need to be resorted to match the target array
    validation_set.samples = sorted(validation_set.samples, key=lambda x: sorter(x[0].split("/")[-1]))
    validation_set.imgs = sorted(validation_set.imgs, key=lambda x: sorter(x[0].split("/")[-1]))

    # Overwrite the bad targets with our new ones.
    validation_set.targets = targets

    # We have to overwrite some class look ups and class arrays to match what we are really representing in the dataset
    validation_set.class_to_idx = train_set.class_to_idx
    validation_set.classes = train_set.classes

    # Each image has a path and a target, we have to overwrite the bad targets and keep the same path
    for idx, target in enumerate(targets):
        validation_set.samples[idx] = (validation_set.samples[idx][0], target)
        validation_set.imgs[idx] = (validation_set.imgs[idx][0], target)

    return validation_set