import os

import numpy as np
import torch
import torchvision

from . import CUB_dataset, custom_dataset


def _get_scale(square: bool) -> float:
    return 10 if square else 20


def db_to_ratio(db: float, square: bool = True) -> float:
    return 10 ** (db / _get_scale(square))


def ratio_to_db(ratio: float, square: bool = True) -> float:
    return _get_scale(square) * np.log10(ratio)


def get_data(transform, dataset):
    if dataset == "imagenet":
        train_ds = torchvision.datasets.DatasetFolder(
            os.path.join("./data/ImageNet_clip/train"),
            loader=lambda path: np.load(path, allow_pickle=True),
            is_valid_file=lambda path: path[-4:] == ".npy",
        )

        test_ds = torchvision.datasets.DatasetFolder(
            os.path.join("./data/ImageNet_clip/val"),
            loader=lambda path: np.load(path, allow_pickle=True),
            is_valid_file=lambda path: path[-4:] == ".npy",
        )

    if dataset == "places365":
        train_ds = custom_dataset.Places365_clip(
            root="data/Places365",
            split="train-standard",
            small=True,
            download=False,
            transform=transform,
        )

        test_ds = custom_dataset.Places365_clip(
            root="data/Places365",
            split="val",
            small=True,
            download=False,
            transform=transform,
        )

    if dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    if dataset == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=False, transform=transform)
        test_ds = torchvision.datasets.CIFAR100(root="./data", train=False, download=False, transform=transform)

    elif dataset == "cub":
        use_attr = True
        no_img = False
        uncertain_label = False
        n_class_atr = 1
        data_dir = "data/"
        image_dir = f"{data_dir}/CUB/CUB_200_2011"
        no_label = False
        prune = False

        train_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/trainclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )

        val_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/valclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )

        train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])

        test_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/testclass_level_all_features.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=transform,
            no_label=no_label,
        )

    return train_ds, test_ds


def get_concepts(filename):
    list_of_concepts = []

    f = open(filename, "r")
    for line in f.readlines():
        list_of_concepts.append(line.strip())

    return list_of_concepts
