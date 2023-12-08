import os

import numpy as np
import torch
import torchvision.datasets as datasets


class Places365_clip(datasets.Places365):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.small and self.split == "train-standard":
            self.attr_dir = os.path.join("./data/Places365_clip/train-standard/")
        elif self.small and self.split == "val":
            self.attr_dir = os.path.join("./data/Places365_clip/val_256/")

    def __getitem__(self, idx):
        label = self.imgs[idx][1]
        if self.small and self.split == "train-standard":
            abbr, cls_name, filename_jpg = self.imgs[idx][0].split("/")[-3:]
            filename = filename_jpg[:-4] + ".npy"
            attr_path = os.path.join(self.attr_dir, abbr, cls_name, filename)
        elif self.small and self.split == "val":
            filename = self.imgs[idx][0].split("/")[-1][:-4] + ".npy"
            attr_path = os.path.join(self.attr_dir, filename)
        attr = torch.from_numpy(np.load(attr_path)).float()
        return attr, label
