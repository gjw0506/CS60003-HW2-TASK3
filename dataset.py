import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class StanfordBackgroundDataset(Dataset):
    def __init__(
        self,
        image_paths,
        label_paths,
        image_size=(256, 256),
        augment=False,
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def _read_label(self, path):
        if path.endswith(".txt"):
            mask = np.loadtxt(path, dtype=np.int64)
        else:
            mask = np.array(Image.open(path), dtype=np.int64)
        return mask

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")

        mask = self._read_label(self.label_paths[idx]).astype(np.int64)


        # 不要用 uint8，否则 -1 会变成 255
        mask = Image.fromarray(mask.astype(np.int32), mode="I")

        img = TF.resize(img, self.image_size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        mask = np.array(mask).astype(np.int64)

        # 再保险
        mask[mask == 255] = -1

        mask = torch.from_numpy(mask).long()

        return img, mask


def build_file_lists(root):
    image_dir = os.path.join(root, "images")
    label_dir = os.path.join(root, "labels")

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )

    label_paths = []

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]

        candidates = [
            os.path.join(label_dir, name + ".regions.txt"),
            os.path.join(label_dir, name + ".png"),
            os.path.join(label_dir, name + ".txt"),
        ]

        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break

        if found is None:
            raise FileNotFoundError(f"No label found for {img_path}")

        label_paths.append(found)

    return image_paths, label_paths


def split_dataset(image_paths, label_paths, seed=42):
    indices = list(range(len(image_paths)))
    random.Random(seed).shuffle(indices)

    n = len(indices)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(idxs):
        return [image_paths[i] for i in idxs], [label_paths[i] for i in idxs]

    return subset(train_idx), subset(val_idx), subset(test_idx)