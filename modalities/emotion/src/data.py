"""
Dataset / dataloader helpers shared across all scripts.
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
)
from src.transforms import get_test_transforms, get_train_transforms


def get_datasets(train_transform=None, test_transform=None):
    """Return (train_ds, val_ds) RAF-DB ImageFolder datasets."""
    train_ds = ImageFolder(
        TRAIN_DIR, transform=train_transform or get_train_transforms()
    )
    val_ds = ImageFolder(
        TEST_DIR, transform=test_transform or get_test_transforms()
    )
    return train_ds, val_ds


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    train_ds=None, val_ds=None, pin_memory=True):
    """Return (train_loader, val_loader)."""
    if train_ds is None or val_ds is None:
        train_ds, val_ds = get_datasets()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_class_weights(dataset):
    """Inverse-frequency class weights (normalized to sum to num_classes)."""
    counts = np.bincount([label for _, label in dataset.samples])
    w = 1.0 / counts.astype(float)
    w = w / w.sum() * len(counts)
    return torch.FloatTensor(w)


def class_counts(dataset):
    """Return a {class_index: count} dict for an ImageFolder dataset."""
    counts = np.bincount([label for _, label in dataset.samples])
    return {i: int(c) for i, c in enumerate(counts)}


# ── Sequence datasets (for the CNN-LSTM temporal model) ──────────────────
class MultiViewImageFolder(ImageFolder):
    """RAF-DB images as pseudo-sequences: ``views`` independently augmented
    copies of the same image, stacked to (T, C, H, W).

    With a stochastic train transform the LSTM learns to aggregate noisy views
    of one expression — the closest RAF-DB (static images) can get to the
    frame-to-frame variation it sees on real clips. With a deterministic test
    transform the views are identical, which just measures the static mapping.
    """

    def __init__(self, root, transform, views=4):
        super().__init__(root, transform=transform)
        self.views = views

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        return torch.stack([self.transform(img) for _ in range(self.views)]), target


class RealWorldClipDataset(Dataset):
    """Face-crop sequences from data/realworld/{train,val}.

    Crops are named ``<clip_id>_f<frame>.jpg`` (extract_realworld_faces.py), so
    grouping by the prefix recovers each clip's frames in temporal order. Each
    item is (seq_len, C, H, W): frames sampled evenly from the clip's available
    crops (repeating frames when a clip has fewer than ``seq_len``).

    Train mode (``train=True``) draws a fresh random (ordered) frame subset per
    access and ``repeats`` multiplies the epoch length, so each clip is seen
    several times per epoch with different frames + augmentations — without it
    ~900 clips/epoch overfit their subjects long before the model generalizes.
    """

    def __init__(self, root, transform, seq_len=8, train=False, repeats=1):
        self.transform = transform
        self.seq_len = seq_len
        self.train = train
        self.repeats = repeats
        self.clips = []  # (list_of_paths_in_order, class_index)
        for folder in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, folder)
            if not os.path.isdir(cls_dir):
                continue
            label = int(folder) - 1  # folders 1..7 -> class index 0..6
            by_clip = defaultdict(list)
            for fname in sorted(os.listdir(cls_dir)):
                clip_id = fname.rsplit("_f", 1)[0]
                by_clip[clip_id].append(os.path.join(cls_dir, fname))
            for clip_id in sorted(by_clip):
                self.clips.append((by_clip[clip_id], label))
        # `samples` mirrors ImageFolder so class-weight helpers work unchanged.
        self.samples = [(paths[0], label) for paths, label in self.clips]

    def __len__(self):
        return len(self.clips) * self.repeats

    def __getitem__(self, index):
        paths, label = self.clips[index % len(self.clips)]
        n = len(paths)
        if self.train:
            pool = range(n)
            idxs = sorted(random.sample(pool, self.seq_len) if n >= self.seq_len
                          else random.choices(pool, k=self.seq_len))
        else:
            idxs = np.linspace(0, n - 1, self.seq_len).round().astype(int)
        frames = [self.transform(default_loader(paths[i])) for i in idxs]
        return torch.stack(frames), label


class RandomSubsetDataset(Dataset):
    """Fixed-length view of a dataset that draws a fresh random sample each
    __getitem__ — used to keep RAF-DB from drowning out the (much smaller)
    real-world clip set during sequence fine-tuning."""

    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        return self.dataset[random.randrange(len(self.dataset))]


def subset_per_class(dataset, max_per_class):
    """
    Small stratified subset for smoke tests / fast trials.
    Returns a torch Subset preserving the dataset's transform.
    """
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    keep = []
    for label, idxs in by_class.items():
        keep.extend(idxs[:max_per_class])
    return Subset(dataset, keep)
