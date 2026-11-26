"""Shared E2 training configuration aligned with E1_a and E1_b.

Keep E2 contact scheduling / method logic separate from data and optimization.
The train-holdout split and cosine schedule reproduce E1_a/train.py.
"""
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

def add_training_arguments(parser):
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--min-lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--val-size', type=int, default=5000)
    parser.add_argument('--validation-source', choices=['train-holdout', 'official-test'],
                        default='train-holdout')
    parser.add_argument('--no-amp', action='store_true')

def validate_training_args(a):
    if not (0 <= a.warmup_epochs < a.epochs):
        raise ValueError('Invalid warmup epochs')
    if a.batch_size < 1 or a.num_workers < 0 or a.val_size < 1:
        raise ValueError('Invalid batch, num-workers or validation size')

def lr_at(epoch, a):
    if a.warmup_epochs and epoch < a.warmup_epochs:
        return a.lr * (epoch + 1) / a.warmup_epochs
    frac = (epoch - a.warmup_epochs) / max(1, a.epochs - a.warmup_epochs - 1)
    return a.min_lr + .5 * (a.lr - a.min_lr) * (
        1 + math.cos(math.pi * min(1, max(0, frac))))

def loaders(a, rank, world):
    """Same stratified 45k/5k train/holdout split and augmentation as E1."""
    mean = (.5071, .4867, .4408)
    std = (.2675, .2565, .2761)
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    t_val = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    if a.dataset == 'synthetic':
        train = datasets.FakeData(size=512, image_size=(3, 32, 32),
                                  num_classes=100, transform=t_train)
        val = datasets.FakeData(size=128, image_size=(3, 32, 32),
                                num_classes=100, transform=t_val,
                                random_offset=50000)
    else:
        original = datasets.CIFAR100(a.data_root, train=True, download=False, transform=t_train)
        evaluation = datasets.CIFAR100(a.data_root, train=True, download=False, transform=t_val)
        if a.validation_source == 'train-holdout':
            labels = np.asarray(original.targets)
            rng = np.random.default_rng(a.seed)
            buckets = [rng.permutation(np.where(labels == c)[0]) for c in range(100)]
            q, r = divmod(a.val_size, 100)
            vi = np.concatenate([v[:q + (c < r)] for c, v in enumerate(buckets)]).tolist()
            ti = np.concatenate([v[q + (c < r):] for c, v in enumerate(buckets)]).tolist()
            train = Subset(original, ti)
            val = Subset(evaluation, vi)
        else:
            train = original
            val = datasets.CIFAR100(a.data_root, train=False, download=False, transform=t_val)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank,
                                 shuffle=True, seed=a.seed, drop_last=False)
    kw = dict(num_workers=a.num_workers, pin_memory=True,
              persistent_workers=a.num_workers > 0)
    tr = DataLoader(train, batch_size=a.batch_size, sampler=sampler, **kw)
    va = DataLoader(val, batch_size=max(128, a.batch_size), shuffle=False, **kw)
    return tr, va, sampler, len(train), len(val)
