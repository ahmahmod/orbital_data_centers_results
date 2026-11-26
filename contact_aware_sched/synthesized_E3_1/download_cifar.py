#!/usr/bin/env python3
"""One-time first-run download. Not part of E1; CIFAR-100 dataset is third-party."""
from pathlib import Path
from torchvision import datasets
p=Path(__file__).resolve().parents[2]/'data'
datasets.CIFAR100(str(p),train=True,download=True)
datasets.CIFAR100(str(p),train=False,download=True)
print('CIFAR-100 ready at',p)
