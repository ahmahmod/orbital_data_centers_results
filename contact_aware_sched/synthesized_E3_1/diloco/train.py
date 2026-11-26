#!/usr/bin/env python3
"""Technique entry point for the shared fixed-H NCCL trainer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nccl_train import main

if __name__ == '__main__':
    main(('diloco8','diloco64'))
