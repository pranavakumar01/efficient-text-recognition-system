"""
Root launcher for Multi-Domain End-to-End Training Pipeline (CNN + Transformer).

Usage:
  python train_all.py
  python train_all.py --model cnn
  python train_all.py --model transformer
  python train_all.py --domain historical
  python train_all.py --domain math
  python train_all.py --domain handwritten
  python train_all.py --max-samples 32 --cnn-epochs 1 --transformer-epochs 1
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train_all import main

if __name__ == "__main__":
    main()
