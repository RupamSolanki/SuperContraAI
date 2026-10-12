"""Quick system check — verifies all packages and GPU are working."""

print("Checking packages...")

# 1. Core packages
import sys
print(f"  Python: {sys.version.split()[0]}")

import numpy
print(f"  NumPy: {numpy.__version__}")

import cv2
print(f"  OpenCV: {cv2.__version__}")

import gymnasium
print(f"  Gymnasium: {gymnasium.__version__}")

import stable_retro
print(f"  Stable-Retro: {stable_retro.__version__}")

import stable_baselines3
print(f"  Stable-Baselines3: {stable_baselines3.__version__}")

import torch
print(f"  PyTorch: {torch.__version__}")

# 2. CUDA / GPU
print()
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  GPU: {name} ({vram:.1f} GB VRAM)")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("  GPU: NOT FOUND (will use CPU)")

# 3. Game ROM
print()
games = stable_retro.data.list_games()
if "SuperC-Nes-v0" in games:
    print("  Super Contra ROM: FOUND")
else:
    print("  Super Contra ROM: NOT FOUND")
    print("  Run: python -m retro.import <rom_folder>")

print()
print("Done.")
