"""Check what observation space the trained model expects."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")

from stable_baselines3 import PPO

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

model = PPO.load(MODEL_PATH, device="cpu")
print(f"Model obs space: {model.observation_space}")
print(f"Model obs shape: {model.observation_space.shape}")
print(f"Model action space: {model.action_space}")
