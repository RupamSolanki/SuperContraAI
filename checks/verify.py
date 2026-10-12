"""
verify.py — Pre-Training System Check
=======================================
Validates environment, CUDA, RAM addresses, observation shape,
and model file integrity before committing to a training run.

USAGE:
    python checks/verify.py           # Full system check
    python checks/verify.py --model   # Also verify model file isn't corrupt
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import stable_retro

from env_wrapper import SuperContraWrapper

GAME = "SuperC-Nes-v0"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")


def check_cuda():
    """Check GPU availability."""
    available = torch.cuda.is_available()
    if available:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  [OK] CUDA: {name} ({vram:.1f} GB)")
    else:
        print("  [!!] CUDA not available — training will use CPU (much slower)")
    return available


def check_environment():
    """Check game loads and wrapper produces correct observations."""
    try:
        env = stable_retro.make(
            game=GAME, state="Level1", render_mode="rgb_array"
        )
        env = SuperContraWrapper(env, frame_skip=4)
        obs, info = env.reset()

        # Observation shape
        assert obs.shape == (84, 84, 1), f"Bad obs shape: {obs.shape}"
        print(f"  [OK] Observation: {obs.shape} uint8")

        # RAM addresses
        required_keys = [
            "lives", "screen_x", "scroll_x", "score_low",
            "score_mid", "game_mode", "level_type", "boss_dead",
        ]
        missing = [k for k in required_keys if k not in info]
        if missing:
            print(f"  [!!] Missing RAM keys: {missing}")
            print("       Update data.json in the game's retro data folder")
            env.close()
            return False
        else:
            print(f"  [OK] RAM data.json: {len(required_keys)} keys verified")
            print(f"       lives={info['lives']}, game_mode={info['game_mode']}")

        # Step test
        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == (84, 84, 1)
        print(f"  [OK] Step execution: reward={reward:.1f}")

        env.close()
        return True

    except Exception as e:
        print(f"  [FAIL] Environment error: {e}")
        print("       Checklist:")
        print(f"         - Is '{GAME}' the correct game folder name?")
        print("         - Is 'Level1.state' in the game data folder?")
        print("         - Is data.json updated with correct RAM addresses?")
        return False


def check_model():
    """Verify model file loads and produces valid predictions."""
    import os
    from stable_baselines3 import PPO

    if not os.path.exists(MODEL_PATH):
        print(f"  [--] No model file found ({MODEL_PATH}) — skip")
        return True

    try:
        model = PPO.load(MODEL_PATH, device="cpu")
        print(f"  [OK] Model loaded: {MODEL_PATH}")
        print(f"       Obs space: {model.observation_space.shape}")
        print(f"       Action space: {model.action_space}")

        # Forward pass test
        dummy_obs = np.zeros(model.observation_space.shape, dtype=np.uint8)
        action, _ = model.predict(dummy_obs, deterministic=True)
        print(f"  [OK] Forward pass: action={action}")

        return True

    except Exception as e:
        print(f"  [FAIL] Model error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Super Contra pre-training verification")
    parser.add_argument(
        "--model", action="store_true", help="Also verify model file"
    )
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("SUPER CONTRA — SYSTEM VERIFICATION")
    print("=" * 50 + "\n")

    results = []

    print("[1/3] CUDA / GPU")
    results.append(check_cuda())

    print("\n[2/3] Environment + RAM")
    results.append(check_environment())

    if args.model:
        print("\n[3/3] Model Integrity")
        results.append(check_model())
    else:
        print("\n[3/3] Model check skipped (use --model to include)")

    # Summary
    print("\n" + "=" * 50)
    if all(results):
        print("ALL CHECKS PASSED — ready to train")
        print(f"  python train.py --phase 1")
    else:
        print("SOME CHECKS FAILED — fix issues above before training")
        sys.exit(1)
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
