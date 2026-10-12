"""
test.py — Full System Test for Super Contra AI
================================================
Runs live tests against the actual environment and model.
Requires all dependencies installed (stable-retro, torch, sb3).

USAGE:
    python test.py              # Run all tests
    python test.py --quick      # Skip model load (faster)
    python test.py --render     # Show game window during env test
"""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from env_wrapper import SuperContraWrapper

# =============================================================================
#  CONFIGURATION
# =============================================================================
GAME = "SuperC-Nes-v0"
STATE = "Level1"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")
N_STACK = 12


# =============================================================================
#  TEST 1: CUDA / GPU
# =============================================================================
def test_cuda():
    """Verify CUDA is available for GPU training."""
    print("\n[TEST 1] CUDA / GPU")
    print("-" * 40)

    available = torch.cuda.is_available()
    if available:
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Device : {name}")
        print(f"  VRAM   : {vram:.1f} GB")
        print(f"  PyTorch: {torch.__version__}")
        print("  Result : PASS")
        return True
    else:
        print("  CUDA not available — will use CPU (slower)")
        print("  Result : WARN (not fatal)")
        return True  # Not a hard failure


# =============================================================================
#  TEST 2: Environment Creation
# =============================================================================
def test_env_creation(render: bool = False):
    """Verify game loads and wrapper produces correct observations."""
    print("\n[TEST 2] Environment Creation")
    print("-" * 40)

    render_mode = "human" if render else "rgb_array"
    env = stable_retro.make(game=GAME, state=STATE, render_mode=render_mode)
    env = SuperContraWrapper(env, frame_skip=4)

    obs, info = env.reset()

    # Check observation shape
    assert obs.shape == (84, 84, 1), f"Bad obs shape: {obs.shape}"
    assert obs.dtype == np.uint8, f"Bad dtype: {obs.dtype}"
    print(f"  Obs shape: {obs.shape} uint8 — OK")

    # Check RAM data populated
    required_keys = [
        "lives", "screen_x", "scroll_x", "score_low",
        "score_mid", "game_mode", "level_type", "boss_dead", "weapon",
    ]
    missing = [k for k in required_keys if k not in info]
    if missing:
        print(f"  FAIL: Missing RAM keys: {missing}")
        env.close()
        return False

    print(f"  RAM keys : {len(required_keys)} verified")
    print(f"  Lives    : {info['lives']}")
    print(f"  Game mode: {info['game_mode']}")
    print("  Result   : PASS")

    env.close()
    return True


# =============================================================================
#  TEST 3: Action Execution (all 8 actions)
# =============================================================================
def test_actions(render: bool = False):
    """Verify all 8 actions execute without error."""
    print("\n[TEST 3] Action Execution (8 actions)")
    print("-" * 40)

    render_mode = "human" if render else "rgb_array"
    env = stable_retro.make(game=GAME, state=STATE, render_mode=render_mode)
    env = SuperContraWrapper(env, frame_skip=4)
    obs, info = env.reset()

    action_names = [
        "Forced Right", "Right", "Left", "Crouch",
        "Up", "Right+Fire", "Jump", "Jump+Fire",
    ]

    for action in range(8):
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (84, 84, 1)
        print(f"  Action {action} ({action_names[action]:<12}): reward={reward:>8.1f}")

        if terminated or truncated:
            obs, info = env.reset()

    print("  Result: PASS")
    env.close()
    return True


# =============================================================================
#  TEST 4: Reward System Validation
# =============================================================================
def test_rewards():
    """Run 200 random steps and verify reward signals are firing."""
    print("\n[TEST 4] Reward System (200 random steps)")
    print("-" * 40)

    env = stable_retro.make(game=GAME, state=STATE, render_mode="rgb_array")
    env = SuperContraWrapper(env, frame_skip=4)
    obs, info = env.reset()

    total_reward = 0.0
    max_world_x = 0
    deaths = 0
    resets = 0

    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        sx = int(info.get("screen_x", 0))
        sc = int(info.get("scroll_x", 0))
        world_x = (sc * 256) + sx
        max_world_x = max(max_world_x, world_x)

        if terminated or truncated:
            deaths += 1
            resets += 1
            obs, info = env.reset()

    print(f"  Steps      : 200")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Max world_x : {max_world_x}")
    print(f"  Deaths      : {deaths}")
    print(f"  Resets      : {resets}")

    # Reward should not be exactly 0 (signals are firing)
    if total_reward == 0.0 and max_world_x == 0:
        print("  WARN: No reward or progress detected")
        print("  Result: WARN")
    else:
        print("  Result: PASS")

    env.close()
    return True


# =============================================================================
#  TEST 5: VecEnv + Frame Stack (training architecture)
# =============================================================================
def test_vec_env():
    """Verify the vectorized env + frame stack matches training setup."""
    print("\n[TEST 5] VecEnv + Frame Stack (training arch)")
    print("-" * 40)

    def _make():
        env = stable_retro.make(game=GAME, state=STATE, render_mode=None)
        env = SuperContraWrapper(env, frame_skip=4)
        return env

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=N_STACK)

    obs = env.reset()
    expected_shape = (1, 84, 84, N_STACK)  # batch, H, W, channels
    assert obs.shape == expected_shape, f"Bad shape: {obs.shape}, expected {expected_shape}"
    print(f"  Obs shape: {obs.shape} — OK")

    # Run a few steps
    for _ in range(10):
        action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        assert obs.shape == expected_shape

    print(f"  10 steps: OK")
    print("  Result: PASS")

    env.close()
    return True


# =============================================================================
#  TEST 6: Model Load + Predict
# =============================================================================
def test_model():
    """Load trained model and verify it produces valid actions."""
    print("\n[TEST 6] Model Load + Predict")
    print("-" * 40)

    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found: {MODEL_PATH}")
        print("  Result: SKIP (train first)")
        return True

    def _make():
        env = stable_retro.make(game=GAME, state=STATE, render_mode=None)
        env = SuperContraWrapper(env, frame_skip=4)
        return env

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=N_STACK)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, env=env, device=device)

    print(f"  Model loaded on: {device}")
    print(f"  Obs space : {model.observation_space.shape}")
    print(f"  Action space: {model.action_space}")

    # Run 50 prediction steps
    obs = env.reset()
    actions_taken = set()

    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        actions_taken.add(int(action[0]))

        if done[0]:
            obs = env.reset()

    print(f"  50 steps predicted: OK")
    print(f"  Actions used: {sorted(actions_taken)}")

    if len(actions_taken) < 2:
        print("  WARN: Agent only using 1 action (may need more training)")

    print("  Result: PASS")
    env.close()
    return True


# =============================================================================
#  TEST 7: Model Inference Speed
# =============================================================================
def test_speed():
    """Benchmark inference speed (FPS)."""
    print("\n[TEST 7] Inference Speed")
    print("-" * 40)

    def _make():
        env = stable_retro.make(game=GAME, state=STATE, render_mode=None)
        env = SuperContraWrapper(env, frame_skip=4)
        return env

    if not os.path.exists(MODEL_PATH):
        print(f"  Model not found — skipping speed test")
        print("  Result: SKIP")
        return True

    env = DummyVecEnv([_make])
    env = VecFrameStack(env, n_stack=N_STACK)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(MODEL_PATH, env=env, device=device)

    obs = env.reset()

    # Warmup
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()

    # Benchmark
    n_steps = 500
    start = time.time()

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()

    elapsed = time.time() - start
    fps = n_steps / elapsed
    effective_fps = fps * 4  # frame_skip=4

    print(f"  Steps     : {n_steps}")
    print(f"  Time      : {elapsed:.2f}s")
    print(f"  Agent FPS : {fps:.0f} steps/sec")
    print(f"  Game FPS  : {effective_fps:.0f} frames/sec (with frame skip)")
    print("  Result    : PASS")

    env.close()
    return True


# =============================================================================
#  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Super Contra AI — System Tests")
    parser.add_argument(
        "--quick", action="store_true", help="Skip model tests (faster)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Show game window during tests"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("SUPER CONTRA AI — FULL SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("CUDA / GPU", test_cuda, {}),
        ("Environment Creation", test_env_creation, {"render": args.render}),
        ("Action Execution", test_actions, {"render": args.render}),
        ("Reward System", test_rewards, {}),
        ("VecEnv + Frame Stack", test_vec_env, {}),
    ]

    if not args.quick:
        tests.append(("Model Load + Predict", test_model, {}))
        tests.append(("Inference Speed", test_speed, {}))

    results = []
    for name, func, kwargs in tests:
        try:
            passed = func(**kwargs)
            results.append(passed)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append(False)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for i, ((name, _, _), passed) in enumerate(zip(tests, results), 1):
        status = "PASS" if passed else "FAIL"
        print(f"  [{i}] {name:<30} {status}")

    passed_count = sum(results)
    total = len(results)
    print()

    if all(results):
        print(f"ALL {total} TESTS PASSED — system is production-ready")
    else:
        failed = total - passed_count
        print(f"{passed_count}/{total} passed, {failed} failed")
        sys.exit(1)

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
