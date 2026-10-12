"""Check RAM values during gameplay to verify addresses are correct."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import stable_retro
from env_wrapper import SuperContraWrapper

env = stable_retro.make(game="SuperC-Nes-v0", state="Level1", render_mode="rgb_array")
env = SuperContraWrapper(env, frame_skip=4)
obs, info = env.reset()

print("Initial state:")
for key, val in sorted(info.items()):
    print(f"  {key:<15} = {val}")

print("\nPlaying 100 steps (moving right)...")
print(f"{'Step':<6} {'world_x':<10} {'lives':<7} {'score':<8} {'game_mode':<11} {'reward':<10}")
print("-" * 60)

for step in range(100):
    obs, reward, terminated, truncated, info = env.step(0)  # Forced Right

    sx = int(info.get("screen_x", 0))
    sc = int(info.get("scroll_x", 0))
    world_x = (sc * 256) + sx
    score = (info.get("score_mid", 0) * 100) + info.get("score_low", 0)

    if step % 10 == 0:
        print(f"{step:<6} {world_x:<10} {info.get('lives', '?'):<7} {score:<8} {info.get('game_mode', '?'):<11} {reward:<10.1f}")

    if terminated or truncated:
        print(f"  >> TERMINATED at step {step}")
        obs, info = env.reset()

env.close()
print("\nDone. Check if world_x increases smoothly (not jumping by thousands).")
