"""
predict_big.py — Live prediction in a BIG window with action + reward bars.

The default src/predict.py uses stable-retro's native viewer (NES size 256x240),
too small to show people. This plays the SAME model but draws each frame with
OpenCV, upscaled by --scale, in a large window — PLUS a side panel showing:
  - which button/action the agent is pressing (bar chart, mild colors)
  - the per-step reward (bar) and running episode reward
src/predict.py is NOT modified.

USAGE (from project root, venv active):
    python predict_big.py                  # 1 episode, scale 3
    python predict_big.py --episodes 3     # watch a few runs
    python predict_big.py --scale 4        # bigger
    python predict_big.py --no-slow        # full speed

Controls: press Q (window focused) to quit early.
NOTE: needs a display (WSLg on Windows 11). Run locally.
"""

import os
import sys
import time
import argparse
from collections import deque

import cv2
import numpy as np
import torch
import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from env_wrapper import SuperContraWrapper

GAME = "SuperC-Nes-v0"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")
VECNORM_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_vecnorm.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW = "Super Contra AI - Live"

ACTION_NAMES = [
    "Forced Right", "Right", "Left", "Crouch",
    "Up", "Right+Fire", "Jump", "Jump+Fire",
]

# Dark panel, soft readable accents (BGR). Dark bg = text/bars stand out.
PANEL_BG     = (45, 42, 40)      # dark charcoal background
BAR_TRACK    = (70, 66, 62)      # bar background track
BAR_IDLE     = (95, 92, 88)      # muted grey (inactive action)
BAR_ACTIVE   = (120, 200, 130)   # clear soft green (chosen action)
REWARD_POS   = (120, 200, 130)   # soft green
REWARD_NEG   = (110, 120, 230)   # soft red/salmon
TEXT_BRIGHT  = (235, 232, 228)   # near-white (readable on dark)
TEXT_MUTED   = (170, 165, 158)   # soft grey text


def make_env(state):
    def _init():
        env = stable_retro.make(game=GAME, state=state, render_mode="rgb_array")
        env = SuperContraWrapper(env, frame_skip=4)
        return env
    return _init


def get_raw_retro_env(vec_env):
    e = vec_env
    while hasattr(e, "venv"):
        e = e.venv
    return e.envs[0].unwrapped


def build_panel(height, action_idx, step_reward, ep_reward, world_x, lives, boss,
                reward_window):
    """Draw the side info panel (action bars + reward bar) as a BGR image."""
    PW = 360  # panel width
    panel = np.full((height, PW, 3), PANEL_BG, dtype=np.uint8)

    def text(img, s, x, y, color=TEXT_BRIGHT, scale=0.5, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)

    y = 34
    text(panel, "AGENT ACTION", 20, y, TEXT_BRIGHT, 0.62, 2)
    y += 18

    # --- Action bar chart (one row per action) ---
    bar_x = 150
    bar_max_w = PW - bar_x - 20
    row_h = 30
    for i, name in enumerate(ACTION_NAMES):
        top = y + i * row_h
        active = (i == action_idx)
        # label
        text(panel, name, 20, top + 20, TEXT_BRIGHT if active else TEXT_MUTED,
             0.5, 2 if active else 1)
        # bar background track
        cv2.rectangle(panel, (bar_x, top + 6), (bar_x + bar_max_w, top + 24),
                      BAR_TRACK, -1)
        # filled bar: full for active, tiny stub for others
        fill = bar_max_w if active else int(bar_max_w * 0.08)
        color = BAR_ACTIVE if active else BAR_IDLE
        cv2.rectangle(panel, (bar_x, top + 6), (bar_x + fill, top + 24), color, -1)

    y = y + len(ACTION_NAMES) * row_h + 24

    # --- Reward bar (per-step), centered zero ---
    text(panel, "STEP REWARD", 20, y, TEXT_BRIGHT, 0.6, 2)
    y += 16
    cx = 20
    cw = PW - 40
    mid = cx + cw // 2
    track_top = y
    cv2.rectangle(panel, (cx, track_top), (cx + cw, track_top + 22),
                  BAR_TRACK, -1)
    cv2.line(panel, (mid, track_top - 2), (mid, track_top + 24), (120, 116, 110), 1)
    # scale step reward to half-width (clip so huge values don't overflow)
    denom = max(1.0, max(abs(r) for r in reward_window) if reward_window else 1.0)
    frac = max(-1.0, min(1.0, step_reward / denom))
    half = cw // 2
    if frac >= 0:
        cv2.rectangle(panel, (mid, track_top), (mid + int(half * frac), track_top + 22),
                      REWARD_POS, -1)
    else:
        cv2.rectangle(panel, (mid + int(half * frac), track_top), (mid, track_top + 22),
                      REWARD_NEG, -1)
    y = track_top + 44

    # --- Numeric readouts ---
    text(panel, f"Step reward : {step_reward:>10.1f}", 20, y, TEXT_BRIGHT, 0.52, 1); y += 26
    text(panel, f"Episode rew : {ep_reward:>10.1f}", 20, y, TEXT_BRIGHT, 0.52, 1); y += 34

    text(panel, "GAME STATE", 20, y, TEXT_BRIGHT, 0.6, 2); y += 24
    text(panel, f"World X : {world_x}", 20, y, TEXT_MUTED, 0.52, 1); y += 24
    text(panel, f"Lives   : {lives}", 20, y, TEXT_MUTED, 0.52, 1); y += 24
    boss_col = REWARD_NEG if boss == "DEAD" else TEXT_MUTED
    text(panel, f"Boss    : {boss}", 20, y, boss_col, 0.52, 1); y += 24

    return panel


def watch(episodes, scale, state, slow, deterministic, win_x=30, win_y=30):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    env = DummyVecEnv([make_env(state)])
    env = VecFrameStack(env, n_stack=12)
    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
    retro_env = get_raw_retro_env(env)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW, win_x, win_y)  # open at a fixed position, not centered
    _win_sized = False
    reward_window = deque(maxlen=120)
    print(f"Watching {episodes} episode(s) at {scale}x | state={state} | press Q to quit")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_step = 0
        ep_reward = 0.0
        ep_max_x = 0
        print(f"\n--- Episode {ep}/{episodes} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            step_r = float(reward[0])
            ep_reward += step_r
            reward_window.append(step_r)
            ep_step += 1

            a_idx = int(action[0])
            sx = int(info[0].get("screen_x", 0)); sc = int(info[0].get("scroll_x", 0))
            world_x = sc * 256 + sx
            ep_max_x = max(ep_max_x, world_x)
            lives = info[0].get("lives", "?")
            boss = "DEAD" if info[0].get("boss_dead", 0) else "alive"

            frame = retro_env.render()  # rgb_array -> HxWx3 (RGB)
            if frame is not None:
                h, w = frame.shape[:2]
                big = cv2.resize(frame, (w * scale, h * scale),
                                 interpolation=cv2.INTER_NEAREST)
                big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)

                panel = build_panel(big.shape[0], a_idx, step_r, ep_reward,
                                    world_x, lives, boss, reward_window)
                # thin white gap between game and panel
                gap = np.full((big.shape[0], 16, 3), 255, dtype=np.uint8)
                combined = np.hstack([big, gap, panel])   # game | gap | panel

                if not _win_sized:
                    cv2.resizeWindow(WINDOW, combined.shape[1], combined.shape[0])
                    cv2.moveWindow(WINDOW, win_x, win_y)  # re-apply after resize
                    _win_sized = True
                cv2.imshow(WINDOW, combined)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Quit by user.")
                env.close(); cv2.destroyAllWindows(); return

            if slow:
                time.sleep(1 / 30)

        print(f"  Episode {ep} done — steps={ep_step}, reward={ep_reward:.1f}, "
              f"max WorldX={ep_max_x}")

    env.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live prediction in a BIG window with bars")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--scale", type=int, default=3, help="window upscale factor (default 3)")
    p.add_argument("--state", type=str, default="Level1", help="start state")
    p.add_argument("--no-slow", dest="slow", action="store_false", help="full speed")
    p.add_argument("--deterministic", type=str, default="true")
    p.add_argument("--x", type=int, default=30, help="window X position (default 30)")
    p.add_argument("--y", type=int, default=30, help="window Y position (default 30)")
    p.set_defaults(slow=True)
    a = p.parse_args()
    watch(
        episodes=a.episodes,
        scale=a.scale,
        state=a.state,
        slow=a.slow,
        deterministic=(a.deterministic.lower() != "false"),
        win_x=a.x,
        win_y=a.y,
    )
