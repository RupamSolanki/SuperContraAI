"""
record_showcase.py — Record the BIG window (game + action/reward panel) to an MP4.

This is the recording companion to predict_big.py. It plays the SAME model and
draws the SAME layout (game upscaled | gap | action/reward/game-state panel),
but instead of only showing it on screen it writes every combined frame straight
to an .mp4 via OpenCV's VideoWriter. No screen recorder, no dropped frames,
clean fixed-FPS output ready for a showcase video.

predict_big.py and src/predict.py are NOT modified.

WHY A SEPARATE SCRIPT?
  - predict_big.py renders live for watching; this one renders to disk for editing.
  - You can point --checkpoint at any of the 178 checkpoints to film the "learning
    montage" (early dumb agent -> final smart agent) with on-screen step labels.

USAGE (from project root, venv active):
    # Final model, Level1 start, one episode -> recordings/showcase_Level1.mp4
    python record_showcase.py

    # Pick a start state (Level1 | Level1_mid | Level1_boss)
    python record_showcase.py --state Level1_boss

    # Film a specific training checkpoint and stamp its step count on the video
    python record_showcase.py --checkpoint checkpoints/super_c_phase2_2000000_steps.zip --label "2M steps"

    # Multiple episodes into one file, custom output name/fps/scale
    python record_showcase.py --episodes 3 --fps 60 --scale 4 --out recordings/final_run.mp4

NOTE: needs a display to render frames (WSLg on Windows 11). Run locally.
      This records VIDEO ONLY (the panel). For game AUDIO, use the .bk2 -> ffmpeg
      path documented in README (src/predict.py --record + playback_movie).
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
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")
VECNORM_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_vecnorm.pkl")
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, "recordings")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_NAMES = [
    "Forced Right", "Right", "Left", "Crouch",
    "Up", "Right+Fire", "Jump", "Jump+Fire",
]

# Same palette as predict_big.py (BGR) so the recording matches the live view.
PANEL_BG     = (45, 42, 40)
BAR_TRACK    = (70, 66, 62)
BAR_IDLE     = (95, 92, 88)
BAR_ACTIVE   = (120, 200, 130)
REWARD_POS   = (120, 200, 130)
REWARD_NEG   = (110, 120, 230)
TEXT_BRIGHT  = (235, 232, 228)
TEXT_MUTED   = (170, 165, 158)
LABEL_BG     = (30, 28, 26)


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
    """Draw the side info panel (action bars + reward bar) as a BGR image.
    Mirrors predict_big.build_panel so recordings match the live view."""
    PW = 360
    panel = np.full((height, PW, 3), PANEL_BG, dtype=np.uint8)

    def text(img, s, x, y, color=TEXT_BRIGHT, scale=0.5, thick=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)

    y = 34
    text(panel, "AGENT ACTION", 20, y, TEXT_BRIGHT, 0.62, 2)
    y += 18

    bar_x = 150
    bar_max_w = PW - bar_x - 20
    row_h = 30
    for i, name in enumerate(ACTION_NAMES):
        top = y + i * row_h
        active = (i == action_idx)
        text(panel, name, 20, top + 20, TEXT_BRIGHT if active else TEXT_MUTED,
             0.5, 2 if active else 1)
        cv2.rectangle(panel, (bar_x, top + 6), (bar_x + bar_max_w, top + 24),
                      BAR_TRACK, -1)
        fill = bar_max_w if active else int(bar_max_w * 0.08)
        color = BAR_ACTIVE if active else BAR_IDLE
        cv2.rectangle(panel, (bar_x, top + 6), (bar_x + fill, top + 24), color, -1)

    y = y + len(ACTION_NAMES) * row_h + 24

    text(panel, "STEP REWARD", 20, y, TEXT_BRIGHT, 0.6, 2)
    y += 16
    cx = 20
    cw = PW - 40
    mid = cx + cw // 2
    track_top = y
    cv2.rectangle(panel, (cx, track_top), (cx + cw, track_top + 22),
                  BAR_TRACK, -1)
    cv2.line(panel, (mid, track_top - 2), (mid, track_top + 24), (120, 116, 110), 1)
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

    text(panel, f"Step reward : {step_reward:>10.1f}", 20, y, TEXT_BRIGHT, 0.52, 1); y += 26
    text(panel, f"Episode rew : {ep_reward:>10.1f}", 20, y, TEXT_BRIGHT, 0.52, 1); y += 34

    text(panel, "GAME STATE", 20, y, TEXT_BRIGHT, 0.6, 2); y += 24
    text(panel, f"World X : {world_x}", 20, y, TEXT_MUTED, 0.52, 1); y += 24
    text(panel, f"Lives   : {lives}", 20, y, TEXT_MUTED, 0.52, 1); y += 24
    boss_col = REWARD_NEG if boss == "DEAD" else TEXT_MUTED
    text(panel, f"Boss    : {boss}", 20, y, boss_col, 0.52, 1); y += 24

    return panel


def draw_label(frame, label):
    """Stamp a caption (e.g. '2M steps') in the top-left of the game view."""
    if not label:
        return frame
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (10, 10), (10 + tw + 20, 10 + th + 20), LABEL_BG, -1)
    cv2.putText(frame, label, (20, 10 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                TEXT_BRIGHT, 2, cv2.LINE_AA)
    return frame


def record(model_path, episodes, scale, state, fps, out_path, label, deterministic,
           preview):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    env = DummyVecEnv([make_env(state)])
    env = VecFrameStack(env, n_stack=12)
    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env, device=DEVICE)
    retro_env = get_raw_retro_env(env)

    writer = None
    reward_window = deque(maxlen=120)
    total_frames = 0

    print(f"Recording {episodes} episode(s) | state={state} | model={os.path.basename(model_path)}")
    print(f"Output: {out_path} @ {fps} FPS, scale {scale}x")

    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        ep_step = 0
        ep_reward = 0.0
        ep_max_x = 0
        print(f"  --- Episode {ep}/{episodes} ---")

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

            frame = retro_env.render()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            big = cv2.resize(frame, (w * scale, h * scale),
                             interpolation=cv2.INTER_NEAREST)
            big = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            big = draw_label(big, label)

            panel = build_panel(big.shape[0], a_idx, step_r, ep_reward,
                                world_x, lives, boss, reward_window)
            gap = np.full((big.shape[0], 16, 3), 255, dtype=np.uint8)
            combined = np.hstack([big, gap, panel])

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps,
                                         (combined.shape[1], combined.shape[0]))
                if not writer.isOpened():
                    print("ERROR: could not open VideoWriter. Check codec/output path.")
                    env.close()
                    return

            writer.write(combined)
            total_frames += 1

            if preview:
                cv2.imshow("record_showcase (preview)", combined)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    done = True

        print(f"    done — steps={ep_step}, reward={ep_reward:.1f}, max WorldX={ep_max_x}")

    if writer is not None:
        writer.release()
    env.close()
    if preview:
        cv2.destroyAllWindows()

    secs = total_frames / fps if fps else 0
    print(f"\nSaved {total_frames} frames (~{secs:.1f}s) -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Record the big window (game + panel) to MP4")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_MODEL,
                   help="model .zip to film (default: models/super_c_production_final.zip)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--scale", type=int, default=3, help="game upscale factor (default 3)")
    p.add_argument("--state", type=str, default="Level1",
                   help="start state: Level1 | Level1_mid | Level1_boss")
    p.add_argument("--fps", type=int, default=30, help="output video FPS (default 30)")
    p.add_argument("--out", type=str, default=None,
                   help="output .mp4 path (default: recordings/showcase_<state>.mp4)")
    p.add_argument("--label", type=str, default="",
                   help="caption stamped on the game view, e.g. '2M steps'")
    p.add_argument("--deterministic", type=str, default="true")
    p.add_argument("--preview", action="store_true",
                   help="also show a live preview window while recording (press Q to stop)")
    a = p.parse_args()

    out = a.out or os.path.join(RECORDINGS_DIR, f"showcase_{a.state}.mp4")
    record(
        model_path=a.checkpoint,
        episodes=a.episodes,
        scale=a.scale,
        state=a.state,
        fps=a.fps,
        out_path=out,
        label=a.label,
        deterministic=(a.deterministic.lower() != "false"),
        preview=a.preview,
    )
