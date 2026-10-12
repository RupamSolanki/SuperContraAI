"""
predict.py — Watch Your Trained Super Contra AI Play
=====================================================
Loads the saved PPO model and runs it live.

Recording uses stable-retro's built-in .bk2 replay system, then converts
to .mp4 with full audio via ffmpeg. This is the cleanest approach since
stable-retro handles audio internally during playback.

USAGE:
    python predict.py                          # Watch 5 episodes
    python predict.py --episodes 10            # Watch 10 episodes
    python predict.py --slow                   # Throttle to 30 FPS
    python predict.py --record                 # Save to super_c_gameplay.mp4 (video + audio)
    python predict.py --checkpoint ./checkpoints/super_c_phase1_500000_steps.zip
    python predict.py --deterministic false    # Stochastic exploration mode
"""

import argparse
import glob
import os
import subprocess
import time

import numpy as np
import torch
import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import SuperContraWrapper

# =============================================================================
#  CONFIGURATION
# =============================================================================
GAME = "SuperC-Nes-v0"
STATE = "Level1"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final.zip")
VECNORM_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_vecnorm.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_OUTPUT = os.path.join(PROJECT_ROOT, "recordings", "super_c_gameplay.mp4")
RECORD_DIR = os.path.join(PROJECT_ROOT, "recordings")

ACTION_NAMES = {
    0: "Forced Right",
    1: "Right",
    2: "Left",
    3: "Crouch",
    4: "Up",
    5: "Right+Fire",
    6: "Jump",
    7: "Jump+Fire",
}


# =============================================================================
#  RECORDING WRAPPER
#  Uses stable-retro's native .bk2 recording (captures button presses).
#  After prediction, converts .bk2 -> .mp4 with audio via ffmpeg.
# =============================================================================
class RecordingWrapper:
    """
    Wraps the environment creation to enable .bk2 recording.
    After episodes complete, converts recordings to .mp4 with audio.
    """

    def __init__(self, output_path: str, record_dir: str):
        self.output_path = output_path
        self.record_dir = record_dir
        os.makedirs(record_dir, exist_ok=True)

    def convert_bk2_to_mp4(self):
        """Print instructions to convert .bk2 files to .mp4 with audio."""
        bk2_files = sorted(glob.glob(os.path.join(self.record_dir, "*.bk2")))

        if not bk2_files:
            print("  No .bk2 recordings found")
            return False

        print(f"  {len(bk2_files)} recording(s) saved.")
        print(f"  Convert to .mp4 with audio using:")
        print()
        for bk2_file in bk2_files:
            print(f"    python -m stable_retro.scripts.playback_movie \"{bk2_file}\"")
        print()
        print(f"  Output will be saved next to each .bk2 file as .mp4")
        return True

    def _concat_videos(self, mp4_parts: list):
        """Concatenate multiple .mp4 files into one using ffmpeg."""
        list_file = os.path.join(self.record_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for part in mp4_parts:
                f.write(f"file '{os.path.abspath(part)}'\n")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    self.output_path,
                ],
                capture_output=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: just use first part
            os.rename(mp4_parts[0], self.output_path)
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)


# =============================================================================
#  ENVIRONMENT FACTORY
# =============================================================================
def make_watch_env(record_dir: str = None):
    """
    Single env with human rendering for live viewing.
    If record_dir is set, enables .bk2 recording (captures button presses + state).
    """

    def _init():
        env = stable_retro.make(
            game=GAME,
            state=STATE,
            render_mode="human",
            record=record_dir if record_dir else False,
        )
        env = SuperContraWrapper(env, frame_skip=4)
        return env

    return _init


# =============================================================================
#  WATCH
# =============================================================================
def watch(
    episodes: int,
    model_path: str,
    slow: bool,
    record: bool,
    deterministic: bool,
):
    """Load model and run episodes with live display."""

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("  Train first: python train.py --phase 1")
        return

    print(f"\n{'=' * 60}")
    print(f"SUPER CONTRA AI — WATCH MODE")
    print(f"  Model   : {model_path}")
    print(f"  Device  : {DEVICE.upper()}")
    print(f"  Episodes: {episodes}")
    print(f"  Mode    : {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"  Record  : {'Yes -> ' + VIDEO_OUTPUT if record else 'No'}")
    print(f"{'=' * 60}\n")

    # --- Setup recording ---
    recorder = None
    record_dir = None
    if record:
        record_dir = RECORD_DIR
        os.makedirs(record_dir, exist_ok=True)
        recorder = RecordingWrapper(VIDEO_OUTPUT, record_dir)
        print(f"  Recording .bk2 replays to {record_dir}")
        print(f"  Will convert to .mp4 with audio after episodes complete\n")

    # --- Build env ---
    env = DummyVecEnv([make_watch_env(record_dir)])
    env = VecFrameStack(env, n_stack=12)

    if os.path.exists(VECNORM_PATH):
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False
        env.norm_reward = False
        print(f"VecNormalize loaded from {VECNORM_PATH}")

    # --- Load model ---
    model = PPO.load(model_path, env=env, device=DEVICE)
    print("Model loaded\n")

    # --- Episode loop ---
    all_rewards = []
    all_steps = []
    all_max_worldx = []

    for ep in range(1, episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_step = 0
        ep_max_worldx = 0
        done = False

        print(f"{'─' * 50}")
        print(f"Episode {ep}/{episodes}")
        print(f"{'─' * 50}")

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            ep_reward += float(reward[0])
            ep_step += 1

            # Track progress
            sx = int(info[0].get("screen_x", 0))
            sc = int(info[0].get("scroll_x", 0))
            world_x = (sc * 256) + sx
            ep_max_worldx = max(ep_max_worldx, world_x)

            # Live stats every 60 steps
            if ep_step % 60 == 0:
                score = (info[0].get("score_mid", 0) * 100) + info[0].get(
                    "score_low", 0
                )
                lives = info[0].get("lives", "?")
                weapon = info[0].get("weapon", "?")
                boss = info[0].get("boss_dead", 0)
                action_name = ACTION_NAMES.get(int(action[0]), "Unknown")

                print(
                    f"  Step {ep_step:>5} | "
                    f"{action_name:<12} | "
                    f"Lives: {lives} | "
                    f"Score: {score:>6} | "
                    f"WorldX: {world_x:>5} | "
                    f"Weapon: {weapon} | "
                    f"Boss: {'DEAD' if boss else 'alive'} | "
                    f"Reward: {ep_reward:>8.1f}"
                )

            if slow:
                time.sleep(1 / 30)

        # Episode summary
        all_rewards.append(ep_reward)
        all_steps.append(ep_step)
        all_max_worldx.append(ep_max_worldx)

        print(f"\n  Episode {ep} done — Steps: {ep_step}, "
              f"Reward: {ep_reward:.1f}, Max WorldX: {ep_max_worldx}\n")

    # --- Session summary ---
    print(f"\n{'=' * 60}")
    print(f"SESSION COMPLETE — {episodes} episodes")
    print(f"  Avg Reward   : {np.mean(all_rewards):.1f}")
    print(f"  Best Reward  : {np.max(all_rewards):.1f}")
    print(f"  Worst Reward : {np.min(all_rewards):.1f}")
    print(f"  Avg Steps    : {np.mean(all_steps):.0f}")
    print(f"  Avg Max WorldX: {np.mean(all_max_worldx):.0f}")
    print(f"{'=' * 60}")

    # --- Convert recordings to video ---
    env.close()

    if recorder:
        print(f"\nConverting recordings to video...")
        recorder.convert_bk2_to_mp4()


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch trained Super Contra AI")

    parser.add_argument(
        "--episodes", type=int, default=5, help="Episodes to watch (default: 5)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=MODEL_PATH, help="Model path"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Throttle to ~30 FPS"
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record gameplay to .mp4 with audio (requires ffmpeg)"
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="true",
        help="true=best action, false=stochastic (default: true)",
    )

    args = parser.parse_args()
    det = args.deterministic.lower() != "false"

    watch(
        episodes=args.episodes,
        model_path=args.checkpoint,
        slow=args.slow,
        record=args.record,
        deterministic=det,
    )
