"""
train.py — Super Contra PPO Training Script
============================================
Game    : Super Contra (NES) via stable-retro 1.0.1
Agent   : PPO + CnnPolicy (Stable-Baselines3 2.9.0)
Hardware: Ryzen 5600H + RTX 3050 (4GB VRAM) / Cloud GPU

TRAINING PHASES:
    Phase 1 ->  2M  steps  (~1 hr)  : Basics — move right, don't die
    Phase 2 ->  10M steps  (~5 hrs) : Combat — reach mid-level, fight enemies
    Phase 3 ->  30M steps  (~15 hrs): Mastery — clear Level 1 consistently
    Phase 4 ->  80M steps  (~40 hrs): Expert — multi-level, near human-level
    Phase 5 ->  150M steps (~80 hrs): Superhuman — better than human on Level 1

USAGE:
    python train.py                 # Auto-resume, Phase 1 by default
    python train.py --phase 2      # Start/resume Phase 2
    python train.py --verify       # Quick RAM verification (500 steps)
"""

import os
import sys
import argparse
import random

import torch
import stable_retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecMonitor,
    VecNormalize,
)
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_wrapper import SuperContraWrapper

# =============================================================================
#  CONFIGURATION
# =============================================================================
GAME = "SuperC-Nes-v0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_production_final")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VECNORM_PATH = os.path.join(PROJECT_ROOT, "models", "super_c_vecnorm.pkl")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Parallel workers. Defaults to the machine's CPU count so it "just works"
# on any host (laptop, Vast, GCP) without needing to set anything.
# Override with the NUM_CPU env var if you want fewer, e.g.  export NUM_CPU=4
NUM_CPU = int(os.environ.get("NUM_CPU", str(os.cpu_count() or 4)))
N_STACK = 12  # Frame stack depth (model expects 12 channels)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Save states for curriculum variety.
# Add more as you create them (Level1_mid, Level1_boss, etc.)
AVAILABLE_STATES = [
    "Level1",
    "Level1_mid",   # curriculum: start near the mid-level obstacle (~world_x 1710)
    "Level1_boss",  # curriculum: start deep in the level toward the boss (~world_x 2301)
]

# =============================================================================
#  PHASE DEFINITIONS
# =============================================================================
PHASES = {
    1: {
        "total_steps": 2_000_000,
        "description": "Basics: move right, don't die",
        "ent_start": 0.10,
        "ent_end": 0.05,
        "lr": 0.0003,
        "save_freq": 100_000,
    },
    2: {
        "total_steps": 10_000_000,
        "description": "Combat: reach mid-level, fight enemies",
        "ent_start": 0.05,
        "ent_end": 0.02,
        "lr": 0.0002,
        "save_freq": 200_000,
    },
    3: {
        "total_steps": 30_000_000,
        "description": "Mastery: clear Level 1 consistently",
        "ent_start": 0.02,
        "ent_end": 0.01,
        "lr": 0.0001,
        "save_freq": 500_000,
    },
    4: {
        "total_steps": 80_000_000,
        "description": "Expert: multi-level, near human-level play",
        "ent_start": 0.01,
        "ent_end": 0.005,
        "lr": 0.00005,
        "save_freq": 1_000_000,
    },
    5: {
        "total_steps": 150_000_000,
        "description": "Superhuman: better than human on Level 1",
        "ent_start": 0.005,
        "ent_end": 0.001,
        "lr": 0.00002,
        "save_freq": 2_000_000,
    },
}


# =============================================================================
#  ENVIRONMENT FACTORY
# =============================================================================
def make_env(rank: int):
    """Create one Super Contra env per SubprocVecEnv worker."""

    def _init():
        state = random.choice(AVAILABLE_STATES)
        env = stable_retro.make(
            game=GAME,
            state=state,
            render_mode=None,  # No display = max speed (400-800 FPS)
        )
        env = SuperContraWrapper(env, frame_skip=4)
        env.reset(seed=rank * 100)
        return env

    return _init


# =============================================================================
#  RAM VERIFICATION
# =============================================================================
def verify_ram():
    """
    Quick 500-step test to verify RAM addresses work correctly.
    Run once before full training to confirm data.json is right.
    """
    print("\n" + "=" * 70)
    print("RAM VERIFICATION — 500 steps with random actions")
    print("Expect: score increases on kill, lives drop on death, world_x increases")
    print("=" * 70)

    env = stable_retro.make(game=GAME, state="Level1", render_mode="human")
    env = SuperContraWrapper(env, frame_skip=4)
    obs, info = env.reset()

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 30 == 0:
            score = (info.get("score_mid", 0) * 100) + info.get("score_low", 0)
            sx = int(info.get("screen_x", 0))
            sc = int(info.get("scroll_x", 0))
            print(
                f"  Step {step:>4} | "
                f"lives={info.get('lives', '?')} | "
                f"game_mode={info.get('game_mode', '?')} | "
                f"level_type={info.get('level_type', '?')} | "
                f"weapon={info.get('weapon', '?')} | "
                f"score={score:>6} | "
                f"world_x={(sc * 256) + sx:>5} | "
                f"boss_dead={info.get('boss_dead', '?')} | "
                f"reward={reward:>8.1f}"
            )

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("\nVerification complete.")
    print("  If score never changes -> check score_low/score_mid in data.json")
    print("  If world_x never changes -> check screen_x/scroll_x in data.json")


# =============================================================================
#  TRAINING
# =============================================================================
def train(phase: int):
    """Run PPO training for the specified phase."""
    cfg = PHASES[phase]

    print(f"\n{'=' * 60}")
    print(f"SUPER CONTRA AI — PHASE {phase}")
    print(f"  Goal   : {cfg['description']}")
    print(f"  Steps  : {cfg['total_steps']:,}")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  Workers: {NUM_CPU} parallel envs")
    print(f"  Stack  : {N_STACK} frames -> obs (84, 84, {N_STACK})")
    print(f"  States : {AVAILABLE_STATES}")
    print(f"{'=' * 60}\n")

    # --- Build environment stack ---
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecNormalize(
        env,
        norm_obs=False,  # Pixels 0-255, no normalization needed
        norm_reward=True,  # Normalize rewards for stable gradients
        clip_reward=10.0,
        gamma=0.99,
    )

    # --- Entropy coefficient ---
    # SB3's PPO uses self.ent_coef directly as a scalar in the loss
    # (loss = ... + self.ent_coef * entropy_loss), so it must be a float,
    # not a schedule callable. Each phase's start/end are close, so we use
    # the phase's start value as a constant entropy coefficient.
    ent_coef = float(cfg["ent_start"])

    # --- Model: resume or fresh ---
    model_zip = MODEL_PATH + ".zip"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if os.path.exists(model_zip):
        print(f"Resuming from {model_zip}")
        model = PPO.load(model_zip, env=env, device=DEVICE)
        model.ent_coef = ent_coef
        model.learning_rate = cfg["lr"]

        if os.path.exists(VECNORM_PATH):
            env = VecNormalize.load(VECNORM_PATH, env)
            env.training = True
            env.norm_reward = True
            print(f"VecNormalize restored from {VECNORM_PATH}")
    else:
        print("Starting fresh model")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            device=DEVICE,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=cfg["lr"],
            n_steps=1024,  # Per worker -> 4x1024=4096 total per update
            batch_size=128,  # RTX 3050 4GB VRAM sweet spot
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=cfg["save_freq"] // NUM_CPU,
        save_path=CHECKPOINT_DIR,
        name_prefix=f"super_c_phase{phase}",
        verbose=1,
    )

    # --- Train ---
    print(f"Training Phase {phase} — {cfg['total_steps']:,} steps")
    print(f"  Checkpoints: every {cfg['save_freq']:,} steps -> {CHECKPOINT_DIR}")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print(f"  Stop: Ctrl+C (progress is always saved)\n")

    try:
        model.learn(
            total_timesteps=cfg["total_steps"],
            reset_num_timesteps=False,
            callback=checkpoint_cb,
            tb_log_name=f"PPO_Phase{phase}",
        )
        print(f"\nPhase {phase} complete!")
        if phase < 5:
            print(f"  Next: python train.py --phase {phase + 1}")

    except KeyboardInterrupt:
        print("\nStopped — saving progress...")

    finally:
        model.save(MODEL_PATH)
        env.save(VECNORM_PATH)
        env.close()
        print(f"  Model saved -> {model_zip}")
        print(f"  Norms saved -> {VECNORM_PATH}")


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Contra PPO Training")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Training phase 1-5 (default: 1)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run RAM verification mode (500 steps with visual)",
    )
    args = parser.parse_args()

    if args.verify:
        verify_ram()
    else:
        train(phase=args.phase)
