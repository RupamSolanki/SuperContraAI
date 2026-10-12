# Super Contra AI — PPO Reinforcement Learning Agent

A production-ready reinforcement learning system that trains an AI agent to play
**Super Contra (NES)** using PPO (Proximal Policy Optimization). The agent learns
entirely from pixel observations and RAM-based reward signals — no human demonstrations needed.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Why WSL (Ubuntu on Windows)](#why-wsl-ubuntu-on-windows)
- [Why PPO Algorithm](#why-ppo-algorithm)
- [Architecture Overview](#architecture-overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Stack](#software-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How Training Works](#how-training-works)
- [How Recording Works (Video + Audio)](#how-recording-works-video--audio)
- [File-by-File Documentation](#file-by-file-documentation)
- [RAM Memory Mapping](#ram-memory-mapping)
- [Reward System Design](#reward-system-design)
- [Training Phases](#training-phases)
- [Commands Reference](#commands-reference)
- [Troubleshooting](#troubleshooting)

---

## Why This Exists

This project trains a neural network to play Super Contra (NES) at superhuman level.
The AI watches the game screen (as 84x84 grayscale pixels), reads game memory (lives,
score, position), and learns which button presses lead to survival and progress.

After ~80 hours of training, the agent can clear Level 1 consistently and fight
enemies better than most human players.

---

## Why WSL (Ubuntu on Windows)

**stable-retro** (the game emulator library) has full support on Linux but limited
support on native Windows. Specifically:

- The emulator cores (NES, SNES, Genesis) are compiled for Linux
- CUDA GPU training works seamlessly on Ubuntu via WSL2
- WSL2 provides near-native Linux performance with full GPU passthrough
- The Python ML ecosystem (PyTorch, stable-baselines3) is better tested on Linux
- Audio recording tools (ffmpeg, PulseAudio) work natively on Linux

**WSL2** (Windows Subsystem for Linux 2) gives you a full Ubuntu environment inside
Windows with direct access to your NVIDIA GPU. Your files live on the Windows drive
(`/mnt/d/`) but Python and training run on Linux.

**Why not native Windows?**
- stable-retro's NES emulator has rendering bugs on Windows
- SubprocVecEnv (parallel training) is unreliable on Windows due to how Python
  multiprocessing works on Windows vs Linux (fork vs spawn)
- ffmpeg audio capture requires PulseAudio which is Linux-only

---

## Why PPO Algorithm

**PPO (Proximal Policy Optimization)** was chosen over other RL algorithms because:

| Algorithm | Pros | Cons | Why not |
|-----------|------|------|---------|
| **DQN** | Simple, proven for Atari | Only works with discrete actions, unstable with large obs | Slower convergence on NES games |
| **A2C** | Fast, simple | High variance, unstable | Crashes during long training |
| **PPO** | Stable, scalable, works with CNN | Slightly slower per step | **Best overall for game AI** |
| **SAC** | Sample efficient | Designed for continuous actions | Super Contra has discrete buttons |

PPO advantages for this project:
- **Stable training** — clips policy updates so it never makes catastrophically bad changes
- **Parallel environments** — scales linearly with CPU cores (4 workers = 4x data)
- **CNN policy support** — processes raw pixel frames through convolutional neural network
- **Entropy bonus** — encourages exploration (tries new strategies instead of getting stuck)
- **Works with VecNormalize** — normalizes rewards for stable gradient descent

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │  NES ROM    │────>│  stable-retro    │────>│ SuperContra    │  │
│  │  (game)     │     │  (emulator)      │     │ Wrapper        │  │
│  └─────────────┘     └──────────────────┘     └────────────────┘  │
│                                                       │            │
│                              ┌─────────────────────────┘            │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Environment Output:                                      │     │
│  │    - Observation: (84, 84, 1) grayscale frame             │     │
│  │    - Reward: shaped from RAM (progress, kills, deaths)    │     │
│  │    - Info: RAM variables (lives, score, position, etc.)   │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  VecFrameStack (n=12)                                     │     │
│  │    Stacks 12 consecutive frames -> (84, 84, 12)           │     │
│  │    Gives the CNN temporal information (motion detection)   │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  VecTransposeImage (auto by SB3)                          │     │
│  │    (84, 84, 12) -> (12, 84, 84) channels-first for CNN   │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  PPO Agent (CnnPolicy on GPU)                             │     │
│  │                                                           │     │
│  │  CNN Encoder:                                             │     │
│  │    Conv2D(12->32, 8x8, stride=4) -> ReLU                 │     │
│  │    Conv2D(32->64, 4x4, stride=2) -> ReLU                 │     │
│  │    Conv2D(64->64, 3x3, stride=1) -> ReLU                 │     │
│  │    Flatten -> Linear(3136->512) -> ReLU                   │     │
│  │                                                           │     │
│  │  Policy Head (Actor):                                     │     │
│  │    Linear(512->64) -> ReLU -> Linear(64->8)              │     │
│  │    Output: probability over 8 actions                     │     │
│  │                                                           │     │
│  │  Value Head (Critic):                                     │     │
│  │    Linear(512->64) -> ReLU -> Linear(64->1)              │     │
│  │    Output: expected future reward                         │     │
│  └───────────────────────────────────────────────────────────┘     │
│                              │                                      │
│                              ▼                                      │
│                        Action (0-7)                                  │
│                              │                                      │
│                              ▼                                      │
│                     Back to Environment                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | This Project |
|-----------|---------|--------------|
| CPU | 4 cores | AMD Ryzen 5 5600H (6 cores, 4 used for training) |
| GPU | 2GB VRAM | NVIDIA RTX 3050 Laptop (4GB VRAM) |
| RAM | 8GB | 16GB recommended |
| Storage | 500MB | ~200MB (model + checkpoints) |
| OS | Ubuntu 20.04+ via WSL2 | Ubuntu 22.04 on WSL2 |

---

## Software Stack

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.12.13 | Runtime language |
| **stable-retro** | 1.0.1 | NES emulator — loads ROM, runs game, exposes frames + RAM |
| **stable-baselines3** | 2.9.0 | PPO implementation — handles training loop, CNN policy, gradient descent |
| **PyTorch** | 2.10.0+cu128 | Deep learning backend — GPU tensor operations, neural network |
| **gymnasium** | 1.3.0 | Environment API standard — defines obs/action/reward interface |
| **OpenCV** | 5.0.0 | Image processing — RGB to grayscale, resize to 84x84 |
| **NumPy** | 2.2.6 | Array operations — frame manipulation, reward math |
| **TensorBoard** | (included) | Training visualization — reward curves, loss plots |
| **ffmpeg** | 4.4.2 | Video encoding — converts .bk2 replay to .mp4 with audio |
| **NumPy** | 2.2.6 | Array operations — frame manipulation, reward math |
| **TensorBoard** | (included) | Training visualization — reward curves, loss plots |
| **ffmpeg** | 4.4.2 | Video encoding — converts .bk2 replay to .mp4 with audio |

---

## Project Structure

```
d:\00stable-contra\
│
├── src/                         # Core production code
│   ├── env_wrapper.py           # Environment wrapper (reward shaping, action mapping)
│   ├── train.py                 # PPO training script (5 phases, auto-resume)
│   └── predict.py               # Watch AI play + record video with audio
│
├── checks/                      # Health check & diagnostic scripts
│   ├── check_system.py          # Quick: packages + GPU + ROM
│   ├── check_game.py            # Quick: list registered games
│   ├── check_model_obs.py       # Quick: model observation shape
│   ├── check_ram.py             # Quick: verify RAM values
│   ├── check_scenario.py        # Quick: inspect game data files
│   ├── verify.py                # Pre-flight hardware/env checks
│   └── test.py                  # Full 7-test system validation
│
├── config/                      # Game configuration
│   ├── data.json                # RAM address mapping (15 variables)
│   └── scenario.json            # Disables built-in reward
│
├── models/                      # Trained models
│   ├── super_c_production_final.zip
│   └── super_c_vecnorm.pkl
│
├── roms/                        # ROM files (gitignored)
│   └── Super C (USA).nes
│
├── docs/                        # Documentation
│   ├── UPGRADE_PLAN.md
│   └── Super contra ram mapping.docx
│
├── checkpoints/                 # Auto-saved during training
├── logs/                        # TensorBoard logs
├── recordings/                  # .bk2 and .mp4 files
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites

1. **Windows 10/11** with WSL2 enabled
2. **Ubuntu 22.04** installed via Microsoft Store
3. **NVIDIA GPU drivers** (Game Ready or Studio drivers on Windows)

### Step-by-step

```bash
# 1. Open Ubuntu terminal (WSL)

# 2. Install Python 3.12 (if not available)
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# 3. Create and activate virtual environment (on Linux filesystem for speed)
python3.12 -m venv ~/contra-venv
source ~/contra-venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch with CUDA
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# 6. Install remaining packages
pip install stable-retro==1.0.1 stable-baselines3==2.9.0 gymnasium==1.3.0 opencv-python numpy==2.2.6 tensorboard

# 7. Install ffmpeg (for video recording with audio)
sudo apt-get install ffmpeg

# 8. Import game ROM into stable-retro
mkdir -p ~/rom_temp
cp "/mnt/d/00stable-contra/roms/Super C (USA).nes" ~/rom_temp/
python -m retro.import ~/rom_temp/

# 9. Copy data.json and scenario.json to game folder
NEW_PATH=$(python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))")
cp /mnt/d/00stable-contra/config/data.json "$NEW_PATH/data.json"
cp /mnt/d/00stable-contra/config/scenario.json "$NEW_PATH/scenario.json"

# 10. Navigate to project and verify
cd /mnt/d/00stable-contra
python checks/check_system.py
python checks/test.py
```

### Activating the Environment (after first setup)

```bash
source ~/contra-venv/bin/activate
cd /mnt/d/00stable-contra
```

---

## How Training Works

### The Training Loop (simplified)

```
1. Four game instances start (no display, maximum speed)
2. Each game produces frames at ~150 FPS per instance (600 total)
3. Every 1024 steps per worker (4096 total), PPO collects a "batch"
4. The batch goes to GPU: CNN processes frames -> predict best action
5. PPO calculates loss: "did this action lead to good reward?"
6. Gradient descent updates the neural network weights
7. Repeat until total_steps reached
8. Save model to .zip file
```

### Why 4 Parallel Environments?

PPO needs lots of diverse experience to learn. Running 4 games simultaneously:
- Provides 4x more training data per second
- Each game is in a different state (one dying, one progressing, etc.)
- Reduces variance — model sees many scenarios per update
- Your Ryzen 5600H has 6 cores — 4 workers + 1 for PPO + 1 for OS = perfect fit

### Why VecNormalize?

Rewards in Super Contra are huge (progress = +100 per pixel, kills = +250 per point).
Raw rewards cause gradient explosions. VecNormalize:
- Tracks running mean/std of rewards
- Normalizes them to roughly [-10, +10]
- Saves statistics to `super_c_vecnorm.pkl` so prediction uses same scale

### Why Frame Stack = 12?

The CNN sees a single image — it can't detect motion from one frame. By stacking
12 consecutive frames into channels (like a short video), the network can:
- Detect bullet trajectories
- See enemy movement direction
- Know if the player is jumping up or falling down
- Understand scrolling speed (am I moving forward?)

12 frames at frame_skip=4 = seeing the last 48 game frames of history (~0.8 seconds).

Your trained model expects observation shape `(12, 84, 84)` — 12 channels of 84x84
grayscale frames.

---

## How Recording Works (Video + Audio)

### The Problem

stable-retro does NOT expose audio through its Python API. You can only get:
- RGB image frames (what you see on screen)
- RAM values (game memory)
- Button press recordings (.bk2 files)

There is no `env.get_audio()` method. Audio is handled internally by the emulator
and sent directly to the system audio device.

### The Solution

We use stable-retro's **native .bk2 replay system**:

1. During `src/predict.py --record`, stable-retro saves button presses + initial game state
   to a tiny `.bk2` file (deterministic replay format — same as TASVideos)
2. After episodes complete, we run `python -m stable_retro.scripts.playback_movie`
3. This script replays the .bk2 through the full emulator with ffmpeg capturing
   both video AND audio output from the emulator core
4. Output: a standard .mp4 file with game video + game audio

### Why This Approach?

| Approach | Video | Audio | Quality | Complexity |
|----------|-------|-------|---------|------------|
| Screen capture (OBS) | Yes | Yes | Good | Manual, external tool |
| OpenCV frame recording | Yes | No | Good | Simple but no audio |
| PulseAudio system capture | Yes | Yes | Variable | Sync issues, depends on audio setup |
| **.bk2 replay + ffmpeg** | **Yes** | **Yes** | **Perfect** | **Built into stable-retro** |

The .bk2 approach gives pixel-perfect video and emulator-native audio because it
replays the game deterministically through the actual emulator. No frame drops,
no audio sync issues.

### Dependencies for Recording

- **ffmpeg** (Ubuntu package) — does the actual video/audio encoding
- **stable-retro's playback_movie script** — replays .bk2 through emulator + pipes to ffmpeg

```bash
# Install ffmpeg
sudo apt-get install ffmpeg

# Record gameplay (saves .bk2 replay)
python src/predict.py --record --episodes 1

# Convert .bk2 to .mp4 with video + audio (run manually after predict)
python -m stable_retro.scripts.playback_movie "./recordings/SuperC-Nes-v0-Level1-000000.bk2"

# Fix video rotation if needed (NES outputs portrait, rotate 90° clockwise)
ffmpeg -i ./recordings/SuperC-Nes-v0-Level1-000000.mp4 -vf "transpose=1" ./recordings/gameplay_fixed.mp4
```

**Note:** On WSL, the automatic .bk2→.mp4 conversion may crash due to a memory bug
in stable-retro's multiprocessing. The workaround is to run the playback_movie
command manually after predict finishes. On native Linux (cloud), it works automatically.

---

## File-by-File Documentation

### env_wrapper.py — Environment Wrapper

This is the heart of the project. It wraps the raw stable-retro NES emulator into
something PPO can learn from effectively.

**Classes:**

1. **FrameSkipWrapper** — Repeats each action for 4 NES frames. NES runs at 60fps
   but meaningful decisions happen every ~4 frames. This gives 4x training speedup
   and smoother control.

2. **SuperContraWrapper** — The main wrapper that:
   - Converts RGB (224x256x3) to grayscale (84x84x1) for faster CNN processing
   - Maps 8 discrete actions to NES button combinations
   - Reads RAM memory to compute custom rewards
   - Tracks progress (furthest X position reached)
   - Detects kills via score changes
   - Penalizes bad behavior (going left, camping, dying)
   - Handles indoor vs outdoor level types differently
   - Terminates episodes on game over

**Why grayscale 84x84?**
- 84x84 is the standard resolution for Atari/NES RL (from DeepMind's DQN paper)
- Grayscale removes irrelevant color info — position/shape matters, not color
- Smaller image = faster CNN processing = more training steps per hour
- INTER_NEAREST resize is 4x faster than INTER_LINEAR on GPU

**Why 8 discrete actions (not all NES buttons)?**
- NES controller has 9 buttons = 2^9 = 512 possible combinations
- Most combinations are useless (pressing Start during gameplay)
- 8 carefully chosen actions cover all meaningful gameplay moves
- Smaller action space = faster learning (less to explore)

---

### train.py — Training Script

**What it does:**
1. Creates 4 parallel game instances (SubprocVecEnv)
2. Wraps them with frame stacking (12 frames) and reward normalization
3. Creates or loads PPO model with CnnPolicy
4. Trains for N million steps with periodic checkpoints
5. Saves model + normalization stats on completion or Ctrl+C

**Key components:**

- **SubprocVecEnv** — Runs 4 separate Python processes, each with its own game.
  Uses OS-level process isolation for true parallelism (bypasses Python GIL).

- **VecMonitor** — Tracks episode rewards/lengths for TensorBoard logging.

- **VecFrameStack(n=12)** — Stacks 12 frames along channel dimension.

- **VecNormalize** — Running normalization of rewards (not observations since pixels
  are already 0-255).

- **get_linear_fn** — Creates entropy decay schedule. High entropy early = explore
  random strategies. Low entropy late = exploit best known strategy.

- **CheckpointCallback** — Saves model every N steps to `./checkpoints/` so you
  never lose more than N steps of progress.

**PPO Hyperparameters explained:**

```python
learning_rate = 0.0003    # How big each weight update step is (too high = unstable, too low = slow)
n_steps = 1024            # Steps collected per worker before update (4 workers = 4096 total)
batch_size = 128          # Mini-batch size for GPU (128 fits in 4GB RTX 3050 VRAM)
n_epochs = 10             # How many times to reuse each batch (more = better but diminishing returns)
gamma = 0.99              # Discount factor (0.99 = care about rewards up to ~100 steps ahead)
gae_lambda = 0.95         # GAE smoothing (balances bias vs variance in advantage estimation)
clip_range = 0.2          # PPO clip (limits how much policy can change per update = stability)
ent_coef = schedule       # Entropy bonus (decays from 0.10 to 0.05 over phase = explore then exploit)
vf_coef = 0.5            # Value function loss weight (critic learning rate relative to actor)
max_grad_norm = 0.5       # Gradient clipping (prevents exploding gradients from huge rewards)
```

---

### predict.py — Watch AI Play + Record

**What it does:**
1. Loads trained model from .zip file
2. Creates single game instance with human rendering (display window)
3. Runs N episodes, printing live stats
4. Optionally records .bk2 replay and converts to .mp4 with audio

**Recording flow:**

```
predict.py --record
    │
    ├── stable_retro.make(record='./recordings/')
    │   └── Saves button presses to .bk2 file per episode
    │
    ├── After episodes: RecordingWrapper.convert_bk2_to_mp4()
    │   └── Calls: python -m stable_retro.scripts.playback_movie file.bk2
    │       └── Replays game through emulator
    │       └── Pipes video+audio to ffmpeg
    │       └── Outputs .mp4 file
    │
    └── If multiple episodes: concatenates all .mp4 files with ffmpeg
```

**Why deterministic=True by default?**
- Shows the model's "best" behavior (always picks highest probability action)
- deterministic=False adds randomness (useful to test if model actually learned
  vs just memorized one sequence)

---

### verify.py — Pre-Flight Check

Runs 3 quick checks before you commit to an hour of training:
1. **CUDA** — confirms GPU is visible and reports VRAM
2. **Environment** — loads game, checks observation shape, verifies RAM keys
3. **Model** (optional) — loads .zip, runs dummy forward pass, confirms not corrupt

---

### test.py — Full System Test (7 checks)

More thorough than verify.py. Tests:
1. CUDA/GPU detection
2. Environment creation + observation shape
3. All 8 actions execute without crash
4. Reward signals fire during random play
5. VecEnv + FrameStack architecture matches training
6. Model loads and produces valid predictions
7. Inference speed benchmark (FPS)

---

### data.json — RAM Address Mapping

This file tells stable-retro which memory addresses to read from the NES RAM.
The NES has 2KB of RAM (addresses 0-2047). Each address holds one byte (0-255).

stable-retro reads these addresses every frame and puts them in the `info` dict.
Our wrapper uses them to compute rewards and detect game events.

This file must be placed in stable-retro's game data folder (found via
`python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))"`).

---

## RAM Memory Mapping

| Variable | Address | Range | Usage in Wrapper |
|----------|---------|-------|------------------|
| lives | 83 ($53) | 0-255 | Death/game over detection |
| current_level | 80 ($50) | 0-7 | Which level we're on |
| game_mode | 24 ($18) | 0-255 | 4=playing, other=menu/demo |
| level_type | 93 ($5d) | 0 or 1 | 0=side-scroll, 1=overhead |
| screen_x | 253 ($fd) | 0-255 | Pixels scrolled within current screen |
| scroll_x | 99 ($63) | 0-255 | Number of screens scrolled into level |
| screen_y | 206 ($ce) | 0-255 | Player Y position on screen |
| scroll_y | 252 ($fc) | 0-255 | Vertical PPU scroll |
| indoor_depth | 130 ($82) | 0-255 | Vertical screen index (overhead levels) |
| score_low | 2019 ($7e3) | 0-255 | P1 Score BCD byte 0 |
| score_mid | 2020 ($7e4) | 0-255 | P1 Score BCD byte 1 |
| boss_dead | 92 ($5c) | 0-255 | Bit 0 = boss defeated |
| weapon | 184 ($b8) | 0-255 | 0-4 weapon type, bit 7=rapid fire |
| game_over | 202 ($ca) | 0 or 1 | 0=alive, 1=game over |
| player_state | 160 ($a0) | 0-8 | 2=alive, 3=hit, 5=dead |

All addresses verified from the official Super C (NES) disassembly:
https://github.com/vermiceli/nes-super-c/blob/main/src/ram.asm

**How world_x is calculated:**
```
world_x = (scroll_x * 256) + screen_x
        = (X_SCREEN * 256) + X_SCROLL
```
This gives absolute horizontal position across the entire level.
Used as the primary progress reward.

---

## Reward System Design

The reward system is the most critical part. It tells the AI what "good" means.

### Reward Signals (ordered by magnitude)

| Signal | Value | When | Purpose |
|--------|-------|------|---------|
| Boss killed | +2000 | boss_dead flips 0→1 | Level clear is the ultimate goal |
| Death penalty | -600 | lives decreases | Dying is very bad |
| Game over | -500 | lives=0 or game_mode=0 | Hard terminate episode |
| Kill reward | +250 × score_delta | score increases | Fighting > avoiding |
| Weapon pickup | +200 | weapon value increases | Better weapons = easier progress |
| Progress | +100 × pixel_delta | world_x increases (outdoor) | Primary objective: move right |
| Indoor progress | +80 × depth_delta | indoor_depth increases | Same but for 3D levels |
| Left penalty | -10 to -20 | action=Left | Don't go backwards |
| Stagnation | -10/step | no progress for 90 steps | Don't stand still |
| Crouch penalty | -5 | action=Crouch | Don't camp/hide |
| Jump reflex | +1.0 | action=Jump when moving | Encourage reactive dodging |
| Forced right | +0.5 | action=ForcedRight | Tiny anti-freeze bonus |
| Fire bribe | +0.1 | action=Right+Fire or Jump+Fire | Encourage shooting |

### Design Philosophy

1. **Progress is king** — +100/pixel means the AI always wants to move right
2. **Killing is better than hiding** — +250/score vs -5/crouch makes fighting optimal
3. **Death is very expensive** — -600 means the AI will learn to dodge bullets
4. **Stagnation is punished** — can't camp in one spot for more than 1.5 seconds
5. **Left is almost never correct** — -10 to -20 prevents backtracking
6. **Rewards scale with magnitude** — boss kill (2000) > progress (100) > fire bribe (0.1)
   so the AI prioritizes correctly

---

## Training Phases

Each phase trains with decreasing exploration (entropy) and learning rate:

| Phase | Steps | Time | Entropy | LR | Goal |
|-------|-------|------|---------|-----|------|
| 1 | 2M | ~1 hr | 0.10→0.05 | 3e-4 | Move right, don't die |
| 2 | 10M | ~5 hrs | 0.05→0.02 | 2e-4 | Reach mid-level, fight enemies |
| 3 | 30M | ~15 hrs | 0.02→0.01 | 1e-4 | Clear Level 1 consistently |
| 4 | 80M | ~40 hrs | 0.01→0.005 | 5e-5 | Multi-level, near human-level |
| 5 | 150M | ~80 hrs | 0.005→0.001 | 2e-5 | Superhuman on Level 1 |

**Why entropy decays:**
- Phase 1: high entropy (0.10) = try random actions = discover that "right" is good
- Phase 5: low entropy (0.001) = almost always pick best known action = consistent play

**Why learning rate decays:**
- Early: big updates are fine because the model knows nothing
- Late: small updates prevent forgetting what was already learned (fine-tuning)

---

## Commands Reference

### Training

```bash
# Start Phase 1 (or resume if model exists)
python src/train.py --phase 1

# Jump to Phase 3 settings (higher steps, lower entropy)
python src/train.py --phase 3

# Verify RAM addresses are correct (visual, 500 steps)
python src/train.py --verify
```

### Watching AI Play

```bash
# Watch 5 episodes (default)
python src/predict.py

# Watch 10 episodes with recording (.bk2 saved)
python src/predict.py --episodes 10 --record

# Slow playback (30 FPS, comfortable to watch)
python src/predict.py --slow

# Use stochastic mode (random exploration)
python src/predict.py --deterministic false

# Use a specific checkpoint
python src/predict.py --checkpoint ./checkpoints/super_c_phase1_500000_steps.zip

# Convert recording to .mp4 with audio (after predict --record)
python -m stable_retro.scripts.playback_movie "./recordings/SuperC-Nes-v0-Level1-000000.bk2"

# Fix video rotation
ffmpeg -i ./recordings/SuperC-Nes-v0-Level1-000000.mp4 -vf "transpose=1" gameplay_fixed.mp4
```

### Verification

```bash
# Quick system check (packages + GPU + ROM)
python checks/check_system.py

# List registered games in stable-retro
python checks/check_game.py

# Verify RAM addresses work during gameplay (200 steps)
python checks/check_ram.py

# Check what observation shape the model expects
python checks/check_model_obs.py

# Inspect game data files (data.json, scenario.json paths)
python checks/check_scenario.py

# Pre-flight hardware + env + model check
python checks/verify.py --model

# Full 7-test live validation (CUDA, env, actions, rewards, model, speed)
python checks/test.py

# Quick tests (skip model load — faster)
python checks/test.py --quick

# Project structure validation (paths, imports, configs)
python checks/run_production_check.py
```

### TensorBoard (monitor training progress)

```bash
tensorboard --logdir ./logs/
# Open http://localhost:6006 in browser
```

---

## Troubleshooting

### "No romfiles found for game: SuperC-Nes-v0"

ROM not imported. Fix:
```bash
mkdir -p ~/rom_temp
cp "/mnt/d/00stable-contra/Super C (USA).nes" ~/rom_temp/
python -m retro.import ~/rom_temp/
```

After import, copy data.json to the game's data folder:
```bash
cp /mnt/d/00stable-contra/data.json /mnt/d/00stable-contra/venv/lib/python3.10/site-packages/stable_retro/data/stable/SuperC-Nes-v0/data.json
```

### "Observation spaces do not match"

Model was trained with different n_stack. Check with:
```bash
python checks/check_model_obs.py
```
Then match N_STACK in src/train.py, src/predict.py, checks/test.py to the model's expected shape.

### "CUDA not available"

NVIDIA driver not installed on Windows, or WSL2 can't see GPU:
```bash
nvidia-smi  # Should show your RTX 3050
```
If not, install latest NVIDIA driver on Windows (not in WSL).

### Training is slow (<100 FPS)

- Make sure `render_mode=None` in train.py (no display window)
- Check if another process is using GPU: `nvidia-smi`
- Reduce NUM_CPU from 4 to 2 if running out of RAM

### Agent only presses one button

Model needs more training. Run at least Phase 1 (2M steps, ~1 hour):
```bash
python src/train.py --phase 1
```

### "apt-get" gives 404 errors

Package lists outdated:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Video recording has no audio

- ffmpeg must be installed: `ffmpeg -version`
- The .bk2 -> .mp4 conversion must be run manually on WSL:
  ```bash
  python -m stable_retro.scripts.playback_movie "./recordings/SuperC-Nes-v0-Level1-000000.bk2"
  ```
- If video is rotated, fix with:
  ```bash
  ffmpeg -i input.mp4 -vf "transpose=1" output_fixed.mp4
  ```
- Auto-conversion crashes on WSL due to a memory bug in multiprocessing. Works fine on native Linux (cloud).

### "score" error after upgrading stable-retro

The default scenario.json references a `score` variable. Fix by copying our custom scenario.json:
```bash
NEW_PATH=$(python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))")
cp /mnt/d/00stable-contra/config/scenario.json "$NEW_PATH/scenario.json"
```

---

## ROM Setup

The US version (Super C) is the primary ROM. All RAM addresses in data.json are
verified against the official Super C disassembly:
https://github.com/vermiceli/nes-super-c/blob/main/src/ram.asm

```bash
# Import ROM
mkdir -p ~/rom_temp
cp "/mnt/d/00stable-contra/roms/Super C (USA).nes" ~/rom_temp/
python -m retro.import ~/rom_temp/

# Copy data.json and scenario.json to game folder
NEW_PATH=$(python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))")
cp /mnt/d/00stable-contra/config/data.json "$NEW_PATH/data.json"
cp /mnt/d/00stable-contra/config/scenario.json "$NEW_PATH/scenario.json"
```

**Why scenario.json?**
stable-retro ships a default scenario.json for SuperC-Nes-v0 that references a
`score` variable not in our custom data.json. Our scenario.json disables the
built-in reward (our wrapper handles all reward shaping) and only uses `lives`
for episode termination.

## Adding Save States

Create additional starting positions for curriculum variety:

1. Play manually to a mid-level or boss position
2. Save state as `Level1_mid.state` or `Level1_boss.state`
3. Place in the game's stable-retro data folder (run the path command above to find it)
4. Uncomment in `src/train.py` → `AVAILABLE_STATES`

---

## Cloud Training (AWS)

For faster training (~2x speed at $0.15/hr with Spot instances):

```bash
# See full guide:
docs/CLOUD_DEPLOYMENT.md
```

Quick summary:
1. Launch AWS g4dn.xlarge Spot instance (Deep Learning AMI)
2. Upload project: `scp contra-project.tar.gz ubuntu@<IP>:~/`
3. Run setup: `bash setup_cloud.sh`
4. Train: `tmux new -s training && python src/train.py --phase 3`
5. Download model when done
6. **Stop instance to avoid charges**

Estimated cost for Phases 3-5: **~$18** ($30 budget gives ample buffer).

---

## License

This project is for educational/research purposes. The game ROM must be legally obtained.
