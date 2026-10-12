# =============================================================================
#  SUPER CONTRA AI — Environment Setup & Launch Notes  (personal cheat-sheet)
# =============================================================================
#
#  Platform : Windows 11 + WSL2 (Ubuntu 24.04)
#  GPU      : RTX 3050 Laptop (4GB) — CUDA 12.8
#  Python   : 3.12  (works fine; supported range is 3.10–3.13, avoid 3.14)
#  Project  : /mnt/d/00stable-contra
#
#  This is documentation to copy/paste from — not a script to execute.
#  Section 1 (SETUP) is one-time. Section 2 (DAILY USE) is what you run
#  each session.
#
# =============================================================================
#  SECTION 1 — ONE-TIME SETUP
# =============================================================================

# --- 1a. (Optional) Install Python 3.13 via deadsnakes ------------------------
#   Only needed if you want 3.13 specifically. Ubuntu 24.04 already ships
#   python3.12, which works — the current venv was built with 3.12.
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.13 python3.13-venv python3.13-dev

# --- 1b. System packages (ffmpeg for recordings, zlib for build) --------------
sudo apt-get install -y ffmpeg zlib1g-dev

# --- 1c. Create the virtual environment ---------------------------------------
#   Use python3.12 (current) or python3.13 if you installed it above.
cd /mnt/d/00stable-contra
python3.12 -m venv ~/contra-venv
source ~/contra-venv/bin/activate
pip install --upgrade pip

# --- 1d. Install PyTorch (GPU/CUDA build) then the rest ------------------------
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# --- 1e. Import the ROM and copy the game config into stable-retro ------------
#   stable-retro needs the ROM imported once, plus our custom data.json /
#   scenario.json copied into its SuperC game folder.
mkdir -p ~/rom_temp
cp "roms/Super C (USA).nes" ~/rom_temp/
python -m retro.import ~/rom_temp/
GAME_DIR=$(python -c "import stable_retro, os; print(os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SuperC-Nes-v0'))")
cp config/data.json "$GAME_DIR/data.json"
cp config/scenario.json "$GAME_DIR/scenario.json"

# --- 1f. Sanity check (prints versions + GPU + confirms ROM found) ------------
python checks/check_system.py


# =============================================================================
#  SECTION 2 — DAILY USE
# =============================================================================

# --- 2a. Activate the venv (do this in EVERY new terminal) --------------------
source ~/contra-venv/bin/activate
cd /mnt/d/00stable-contra

# --- 2b. Train ----------------------------------------------------------------
#   Auto-resumes from models/super_c_production_final.zip if it exists.
#   Ctrl+C to stop — progress is saved to models/ on exit.
python src/train.py --phase 3          # Phase 3: Mastery (30M steps)
#   Phases: 1=Basics  2=Combat  3=Mastery  4=Expert  5=Superhuman
#           --phase 1 | 2 | 3 | 4 | 5

# --- 2c. Quick RAM verification (500 steps, opens a game window) --------------
python src/train.py --verify

# --- 2d. Predict / watch the trained agent play -------------------------------
python src/predict.py --episodes 2 --slow      # watch 2 episodes, ~30 FPS  (CONFIRMED WORKING)
python src/predict.py --episodes 5             # watch 5 episodes, full speed
python src/predict.py --record                 # save .bk2 replays -> convert to mp4 (needs ffmpeg)
python src/predict.py --deterministic false    # stochastic (exploratory) actions
python src/predict.py --checkpoint ./checkpoints/super_c_phase2_10000000_steps.zip

# --- 2e. Monitor (run in a separate terminal) ---------------------------------
nvidia-smi                                     # GPU usage / VRAM
tensorboard --logdir /mnt/d/00stable-contra/logs   # then open http://localhost:6006

# --- 2f. Record ONE full episode and convert to .mp4 --------------------------
#   Goal: one window, full playthrough, saved as a single mp4.
source ~/contra-venv/bin/activate
cd /mnt/d/00stable-contra

# 1) Play ONE episode with recording (one window; --slow = real-time speed):
python src/predict.py --record --episodes 1 --slow

# 2) Convert the .bk2 replay to .mp4 (video + audio):
python -m stable_retro.scripts.playback_movie recordings/*.bk2
#   -> the .mp4 is created next to the .bk2 in recordings/

#   NOTE 1: --episodes N opens N windows one-by-one (default is 5). Use 1 for
#           a single window/episode.
#   NOTE 2: A .bk2 is a button-press replay, NOT a video — do NOT use
#           `ffmpeg -i file.bk2`. playback_movie re-runs the emulator to render
#           frames + audio, then muxes to mp4 (needs ffmpeg installed).
#   NOTE 3: Do recording + conversion on your LOCAL machine (needs display +
#           the ROM imported). It won't work on a headless Vast/cloud instance.
#   NOTE 4: If the recording is only ~1 second, the episode is instant-
#           terminating on startup (game_mode==0 during load frames). Fix in
#           env_wrapper.py: add a startup grace period so game_mode==0 only
#           counts after ~15 steps:
#             if curr_lives == 0 or (game_mode == 0 and self.episode_steps > 15):

# Convert ALL .bk2 files in recordings/ (if you recorded several):
for f in recordings/*.bk2; do
    python -m stable_retro.scripts.playback_movie "$f"
done


# =============================================================================
#  NOTES / GOTCHAS
# =============================================================================
#  - predict.py and --verify use render_mode="human" -> need a display.
#    On Windows 11 WSL2 this works via WSLg out of the box.
#  - train.py --phase N runs headless (render_mode=None) = max speed.
#  - The saved model was originally trained with SB3 2.7.1; it resumes fine
#    under SB3 2.9.0 (may print a harmless version-mismatch warning).
#  - NUM_CPU auto-detects the machine's CPU count (os.cpu_count()). No need to
#    set it. To use fewer workers:  export NUM_CPU=<n>  before training.


# =============================================================================
#  PACKAGE THE PROJECT (tarball for cloud upload)
# =============================================================================
#  Only the files needed to train (includes models/ so it resumes).

cd /mnt/d/00stable-contra
# WSL2 / Linux:
tar -czf contra-project.tar.gz src/ checks/ config/ models/ roms/ states/ requirements.txt setup_vast.sh
# Windows PowerShell (zip alternative):
Compress-Archive -Path src, checks, config, models, roms, states, requirements.txt, setup_vast.sh -DestinationPath contra-project.zip


# =============================================================================
#  SECTION 3 — TRAIN ON VAST.AI  (full guide: VAST_README.md)
# =============================================================================
#
#  Vast.ai = rent a GPU by the hour, no quota walls. Steps summarized here;
#  full walkthrough in VAST_README.md at the project root.

# --- 3a. One-time: SSH key (so scp/ssh work) ----------------------------------
#   Generate if you don't have one, then paste the .pub into Vast > Account > SSH Keys
ssh-keygen -t ed25519            # press Enter through all prompts
cat ~/.ssh/id_ed25519.pub        # copy this whole line into Vast

# --- 3b. Rent: Vast Console > Search ------------------------------------------
#   - Disk ~30 GB, GPU RTX 3090/4090/A5000/L4, PyTorch or CUDA 12.x template.
#   - Rent -> Instances page -> wait for green "running".
#   - Click instance > Connect -> copy the SSH command:  ssh -p <PORT> root@<HOST>

# --- 3c. Upload the tarball (local WSL2) --------------------------------------
scp -P <PORT> contra-project.tar.gz root@<HOST>:~/

# --- 3d. On the instance: extract + setup -------------------------------------
#   (SSH in first: ssh -p <PORT> root@<HOST>)
mkdir -p ~/super-contra && cd ~/super-contra
tar -xzf ~/contra-project.tar.gz
bash setup_vast.sh
#   If torch fails on CUDA mismatch: check `nvidia-smi` CUDA version and edit
#   setup_vast.sh (cu128 -> cu121 for CUDA 12.1), then re-run.

# --- 3e. Train (inside tmux) --------------------------------------------------
source ~/contra-venv/bin/activate
cd ~/super-contra
tmux new -s training
python src/train.py --phase 3    # NUM_CPU auto-detects all cores
#   detach: Ctrl+B then D   |   reattach: tmux attach -t training   |   stop: Ctrl+C

# --- 3f. Back up the model (LOCAL WSL2 — do this every few hours!) -------------
#   Vast hosts are ephemeral; if reclaimed you lose local data.
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_production_final.zip /mnt/d/00stable-contra/models/
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_vecnorm.pkl /mnt/d/00stable-contra/models/

# --- 3g. When done: download final model, then DESTROY the instance -----------
#   (same scp as 3f), then Vast Instances page > your instance > DESTROY.
#   A running GPU charges by the second — destroy it once downloaded.




rom_temp

cd /mnt/d/00stable-contra
tar -czf contra-project.tar.gz \
  src/ checks/ config/ models/ roms/ states/ requirements.txt setup_vast.sh

