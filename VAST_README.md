# Vast.ai — Train Super Contra AI (step-by-step)

Complete guide to train this project on a rented Vast.ai GPU. No quota
requests, no capacity waiting — GPUs are available on demand, billed per hour.

This resumes your existing model (`models/super_c_production_final.zip`) — it
does NOT start from scratch.

---

## 0. What you'll do (overview)
1. Make a Vast.ai account + add credit + add your SSH key.
2. Rent a GPU with a PyTorch/CUDA template.
3. Upload the project (a tarball).
4. Run `setup_vast.sh` on the instance (installs everything).
5. Train in tmux, back up the model, destroy the instance when done.

---

## 1. Account setup (one time)
1. Sign up at https://vast.ai and verify your email.
2. **Billing** → add credit. **$10** is plenty to start.
3. **Account → SSH Keys** → add your PUBLIC key so you can `ssh`/`scp`.
   - If you don't have a key, make one locally (WSL2):
     ```bash
     ssh-keygen -t ed25519          # press Enter through the prompts
     cat ~/.ssh/id_ed25519.pub      # copy this whole line into Vast
     ```

---

## 2. Rent a GPU
1. Go to **Console → Search / Create**.
2. Set filters:
   - **Disk space:** ~30 GB (project + model + checkpoints).
   - **GPU:** RTX 3090 / 4090 / A5000 / L4 are cheap and great. Any modern
     GPU beats the laptop. (The CNN is small — no need for A100.)
   - Prefer higher **reliability** and good **upload/download** speed.
3. **Template / Image:** pick a **PyTorch** or **CUDA 12.x** template
   (driver + CUDA preinstalled). Do NOT pick a bare/blank image.
4. Click **RENT**. Instance boots in ~1–2 minutes.

---

## 3. Connect (SSH)
1. **Instances** page → wait for status = running (green).
2. Click the instance → **Connect** → copy the SSH command, e.g.:
   ```
   ssh -p 41234 root@ssh5.vast.ai
   ```
   (Your PORT and HOST will differ. You usually get a ROOT shell.)
3. Test it from your local WSL2 terminal:
   ```bash
   ssh -p <PORT> root@<HOST>
   ```
   Note your `<PORT>` and `<HOST>` — you'll reuse them for upload/download.

---

## 4. Package the project (local WSL2)
From the project folder. Only the files needed to train (includes `models/`
so it resumes; skips docs/recordings/venv):
```bash
cd /mnt/d/00stable-contra
tar -czf contra-project.tar.gz \
  src/ checks/ config/ models/ roms/ requirements.txt setup_vast.sh
```

---

## 5. Upload to the instance
```bash
scp -P <PORT> contra-project.tar.gz root@<HOST>:~/
```
(If scp is slow or blocked, Vast's web UI / Jupyter also has a file-upload
button — upload the tarball to the home directory.)

---

## 6. Run setup_vast.sh (on the instance)
SSH into the instance, then:
```bash
mkdir -p ~/super-contra && cd ~/super-contra
tar -xzf ~/contra-project.tar.gz
bash setup_vast.sh
```

### What setup_vast.sh does (7 steps, automatic):
1. Checks the GPU is visible (`nvidia-smi`).
2. Installs system packages (ffmpeg, zlib, python venv, git).
3. Creates a Python venv at `~/contra-venv` and upgrades pip.
4. Installs PyTorch (CUDA build) + everything in `requirements.txt`.
5. Imports the ROM and copies `data.json` / `scenario.json` into stable-retro.
6. Creates `checkpoints/ logs/ recordings/`.
7. Runs `checks/check_system.py` to verify GPU + ROM + versions.

**When it finishes**, it prints the train/monitor commands.

### If the torch install fails (CUDA mismatch)
Check the instance CUDA:
```bash
nvidia-smi          # top-right shows CUDA version
```
If it's CUDA 12.1 (not 12.8), edit setup_vast.sh line with the torch install:
- change `cu128` → `cu121`, then re-run `bash setup_vast.sh`.

---

## 7. Train
```bash
source ~/contra-venv/bin/activate
cd ~/super-contra

nproc                       # how many CPUs this instance has
tmux new -s training        # so training survives SSH disconnect
export NUM_CPU=8            # set to the nproc value (or slightly less)
python src/train.py --phase 3
```
- Resumes from your uploaded model and continues.
- **Detach (leave running):** press `Ctrl+B` then `D`.
- **Reattach later:** `tmux attach -t training`.
- **Stop:** `Ctrl+C` (saves model + vecnorm to `models/`).

Watch the `fps` in the output — on a real GPU it should beat the laptop's ~250.

### Monitor (second SSH tab, or after detaching)
```bash
nvidia-smi                  # confirm the GPU is busy
```

---

## 8. Back up the model (IMPORTANT — do this often)
Vast hosts are ephemeral. If the host is reclaimed, local data is lost.
Pull the model down to your laptop every few hours and when done:
```bash
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_production_final.zip /mnt/d/00stable-contra/models/
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_vecnorm.pkl /mnt/d/00stable-contra/models/
```
Checkpoints (every 500K steps) also live in `~/super-contra/checkpoints/`.

---

## 9. DESTROY the instance when done (stop billing!)
Instances page → your instance → **DESTROY**. A running GPU charges by the
second, so destroy it the moment the model is downloaded.

---

## Quick reference (copy/paste)
```bash
# LOCAL: package + upload
cd /mnt/d/00stable-contra
tar -czf contra-project.tar.gz src/ checks/ config/ models/ roms/ states/ requirements.txt setup_vast.sh
scp -P <PORT> contra-project.tar.gz root@<HOST>:~/

# ON INSTANCE: setup + train
mkdir -p ~/super-contra && cd ~/super-contra
tar -xzf ~/contra-project.tar.gz
bash setup_vast.sh
source ~/contra-venv/bin/activate
tmux new -s training
export NUM_CPU=8 && python src/train.py --phase 3
# Ctrl+B then D to detach

# LOCAL: back up model (repeat every few hours)
scp -P <PORT> root@<HOST>:~/super-contra/models/*.zip /mnt/d/00stable-contra/models/
scp -P <PORT> root@<HOST>:~/super-contra/models/*.pkl /mnt/d/00stable-contra/models/
```

---

## Notes / gotchas
- Training runs headless. Do NOT run `predict.py` or `--verify` on the
  instance — they need a display.
- Set `NUM_CPU` to the instance's real CPU count (`nproc`), not a fixed number.
- The saved model was trained with SB3 2.7.1; it resumes fine under 2.9.0
  (harmless version-mismatch warning).
- Back up `models/` regularly — ephemeral hosts are the one real risk on Vast.
