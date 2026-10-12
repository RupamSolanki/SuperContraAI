# Cloud Deployment Guide — Vast.ai

Train Super Contra AI on a rented Vast.ai GPU. No quota requests, no capacity
walls — GPUs are available on demand and billed per hour. Mirrors the verified
local setup (SB3 2.9.0 / gymnasium 1.3.0 / stable-retro 1.0.1).

The model resumes automatically from `models/super_c_production_final.zip`.

> Vast is a marketplace of individual hosts — you pay real money (no free
> credit). For this project it's cheap (a few dollars for Phase 3).

---

## Step 0 — Account + credit
1. Sign up at vast.ai, verify email.
2. Add credit (Billing) — $10 is plenty to start.

---

## Step 1 — Pick a GPU instance
1. Go to **Search / Console → Create**.
2. Filters that matter:
   - **GPU:** an RTX 3090 / 4090 / A5000 / L4 is great and cheap. Even an RTX
     3060/3070 beats your laptop. (Your CNN is small — you don't need an A100.)
   - **Disk space:** set to **~30 GB** (project + model + checkpoints).
   - **Reliability:** prefer higher reliability score + good upload/download.
3. **Template / Image:** choose a **PyTorch** or **CUDA 12.x** template (these
   have the NVIDIA driver + CUDA preinstalled). Avoid bare images.
4. Click **Rent**. The instance boots in a minute or two.

---

## Step 2 — Connect (SSH)
1. On the **Instances** page, wait for status to go green/"running".
2. Click the instance → **Connect / SSH** — Vast shows an SSH command like:
   `ssh -p <PORT> root@<HOST>` (and often a Jupyter link too).
3. Vast usually gives you a **root** shell. Note the SSH host + port.

You'll need your SSH key added to your Vast account (Account → SSH Keys) for
`ssh`/`scp` to work. Generate one locally if needed:
`ssh-keygen -t ed25519` then paste the `.pub` into Vast.

---

## Step 3 — Package the project (local, WSL2)
```bash
cd /mnt/d/00stable-contra
tar -czf contra-project.tar.gz \
  src/ checks/ config/ models/ roms/ requirements.txt setup_vast.sh
```
(Only the files needed to train — includes models/ so it resumes.)

---

## Step 4 — Upload to the instance
Use the SSH host + port Vast gave you:
```bash
scp -P <PORT> contra-project.tar.gz root@<HOST>:~/
```
(Alternatively, Vast's web UI has a file upload, or use the Jupyter interface
to upload the tarball.)

---

## Step 5 — Set up on the instance
SSH in, then:
```bash
mkdir -p ~/super-contra && cd ~/super-contra
tar -xzf ~/contra-project.tar.gz
bash setup_vast.sh
```
The script checks the GPU, builds the venv, installs torch + requirements,
imports the ROM, copies config, and verifies.

> If torch fails on CUDA mismatch: check the image's CUDA with `nvcc --version`
> or `nvidia-smi`, and change the index in setup_vast.sh (cu128 for CUDA 12.8,
> cu121 for CUDA 12.1, etc.), then re-run step 4 of the script.

---

## Step 6 — Train (inside tmux)
```bash
source ~/contra-venv/bin/activate
cd ~/super-contra
nproc                      # shows how many CPUs you have
tmux new -s training
export NUM_CPU=8           # set to the nproc value (or a bit less)
python src/train.py --phase 3
```
Detach: `Ctrl+B` then `D`. Reattach: `tmux attach -t training`.
Stop: `Ctrl+C` (saves to models/).

---

## Step 7 — Back up the model (IMPORTANT)
Vast instances are ephemeral — if the host is reclaimed you lose local data.
Periodically pull the model down to your machine:
```bash
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_production_final.zip D:\00stable-contra\models\
scp -P <PORT> root@<HOST>:~/super-contra/models/super_c_vecnorm.pkl D:\00stable-contra\models\
```
Do this every few hours, and again when training finishes.

---

## Step 8 — DESTROY the instance when done
On the Instances page → your instance → **Destroy** (stop billing). A running
GPU keeps charging by the second, so destroy it once the model is downloaded.

---

## Notes
- Training runs headless. Don't run predict.py / --verify (needs a display).
- Cheapest reliable pick: an RTX 3090/4090 or L4 with a PyTorch template.
- Because Vast hosts can disappear, the "back up models/ often" step matters
  more here than on GCP. Checkpoints every 500K steps also help.
