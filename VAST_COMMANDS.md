# Vast.ai — Command Cheat Sheet (copy/paste)

Your live instance details (update these if you rent a new one):
- **PORT:** `23052`
- **HOST:** `79.117.120.96`
- **GPU:** RTX 4080S (16GB), 8 CPUs, PyTorch CUDA 12.8 image
- **Price:** ~$0.248/hr

> `ssh` uses lowercase `-p`. `scp` uses uppercase `-P`. (Easy to mix up.)

---

## 1. Connect (SSH) — from LOCAL WSL2
```bash
ssh -p 23052 root@79.117.120.96
```
First time it asks to trust the host → type `yes`.
The instance auto-starts in tmux. Detach: `Ctrl+B` then `D`. Reattach: `tmux attach`.

---

## 2. Upload the project — from LOCAL WSL2
Build the tarball (only if code changed), then upload:
```bash
cd /mnt/d/00stable-contra
tar -czf contra-project.tar.gz src/ checks/ config/ models/ roms/ states/ requirements.txt setup_vast.sh
scp -P 23052 contra-project.tar.gz root@79.117.120.96:~/
```

---

## 3. First-time setup — ON THE INSTANCE
```bash
mkdir -p ~/super-contra && cd ~/super-contra
tar -xzf ~/contra-project.tar.gz
bash setup_vast.sh
```
(If torch fails on CUDA mismatch: check `nvidia-smi` CUDA version, edit
setup_vast.sh `cu128` -> `cu124`/`cu121`, re-run.)

---

## 4. Train — ON THE INSTANCE
```bash
source ~/contra-venv/bin/activate
cd ~/super-contra
python src/train.py --phase 3        # NUM_CPU auto-detects all 8 cores
```
- Resumes from models/super_c_production_final.zip automatically.
- Phases: `--phase 1|2|3|4|5` (1=Basics 2=Combat 3=Mastery 4=Expert 5=Superhuman).
- **Detach (keep training after you disconnect):** `Ctrl+B` then `D`.
- **Reattach later:** SSH in, then `tmux attach`.
- **Stop training:** `Ctrl+C` (saves model + vecnorm to models/).

---

## 5. Monitor — ON THE INSTANCE
```bash
nvidia-smi                 # GPU usage / temp / VRAM
nproc                      # CPU count
```
Watch the `fps` and `ep_rew_mean` in the training output.

---

## 6. Back up the model — from LOCAL WSL2 (DO THIS EVERY FEW HOURS!)
Vast hosts are ephemeral — if reclaimed you lose instance data.
```bash
scp -P 23052 root@79.117.120.96:~/super-contra/models/super_c_production_final.zip /mnt/d/00stable-contra/models/
scp -P 23052 root@79.117.120.96:~/super-contra/models/super_c_vecnorm.pkl /mnt/d/00stable-contra/models/
```
Optional — grab checkpoints too:
```bash
scp -P 23052 root@79.117.120.96:~/super-contra/checkpoints/*.zip /mnt/d/00stable-contra/checkpoints/
```

---

## 7. Download final model when done — from LOCAL WSL2
(Same as backup — pull the final trained model home:)
```bash
scp -P 23052 root@79.117.120.96:~/super-contra/models/super_c_production_final.zip /mnt/d/00stable-contra/models/
scp -P 23052 root@79.117.120.96:~/super-contra/models/super_c_vecnorm.pkl /mnt/d/00stable-contra/models/
```

---

## 8. DESTROY the instance (STOP BILLING) — in Vast browser
Vast Console -> **Instances** -> your instance -> **DESTROY** (trash icon) -> confirm.
A running GPU charges by the second. Destroy it the moment the model is downloaded.

---

## Quick timing / cost (RTX 4080S @ ~822 FPS, $0.248/hr, from ~10.4M steps)
| Target | Steps left | Time (~822 fps) | Cost |
|--------|-----------|-----------------|------|
| Phase 3 (30M) | ~19.6M | ~6.5 hrs | ~$1.65 |
| Phase 5 (150M) | ~139.6M | ~47 hrs (~2 days) | ~$12 |

---

## Watch the trained model locally (after downloading)
On your LOCAL machine (needs display / WSLg), NOT the instance:
```bash
source ~/contra-venv/bin/activate
cd /mnt/d/00stable-contra
python src/predict.py --episodes 2 --slow
```

---

## If you rent a NEW instance later
1. Update PORT + HOST at the top of this file (from Vast -> instance -> SSH Connect).
2. Redo steps 1-4 (connect, upload, setup, train).
3. Your model resumes from whatever is in local models/ (upload it in the tarball).
