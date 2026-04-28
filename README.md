# ICBHI AST-SAM — Optimized Respiratory Sound Classification

Re-implementation and optimization of:
> *Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers*  
> Atakan Işık et al., arXiv 2512.22564

---

## What's new compared to the original repo

| Area | Original | This version |
|---|---|---|
| Feature extraction | Inside `__getitem__` — rerun every epoch | Pre-computed once in `preprocess.py` |
| Optimizer | Vanilla SAM (2× cost per step) | **LookSAM k=5** (~1.2× cost, same generalization) |
| Mixed precision | None (FP32) | **AMP autocast + GradScaler** |
| DataLoader | 0 workers, no pin_memory | `num_workers=4`, `pin_memory=True`, `persistent_workers=True` |
| Batch size | 8 | 16 (VRAM freed by AMP) |
| LR schedule | Flat 1e-5 | **CosineAnnealingWarmRestarts** |
| Checkpointing | Best model only | **Full resume checkpoint every epoch** (survives Colab disconnect) |
| Training plots | None | 4-panel dashboard saved every epoch |
| Evaluation figures | Confusion matrix | + t-SNE, per-class bar chart, per-device breakdown, probability matrix |
| Augmentation | Gain + noise on waveform | Gain + noise + **SpecAugment** (freq/time masking) on spectrogram |

**Expected speedup on RTX 4050 (6 GB):** ~6–10× vs original → ~30–50 min/epoch.

---

## Project structure

```
/
├── src/
│   ├── __init__.py
│   ├── model.py       # CustomAST (unchanged architecture from paper)
│   ├── dataset.py     # ASTDataset — loads pre-computed spectrograms + SpecAugment
│   └── look_sam.py    # LookSAM optimizer (Liu et al. 2022)
├── data/              # Place ICBHI_final_database/ and split file here
├── checkpoints/       # Saved models (symlinked to Drive in Colab)
├── results/           # Training plots and evaluation figures
├── preprocess.py      # One-time offline feature extraction
├── train.py           # Optimized training loop
├── evaluate.py        # Comprehensive evaluation + 5 figures
├── colab_setup.ipynb  # Self-contained Colab notebook
└── requirements.txt
```

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Dataset

Download from the [official ICBHI website](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge):
- `ICBHI_final_database/` → place inside `data/`
- `ICBHI_challenge_train_test.txt` → place inside `data/`

### 3. Preprocess (once)

```bash
python preprocess.py \
    --data_dir  ./data/ICBHI_final_database \
    --split_file ./data/ICBHI_challenge_train_test.txt \
    --output    ./icbhi_ast_16k_8s_spectrograms.npz
```

This runs `ASTFeatureExtractor` **once** over all ~6 900 cycles and saves the
resulting spectrogram tensors.  Takes ~5–10 min but eliminates the biggest
training bottleneck.

### 4. Train

```bash
# First run
python train.py --epochs 20 --batch_size 16 --lr 1e-5

# Resume after interruption (Colab disconnect, etc.)
python train.py --epochs 20 --batch_size 16 --lr 1e-5 --resume
```

Training plots (`results/training_curves.png`) are updated after every epoch.

### 5. Evaluate

```bash
python evaluate.py \
    --model_path ./checkpoints/best_model.pth \
    --output_dir ./results
```

Produces:
- `confusion_matrix.png`    — paper Figure 2 style
- `tsne_embeddings.png`     — paper Figure 3 style  
- `per_class_metrics.png`   — precision / recall / F1 per class
- `per_device_metrics.png`  — breakdown by recording device
- `probability_matrix.png`  — row-normalised conditional prediction matrix

---

## Colab + VS Code workflow

Open `colab_setup.ipynb` in Colab.  It:
1. Mounts Google Drive (checkpoints persist across disconnects)
2. Clones your GitHub repo
3. Optionally opens a VS Code Remote Tunnel (edit code locally, GPU runs in Colab)
4. Symlinks `checkpoints/` and `results/` to Drive
5. Runs preprocessing / training / evaluation

---

## Key hyperparameters

| Argument | Default | Notes |
|---|---|---|
| `--batch_size` | 16 | Increase to 24 if VRAM allows |
| `--lr` | 1e-5 | AdamW base LR |
| `--rho` | 0.05 | SAM perturbation radius |
| `--looksam_k` | 5 | SAM full-pass every k steps; k=1 = vanilla SAM |
| `--scheduler_t0` | 5 | CosineAnnealingWarmRestarts first cycle length |
| `--num_workers` | 4 | Set to 2 on Colab |
| `--compile` | off | `--compile` flag to enable `torch.compile` |

---

## Target results (paper baseline to beat)

| Metric | Paper (AST + SAM) | Your target |
|---|---|---|
| Sensitivity (Se) | 68.31% | > 68.31% |
| Specificity (Sp) | 67.89% | — |
| ICBHI Score | 68.10% | > 68.10% |
