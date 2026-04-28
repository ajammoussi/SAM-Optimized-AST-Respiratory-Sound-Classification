"""
train.py
========
Optimized training loop for ICBHI AST + LookSAM.

Key improvements over the original paper code
----------------------------------------------
1. Pre-computed spectrograms  — DataLoader no longer calls ASTFeatureExtractor
   per sample per epoch (biggest bottleneck eliminated).
2. LookSAM (k=5)             — full SAM second-pass only every 5 steps; same
   geometry-aware generalization at ~1.2× first-order cost vs 2× for vanilla SAM.
3. AMP (Automatic Mixed Precision) — autocast + GradScaler runs attention/linear
   layers in FP16 on the RTX 4050's tensor cores: ~1.5–2× speedup, ~30% less VRAM.
4. DataLoader tuning         — num_workers, pin_memory, persistent_workers.
5. Larger batch size         — VRAM freed by AMP enables batch_size 16-24.
 6. torch.compile (optional)  — fuse kernels for extra ~10-20% on PyTorch >= 2.0.
7. Robust checkpointing      — saves every epoch to Google Drive (or local path)
   including model, optimizer, epoch index, best score and full history so that
   training can be resumed exactly after a Colab disconnect.
8. Correct best-model logic  — best_model.pth is only overwritten when the ICBHI
   Score (Se+Sp)/2 strictly improves; the resume checkpoint always reflects the
   latest epoch regardless of score.

Usage (local / Colab)
---------------------
  # First run (or after Colab disconnect):
  python scripts/train.py \\
      --data_path ./icbhi_ast_16k_8s_spectrograms.npz \\
      --checkpoint_dir ./checkpoints \\
      --epochs 20 --batch_size 16 --lr 1e-5

  # Resume after disconnect (auto-detected from latest_checkpoint.pt):
  python scripts/train.py \\
      --data_path ./icbhi_ast_16k_8s_spectrograms.npz \\
      --checkpoint_dir ./checkpoints \\
      --epochs 20 --batch_size 16 --lr 1e-5 --resume
"""

import os
import argparse
import multiprocessing as mp
import sys
import contextlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in scripts & Colab
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import ASTDataset
from src.model import CustomAST
from src.look_sam import LookSAM


def _autocast_ctx(enabled: bool, dtype: torch.dtype):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", enabled=True, dtype=dtype)


def _all_grads_finite(params) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _tensor_stats(x: torch.Tensor) -> str:
    finite = torch.isfinite(x)
    if not finite.any():
        return f"shape={tuple(x.shape)} dtype={x.dtype} (no finite values)"
    x_f = x[finite]
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"min={x_f.min().item():.4g} max={x_f.max().item():.4g} "
        f"mean={x_f.mean().item():.4g} std={x_f.std(unbiased=False).item():.4g}"
    )


# ======================================================================
# Metrics helpers
# ======================================================================

def compute_icbhi_metrics(all_labels, all_preds):
    """
    Official ICBHI binary evaluation:
      Sensitivity (Se) = correctly identified abnormal / all abnormal
      Specificity (Sp) = correctly identified normal   / all normal
      Score            = (Se + Sp) / 2

    Note: intra-adventitious misclassifications (e.g. Crackle predicted
    as Wheeze) do NOT count as False Negatives per the challenge protocol.
    """
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    # Abnormal rows/cols: indices 1, 2, 3
    se_num   = np.sum(cm[1:, 1:])   # abnormal predicted as any abnormal
    se_denom = np.sum(cm[1:, :])    # all true abnormal samples
    se = se_num / se_denom if se_denom > 0 else 0.0

    sp_num   = cm[0, 0]             # normal predicted as normal
    sp_denom = np.sum(cm[0, :])     # all true normal samples
    sp = sp_num / sp_denom if sp_denom > 0 else 0.0

    score = (se + sp) / 2.0
    return se, sp, score, cm


# ======================================================================
# Plotting helpers
# ======================================================================

def save_training_plots(history: dict, output_dir: str, epoch: int):
    """
    Save a 4-panel training dashboard (inspired by the uploaded reference plots):
      • Loss (Train vs Validation)
      • ICBHI Score over epochs
      • Sensitivity vs Specificity
      • Learning Rate schedule

    Saved to output_dir/training_curves.png  (overwritten each epoch).
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs_axis = list(range(1, epoch + 2))   # 1-indexed

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=1.01)

    # ---- Loss ----
    ax = axes[0, 0]
    ax.plot(epochs_axis, history["train_loss"], "b-o", label="Train", markersize=4)
    ax.plot(epochs_axis, history["val_loss"],   "o-", color="orange",
            label="Validation", markersize=4)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- ICBHI Score ----
    ax = axes[0, 1]
    ax.plot(epochs_axis, [s * 100 for s in history["score"]],
            "g-o", label="ICBHI Score", markersize=4)
    ax.axhline(68.10, color="red", linestyle="--", label="Paper SOTA (68.10%)", alpha=0.7)
    ax.set_title("ICBHI Score ((Se+Sp)/2)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Sensitivity & Specificity ----
    ax = axes[1, 0]
    ax.plot(epochs_axis, [s * 100 for s in history["sensitivity"]],
            "b-o", label="Sensitivity (Se)", markersize=4)
    ax.plot(epochs_axis, [s * 100 for s in history["specificity"]],
            "o-", color="orange", label="Specificity (Sp)", markersize=4)
    ax.axhline(68.31, color="blue",   linestyle="--", alpha=0.5, label="Paper Se 68.31%")
    ax.axhline(67.89, color="orange", linestyle="--", alpha=0.5, label="Paper Sp 67.89%")
    ax.set_title("Sensitivity vs Specificity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Learning Rate ----
    ax = axes[1, 1]
    ax.semilogy(epochs_axis, history["lr"], "r-o", label="LR", markersize=4)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Checkpoint helpers
# ======================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: LookSAM,
    scaler: GradScaler,
    best_score: float,
    history: dict,
):
    """Save a full resume checkpoint (model + optimizer + scaler + metadata)."""
    torch.save(
        {
            "epoch": epoch,                               # next epoch to run
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_score": best_score,
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: LookSAM, scaler: GradScaler, device):
    """
    Load a resume checkpoint.

    Returns
    -------
    start_epoch : int — the epoch to start from (0-based)
    best_score : float — best ICBHI Score seen so far
    history : dict — accumulated per-epoch metrics
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Move optimizer tensors to the correct device
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for state in optimizer.base_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Move LookSAM's v_cache to the correct device
    for p, v in optimizer._v_cache.items():
        if isinstance(v, torch.Tensor):
            optimizer._v_cache[p] = v.to(device)

    scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"]
    best_score = ckpt["best_score"]
    history = ckpt["history"]
    return start_epoch, best_score, history


# ======================================================================
# Evaluation pass
# ======================================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
):
    """Run the model on loader, return (avg_loss, se, sp, score, cm)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with _autocast_ctx(autocast_enabled, autocast_dtype):
                logits = model(inputs)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    se, sp, score, cm = compute_icbhi_metrics(all_labels, all_preds)
    return avg_loss, se, sp, score, cm


def _to_shared_cpu_tensor(array):
    tensor = torch.as_tensor(array)
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    if hasattr(tensor, "is_shared") and not tensor.is_shared():
        tensor = tensor.contiguous()
        tensor.share_memory_()
    return tensor


# ======================================================================
# Training loop
# ======================================================================

def train(args):
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")

    amp_mode = args.amp_dtype.lower()
    autocast_enabled = bool(has_cuda and amp_mode != "off")
    if amp_mode == "bf16":
        autocast_dtype = torch.bfloat16
    elif amp_mode in ("fp16", "off"):
        autocast_dtype = torch.float16
    else:
        raise ValueError(f"Unknown --amp_dtype '{args.amp_dtype}' (use: fp16, bf16, off)")

    # GradScaler is only required for fp16 autocast.
    scaler_enabled = bool(autocast_enabled and autocast_dtype == torch.float16)
    print(
        f"Device : {device}  |  autocast : {autocast_enabled} ({amp_mode})  |  GradScaler : {scaler_enabled}"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir,    exist_ok=True)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    print(f"\nLoading data from: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"{args.data_path} not found.  Run scripts/preprocess.py first."
        )

    data = np.load(args.data_path)
    X_train = _to_shared_cpu_tensor(data["X_train"])
    y_train = _to_shared_cpu_tensor(data["y_train"])
    d_train = _to_shared_cpu_tensor(data["device_train"])
    X_test  = _to_shared_cpu_tensor(data["X_test"])
    y_test  = _to_shared_cpu_tensor(data["y_test"])
    d_test  = _to_shared_cpu_tensor(data["device_test"])
    print(f"   Train: {X_train.shape}   Test: {X_test.shape}")

    # Weighted sampler to counteract class imbalance
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    counts = np.bincount(y_train_np, minlength=4)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[y_train_np]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y_train_np),
        replacement=True,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context=mp.get_context("spawn") if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        ASTDataset(X_train, y_train, d_train, train=True),
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, train=False),
        shuffle=False,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    print("\nBuilding model …")
    model = CustomAST(num_classes=4).to(device)

    # Optional: torch.compile for extra kernel fusion (PyTorch >= 2.0)
    if args.compile and hasattr(torch, "compile"):
        print("   torch.compile() enabled …")
        model = torch.compile(model, mode="reduce-overhead")

    # ------------------------------------------------------------------ #
    # Optimizer, loss, scaler
    # ------------------------------------------------------------------ #
    optimizer = LookSAM(
        model.parameters(),
        torch.optim.AdamW,
        lr=args.lr,
        rho=args.rho,
        k=args.looksam_k,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler("cuda", enabled=scaler_enabled)

    # Cosine annealing with warm restarts (improves convergence vs flat LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer.base_optimizer,
        T_0=args.scheduler_t0,
        T_mult=2,
        eta_min=1e-7,
    )

    # ------------------------------------------------------------------ #
    # Resume from checkpoint if requested
    # ------------------------------------------------------------------ #
    latest_ckpt = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    start_epoch = 0
    best_score  = 0.0
    history = {
        "train_loss":  [],
        "val_loss":    [],
        "sensitivity": [],
        "specificity": [],
        "score":       [],
        "lr":          [],
    }

    if args.resume and os.path.exists(latest_ckpt):
        print(f"\nResuming from {latest_ckpt} …")
        start_epoch, best_score, history = load_checkpoint(
            latest_ckpt, model, optimizer, scaler, device
        )
        # Fast-forward the scheduler to the right step
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
        print(f"   Resumed at epoch {start_epoch + 1}  |  Best score so far: {best_score:.4f}")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    print(f"\nTraining starts at epoch {start_epoch + 1} / {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        n_batches    = 0

        nonfinite_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        for inputs, labels, _ in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ================================================================
            # AMP + SAM correct pattern
            # ---------------------------------------------------------------
            # For SAM + AMP, we need to handle the two backward passes carefully.
            # The key insight is that we should only call scaler.update() once per
            # optimization step, after both SAM passes are complete.
            #
            # Pattern:
            #   1. autocast + scaled backward to first_step (perturbs weights)
            #   2. autocast + scaled backward to second_step (restores + updates)
            #   3. scaler.update() (once per full SAM step)
            # ================================================================

            # ---- Pass 1: gradient at current weights w ----
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(autocast_enabled, autocast_dtype):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss detected. "
                    f"inputs: {_tensor_stats(inputs)} | logits: {_tensor_stats(logits)}"
                )

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer.base_optimizer)
            else:
                loss.backward()

            # If gradients overflowed/are NaN, do NOT call SAM first_step.
            if not _all_grads_finite(model.parameters()):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                pbar.set_postfix({"loss": "nonfinite_grad(skip)"})
                if nonfinite_batches >= args.max_nonfinite_batches and autocast_enabled:
                    print(
                        f"\n[WARN] {nonfinite_batches} non-finite batches this epoch. "
                        "Consider '--amp_dtype bf16' or '--amp_dtype off'."
                    )
                continue

            optimizer.first_step(zero_grad=False)       # perturbs w to w + epsilon; keeps grads

            # ---- Pass 2: gradient at perturbed weights w + ε̂ ----
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(autocast_enabled, autocast_dtype):
                logits2 = model(inputs)
                loss2 = criterion(logits2, labels)

            if not torch.isfinite(loss2):
                optimizer.second_step(scaler=None, zero_grad=True, skip_update=True)
                nonfinite_batches += 1
                pbar.set_postfix({"loss": "nonfinite_loss2(skip)"})
                continue

            if scaler.is_enabled():
                scaler.scale(loss2).backward()
                optimizer.second_step(scaler=scaler, zero_grad=True)
                scaler.update()
            else:
                loss2.backward()
                if not _all_grads_finite(model.parameters()):
                    optimizer.second_step(scaler=None, zero_grad=True, skip_update=True)
                    nonfinite_batches += 1
                    pbar.set_postfix({"loss": "nonfinite_grad2(skip)"})
                    continue
                optimizer.second_step(scaler=None, zero_grad=True)

            scheduler.step()

            running_loss += loss.item()
            n_batches    += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ---- Validation ----
        avg_train_loss = running_loss / n_batches
        avg_val_loss, se, sp, score, cm = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

        current_lr = optimizer.base_optimizer.param_groups[0]["lr"]
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["sensitivity"].append(se)
        history["specificity"].append(sp)
        history["score"].append(score)
        history["lr"].append(current_lr)

        print(
            f"\nEpoch {epoch+1:>3d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Se: {se*100:.2f}% | "
            f"Sp: {sp*100:.2f}% | "
            f"Score: {score*100:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        # ---- Save best model (strict improvement on ICBHI Score) ----
        if score > best_score:
            best_score = score
            best_path  = os.path.join(args.checkpoint_dir, "best_model.pth")
            # Save only model weights for the best model (lighter file)
            torch.save(model.state_dict(), best_path)
            print(f"    New best model saved -> {best_path}  (Score: {best_score*100:.2f}%)")

        # ---- Save full resume checkpoint every epoch ----
        save_checkpoint(
            latest_ckpt,
            epoch=epoch + 1,          # next epoch to run on resume
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            best_score=best_score,
            history=history,
        )

        # ---- Update training plots ----
        save_training_plots(history, args.results_dir, epoch)

    print(f"\nTraining complete.  Best ICBHI Score: {best_score*100:.2f}%")
    print(f"   Best model    : {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"   Training plots: {os.path.join(args.results_dir, 'training_curves.png')}")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Train AST + LookSAM on ICBHI 2017")

    parser.add_argument("--data_path",     type=str,   default="./data/icbhi_ast_16k_8s_spectrograms.npz")
    parser.add_argument("--checkpoint_dir",type=str,   default="./checkpoints")
    parser.add_argument("--results_dir",   type=str,   default="./results")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch_size",    type=int,   default=4)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--rho",           type=float, default=0.05,
                        help="SAM neighbourhood radius")
    parser.add_argument("--looksam_k",     type=int,   default=5,
                        help="LookSAM update frequency (k=1 to vanilla SAM)")
    parser.add_argument("--num_workers",   type=int,   default=0,
                        help="DataLoader worker processes")
    parser.add_argument("--scheduler_t0", type=int,   default=5,
                        help="CosineAnnealingWarmRestarts T_0")
    parser.add_argument("--compile",       action="store_true",
                        help="Enable torch.compile() (PyTorch >= 2.0)")
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from latest_checkpoint.pt if it exists")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "off"],
        help="Autocast dtype on CUDA: fp16 (fast), bf16 (more stable), off (fp32)",
    )
    parser.add_argument(
        "--max_nonfinite_batches",
        type=int,
        default=10,
        help="Warn after this many non-finite batches in an epoch.",
    )

    args = parser.parse_args()
    train(args)
