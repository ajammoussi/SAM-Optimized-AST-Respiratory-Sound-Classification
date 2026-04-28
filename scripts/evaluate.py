"""
evaluate.py
===========
Comprehensive evaluation of a trained ICBHI AST model, producing:

  1. Confusion Matrix  (paper Figure 2 style, with ICBHI metrics footer)
  2. t-SNE Embedding   (paper Figure 3 style, 4-class colouring)
  3. Per-class Precision / Recall / F1 table (console + saved as PNG)
  4. Per-Device breakdown  (which recording device benefits / suffers most)
  5. Misclassification heatmap — conditional probability matrix

All figures are saved to --output_dir (default ./results).

Usage:
    python scripts/evaluate.py \\
            --data_path  ./icbhi_ast_16k_8s_spectrograms.npz \\
            --model_path ./checkpoints/best_model.pth \\
            --output_dir ./results
"""

import os
import argparse
import gc
import multiprocessing as mp
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataset import ASTDataset
from src.model import CustomAST


# ======================================================================
# Constants
# ======================================================================

CLASSES      = ["Normal", "Crackle", "Wheeze", "Both"]
DEVICE_NAMES = {0: "AKGC417L", 1: "LittC2SE", 2: "Litt3200", 3: "Meditron"}

# Colour palette matching the paper's style (muted, publication-friendly)
CLASS_COLORS = {
    0: "#2ecc71",   # Normal  — green
    1: "#e74c3c",   # Crackle — red
    2: "#f39c12",   # Wheeze  — amber
    3: "#9b59b6",   # Both    — purple
}


# ======================================================================
# ICBHI binary metrics
# ======================================================================

def compute_icbhi_metrics(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    se_num   = np.sum(cm[1:, 1:])
    se_denom = np.sum(cm[1:, :])
    se = se_num / se_denom if se_denom > 0 else 0.0

    sp_num   = cm[0, 0]
    sp_denom = np.sum(cm[0, :])
    sp = sp_num / sp_denom if sp_denom > 0 else 0.0

    score = (se + sp) / 2.0
    return se, sp, score, cm


# ======================================================================
# Figure 1 — Confusion Matrix  (paper style)
# ======================================================================

def plot_confusion_matrix(cm, se, sp, score, output_dir):
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"label": "Number of Samples"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold", pad=16)

    metrics_text = (
        f"Sensitivity (Se): {se*100:.2f}%  |  "
        f"Specificity (Sp): {sp*100:.2f}%  |  "
        f"Score: {score*100:.2f}%"
    )
    fig.text(
        0.5, 0.01, metrics_text,
        ha="center", fontsize=11, fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black",
                  boxstyle="round,pad=0.4", alpha=0.85),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" [OK] Confusion matrix saved to {path}")


# ======================================================================
# Figure 2 — t-SNE  (paper style)
# ======================================================================

def plot_tsne(embeddings, labels, output_dir, perplexity=40, n_iter=1200):
    print("   Running t-SNE … (this may take 1-3 min)")
    tsne  = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                 random_state=42, init="pca", learning_rate="auto")
    proj  = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = labels == cls_idx
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=CLASS_COLORS[cls_idx],
            label=cls_name,
            s=18, alpha=0.75, edgecolors="none",
        )
    ax.legend(title="True Classes", fontsize=11, title_fontsize=11,
              markerscale=2)
    ax.set_title("t-SNE Visualization of Learned Embeddings",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "tsne_embeddings.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" [OK] t-SNE plot saved to {path}")


# ======================================================================
# Figure 3 — Per-class Precision / Recall / F1
# ======================================================================

def plot_per_class_metrics(all_labels, all_preds, output_dir):
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3], zero_division=0
    )
    x     = np.arange(len(CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_p = ax.bar(x - width, prec * 100, width, label="Precision", color="#3498db", alpha=0.85)
    bars_r = ax.bar(x,         rec  * 100, width, label="Recall",    color="#e67e22", alpha=0.85)
    bars_f = ax.bar(x + width, f1   * 100, width, label="F1-Score",  color="#2ecc71", alpha=0.85)

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Per-Class Precision / Recall / F1",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add support counts below x-axis labels
    for i, s in enumerate(sup):
        ax.text(i, -8, f"n={s}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_metrics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Per-class metrics saved to {path}")


# ======================================================================
# Figure 4 — Per-device breakdown
# ======================================================================

def plot_per_device_metrics(all_labels, all_preds, all_devices, output_dir):
    device_ids = sorted(set(all_devices))
    dev_scores = {"Se": [], "Sp": [], "Score": [], "Name": []}

    for did in device_ids:
        mask = np.array(all_devices) == did
        if mask.sum() == 0:
            continue
        lbl = np.array(all_labels)[mask]
        prd = np.array(all_preds)[mask]
        se, sp, sc, _ = compute_icbhi_metrics(lbl, prd)
        dev_scores["Se"].append(se * 100)
        dev_scores["Sp"].append(sp * 100)
        dev_scores["Score"].append(sc * 100)
        dev_scores["Name"].append(DEVICE_NAMES.get(did, f"Dev {did}"))

    if not dev_scores["Name"]:
        return

    x     = np.arange(len(dev_scores["Name"]))
    width = 0.28

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, dev_scores["Se"],    width, label="Sensitivity", color="#3498db", alpha=0.85)
    ax.bar(x,         dev_scores["Sp"],    width, label="Specificity", color="#e67e22", alpha=0.85)
    ax.bar(x + width, dev_scores["Score"], width, label="Score",       color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(dev_scores["Name"], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Per-Device ICBHI Metrics", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "per_device_metrics.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Per-device metrics saved to {path}")


# ======================================================================
# Figure 5 — Normalised conditional probability matrix
# ======================================================================

def plot_probability_matrix(cm, output_dir):
    """Row-normalised confusion matrix as a heatmap (shows error patterns)."""
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=CLASSES, yticklabels=CLASSES,
        vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": "P(predicted | true)"},
    )
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=11, fontweight="bold")
    ax.set_title("Normalised Prediction Probability Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "probability_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   [OK] Probability matrix saved to {path}")


# ======================================================================
# Main evaluation routine
# ======================================================================

def evaluate(args):
    gc.collect()
    torch.cuda.empty_cache()

    use_amp = torch.cuda.is_available()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}  |  AMP: {use_amp}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    print(f"\n Loading data: {args.data_path}")
    data = np.load(args.data_path)
    X_test  = torch.as_tensor(data["X_test"])
    y_test  = torch.as_tensor(data["y_test"])
    d_test  = torch.as_tensor(data["device_test"])

    if args.num_workers > 0:
        X_test = X_test.contiguous().share_memory_()
        y_test = y_test.contiguous().share_memory_()
        d_test = d_test.contiguous().share_memory_()

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context=mp.get_context("spawn") if args.num_workers > 0 else None,
    )

    test_loader = DataLoader(
        ASTDataset(X_test, y_test, d_test, train=False),
        **loader_kwargs,
    )

    # ---- Model ----
    print(f" Loading model: {args.model_path}")
    model = CustomAST(num_classes=4).to(device)

    ckpt = torch.load(args.model_path, map_location="cpu")
    # Support both a raw state_dict and a full checkpoint dict
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print("[OK] Model loaded successfully.")

    # ---- Inference + embedding extraction ----
    print("\n Running inference …")
    all_preds, all_labels, all_devices = [], [], []
    all_embeddings = []   # for t-SNE

    # Hook to capture mean-pooled embeddings before the classifier head
    embeddings_buffer = []

    def _hook(module, input, output):
        # output is last_hidden_state: (B, seq_len, 768) — we mean-pool here
        embeddings_buffer.append(output.last_hidden_state.mean(dim=1).detach().cpu())

    hook_handle = model.ast.register_forward_hook(_hook)

    with torch.no_grad():
        for inputs, labels, devices in test_loader:
            inputs = inputs.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(inputs)
            else:
                logits = model(inputs)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_devices.extend(devices.numpy() if hasattr(devices, "numpy")
                                else list(devices))

    hook_handle.remove()
    all_embeddings = torch.cat(embeddings_buffer, dim=0).numpy()

    all_labels  = np.array(all_labels)
    all_preds   = np.array(all_preds)
    all_devices = np.array(all_devices)

    # ---- Compute metrics ----
    se, sp, score, cm = compute_icbhi_metrics(all_labels, all_preds)

    print("\n" + "=" * 55)
    print("  ICBHI 2017 Evaluation Results")
    print("=" * 55)
    print(f"  Sensitivity (Se) : {se*100:.2f}%")
    print(f"  Specificity (Sp) : {sp*100:.2f}%")
    print(f"  ICBHI Score      : {score*100:.2f}%")
    print("=" * 55)
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=CLASSES, zero_division=0))

    # ---- Figures ----
    print("\n Generating figures …")
    plot_confusion_matrix(cm, se, sp, score, args.output_dir)
    plot_probability_matrix(cm, args.output_dir)
    plot_per_class_metrics(all_labels, all_preds, args.output_dir)
    plot_per_device_metrics(all_labels, all_preds, all_devices, args.output_dir)
    plot_tsne(all_embeddings, all_labels, args.output_dir,
              perplexity=args.tsne_perplexity)

    print(f"\n[OK] All figures saved to: {args.output_dir}/")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of ICBHI AST model"
    )
    parser.add_argument("--data_path",     type=str,   default="./data/icbhi_ast_16k_8s_spectrograms.npz")
    parser.add_argument("--model_path",      type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--output_dir",      type=str, default="./results")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--num_workers",     type=int, default=2)
    parser.add_argument("--tsne_perplexity", type=int, default=40)
    args = parser.parse_args()
    evaluate(args)
