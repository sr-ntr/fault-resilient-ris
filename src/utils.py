"""
utils.py
--------
Shared helper functions used across the project.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
# PATHS  — edit ROOT if your structure differs
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")
OUTPUTS = os.path.join(ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS, "models")
PLOTS_DIR = os.path.join(OUTPUTS, "plots")
RESULTS_DIR = os.path.join(OUTPUTS, "results")

for d in [DATA_PROC, MODELS_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# COLUMN DEFINITIONS
# ─────────────────────────────────────────────
T = 10  # pilot sequence length
MW = 10  # RIS rows
ML = 10  # RIS cols
M = MW * ML  # total pixels = 100

FEATURE_COLS = (
    [f"y_real_{t}" for t in range(T)]
    + [f"y_imag_{t}" for t in range(T)]
    + [f"y_mag_{t}" for t in range(T)]
    + [f"y_phase_{t}" for t in range(T)]
    + ["direct_ch_real", "direct_ch_imag", "rx_power", "y_mag_mean", "y_mag_std"]
)  # 45 features

LABEL_COLS = [f"pixel_{r}_{c}" for r in range(MW) for c in range(ML)]
# 100 labels

META_COLS = [
    "ue_x",
    "ue_y",
    "ue_z",
    "p_fail_used",
    "n_failed_pixels",
    "pct_failed_pixels",
]


# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
def get_device():
    """Returns CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Using device: {device}")
    return device


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Computes pixel-level classification metrics.

    Args:
        y_true      : (N, 100) binary ground truth
        y_pred_prob : (N, 100) sigmoid probabilities from model
        threshold   : decision boundary (paper uses AUC-tuned ~0.5)

    Returns:
        dict with accuracy, precision, recall, f1, per-pixel f1
    """
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
        roc_auc_score,
    )

    y_pred = (y_pred_prob >= threshold).astype(int)

    # Flatten all pixels for global metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    y_prob_flat = y_pred_prob.flatten()

    metrics = {
        "accuracy": float(accuracy_score(y_true_flat, y_pred_flat)),
        "precision": float(precision_score(y_true_flat, y_pred_flat, zero_division=0)),
        "recall": float(recall_score(y_true_flat, y_pred_flat, zero_division=0)),
        "f1": float(f1_score(y_true_flat, y_pred_flat, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true_flat, y_prob_flat)),
        # Per-pixel F1: shape (100,)
        "per_pixel_f1": f1_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
    }
    return metrics


def save_metrics(metrics: dict, filename: str = "metrics.json"):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[utils] Metrics saved → {path}")


def find_best_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Sweep thresholds 0.1–0.9, pick the one with best macro F1.
    Paper mentions using AUC=92% to threshold predictions.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_pred_prob >= t).astype(int)
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print(f"[utils] Best threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return float(best_t)


# ─────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────
def plot_training_curves(train_losses, val_losses, train_f1s, val_f1s, save=True):
    """Replicates the loss-over-epochs plot from paper Fig.9b."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", linewidth=1.5)
    ax1.plot(val_losses, label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_f1s, label="Train F1", linewidth=1.5)
    ax2.plot(val_f1s, label="Val F1", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Training & Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[utils] Plot saved → {path}")
    plt.show()


def plot_pixel_heatmap(
    y_true_sample: np.ndarray,
    y_pred_sample: np.ndarray,
    sample_idx: int = 0,
    pct_failed: float = None,
    save=True,
):
    """
    Replicates paper Fig.7 — side-by-side True vs Predicted
    RIS pixel failure heatmap for a single sample.

    y_true_sample : (100,) binary array
    y_pred_sample : (100,) binary array
    """
    true_grid = y_true_sample.reshape(MW, ML)
    pred_grid = y_pred_sample.reshape(MW, ML)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    cmap = plt.cm.RdYlGn_r  # green=healthy, red=failed

    sns.heatmap(
        true_grid,
        ax=ax1,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "0=Healthy, 1=Failed"},
        annot=True,
        fmt="d",
    )
    title = "True RIS"
    if pct_failed is not None:
        title += f"  ({pct_failed:.0f}% failed)"
    ax1.set_title(title, fontsize=12, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    sns.heatmap(
        pred_grid,
        ax=ax2,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "0=Healthy, 1=Failed"},
        annot=True,
        fmt="d",
    )
    ax2.set_title("Predicted RIS", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")

    plt.suptitle(f"Sample #{sample_idx}", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f"pixel_heatmap_sample{sample_idx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[utils] Heatmap saved → {path}")
    plt.show()


def plot_f1_vs_failure_rate(results_by_pct: dict, save=True):
    """
    Replicates paper Fig.8 — F1 score vs percentage of failed pixels.
    results_by_pct : { pct_bin_label: {"DNN": f1, "RF": f1, ...} }
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    pcts = sorted(results_by_pct.keys())
    methods = list(next(iter(results_by_pct.values())).keys())
    markers = ["o", "s", "^", "D"]

    for i, method in enumerate(methods):
        f1s = [results_by_pct[p][method] for p in pcts]
        ax.plot(
            pcts, f1s, marker=markers[i % 4], label=method, linewidth=1.8, markersize=6
        )

    ax.set_xlabel("Percentage of Failed Pixels (%)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Pixel Failure Detection: F1 vs Failure Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "f1_vs_failure_rate.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[utils] Plot saved → {path}")
    plt.show()


def plot_confusion_matrix_summary(y_true: np.ndarray, y_pred: np.ndarray, save=True):
    """Aggregate confusion matrix across all 100 pixels."""
    from sklearn.metrics import confusion_matrix

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    cm = confusion_matrix(y_true_flat, y_pred_flat)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred Healthy", "Pred Failed"],
        yticklabels=["True Healthy", "True Failed"],
    )
    ax.set_title("Confusion Matrix (all pixels, all samples)")
    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[utils] Confusion matrix saved → {path}")
    plt.show()
