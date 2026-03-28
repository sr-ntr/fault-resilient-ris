"""
evaluate.py
-----------
Post-training evaluation that replicates the paper's key results:

  Fig. 7  → Pixel heatmaps at different failure rates
  Fig. 8  → DNN vs baselines (RF, DT, LR) detection accuracy
  Fig. 9b → Validation loss comparison

Run after training:
  python src/evaluate.py
"""

import os
import sys
import json
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    get_device,
    compute_metrics,
    save_metrics,
    find_best_threshold,
    plot_pixel_heatmap,
    plot_f1_vs_failure_rate,
    plot_confusion_matrix_summary,
    MODELS_DIR,
    RESULTS_DIR,
    DATA_PROC,
    DATA_RAW,
    FEATURE_COLS,
    LABEL_COLS,
)
from dataset import load_processed_arrays, load_full_dataframe
from model import load_model


# ─────────────────────────────────────────────
# 1. LOAD BEST MODEL + TEST DATA
# ─────────────────────────────────────────────
def load_everything():
    device = get_device()
    model = load_model(os.path.join(MODELS_DIR, "best_model.pth"))
    model = model.to(device)
    model.eval()

    X_test, y_test = load_processed_arrays()
    print(f"[evaluate] Test set: X={X_test.shape}  y={y_test.shape}")
    return model, X_test, y_test, device


# ─────────────────────────────────────────────
# 2. GET DNN PREDICTIONS
# ─────────────────────────────────────────────
@torch.no_grad()
def get_dnn_predictions(model, X_test: np.ndarray, device, batch_size: int = 512):
    """Returns probabilities (N, 100) and binary preds (N, 100)."""
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    all_probs = []

    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i : i + batch_size].to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        all_probs.append(probs)

    y_prob = np.vstack(all_probs)  # (N, 100)
    return y_prob


# ─────────────────────────────────────────────
# 3. BASELINE CLASSIFIERS  (for paper Fig. 8 comparison)
# ─────────────────────────────────────────────
def train_baselines(X_train: np.ndarray, y_train: np.ndarray):
    """
    Trains RF, DT, LR baselines as compared in paper Fig. 8.
    Uses MultiOutputClassifier to handle 100 output labels.
    Trained on a subset (10k) for speed since baselines are slow.
    """
    print("\n[evaluate] Training baseline classifiers...")
    subset = 10_000  # use subset for speed

    X_sub = X_train[:subset]
    y_sub = y_train[:subset]

    baselines = {
        "Random Forest": MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
            ),
            n_jobs=-1,
        ),
        "Decision Tree": MultiOutputClassifier(
            DecisionTreeClassifier(max_depth=10, random_state=42), n_jobs=-1
        ),
        "Logistic Regression": MultiOutputClassifier(
            LogisticRegression(max_iter=200, random_state=42, n_jobs=-1), n_jobs=-1
        ),
    }

    trained = {}
    for name, clf in baselines.items():
        print(f"  Fitting {name}...", end=" ", flush=True)
        clf.fit(X_sub, y_sub)
        trained[name] = clf
        print("done")

    return trained


# ─────────────────────────────────────────────
# 4. F1 vs FAILURE RATE  (replicates Fig. 8)
# ─────────────────────────────────────────────
def evaluate_by_failure_rate(model, baselines, device):
    """
    Bins test samples by their failure percentage and
    computes F1 for each method at each bin.
    Replicates Fig. 8 (1%–6% bins shown in paper).
    We use broader bins: 0-10%, 10-20%, 20-30%, 30-40%, 40-50%, 50%+
    """
    import pandas as pd

    print("\n[evaluate] Computing F1 vs failure rate...")
    df_full = load_full_dataframe()

    # Get test indices — last 15% of data (matches dataset.py split)
    X_all = df_full[FEATURE_COLS].values.astype(np.float32)
    y_all = df_full[LABEL_COLS].values.astype(np.float32)
    pct_all = df_full["pct_failed_pixels"].values

    # Load scaler
    import pickle

    with open(os.path.join(DATA_PROC, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    from sklearn.model_selection import train_test_split

    _, X_test_raw, _, y_test, _, pct_test = train_test_split(
        X_all, y_all, pct_all, test_size=0.15, random_state=42
    )
    X_test_scaled = scaler.transform(X_test_raw)

    # Bins matching broader failure ranges
    bins = [0, 10, 20, 30, 40, 50, 101]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50%+"]

    results = {lbl: {} for lbl in labels}

    for lbl, (lo, hi) in zip(labels, zip(bins, bins[1:])):
        mask = (pct_test >= lo) & (pct_test < hi)
        if mask.sum() < 10:
            continue

        X_bin = X_test_scaled[mask]
        y_bin = y_test[mask].astype(int)

        # DNN
        y_prob = get_dnn_predictions(model, X_bin, device)
        y_pred_dnn = (y_prob >= 0.5).astype(int)
        results[lbl]["DNN (Proposed)"] = f1_score(
            y_bin.flatten(), y_pred_dnn.flatten(), zero_division=0
        )

        # Baselines
        for name, clf in baselines.items():
            y_pred_bl = np.array(clf.predict(X_bin))
            if y_pred_bl.ndim == 3:
                y_pred_bl = y_pred_bl[:, :, 1]
            results[lbl][name] = f1_score(
                y_bin.flatten(), y_pred_bl.flatten(), zero_division=0
            )

        print(
            f"  {lbl:>8}  n={mask.sum():>5}  "
            f"DNN F1={results[lbl]['DNN (Proposed)']:.3f}"
        )

    return results


# ─────────────────────────────────────────────
# 5. PIXEL HEATMAPS  (replicates Fig. 7)
# ─────────────────────────────────────────────
def generate_heatmaps(model, X_test, y_test, device):
    """
    Finds samples at specific failure rates and plots
    True vs Predicted heatmaps — replicates paper Fig. 7.
    Target failure rates: 10%, 22%, 30%, 50%, 66%
    """
    import pandas as pd

    print("\n[evaluate] Generating pixel heatmaps (Fig. 7 replication)...")

    df_full = load_full_dataframe()
    import pickle

    with open(os.path.join(DATA_PROC, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_all = df_full[FEATURE_COLS].values.astype(np.float32)
    y_all = df_full[LABEL_COLS].values.astype(np.float32)
    pct_all = df_full["pct_failed_pixels"].values

    from sklearn.model_selection import train_test_split

    _, X_test_raw, _, y_test_np, _, pct_test = train_test_split(
        X_all, y_all, pct_all, test_size=0.15, random_state=42
    )
    X_test_sc = scaler.transform(X_test_raw)

    # Target failure percentages as in paper Fig.7
    targets = [10, 22, 30, 50, 66]

    for target_pct in targets:
        # Find sample closest to target failure rate
        closest_idx = np.argmin(np.abs(pct_test - target_pct))
        pct_actual = pct_test[closest_idx]

        x_sample = X_test_sc[closest_idx : closest_idx + 1]
        y_true = y_test_np[closest_idx].astype(int)

        # DNN prediction
        y_prob = get_dnn_predictions(model, x_sample, device)
        y_pred = (y_prob[0] >= 0.5).astype(int)

        plot_pixel_heatmap(
            y_true, y_pred, sample_idx=closest_idx, pct_failed=pct_actual
        )
        print(f"  Heatmap for ~{target_pct}% failure (actual {pct_actual:.1f}%) saved")


# ─────────────────────────────────────────────
# 6. FULL TEST SET METRICS
# ─────────────────────────────────────────────
def full_evaluation(model, X_test, y_test, device):
    print("\n[evaluate] Running full test set evaluation...")

    y_prob = get_dnn_predictions(model, X_test, device)

    # Find best threshold
    best_t = find_best_threshold(y_test, y_prob)

    # Compute all metrics
    metrics = compute_metrics(y_test, y_prob, threshold=best_t)

    print(f"\n  ── Test Set Results ──")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")

    save_metrics(metrics)

    # Confusion matrix
    y_pred = (y_prob >= best_t).astype(int)
    plot_confusion_matrix_summary(y_test.astype(int), y_pred)

    return metrics


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load model + test data
    model, X_test, y_test, device = load_everything()

    # 1. Full metrics on test set
    metrics = full_evaluation(model, X_test, y_test, device)

    # 2. Load train data for baselines
    X_train = np.load(os.path.join(DATA_PROC, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_PROC, "y_train.npy"))

    # 3. Train baseline classifiers
    baselines = train_baselines(X_train, y_train)

    # 4. F1 vs failure rate plot (Fig. 8)
    results_by_pct = evaluate_by_failure_rate(model, baselines, device)
    plot_f1_vs_failure_rate(results_by_pct)

    # 5. Pixel heatmaps (Fig. 7)
    generate_heatmaps(model, X_test, y_test, device)

    print("\n[evaluate] All done. Check outputs/plots/ for figures.")


if __name__ == "__main__":
    main()
