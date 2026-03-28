"""
train.py
--------
Training loop for RISFaultDetector.

Paper settings (Section V-A):
  Optimizer  : Adam
  lr         : 0.1
  Epochs     : 500
  Loss       : Binary Cross Entropy
  Monitor    : F1-score
  Batch size : 256

Run this file directly:
  python src/train.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_device, MODELS_DIR, RESULTS_DIR, plot_training_curves
from dataset import prepare_data
from model import RISFaultDetector, print_model_summary, save_model


# ─────────────────────────────────────────────
# HYPERPARAMETERS  (paper Section V-A)
# ─────────────────────────────────────────────
CONFIG = {
    "lr": 0.1,
    "epochs": 500,
    "batch_size": 256,
    "dropout": 0.0,  # paper doesn't mention dropout
    "threshold": 0.5,  # adjusted during eval via AUC
    "random_state": 42,
    "n_features": 45,
    "n_pixels": 100,
}


# ─────────────────────────────────────────────
# ONE EPOCH OF TRAINING
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)  # (batch, 100) raw logits
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

        # Collect predictions for F1
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs >= CONFIG["threshold"]).astype(int)
        all_preds.append(preds)
        all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)

    return avg_loss, f1


# ─────────────────────────────────────────────
# ONE EPOCH OF VALIDATION
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= CONFIG["threshold"]).astype(int)
        all_preds.append(preds)
        all_labels.append(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    f1 = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)

    return avg_loss, f1


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def train(config: dict = CONFIG):

    device = get_device()

    # ── Data ──
    train_loader, val_loader, test_loader, scaler, pos_weight = prepare_data(
        batch_size=config["batch_size"],
        random_state=config["random_state"],
    )

    # ── Model ──
    model = RISFaultDetector(
        n_features=config["n_features"],
        n_pixels=config["n_pixels"],
        dropout=config["dropout"],
    ).to(device)
    print_model_summary(model)

    # ── Loss: BCEWithLogitsLoss with pos_weight for class imbalance ──
    # pos_weight tells the loss to penalise missed failures more heavily
    # This is critical since 74.5% of pixels are healthy
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ── Optimizer: Adam, lr=0.1  (paper Section V-A) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ── LR Scheduler: reduce on plateau to stabilise training ──
    # Paper uses fixed lr=0.1 but in practice this helps convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, verbose=True
    )

    # ── Training loop ──
    best_val_f1 = 0.0
    best_epoch = 0
    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    print(f"\n{'=' * 55}")
    print(f"  Training RISFaultDetector for {config['epochs']} epochs")
    print(f"{'=' * 55}")

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # Scheduler step on val F1
        scheduler.step(val_f1)

        # ── Save best model ──
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_model(model, os.path.join(MODELS_DIR, "best_model.pth"))

        # ── Logging every 10 epochs ──
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4}/{config['epochs']}  |  "
                f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}"
                + (" ← best" if epoch == best_epoch else "")
            )

    print(f"\n  Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Save training history ──
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1s": train_f1s,
        "val_f1s": val_f1s,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "config": config,
    }
    hist_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved → {hist_path}")

    # ── Plot training curves ──
    plot_training_curves(train_losses, val_losses, train_f1s, val_f1s)

    return model, history, test_loader


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model, history, test_loader = train()
    print("\n[train] Done. Run evaluate.py for test metrics and plots.")
