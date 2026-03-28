"""
dataset.py
----------
Loads ris_dataset.csv, preprocesses it, and exposes PyTorch
Dataset objects for train / val / test splits.

Paper alignment:
  - Features  : 45 columns extracted from received BS signal
  - Labels    : 100 binary columns (one per RIS pixel)
  - Normalisation: StandardScaler on features (Algorithm 1, Step 4)
  - Split     : 70% train / 15% val / 15% test
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Import shared constants
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURE_COLS, LABEL_COLS, META_COLS, DATA_RAW, DATA_PROC


# ─────────────────────────────────────────────
# PYTORCH DATASET CLASS
# ─────────────────────────────────────────────
class RISDataset(Dataset):
    """
    PyTorch Dataset for RIS pixel failure detection.

    Each item:
        X : FloatTensor of shape (45,)  — normalised BS features
        y : FloatTensor of shape (100,) — binary pixel failure labels
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────────
def prepare_data(
    csv_filename: str = "ris_dataset.csv",
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 256,
    random_state: int = 42,
    force_reprocess: bool = False,
):
    """
    Full preprocessing pipeline:
      1. Load CSV
      2. Extract features (X) and labels (y)
      3. Train/val/test split
      4. Fit StandardScaler on train set only
      5. Save processed tensors to data/processed/
      6. Return DataLoaders

    Args:
        csv_filename    : name of CSV file inside data/raw/
        val_size        : fraction for validation set
        test_size       : fraction for test set
        batch_size      : DataLoader batch size
        random_state    : reproducibility seed
        force_reprocess : if True, reprocess even if .pt files exist

    Returns:
        train_loader, val_loader, test_loader, scaler, pos_weight
    """

    # ── Check if processed files already exist ──
    proc_files = [
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "scaler.pkl",
    ]
    all_exist = all(os.path.exists(os.path.join(DATA_PROC, f)) for f in proc_files)

    if all_exist and not force_reprocess:
        print("[dataset] Loading preprocessed data from disk...")
        X_train = np.load(os.path.join(DATA_PROC, "X_train.npy"))
        X_val = np.load(os.path.join(DATA_PROC, "X_val.npy"))
        X_test = np.load(os.path.join(DATA_PROC, "X_test.npy"))
        y_train = np.load(os.path.join(DATA_PROC, "y_train.npy"))
        y_val = np.load(os.path.join(DATA_PROC, "y_val.npy"))
        y_test = np.load(os.path.join(DATA_PROC, "y_test.npy"))
        with open(os.path.join(DATA_PROC, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

    else:
        # ── Step 1: Load CSV ──
        csv_path = os.path.join(DATA_RAW, csv_filename)
        print(f"[dataset] Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"[dataset] Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

        # ── Step 2: Extract X and y ──
        X = df[FEATURE_COLS].values.astype(np.float32)  # (50000, 45)
        y = df[LABEL_COLS].values.astype(np.float32)  # (50000, 100)
        print(f"[dataset] Features X: {X.shape}  Labels y: {y.shape}")

        # ── Step 3: Train / val / test split ──
        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        # Then split remaining into train and val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, random_state=random_state
        )
        print(
            f"[dataset] Train: {len(X_train):,} | "
            f"Val: {len(X_val):,} | Test: {len(X_test):,}"
        )

        # ── Step 4: Normalise features ──
        # Fit ONLY on train — prevents data leakage into val/test
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        print("[dataset] Features normalised (StandardScaler fit on train)")

        # ── Step 5: Save to disk ──
        np.save(os.path.join(DATA_PROC, "X_train.npy"), X_train)
        np.save(os.path.join(DATA_PROC, "X_val.npy"), X_val)
        np.save(os.path.join(DATA_PROC, "X_test.npy"), X_test)
        np.save(os.path.join(DATA_PROC, "y_train.npy"), y_train)
        np.save(os.path.join(DATA_PROC, "y_val.npy"), y_val)
        np.save(os.path.join(DATA_PROC, "y_test.npy"), y_test)
        with open(os.path.join(DATA_PROC, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        print(f"[dataset] Processed data saved to {DATA_PROC}/")

    # ── pos_weight for class imbalance in BCEWithLogitsLoss ──
    # pos_weight = n_negative / n_positive  (per pixel, averaged)
    # Since ~74.5% healthy and ~25.5% failed:
    # pos_weight ≈ 0.745 / 0.255 ≈ 2.92
    n_pos = y_train.sum()
    n_neg = y_train.size - n_pos
    pos_weight = torch.tensor(n_neg / (n_pos + 1e-8), dtype=torch.float32)

    print(
        f"[dataset] pos_weight = {pos_weight.item():.3f}  "
        f"(handles {y_train.mean() * 100:.1f}% failure rate imbalance)"
    )

    # ── Step 6: Build DataLoaders ──
    train_ds = RISDataset(X_train, y_train)
    val_ds = RISDataset(X_val, y_val)
    test_ds = RISDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"[dataset] DataLoaders ready — batch size: {batch_size}")
    return train_loader, val_loader, test_loader, scaler, pos_weight


# ─────────────────────────────────────────────
# HELPER: load raw arrays (for evaluation scripts)
# ─────────────────────────────────────────────
def load_processed_arrays():
    """Returns raw numpy arrays for evaluation without DataLoaders."""
    X_test = np.load(os.path.join(DATA_PROC, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_PROC, "y_test.npy"))
    return X_test, y_test


def load_full_dataframe():
    """Returns the full CSV as a DataFrame (for analysis notebooks)."""
    csv_path = os.path.join(DATA_RAW, "ris_dataset.csv")
    return pd.read_csv(csv_path)
