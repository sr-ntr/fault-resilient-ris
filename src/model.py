"""
model.py
--------
DNN architecture exactly as specified in the paper (Section V-A):

  Input  → 45 neurons
  Dense  → 32  + ReLU
  Dense  → 64  + ReLU
  Dense  → 128 + ReLU
  Output → 100 + Sigmoid

This is a multi-label binary classifier:
  - 100 independent binary outputs (one per RIS pixel)
  - Each output = probability that pixel is FAILED
  - Loss: BCEWithLogitsLoss (numerically stable BCE)
  - Note: Sigmoid is NOT applied inside the model when using
    BCEWithLogitsLoss — it is applied at inference time.
"""

import torch
import torch.nn as nn


class RISFaultDetector(nn.Module):
    """
    Deep Neural Network for RIS pixel failure detection.

    Exactly matches paper Algorithm 1 / Section V-A architecture.
    Uses BCEWithLogitsLoss during training (no Sigmoid in forward).
    At inference, apply torch.sigmoid() to get probabilities.

    Args:
        n_features : input size  (default 45)
        n_pixels   : output size (default 100 = 10×10 RIS)
        dropout    : dropout rate for regularisation (0 = disabled)
    """

    def __init__(self, n_features: int = 45, n_pixels: int = 100, dropout: float = 0.0):
        super(RISFaultDetector, self).__init__()

        self.network = nn.Sequential(
            # ── Hidden Layer 1: 45 → 32 ──
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Hidden Layer 2: 32 → 64 ──
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Hidden Layer 3: 64 → 128 ──
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            # ── Output Layer: 128 → 100 ──
            # No Sigmoid here — BCEWithLogitsLoss handles it
            nn.Linear(128, n_pixels),
        )

        # Initialise weights (Xavier uniform — good for ReLU networks)
        self._init_weights()

    def _init_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input  : (batch, 45)
        Output : (batch, 100)  — raw logits, NOT probabilities
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns sigmoid probabilities for inference.
        Input  : (batch, 45)
        Output : (batch, 100) — probabilities in [0, 1]
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Returns binary predictions.
        Input  : (batch, 45)
        Output : (batch, 100) — 0 or 1
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()


# ─────────────────────────────────────────────
# MODEL SUMMARY UTILITY
# ─────────────────────────────────────────────
def print_model_summary(model: nn.Module):
    """Prints layer-by-layer parameter count."""
    print("\n" + "=" * 50)
    print("  RISFaultDetector — Architecture Summary")
    print("=" * 50)
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        print(f"  {name:<30} {str(list(param.shape)):<20} {n:,} params")
    print("-" * 50)
    print(f"  Total trainable parameters: {total_params:,}")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────
def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    print(f"[model] Saved → {path}")


def load_model(
    path: str, n_features: int = 45, n_pixels: int = 100, dropout: float = 0.0
) -> nn.Module:
    model = RISFaultDetector(n_features, n_pixels, dropout)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"[model] Loaded ← {path}")
    return model
