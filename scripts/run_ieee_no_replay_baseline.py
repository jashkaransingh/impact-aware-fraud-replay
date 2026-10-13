"""Run one fixed, development-only IEEE-CIS no-replay baseline.

Run: .venv/bin/python scripts/run_ieee_no_replay_baseline.py
Reads only prepared development-window matrices and metadata.
"""

import json
import math
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/ieee-preparation/windows"
OUTPUT = ROOT / "results/ieee-no-replay-baseline-501"
SEED = 501
BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 0.001
REVIEW_FRACTION = 0.05


class Network(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, features):
        return self.layers(features).squeeze(1)


def load_subset(window, subset, width):
    stem = f"window-{window:02d}-{subset}"
    matrix = sparse.load_npz(INPUT / f"{stem}-features.npz").tocsr()
    metadata = pd.read_csv(INPUT / f"{stem}-metadata.csv")
    if matrix.shape != (len(metadata), width) or matrix.dtype != np.float32:
        raise ValueError(f"Feature shape or dtype invalid for {stem}")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"Nonfinite features in {stem}")
    if not set(metadata["isFraud"].unique()).issubset({0, 1}):
        raise ValueError(f"Invalid labels in {stem}")
    return matrix, metadata


def dense_batch(matrix, indices):
    return torch.from_numpy(np.asarray(matrix[indices].toarray(), dtype=np.float32))


def smoke_check(matrix, labels, width, positive_weight):
    """One optimizer step on window-1 training only; model is discarded."""
    torch.manual_seed(SEED)
    model = Network(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    indices = np.arange(min(BATCH_SIZE, matrix.shape[0]))
    x = dense_batch(matrix, indices)
    y = torch.from_numpy(labels[indices].astype(np.float32))
    if x.shape != (len(indices), width) or y.shape != (len(indices),):
        raise AssertionError("Smoke batch shape invalid")
    logits = model(x)
    if logits.shape != y.shape:
        raise AssertionError("Smoke output shape invalid")
    loss = loss_fn(logits, y)
    if not torch.isfinite(loss):
        raise AssertionError("Smoke loss nonfinite")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"batch_rows": len(indices), "input_shape": list(x.shape),
            "output_shape": list(logits.shape), "loss": float(loss.item()),
            "finite_loss": True}


def evaluate(model, matrix, metadata):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, matrix.shape[0], BATCH_SIZE):
            batch = dense_batch(matrix, np.arange(start, min(start + BATCH_SIZE, matrix.shape[0])))
            scores.append(torch.sigmoid(model(batch)).numpy())
    scores = np.concatenate(scores)
    labels = metadata["isFraud"].to_numpy(dtype=np.int8)
    amounts = metadata["TransactionAmt"].to_numpy(dtype=np.float64)
    ids = metadata["TransactionID"].to_numpy(dtype=np.int64)
    quota = math.ceil(REVIEW_FRACTION * len(labels))
    # Stable quota resolution: descending score, then ascending TransactionID.
    top = np.lexsort((ids, -scores))[:quota]
    fraud_total = int(labels.sum())
    fraud_amount_total = float(amounts[labels == 1].sum())
    fraud_at_quota = int(labels[top].sum())
    fraud_amount_at_quota = float(amounts[top][labels[top] == 1].sum())
    if fraud_total == 0 or fraud_amount_total <= 0 or not np.isfinite(amounts).all():
        raise ValueError("Held-out labels or amounts invalid for evaluation")
    return {"rows": len(labels), "fraud_count": fraud_total,
            "ap": float(average_precision_score(labels, scores)),
            "fraud_prevalence": fraud_total / len(labels),
            "review_quota_rows": quota,
            "fraud_recall_at_5pct": fraud_at_quota / fraud_total,
            "amount_weighted_fraud_recall_at_5pct": fraud_amount_at_quota / fraud_amount_total}


def plot_retention(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    for evaluation_window in range(1, 7):
        points = [(step["after_training_window"], metric["ap"])
                  for step in results for metric in step["held_out_evaluations"]
                  if metric["held_out_window"] == evaluation_window]
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                label=f"Held-out window {evaluation_window}")
    ax.set(xlabel="Training window completed", ylabel="Average precision",
           title="Development held-out AP retention (no replay)", xticks=range(1, 7))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "ap-retention.png", dpi=160)
    plt.close(fig)


def main():
    prepared = json.loads((INPUT / "verification.json").read_text())
    if not all((prepared["verification"]["development_rows_assigned_once"],
                prepared["verification"]["training_and_held_out_ids_disjoint"],
                prepared["verification"]["both_classes_in_each_subset"])):
        raise ValueError("Prepared window checks did not pass")
    width = prepared["preprocessing"]["feature_count"]
    first_matrix, first_metadata = load_subset(1, "training", width)
    first_labels = first_metadata["isFraud"].to_numpy(dtype=np.int8)
    positives = int(first_labels.sum())
    negatives = len(first_labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Window-1 training lacks both classes")
    positive_weight = negatives / positives

    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    smoke = smoke_check(first_matrix, first_labels, width, positive_weight)
    print(f"Smoke check passed: input {smoke['input_shape']}, loss {smoke['loss']:.6f}", flush=True)

    # Fresh initialization after smoke; one model and optimizer persist across windows.
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = Network(width).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    config = {
        "description": "Initial development settings; not a final protocol or tuned optimum",
        "source_preparation_sha256": prepared["source_sha256_from_preparation"],
        "seed": SEED, "device": "cpu", "input_width": width,
        "network": "Linear(input_width,32), ReLU, Linear(32,1); one output logit",
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "epochs_per_window": EPOCHS,
        "loss": "BCEWithLogitsLoss", "fixed_positive_weight": positive_weight,
        "positive_weight_source": "window 1 training labels: negatives / positives",
        "training_rule": "Current window training subset only; same model and optimizer carry forward; no replay or earlier-window retraining",
        "evaluation_rule": "After each window, evaluate held-out subsets from windows seen so far; no held-out fitting or training",
        "review_rule": "Highest-ranked ceil(5% of rows) by sigmoid score; ties by ascending TransactionID. Fraud recall and fraud-amount-weighted recall use all fraud in that held-out set as denominator. Fixed review quota, not recovered money.",
        "versions": {name: version(name) for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUTPUT / "smoke-check.json").write_text(json.dumps(smoke, indent=2) + "\n")
    held_out = {}
    results = []
    for window in range(1, 7):
        matrix, metadata = ((first_matrix, first_metadata) if window == 1
                            else load_subset(window, "training", width))
        labels = metadata["isFraud"].to_numpy(dtype=np.float32)
        model.train()
        epoch_losses = []
        for _ in range(EPOCHS):
            permutation = torch.randperm(len(labels), generator=generator).numpy()
            weighted_loss = 0.0
            for start in range(0, len(labels), BATCH_SIZE):
                indices = permutation[start:start + BATCH_SIZE]
                x = dense_batch(matrix, indices)
                y = torch.from_numpy(labels[indices])
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise AssertionError(f"Nonfinite training loss in window {window}")
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.item()) * len(indices)
            epoch_losses.append(weighted_loss / len(labels))
        del matrix, metadata
        held_out[window] = load_subset(window, "held_out", width)
        evaluations = []
        for past_window in range(1, window + 1):
            metric = evaluate(model, *held_out[past_window])
            evaluations.append({"held_out_window": past_window, **metric})
        torch.save({"after_training_window": window, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "seed": SEED,
                    "input_width": width}, OUTPUT / f"checkpoint-window-{window:02d}.pt")
        results.append({"after_training_window": window,
                        "training_rows": len(labels), "epoch_mean_losses": epoch_losses,
                        "held_out_evaluations": evaluations})
        print(f"Window {window}: trained {len(labels)} rows; evaluated {len(evaluations)} held-out set(s)", flush=True)
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    plot_retention(results)
    print(f"Saved baseline to {OUTPUT}")


if __name__ == "__main__":
    main()
