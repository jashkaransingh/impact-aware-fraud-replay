"""One fixed, development-only, class-balanced random-replay comparison.

Run: .venv/bin/python scripts/run_ieee_balanced_replay.py
Uses the baseline model and evaluation code, and reads prepared development sets only.
"""

import json
import math
import random
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from run_ieee_no_replay_baseline import (ROOT, INPUT, SEED, BATCH_SIZE, EPOCHS,
                                         LEARNING_RATE, Network, dense_batch,
                                         evaluate, load_subset)


BASELINE = ROOT / "results/ieee-no-replay-baseline-501"
OUTPUT = ROOT / "results/ieee-class-balanced-replay-501"
SLOTS_PER_CLASS = 128
RETRIEVE_PER_CLASS = 64


class StratifiedReservoir:
    """Independent algorithm-R reservoir for each class, up to 128 each."""

    def __init__(self, admission_rng, retrieval_rng):
        self.admission_rng = admission_rng
        self.retrieval_rng = retrieval_rng
        self.seen = {0: 0, 1: 0}
        self.slots = {0: [], 1: []}

    def admit_window(self, matrix, metadata, window, held_out_ids):
        for index, row in metadata.iterrows():
            label = int(row["isFraud"])
            transaction_id = int(row["TransactionID"])
            if label not in (0, 1) or transaction_id in held_out_ids:
                raise AssertionError("Only training IDs may enter replay memory")
            self.seen[label] += 1
            pool = self.slots[label]
            position = (len(pool) if len(pool) < SLOTS_PER_CLASS else
                        self.admission_rng.randrange(self.seen[label]))
            if position < SLOTS_PER_CLASS:
                example = {"transaction_id": transaction_id, "class": label,
                           "source_window": window,
                           "features": np.asarray(matrix.getrow(index).toarray().ravel(), dtype=np.float32)}
                if position == len(pool):
                    pool.append(example)
                else:
                    pool[position] = example
        self.check_limits()

    def retrieve(self, current_window, training_ids_by_window, held_out_ids):
        selected = []
        for label in (0, 1):
            pool = self.slots[label]
            selected.extend(self.retrieval_rng.sample(pool, min(RETRIEVE_PER_CLASS, len(pool))))
        self.retrieval_rng.shuffle(selected)
        ids = [item["transaction_id"] for item in selected]
        if len(ids) != len(set(ids)) or len(ids) > 2 * RETRIEVE_PER_CLASS:
            raise AssertionError("Retrieval is not unique or exceeds limit")
        for item in selected:
            if (item["source_window"] >= current_window
                    or item["transaction_id"] not in training_ids_by_window[item["source_window"]]
                    or item["transaction_id"] in held_out_ids):
                raise AssertionError("Replay ID is not from an earlier training window")
        return selected

    def check_limits(self):
        if any(len(self.slots[label]) > SLOTS_PER_CLASS for label in (0, 1)):
            raise AssertionError("Replay memory exceeds class slot cap")
        ids = [item["transaction_id"] for pool in self.slots.values() for item in pool]
        if len(ids) != len(set(ids)):
            raise AssertionError("Replay memory has repeated IDs")

    def snapshot(self, after_window):
        return {"after_training_window": after_window,
                "seen_by_class": {str(label): self.seen[label] for label in (0, 1)},
                "slots_by_class": {str(label): [{"transaction_id": item["transaction_id"],
                                                  "source_window": item["source_window"]}
                                                 for item in self.slots[label]] for label in (0, 1)}}


def load_held_out_ids(width):
    ids = set()
    for window in range(1, 7):
        _, metadata = load_subset(window, "held_out", width)
        new_ids = set(metadata["TransactionID"].astype(int))
        if ids.intersection(new_ids):
            raise AssertionError("Held-out IDs overlap across windows")
        ids.update(new_ids)
    return ids


def smoke_check(width, weight, first_matrix, first_metadata, held_out_ids):
    """Isolated one-batch current step and one-batch replay step; then discard."""
    torch.manual_seed(SEED)
    model = Network(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    labels = first_metadata["isFraud"].to_numpy(dtype=np.float32)
    indices = np.arange(BATCH_SIZE)
    x = dense_batch(first_matrix, indices)
    y = torch.from_numpy(labels[indices])
    if x.shape != (BATCH_SIZE, width) or model(x).shape != y.shape:
        raise AssertionError("Current smoke shapes invalid")
    loss = loss_fn(model(x), y)
    if not torch.isfinite(loss):
        raise AssertionError("Current smoke loss nonfinite")
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    memory = StratifiedReservoir(random.Random(SEED), random.Random(SEED))
    memory.admit_window(first_matrix, first_metadata, 1, held_out_ids)
    second_matrix, second_metadata = load_subset(2, "training", width)
    training_ids = {1: set(first_metadata["TransactionID"].astype(int))}
    replay = memory.retrieve(2, training_ids, held_out_ids)
    current_x = dense_batch(second_matrix, indices)
    current_y = torch.from_numpy(second_metadata["isFraud"].to_numpy(dtype=np.float32)[indices])
    replay_x = torch.from_numpy(np.stack([item["features"] for item in replay]))
    replay_y = torch.tensor([item["class"] for item in replay], dtype=torch.float32)
    combined_x = torch.cat([current_x, replay_x])
    combined_y = torch.cat([current_y, replay_y])
    if combined_x.shape != (BATCH_SIZE + len(replay), width):
        raise AssertionError("Replay smoke shapes invalid")
    replay_loss = loss_fn(model(combined_x), combined_y)
    if not torch.isfinite(replay_loss):
        raise AssertionError("Replay smoke loss nonfinite")
    optimizer.zero_grad(); replay_loss.backward(); optimizer.step()
    return {"current_batch_rows": BATCH_SIZE, "retrieved_rows": len(replay),
            "retrieved_class_counts": {str(label): sum(item["class"] == label for item in replay)
                                       for label in (0, 1)},
            "combined_shape": list(combined_x.shape),
            "current_loss": float(loss.item()), "replay_loss": float(replay_loss.item()),
            "finite_losses": True}


def comparison(replay_results, baseline_results):
    windows = []
    for window in range(1, 7):
        def series(results):
            return [next(metric for metric in step["held_out_evaluations"]
                         if metric["held_out_window"] == window)
                    for step in results[window - 1:]]
        replay = series(replay_results)
        baseline = series(baseline_results)
        windows.append({
            "held_out_window": window,
            "fraud_prevalence": replay[0]["fraud_prevalence"],
            "replay_initial_ap": replay[0]["ap"],
            "replay_final_ap": replay[-1]["ap"],
            "replay_best_ap": max(item["ap"] for item in replay),
            "replay_initial_to_final_ap_change": replay[-1]["ap"] - replay[0]["ap"],
            "replay_best_to_final_forgetting": max(item["ap"] for item in replay) - replay[-1]["ap"],
            "baseline_final_ap": baseline[-1]["ap"],
            "final_ap_replay_minus_baseline": replay[-1]["ap"] - baseline[-1]["ap"],
            "replay_final_amount_weighted_recall_at_5pct": replay[-1]["amount_weighted_fraud_recall_at_5pct"],
            "baseline_final_amount_weighted_recall_at_5pct": baseline[-1]["amount_weighted_fraud_recall_at_5pct"],
        })
    return {"per_held_out_window": windows,
            "historical_windows": [1, 2, 3, 4, 5],
            "mean_best_to_final_forgetting_replay_1_to_5": sum(
                item["replay_best_to_final_forgetting"] for item in windows[:5]) / 5,
            "mean_best_to_final_forgetting_baseline_1_to_5": sum(
                max(next(metric for metric in step["held_out_evaluations"]
                         if metric["held_out_window"] == window)["ap"]
                    for step in baseline_results[window - 1:])
                - next(metric for metric in baseline_results[-1]["held_out_evaluations"]
                       if metric["held_out_window"] == window)["ap"]
                for window in range(1, 6)) / 5,
            "mean_final_ap_replay_1_to_5": sum(item["replay_final_ap"] for item in windows[:5]) / 5,
            "mean_final_ap_baseline_1_to_5": sum(item["baseline_final_ap"] for item in windows[:5]) / 5,
            "window_6_ap_replay": windows[5]["replay_final_ap"],
            "window_6_ap_baseline": windows[5]["baseline_final_ap"]}


def plot_comparison(replay_results, baseline_results):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for window, ax in enumerate(axes.flat, 1):
        for results, label, color in ((baseline_results, "No replay", "#3578b5"),
                                      (replay_results, "Balanced replay", "#d56a28")):
            points = [(step["after_training_window"], metric["ap"])
                      for step in results for metric in step["held_out_evaluations"]
                      if metric["held_out_window"] == window]
            ax.plot([x for x, _ in points], [y for _, y in points], marker="o",
                    color=color, label=label)
        ax.set_title(f"Held-out window {window}")
        ax.set_xticks(range(window, 7))
        ax.grid(alpha=0.25)
        if window > 3:
            ax.set_xlabel("Training window completed")
        if window in (1, 4):
            ax.set_ylabel("Average precision")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Development AP retention: no replay vs balanced random replay")
    fig.tight_layout()
    fig.savefig(OUTPUT / "ap-retention-comparison.png", dpi=160)
    plt.close(fig)


def verify_saved_run():
    """Audit saved replay IDs and step counts without running training again."""
    replay_log = json.loads((OUTPUT / "replay-ids.json").read_text())
    results = json.loads((OUTPUT / "results.json").read_text())
    baseline = json.loads((BASELINE / "results.json").read_text())
    training_ids = {}
    training_labels = {}
    held_out_ids = set()
    for window in range(1, 7):
        training_rows = pd.read_csv(
            INPUT / f"window-{window:02d}-training-metadata.csv",
            usecols=["TransactionID", "isFraud"])
        training_ids[window] = set(training_rows["TransactionID"].astype(int))
        training_labels.update(zip(training_rows["TransactionID"].astype(int),
                                   training_rows["isFraud"].astype(int)))
        held_out_ids.update(pd.read_csv(
            INPUT / f"window-{window:02d}-held_out-metadata.csv",
            usecols=["TransactionID"])["TransactionID"].astype(int))
    if any(ids.intersection(held_out_ids) for ids in training_ids.values()):
        raise AssertionError("Training IDs overlap held-out IDs")
    retrievals = replay_log["retrievals"]
    snapshots = replay_log["memory_snapshots"]
    if len(retrievals) != 18 or len(snapshots) != 6 or len(results) != 6:
        raise AssertionError("Missing epoch or window records")
    maximum_retrieval = 0
    maximum_memory = 0
    for record in retrievals:
        window = record["training_window"]
        items = record["retrieved"]
        ids = [item["transaction_id"] for item in items]
        counts = {label: sum(item["class"] == label for item in items) for label in (0, 1)}
        prior_slots = ({"0": [], "1": []} if window == 1 else
                       snapshots[window - 2]["slots_by_class"])
        if (len(ids) != len(set(ids)) or len(ids) > 128 or max(counts.values()) > 64
                or any(counts[label] != min(64, len(prior_slots[str(label)])) for label in (0, 1))
                or record["current_rows_exposed"] != len(training_ids[window])
                or record["optimizer_steps"] != math.ceil(len(training_ids[window]) / BATCH_SIZE)):
            raise AssertionError("Retrieval, exposure, or optimizer-step limit failed")
        for item in items:
            if (item["source_window"] >= window
                    or item["transaction_id"] not in training_ids[item["source_window"]]
                    or item["class"] != training_labels[item["transaction_id"]]
                    or item["transaction_id"] in held_out_ids):
                raise AssertionError("Replay contains nonhistorical or held-out ID")
        maximum_retrieval = max(maximum_retrieval, len(ids))
    for snapshot in snapshots:
        window = snapshot["after_training_window"]
        pools = snapshot["slots_by_class"]
        ids = [item["transaction_id"] for pool in pools.values() for item in pool]
        if (any(len(pool) > 128 for pool in pools.values())
                or len(ids) != len(set(ids)) or len(ids) > 256):
            raise AssertionError("Memory slot or uniqueness limit failed")
        for label, pool in pools.items():
            for item in pool:
                if (item["source_window"] > window
                        or item["transaction_id"] not in training_ids[item["source_window"]]
                        or int(label) != training_labels[item["transaction_id"]]
                        or item["transaction_id"] in held_out_ids):
                    raise AssertionError("Memory contains nontraining or future ID")
        maximum_memory = max(maximum_memory, len(ids))
    for window, (result, base) in enumerate(zip(results, baseline), 1):
        logs = [entry for entry in retrievals if entry["training_window"] == window]
        if (result["training_rows"] != base["training_rows"]
                or len(result["epoch_mean_losses"]) != EPOCHS
                or result["epoch_replay_counts"] != [len(entry["retrieved"]) for entry in logs]):
            raise AssertionError("Current-data exposure or replay count mismatch")
    first_window_exact_match = (results[0]["epoch_mean_losses"] == baseline[0]["epoch_mean_losses"]
                                and results[0]["held_out_evaluations"] == baseline[0]["held_out_evaluations"])
    if not first_window_exact_match:
        raise AssertionError("No-replay first window differs from baseline")
    report = {"historical_training_ids_only": True, "held_out_ids_excluded": True,
              "retrieval_ids_unique_per_epoch": True, "memory_ids_unique": True,
              "memory_limit_256_and_128_per_class": True,
              "retrieval_limit_128_and_64_per_class": True,
              "retrieval_class_targets_match_available_memory": True,
              "current_rows_and_optimizer_steps_match_baseline": True,
              "window_1_exact_baseline_match": True,
              "maximum_memory_rows": maximum_memory,
              "maximum_retrieved_rows_per_epoch": maximum_retrieval,
              "retrieval_epoch_records": len(retrievals)}
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved replay verification without retraining", flush=True)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--verify-existing":
        verify_saved_run()
        return
    if len(sys.argv) != 1:
        raise SystemExit("Usage: run_ieee_balanced_replay.py [--verify-existing]")
    baseline_config = json.loads((BASELINE / "configuration.json").read_text())
    baseline_results = json.loads((BASELINE / "results.json").read_text())
    prepared = json.loads((INPUT / "verification.json").read_text())
    width = prepared["preprocessing"]["feature_count"]
    first_matrix, first_metadata = load_subset(1, "training", width)
    first_labels = first_metadata["isFraud"].to_numpy(dtype=np.int8)
    weight = (len(first_labels) - int(first_labels.sum())) / int(first_labels.sum())
    if (baseline_config["seed"] != SEED or baseline_config["input_width"] != width
            or baseline_config["batch_size"] != BATCH_SIZE
            or baseline_config["epochs_per_window"] != EPOCHS
            or baseline_config["learning_rate"] != LEARNING_RATE
            or not math.isclose(baseline_config["fixed_positive_weight"], weight)):
        raise ValueError("Requested settings differ from saved baseline")
    held_out_ids = load_held_out_ids(width)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    smoke = smoke_check(width, weight, first_matrix, first_metadata, held_out_ids)
    print(f"Replay smoke check passed: {smoke['combined_shape']}, loss {smoke['replay_loss']:.6f}", flush=True)

    # Reset after smoke. Current-order generator matches the baseline exactly;
    # admission and retrieval each own a separate independent random stream.
    torch.manual_seed(SEED)
    current_rng = torch.Generator().manual_seed(SEED)
    memory = StratifiedReservoir(random.Random(SEED), random.Random(SEED))
    model = Network(width).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float32))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = {
        "description": "One initial development-only class-balanced random-replay comparison; not tuned",
        "baseline_configuration": str((BASELINE / "configuration.json").relative_to(ROOT)),
        "prepared_windows": str((INPUT / "verification.json").relative_to(ROOT)),
        "source_sha256_from_preparation": prepared["source_sha256_from_preparation"],
        "model_optimizer_loss_current_order": "Same imported model, evaluation, seed, Adam, BCEWithLogitsLoss positive weight, current-data torch.randperm stream, learning rate, batch size, and epochs as baseline",
        "seed": SEED, "input_width": width, "positive_weight": weight,
        "device": "cpu", "batch_size_current_examples": BATCH_SIZE,
        "epochs_per_window": EPOCHS, "learning_rate": LEARNING_RATE,
        "memory_rule": "Independent classwise algorithm-R reservoirs, 128 slots per class; admit current-window training rows in source order only after that window finishes training; never admit held-out rows",
        "retrieval_rule": "Before each later epoch, uniformly sample without replacement min(64, available) from each class, then shuffle and distribute round-robin across existing current-data batches. No oversampling or backfilling when a class has fewer than 64 examples; total can be under 128. Each retrieved ID appears at most once per epoch.",
        "batch_rule": "Append replay examples to existing batches. Keep exactly the baseline's number of current examples, current order, and optimizer steps; combined batch can exceed 256.",
        "random_generators": {"current_order": "torch.Generator seed 501, matching baseline",
                              "memory_admission": "independent random.Random seed 501",
                              "retrieval": "independent random.Random seed 501"},
        "review_rule": baseline_config["review_rule"],
        "versions": {name: version(name) for name in ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
    }
    (OUTPUT / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUTPUT / "smoke-check.json").write_text(json.dumps(smoke, indent=2) + "\n")
    training_ids_by_window = {}
    held_out = {}
    retrieval_log = []
    memory_log = []
    results = []
    for window in range(1, 7):
        matrix, metadata = ((first_matrix, first_metadata) if window == 1
                            else load_subset(window, "training", width))
        ids = set(metadata["TransactionID"].astype(int))
        if ids.intersection(held_out_ids):
            raise AssertionError("Training and held-out IDs overlap")
        training_ids_by_window[window] = ids
        labels = metadata["isFraud"].to_numpy(dtype=np.float32)
        model.train()
        epoch_losses = []
        epoch_replay_counts = []
        for epoch in range(1, EPOCHS + 1):
            selected = memory.retrieve(window, training_ids_by_window, held_out_ids)
            if window == 1 and selected:
                raise AssertionError("Window 1 unexpectedly retrieved replay")
            permutation = torch.randperm(len(labels), generator=current_rng).numpy()
            batches = math.ceil(len(labels) / BATCH_SIZE)
            distributed = [[] for _ in range(batches)]
            for index, item in enumerate(selected):
                distributed[index % batches].append(item)
            weighted_loss = 0.0
            steps = 0
            for batch_index, start in enumerate(range(0, len(labels), BATCH_SIZE)):
                indices = permutation[start:start + BATCH_SIZE]
                x = dense_batch(matrix, indices)
                y = torch.from_numpy(labels[indices])
                replay_batch = distributed[batch_index]
                if replay_batch:
                    x = torch.cat((x, torch.from_numpy(np.stack([item["features"] for item in replay_batch]))))
                    y = torch.cat((y, torch.tensor([item["class"] for item in replay_batch], dtype=torch.float32)))
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                if not torch.isfinite(loss):
                    raise AssertionError(f"Nonfinite loss in window {window} epoch {epoch}")
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.item()) * len(indices)
                steps += 1
            if steps != batches or sum(map(len, distributed)) != len(selected):
                raise AssertionError("Replay distribution changed optimizer step count")
            epoch_losses.append(weighted_loss / len(labels))
            epoch_replay_counts.append(len(selected))
            retrieval_log.append({"training_window": window, "epoch": epoch,
                                  "current_rows_exposed": len(labels), "optimizer_steps": steps,
                                  "retrieved": [{"transaction_id": item["transaction_id"],
                                                 "class": item["class"],
                                                 "source_window": item["source_window"]}
                                                for item in selected]})
        memory.admit_window(matrix, metadata, window, held_out_ids)
        memory_log.append(memory.snapshot(window))
        del matrix, metadata
        held_out[window] = load_subset(window, "held_out", width)
        evaluations = []
        for prior in range(1, window + 1):
            evaluations.append({"held_out_window": prior, **evaluate(model, *held_out[prior])})
        torch.save({"after_training_window": window, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "seed": SEED,
                    "input_width": width}, OUTPUT / f"checkpoint-window-{window:02d}.pt")
        results.append({"after_training_window": window, "training_rows": len(labels),
                        "epoch_mean_losses": epoch_losses,
                        "epoch_replay_counts": epoch_replay_counts,
                        "held_out_evaluations": evaluations})
        print(f"Window {window}: {len(labels)} current rows, replay {epoch_replay_counts} per epoch", flush=True)
    (OUTPUT / "replay-ids.json").write_text(json.dumps({"retrievals": retrieval_log,
                                                        "memory_snapshots": memory_log}, indent=2) + "\n")
    (OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    comparison_report = comparison(results, baseline_results)
    (OUTPUT / "comparison.json").write_text(json.dumps(comparison_report, indent=2) + "\n")
    plot_comparison(results, baseline_results)
    verify_saved_run()
    print(f"Saved comparison to {OUTPUT}")


if __name__ == "__main__":
    main()
