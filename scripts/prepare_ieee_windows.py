"""Prepare six IEEE-CIS development windows without reading later partitions.

Run: python3 scripts/prepare_ieee_windows.py
The first-window training subset alone fits the frozen sklearn preprocessor.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "results/ieee-preparation/preparation.json"
OUTPUT = ROOT / "results/ieee-preparation/windows"
CATEGORICAL = ["ProductCD", *(f"card{i}" for i in range(1, 7)),
               "addr1", "addr2", "P_emaildomain", "R_emaildomain"]
NUMERIC = ["dist1", "dist2"]
PREDICTORS = CATEGORICAL + NUMERIC
METADATA = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
SEED = 501


def window_for(dt, start, end):
    """Six equal-duration intervals; the final interval includes the maximum."""
    return min(5, 6 * (dt - start) // (end - start))


def subset_for(dt):
    """SHA-256 of ASCII '501:<TransactionDT>'; first 8 bytes, big endian."""
    digest = hashlib.sha256(f"{SEED}:{dt}".encode("ascii")).digest()
    return "training" if int.from_bytes(digest[:8], "big") % 100 < 80 else "held_out"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_development(source, count, start, end):
    """Consume exactly the reported development row count, never later rows."""
    columns = {name: [] for name in PREDICTORS + METADATA}
    window_indices = [[] for _ in range(6)]
    subset_indices = {(w, s): [] for w in range(6) for s in ("training", "held_out")}
    timestamp_assignments = {}
    ids = set()
    previous_dt = None
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        needed = PREDICTORS + METADATA
        if len(header) != len(set(header)) or any(name not in header for name in needed):
            raise ValueError("Required CSV columns missing or repeated")
        positions = {name: header.index(name) for name in needed}
        for index in range(count):
            try:
                row = next(reader)
            except StopIteration as error:
                raise ValueError("CSV ended before reported development count") from error
            if len(row) != len(header):
                raise ValueError(f"Development row {index + 1} has wrong column count")
            dt = int(row[positions["TransactionDT"]])
            transaction_id = int(row[positions["TransactionID"]])
            if previous_dt is not None and dt < previous_dt:
                raise ValueError(f"Development source is not chronological at row {index + 1}")
            previous_dt = dt
            if not start <= dt <= end or transaction_id in ids:
                raise ValueError("Development range or ID uniqueness disagrees with preparation")
            ids.add(transaction_id)
            window = window_for(dt, start, end)
            subset = subset_for(dt)
            prior = timestamp_assignments.setdefault(dt, (window, subset))
            if prior != (window, subset):
                raise AssertionError("Timestamp group was split")
            window_indices[window].append(index)
            subset_indices[(window, subset)].append(index)
            for name in needed:
                value = row[positions[name]]
                columns[name].append(value if name not in CATEGORICAL or value else "__MISSING__")
    return pd.DataFrame(columns), window_indices, subset_indices, len(ids), len(timestamp_assignments)


def verify_existing(frame, subsets, report, source_hash):
    """Recompute development artifacts in memory and compare; write nothing."""
    if report["source_sha256_from_preparation"] != source_hash:
        raise ValueError("Window report source SHA-256 disagrees with current source")
    names = json.loads((OUTPUT / "feature_names.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(OUTPUT / "preprocessor.joblib")
    if names != preprocessor.get_feature_names_out().tolist():
        raise ValueError("Saved feature names disagree with preprocessor")
    features = frame[PREDICTORS].copy()
    for name in NUMERIC:
        features[name] = pd.to_numeric(features[name], errors="coerce")
    for window in range(6):
        for subset in ("training", "held_out"):
            indices = subsets[(window, subset)]
            stem = f"window-{window + 1:02d}-{subset}"
            saved_meta = pd.read_csv(OUTPUT / f"{stem}-metadata.csv", dtype=str, keep_default_na=False)
            expected_meta = frame.iloc[indices][METADATA].reset_index(drop=True)
            if not saved_meta.equals(expected_meta):
                raise ValueError(f"Metadata differs from source: {stem}")
            saved_matrix = sparse.load_npz(OUTPUT / f"{stem}-features.npz")
            expected_matrix = sparse.csr_matrix(preprocessor.transform(features.iloc[indices]), dtype=np.float32)
            if saved_matrix.shape != expected_matrix.shape or saved_matrix.dtype != np.float32:
                raise ValueError(f"Feature shape or dtype differs: {stem}")
            difference = saved_matrix - expected_matrix
            if difference.nnz and np.any(difference.data != 0):
                raise ValueError(f"Features differ from source and preprocessor: {stem}")
            recorded = report["windows"][window][subset]
            labels = frame.iloc[indices]["isFraud"].astype(np.int8).to_numpy()
            if (recorded["row_count"] != len(indices)
                    or recorded["fraud_count"] != int(labels.sum())
                    or recorded["nonfraud_count"] != len(indices) - int(labels.sum())):
                raise ValueError(f"Window counts differ: {stem}")
    print(f"Verified existing artifacts against source SHA-256 and {len(frame)} development rows; no files rewritten")


def build_preprocessor():
    return ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                       max_categories=64, dtype=np.float32), CATEGORICAL),
         ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), NUMERIC)],
        sparse_threshold=1.0,
    )


def main():
    verify_only = len(sys.argv) == 2 and sys.argv[1] == "--verify-existing"
    if len(sys.argv) > 1 and not verify_only:
        raise SystemExit("Usage: python3 scripts/prepare_ieee_windows.py [--verify-existing]")
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    source = ROOT / preparation["source"]
    if source.stat().st_size != preparation["source_bytes"]:
        raise ValueError("Source size disagrees with preparation report")
    source_hash = sha256(source)
    if source_hash != preparation["source_sha256"]:
        raise ValueError("Source SHA-256 disagrees with preparation report")
    if preparation["source_checks"]["source_time_order_violations"] != 0:
        raise ValueError("Preparation report recorded nonchronological source data")
    development = preparation["partitions"]["development"]
    start, end = development["min_transaction_dt"], development["max_transaction_dt"]
    count = development["row_count"]
    frame, windows, subsets, unique_ids, timestamp_groups = read_development(source, count, start, end)
    if verify_only:
        verify_existing(frame, subsets, json.loads((OUTPUT / "verification.json").read_text()), source_hash)
        return

    # The model matrix is built only from the selected predictors. Amounts,
    # labels, IDs, and absolute time are retained in separate metadata files.
    features = frame[PREDICTORS].copy()
    for name in NUMERIC:
        features[name] = pd.to_numeric(features[name], errors="coerce")
    labels = frame["isFraud"].astype(np.int8).to_numpy()
    ids = frame["TransactionID"].astype(np.int64).to_numpy()
    times = frame["TransactionDT"].astype(np.int64).to_numpy()
    report_windows = []
    training_ids, held_out_ids = set(), set()
    for window in range(6):
        entry = {"window": window + 1,
                 "exact_start_seconds": {"numerator": 6 * start + window * (end - start), "denominator": 6},
                 "exact_end_seconds": {"numerator": 6 * start + (window + 1) * (end - start), "denominator": 6},
                 "actual_min_transaction_dt": int(times[windows[window]].min()),
                 "actual_max_transaction_dt": int(times[windows[window]].max()),
                 "row_count": len(windows[window])}
        for subset, destination in (("training", training_ids), ("held_out", held_out_ids)):
            indices = subsets[(window, subset)]
            destination.update(ids[indices].tolist())
            class_counts = np.bincount(labels[indices], minlength=2)
            entry[subset] = {"row_count": len(indices), "nonfraud_count": int(class_counts[0]),
                             "fraud_count": int(class_counts[1])}
            if min(class_counts) == 0:
                raise ValueError(f"Window {window + 1} {subset} lacks a class")
        report_windows.append(entry)

    assigned_once = sum(len(indices) for indices in subsets.values()) == count
    ids_disjoint = not training_ids.intersection(held_out_ids)
    ids_complete = len(training_ids | held_out_ids) == unique_ids == count
    if not (assigned_once and ids_disjoint and ids_complete):
        raise AssertionError("Development assignment failed coverage or disjointness checks")

    fit_indices = subsets[(0, "training")]
    preprocessor = build_preprocessor()
    preprocessor.fit(features.iloc[fit_indices])
    feature_names = preprocessor.get_feature_names_out().tolist()
    if len(feature_names) != len(set(feature_names)):
        raise AssertionError("Encoded feature names are not unique")
    # max_categories caps output width, including the infrequent-category bin.
    encoded_widths = {}
    for name in CATEGORICAL:
        prefix = f"categorical__{name}_"
        encoded_widths[name] = sum(value.startswith(prefix) for value in feature_names)
    if any(width > 64 for width in encoded_widths.values()):
        raise AssertionError("Categorical encoding exceeded the per-field cap")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, OUTPUT / "preprocessor.joblib")
    (OUTPUT / "feature_names.json").write_text(json.dumps(feature_names, indent=2) + "\n", encoding="utf-8")
    transformed_checks = []
    for window in range(6):
        for subset in ("training", "held_out"):
            indices = subsets[(window, subset)]
            matrix = preprocessor.transform(features.iloc[indices])
            matrix = sparse.csr_matrix(matrix, dtype=np.float32)
            if matrix.shape != (len(indices), len(feature_names)) or not np.isfinite(matrix.data).all():
                raise AssertionError("Transformed feature matrix has invalid shape or values")
            stem = f"window-{window + 1:02d}-{subset}"
            sparse.save_npz(OUTPUT / f"{stem}-features.npz", matrix)
            frame.iloc[indices][METADATA].to_csv(OUTPUT / f"{stem}-metadata.csv", index=False)
            transformed_checks.append({"window": window + 1, "subset": subset,
                                       "rows": matrix.shape[0], "columns": matrix.shape[1],
                                       "dtype": str(matrix.dtype), "finite": True})

    report = {
        "source_preparation": str(PREPARATION.relative_to(ROOT)),
        "source_sha256_from_preparation": preparation["source_sha256"],
        "development_rows_read": count,
        "development_time_span": {"min_transaction_dt": start, "max_transaction_dt": end},
        "window_rule": "Six equal elapsed-time intervals from development min to max TransactionDT. "
                       "For window i=0..5, assign min(5, floor(6*(TransactionDT-min)/(max-min))); "
                       "exact boundaries go to the later window, and the maximum belongs to window 6.",
        "holdout_rule": "SHA-256 of UTF-8 ASCII '501:<TransactionDT>'; interpret first eight digest bytes "
                        "as an unsigned big-endian integer. Remainder modulo 100 below 80 is training; "
                        "otherwise held_out. Seed=501; labels and amounts are not used.",
        "predictors": {"categorical": CATEGORICAL, "numeric": NUMERIC,
                       "excluded_from_model_inputs": METADATA,
                       "description": "Initial limited feature set, not final feature selection"},
        "preprocessing": {"fit_scope": "window 1 training subset only",
                          "categorical": "OneHotEncoder(max_categories=64, handle_unknown='infrequent_if_exist', dtype=float32); empty string becomes __MISSING__",
                          "numeric": "median imputation then StandardScaler, fit on window 1 training only",
                          "frozen_for_later_windows": True,
                          "encoded_widths_by_field": encoded_widths,
                          "feature_count": len(feature_names),
                          "output_dtype": "float32"},
        "verification": {"development_rows_assigned_once": assigned_once,
                         "training_and_held_out_ids_disjoint": ids_disjoint,
                         "all_development_ids_preserved": ids_complete,
                         "timestamp_groups_kept_together": True,
                         "timestamp_group_count": timestamp_groups,
                         "both_classes_in_each_subset": True,
                         "transformed_matrices": transformed_checks},
        "windows": report_windows,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {count} development rows in {OUTPUT}")


if __name__ == "__main__":
    main()
