"""Validate IEEE-CIS transactions and record a chronological time split.

Run from anywhere: python3 scripts/prepare_ieee_cis.py
Only a JSON preparation report is written; the source CSV is read-only.
"""

import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/ieee-cis/train_transaction.csv"
OUTPUT = ROOT / "results/ieee-preparation/preparation.json"
FIELDS = ("TransactionID", "TransactionDT", "TransactionAmt", "isFraud")


def rows():
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(header) != len(set(header)) or any(name not in header for name in FIELDS):
            raise ValueError("Required columns are missing or header names repeat")
        positions = [header.index(name) for name in FIELDS]
        for line_number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {line_number} has {len(row)} columns; expected {len(header)}")
            yield line_number, tuple(row[index].strip() for index in positions)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source():
    missing = dict.fromkeys(FIELDS, 0)
    invalid = {"TransactionID": 0, "TransactionDT": 0,
               "TransactionAmt": 0, "isFraud": 0}
    ids = set()
    duplicates = 0
    count = 0
    first_dt = last_dt = None
    source_order_violations = 0
    previous_dt = None
    for _, values in rows():
        count += 1
        transaction_id, dt_text, amount_text, label = values
        for name, value in zip(FIELDS, values):
            missing[name] += not bool(value)
        if transaction_id:
            try:
                parsed_id = int(transaction_id)
            except ValueError:
                invalid["TransactionID"] += 1
            else:
                if parsed_id in ids:
                    duplicates += 1
                ids.add(parsed_id)
        if dt_text:
            try:
                dt = int(dt_text)
                if str(dt) != dt_text:
                    raise ValueError("noncanonical integer")
            except ValueError:
                invalid["TransactionDT"] += 1
            else:
                first_dt = dt if first_dt is None else min(first_dt, dt)
                last_dt = dt if last_dt is None else max(last_dt, dt)
                if previous_dt is not None and dt < previous_dt:
                    source_order_violations += 1
                previous_dt = dt
        if amount_text:
            try:
                amount = Decimal(amount_text)
                if not amount.is_finite() or amount < 0:
                    raise InvalidOperation
            except InvalidOperation:
                invalid["TransactionAmt"] += 1
        if label and label not in ("0", "1"):
            invalid["isFraud"] += 1
    checks = {
        "missing_core_values": missing,
        "invalid_core_values": invalid,
        "duplicate_transaction_ids": duplicates,
        "unique_transaction_ids": len(ids),
        "source_time_order_violations": source_order_violations,
        "ids_unique": duplicates == 0 and len(ids) == count,
        "labels_binary": missing["isFraud"] == 0 and invalid["isFraud"] == 0,
        "amounts_finite_nonnegative": missing["TransactionAmt"] == 0 and invalid["TransactionAmt"] == 0,
        "core_values_complete_and_valid": all(v == 0 for v in missing.values()) and all(v == 0 for v in invalid.values()),
    }
    if (not count or not checks["ids_unique"] or not checks["core_values_complete_and_valid"]
            or source_order_violations != 0):
        raise ValueError(f"Source validation failed: {json.dumps(checks, sort_keys=True)}")
    if first_dt == last_dt:
        raise ValueError("TransactionDT has no elapsed time span")
    return count, first_dt, last_dt, checks


def partition_source(total, start, end):
    span = end - start
    parts = {name: {"row_count": 0, "min_transaction_dt": None,
                    "max_transaction_dt": None}
             for name in ("development", "validation", "reserved_evaluation")}
    parts["development"]["fraud_count"] = 0
    parts["validation"]["fraud_count"] = 0
    for _, (_, dt_text, _, label) in rows():
        dt = int(dt_text)
        offset5 = 5 * (dt - start)
        name = ("development" if offset5 < 3 * span else
                "validation" if offset5 < 4 * span else "reserved_evaluation")
        part = parts[name]
        part["row_count"] += 1
        part["min_transaction_dt"] = dt if part["min_transaction_dt"] is None else min(part["min_transaction_dt"], dt)
        part["max_transaction_dt"] = dt if part["max_transaction_dt"] is None else max(part["max_transaction_dt"], dt)
        if name != "reserved_evaluation":
            part["fraud_count"] += int(label)
    all_assigned = sum(part["row_count"] for part in parts.values()) == total
    nonempty = all(part["row_count"] > 0 for part in parts.values())
    ordered = nonempty and (
        parts["development"]["max_transaction_dt"] < parts["validation"]["min_transaction_dt"]
        and parts["validation"]["max_transaction_dt"] < parts["reserved_evaluation"]["min_transaction_dt"]
    )
    # Strictly separated time ranges also prove timestamp groups cannot overlap.
    disjoint = ordered
    if not all_assigned or not ordered or not disjoint:
        raise AssertionError("Partition coverage or chronological separation failed")
    return parts, {"all_rows_preserved": all_assigned,
                   "partitions_chronological": ordered,
                   "partitions_nonoverlapping": disjoint,
                   "identical_timestamps_kept_together": disjoint}


def main():
    source_size = SOURCE.stat().st_size
    source_hash = sha256(SOURCE)
    total, start, end, checks = scan_source()
    parts, partition_checks = partition_source(total, start, end)
    if SOURCE.stat().st_size != source_size or sha256(SOURCE) != source_hash:
        raise RuntimeError("Source CSV changed during preparation")
    span = end - start
    report = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_bytes": source_size,
        "source_sha256": source_hash,
        "source_row_count": total,
        "time_unit": "TransactionDT elapsed seconds; no absolute calendar origin assumed",
        "time_span": {"min_transaction_dt": start, "max_transaction_dt": end,
                      "elapsed_seconds": span},
        "boundaries": {
            "development_validation": {"exact_seconds_numerator": 5 * start + 3 * span,
                                       "denominator": 5},
            "validation_reserved_evaluation": {"exact_seconds_numerator": 5 * start + 4 * span,
                                                "denominator": 5},
        },
        "partition_rule": (
            "Let d = TransactionDT - min(TransactionDT) and S = max(TransactionDT) - min(TransactionDT). "
            "Assign development if 5*d < 3*S; otherwise validation if 5*d < 4*S; "
            "otherwise reserved_evaluation. No random shuffle. Boundary equality goes to the later partition."
        ),
        "source_checks": checks,
        "partition_checks": partition_checks,
        "partitions": parts,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
