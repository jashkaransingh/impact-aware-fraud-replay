# IEEE-CIS experimental closeout: one-time reserved evaluation

Experimental coding is COMPLETE. The explicitly authorized real reserved
inference command was executed exactly once, from clean commit 9de6406, using
all 25 existing models and the contract frozen under implementation 384e661.
There were no failed real attempts, execution fixes, retraining, preprocessing
refits, calibration changes, new methods, replacement seeds, or tuning. The
scientific protocol and all contract-covered code remain unchanged.

## Final reserved results

The reserved period has 104637 transactions, including 3562 fraud, at times
12666219–15811131. Every method/seed reviews exactly 5232 transactions
(ceil(5% of 104637)), with ties broken by ascending TransactionID. Values below
are mean ± sample SD across training seeds 501–505, not uncertainty across
independent datasets or time periods.

| Method | AP | Fraud recall at 5% | Amount-weighted fraud recall at 5% |
| --- | ---: | ---: | ---: |
| No replay | 0.145288 ± 0.004241 | 0.275014 ± 0.018053 | 0.150775 ± 0.012569 |
| Random | 0.142698 ± 0.004862 | 0.261258 ± 0.013726 | 0.143533 ± 0.012747 |
| Forgetting-only | 0.144705 ± 0.004082 | 0.258057 ± 0.013350 | 0.140939 ± 0.012906 |
| Original combined | 0.143117 ± 0.006216 | 0.266199 ± 0.027449 | 0.148068 ± 0.022017 |
| Class-normalized combined | 0.141824 ± 0.006669 | 0.255924 ± 0.008632 | 0.139485 ± 0.010171 |

No replay has the highest observed mean on all three outcomes. Reviewing 5%
of records catches about 27.5% of fraud transactions and 15.1% of total fraud
amount with that method. These mean rankings do not by themselves establish
population superiority.

## Prespecified paired contrast and uncertainty

Class-normalized minus original combined, retaining every seed:

| Seed | AP difference | Fraud recall difference | Amount recall difference |
| --- | ---: | ---: | ---: |
| 501 | +0.000987 | +0.010387 | +0.003663 |
| 502 | +0.001349 | -0.000842 | -0.019323 |
| 503 | -0.006000 | -0.064851 | -0.033408 |
| 504 | +0.000313 | +0.007019 | +0.009451 |
| 505 | -0.003114 | -0.003088 | -0.003300 |

| Metric | Mean difference | Sample SD | Exploratory paired t 95% interval |
| --- | ---: | ---: | --- |
| AP | -0.001293 | 0.003171 | [-0.005231, +0.002645] |
| Fraud recall | -0.010275 | 0.031005 | [-0.048773, +0.028222] |
| Amount-weighted fraud recall (primary) | -0.008584 | 0.017569 | [-0.030399, +0.013232] |

The primary change is -0.858 percentage points, with interval [-3.040, +1.323]
points. Only two of five seeds improve primary amount recall. All three
intervals include zero. No superiority of class normalization is supported;
these results also do not establish a precise population-level harm estimate.
AP rises in three seeds but its mean falls because the other two losses are
larger. No seed is excluded or substituted after observing its outcome.

Original combined has higher descriptive means than random replay on AP,
fraud recall and amount recall, but all remain below no replay's means. These
secondary comparisons are not multiplicity-adjusted superiority tests. No
replay-based method improves a reserved mean metric over no replay here.

## Relationship to development and validation

The five-seed historical development contrast for class normalization was
-0.001793 amount recall, with interval [-0.016624,+0.013039]. Validation's
contrast was +0.005501, interval [-0.005690,+0.016692], four seeds positive.
The final reserved contrast is negative, with two seeds positive. The favorable
validation mean did not persist. The original hypothesis remains the primary
contrast throughout; no outcome was used to select another primary metric.

Development historical retention is a macro-average of held-out windows 1–5;
validation and reserved metrics pool distinct later chronological partitions.
They are different populations and aggregation units. The historical replay
retention findings do not establish future-transfer benefits. Forgetting-only
led development historical AP; no replay has the largest validation and
reserved AP means. No replay also led development historical amount recall
and has the largest means for that metric in both later partitions.

All five frozen models have lower mean AP, fraud recall and amount recall on
reserved than validation. The validation-reserved figure shows the change
without retraining either model. Different prevalence, case mix, amount
patterns and temporal distribution can contribute; the experiment does not
identify which mechanism caused deterioration. Amount only influenced past
replay selection and evaluation, never classifier inputs or new loss weights.
A plausible selection-score normalization improvement did not produce a stable
value-recall improvement across later data. This is a negative primary result,
not a reason to reopen method search after reserved outcomes are known.

## Execution and audit verification

Contract: e868c8586e5d2df7c94649b30dc5ffe244253b5f50c861809e3b64af25bc158f.
The contract and completed validation receipt passed before reserved access.
The unchanged existing runner then executed the user-authorized command:

```sh
export PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache
.venv/bin/python scripts/run_ieee_final_evaluation.py reserved --contract results/ieee-final-contract-v1/contract.json --validation results/ieee-final-validation-v1/receipt.json --output results/ieee-final-reserved-v1
```

The full original source SHA-256 matches the preparation report:
3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642.
The previously verified development+validation prefix remains
c33a5b33854efd27d94388b21aa8164406b0db788c0b02a2e60aa9fb60cc1daa.
All 590540 source rows were consumed and verified during authorized reserved
loading, including the exact 104637 reserved records. The new read-only audit
independently reread these data and checked saved predictions; it did not
rerun inference or produce another set of model predictions.

All checks passed:

- All 25 prediction/metric hashes and completion receipts match. Every saved
  array has the correct membership/order, 104637 finite scores in [0,1], and
  the fixed 5232-record quota. Predictions were persisted before metrics.
- Independent stable-sort ranking and official sklearn AP reproduce per-seed
  metrics, means/sample SD, paired contrasts and intervals within 1e-12.
- Raw IDs are unique, reserved IDs are disjoint from validation and all 12
  prepared development subsets, and reserved time follows the validation end
  12666174 and the development training cutoff 9521218.
- All 25 model checkpoints, configurations, frozen preprocessor, training
  sources, package versions and scientific protocol retain their contract
  hashes. No fitting, optimization, calibration or replay occurred in final
  inference. Labels/amounts enter scoring only, not classifier features.
- All 547 pre-existing artifacts snapshotted before this run remain unchanged,
  including development/multi-seed outputs, validation predictions/reports,
  and the frozen contract. Earlier experiments were not repeated.
- All 39 tests pass (the existing 37 plus two synthetic metadata tests for the
  new reserved audit). There were zero failed real reserved attempts.

## Final artifacts and reproducibility

All paths below are relative to /Users/jas/Downloads/impact-replay/.

- Local frozen contract: results/ieee-final-contract-v1/contract.json.
- Local restricted reserved predictions, metrics and completion receipt:
  results/ieee-final-reserved-v1/ (never committed).
- Local verification logs and preservation manifest:
  results/ieee-reserved-report-v1/reserved-execution.log,
  audit-and-report.log, final-tests.log, prior-artifacts-sha256.json,
  closeout-record.json.
- Local paper outputs: results/ieee-reserved-report-v1/paper/.
- Versioned paper table, full-precision aggregate audit and paired table:
  summaries/ieee-reserved-v1/comparison.md and verification-and-results.json.
- Versioned CSVs: summaries/ieee-reserved-v1.csv,
  ieee-reserved-v1-per-seed.csv, ieee-reserved-v1-paired.csv,
  ieee-reserved-v1-validation-comparison.csv (all under summaries/).
- Final figures: summaries/ieee-reserved-v1/reserved-comparison.{pdf,png},
  paired-reserved.{pdf,png}, validation-reserved.{pdf,png}.
- Safe execution/hash record: summaries/ieee-reserved-v1/closeout-record.json.

The committed contract/protocol document remains the unchanged pre-evaluation
record. This document supersedes its historical statements about execution
being pending. The full contract stays local because it includes data-derived
feature categories. No raw rows, transaction IDs, prediction arrays, fitted
preprocessor or checkpoints are committed.

Tests and read-only reporting can be run without repeating final inference:

```sh
export PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/report_ieee_reserved.py --contract results/ieee-final-contract-v1/contract.json --validation results/ieee-final-validation-v1/receipt.json --reserved results/ieee-final-reserved-v1 --prior-manifest results/ieee-reserved-report-v1/prior-artifacts-sha256.json --output NEW_REPORT_DIRECTORY
```

The reporter requires local restricted artifacts and refuses an existing report
path. Published aggregates alone cannot satisfy raw-membership audits. Do not
repeat development, validation or the one-time reserved inference to improve
numbers. Independent reproduction in another workspace would require the
restricted dataset under its original access terms, the recorded dependencies,
and checkpoint/provenance reconstruction; a clean-environment full rebuild has
not been verified and is not claimed.

## Final conclusion and limitations

Experimental coding and the bounded experimental execution are COMPLETE.
The final primary hypothesis of improved amount recall from class-normalized
combined replay is unsupported. Retention benefits in development and synthetic
experiments do not establish better frozen future classification or recovered
money on IEEE-CIS. No method or parameter was changed after the final result.

Five training seeds and one fixed temporal split do not quantify uncertainty
over future temporal populations; seed 501 informed development selection.
The feature set is limited, training assumed retrospective immediate labels,
and future inference remained frozen rather than adapting. The 5% quota is an
offline batch assessment, and transaction amount is a proxy. These constraints
and the negative result belong in the final DS 340W paper. Remaining work is
paper writing/integration and presentation, not additional experiments or
experimental coding.
