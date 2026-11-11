# IEEE-CIS real validation: frozen five-method comparison

Execution began from clean Git commit 384e661. The unchanged final-evaluation
runner completed validation for all 25 existing development models on 105088
validation transactions (4090 fraud). All five training seeds 501–505 and all
five methods were retained. No training, refitting, tuning, policy changes,
replacement seeds, or execution bug fixes were required.

## Frozen artifacts and execution

The existing 35 tests passed and the metadata-only check passed before freeze.
Contract e868c8586e5d2df7c94649b30dc5ffe244253b5f50c861809e3b64af25bc158f
was saved in results/ieee-final-contract-v1/contract.json and verified before
and after inference and reporting. This binds all 25 model checkpoints,
configurations, prepared development artifacts, frozen preprocessor, package
versions and the implementation/scientific protocol from 384e661. No file
covered by that contract was edited. The final suite has 37 passing tests:
original tests unchanged, plus two checks for the independent reporter.

The real validation command was:

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python scripts/run_ieee_final_evaluation.py validation --contract results/ieee-final-contract-v1/contract.json --output results/ieee-final-validation-v1
```

All ID/score arrays were saved before outcome metrics. Each method/seed used
exactly ceil(.05*105088)=5255 reviewed records, ties resolved by ascending ID.
Models remained frozen after development. Preprocessing was transform-only,
using the previously fitted window-1 training preprocessor. TransactionAmt,
labels, IDs and time did not enter the classifier. No replay admission or
retrieval occurred during inference.

## Actual aggregate results

The complete mean ± sample SD table is in
summaries/ieee-validation-v1/comparison.md. The sample SD is across the five
training seeds, not across records or temporal windows. Full-precision JSON,
method/per-seed/paired CSV tables and PNG/PDF plots are retained.

No replay has the largest observed mean on all three validation outcomes:
AP 0.177740, fraud recall 0.329095, amount-weighted fraud recall 0.171268.
At the fixed review quota, it finds about 33% of fraud transactions and 17%
of fraud transaction amount on average. These are descriptive mean rankings,
not statistical superiority claims over all other methods.

Class-normalized combined is the strongest replay method by mean fraud recall
(0.327579) and amount recall (0.169214); random replay has the highest replay
AP mean (0.177066). Differences between some means are small relative to the
training-seed variation. Forgetting-only's development historical AP lead does
not carry into this later validation period. No replay's development lead in
historical amount recall also appears in the validation mean ranking.

The prespecified class-normalized minus original combined diagnostic is:

| Validation metric | Mean paired change | Sample SD | Paired t 95% interval | Positive / tie / negative seeds |
| --- | ---: | ---: | --- | --- |
| AP | +0.000282 | 0.001845 | [-0.002009, +0.002573] | 2 / 0 / 3 |
| Fraud recall at 5% | +0.002396 | 0.003075 | [-0.001422, +0.006215] | 3 / 1 / 1 |
| Amount-weighted fraud recall at 5% | +0.005501 | 0.009013 | [-0.005690, +0.016692] | 4 / 0 / 1 |

Per-seed amount-recall changes are 501 +0.002380; 502 -0.001395;
503 +0.003502; 504 +0.001724; 505 +0.021293. Every seed remains included;
seed 505 supplies the largest positive difference. The mean change is +0.550
percentage points, with interval [-0.569, +1.669] percentage points.

The earlier development historical amount contrast was negative on average
(-0.001793, two of five seeds positive). Validation's contrast is positive
(four of five positive), while both exploratory intervals include zero. This
is mixed evidence across different temporal populations and metric aggregation
units, not a persistent superiority result. Development historical recall was
a macro-average over five held-out windows; validation pools one later
partition. The underlying frozen models and policies were not altered in
response to these results, and no best seed/method was selected for reserved
execution. All five methods remain in the frozen contract.

## Independent audits

scripts/report_ieee_validation.py does not predict, train or fit. It re-reads
only the permitted development+validation prefix, verifies the saved arrays,
and recomputes the ranking with a separate pandas stable sort plus official
sklearn AP. Independent metrics and aggregate mean/SD agree with the inference
runner within 1e-12. All 25 arrays have exactly 105088 finite scores in [0,1],
matching the validation IDs in the same order and the 5255-record quota.

Chronological checks show validation times 9521270–12666174, strictly after
the training cutoff 9521218 and before the existing preparation metadata's
reserved minimum 12666219. Validation IDs are disjoint from all 12 prepared
development training/held-out subsets. The loader checks ID uniqueness and
partition/time membership throughout the consumed prefix. Prediction hashes,
prediction-completion marker, and timestamps show all prediction files were
persisted before metrics. The completion receipt and all frozen hashes match.
All protected prior study outputs and prepared inputs also retain their hashes.

Exactly 485903 logical source rows (380815 development + 105088 validation)
were consumed, with prefix SHA-256
c33a5b33854efd27d94388b21aa8164406b0db788c0b02a2e60aa9fb60cc1daa.
The independent audit obtained the same prefix hash. No reserved logical row
was requested, decoded, scored or used. Ordinary filesystem buffering may read
ahead, as documented in the scientific protocol. The shared source CSV's full
SHA is deliberately not recomputed: doing so would consume reserved content.
Source size and consumed-prefix integrity are verified; full source-content
attestation remains deferred to explicitly authorized reserved execution.

Detailed prediction-level audit outcomes contain only seed/method identifiers
and aggregate counts in the versioned report. Actual transaction IDs and
scores remain solely in the ignored prediction files. The complete frozen
contract stays local because it contains data-derived feature categories.
Only its digest and safe artifact/hash provenance are versioned.

## Artifacts, logs and commands

All local paths are relative to /Users/jas/Downloads/impact-replay/:

- Frozen contract: results/ieee-final-contract-v1/contract.json.
- Restricted prediction arrays, metrics and completed validation receipt:
  results/ieee-final-validation-v1/. These must not be committed.
- Execution logs: results/ieee-validation-report-v1/freeze.log,
  validation-execution.log, audit-and-report.log, post-report-tests.log.
- Execution/hash record: results/ieee-validation-report-v1/execution-record.json.
- Paper table, full-precision aggregate audit JSON, CSVs and PNG/PDF figures:
  results/ieee-validation-report-v1/paper/.
- Versioned aggregate reports/figures/audit record: summaries/ieee-validation-v1/.
- Versioned CSVs: summaries/ieee-validation-v1.csv,
  summaries/ieee-validation-v1-per-seed.csv, summaries/ieee-validation-v1-paired.csv.

The initial freeze followed the documented command, using an unused path:

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python scripts/run_ieee_final_evaluation.py freeze --output results/ieee-final-contract-v1
```

Do not repeat freeze/inference into existing directories. To independently
audit and regenerate aggregate reports without rerunning predictions, use a
new report path:

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python scripts/report_ieee_validation.py --contract results/ieee-final-contract-v1/contract.json --validation results/ieee-final-validation-v1 --output NEW_REPORT_DIRECTORY
```

## Limitations and reserved readiness

This is one later chronological validation population, with five training
seeds and a limited feature set. Training-seed SD does not represent future
population uncertainty. Small mean rankings and intervals spanning zero do
not establish general superiority. Top-5% review is an offline batch quota,
not an online capacity policy. Amount-weighted recall is a transaction-value
proxy, not measured recovered money. Frozen future transfer does not test
adaptation or realistic label-delay handling.

The unchanged scientific protocol, contract and completed validation receipt
pass every implemented gate for one-time reserved evaluation. There are no
execution blockers or code fixes required. Real reserved data/schema and full
source SHA remain intentionally unverified until authorization permits access.
Reserved evaluation was not executed in this task. Its eventual execution must
retain all five methods, the same models/seeds, and the prespecified primary
contrast; validation outcomes must not change those choices.
