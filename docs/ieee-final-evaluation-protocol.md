# Frozen IEEE-CIS final evaluation protocol

Status: implementation protocol; no validation or reserved predictions have been
made by this implementation. This coding task does not authorize opening the
reserved records. Freeze the implementation and contract before future execution.
The five-seed development result did not establish a historical amount-recall
improvement from class normalization. No new method or tuning is introduced.

## Data and chronological timeline

Use the existing preparation report; do not recompute boundaries from outcomes.
For d = TransactionDT − 86400 and S = 15724731, development has 5d < 3S,
validation has 3S ≤ 5d < 4S, and reserved evaluation has 5d ≥ 4S. Equality goes
to the later partition. Reported actual ranges are development 86400–9521218
(380815 rows), validation 9521270–12666174 (105088 rows), and reserved
12666219–15811131 (104637 rows). These are existing preparation metadata,
not newly inspected reserved records. Timestamp groups cannot cross partitions.

| Stage | Model/preprocessor actions | Permitted outcome use |
| --- | --- | --- |
| Development (completed) | Reuse six-window training and final checkpoints; no retraining | Prior training labels trained the models; held-out development labels were used only in earlier development metrics |
| Freeze (metadata only) | Verify 25 checkpoints/configurations, prepared feature hashes, versions and policy settings; bind paths and hashes in an immutable contract | Read saved aggregate reports/configurations; no new raw partition records |
| Validation (future explicit command) | Transform using the frozen preprocessor; predict with each final development checkpoint; no fitting, optimization, calibration, early stopping or method selection | After all predictions are saved, validation labels and amounts are used for diagnostic metrics only |
| Reserved (future explicit command) | Verify the same contract and completed validation receipt; predict once using exactly the same frozen states | After all 25 prediction files are saved, reserved labels and amounts are used only for final metrics |

All development training timestamps precede every validation prediction; all
validation timestamps precede reserved predictions. The model is unchanged
throughout validation/reserved. Validation labels are never promoted into
training or replay. There is no sequential adaptation inside the reserved
partition, no final refit on development+validation, and no training on prior
reserved blocks. This supports a clean future-transfer comparison and reuses
completed work; it does not measure adaptation to the later partitions.

Development held-out retention metrics are retrospective within the development
period and are not out-of-time final predictions. They remain separately labeled
archived diagnostics. Final forgetting is not reported as zero: with no updates
in the later stages, it would not be a meaningful retention experiment.

## Frozen methods and training settings

Exactly no replay, random, forgetting-only, original combined, class-normalized
combined, each with seeds 501–505. Reuse all 25 final development checkpoints
from the paired study. The imported network remains Linear(495,32), ReLU,
Linear(32,1); CPU inference. Training already used Adam lr=.001, weighted BCE
with the fixed window-1 training weight, three epochs/window, current batch 256,
six chronological windows, separate same-seed current/admission/retrieval RNGs.
Within development, keep the completed six equal-elapsed-time windows and
hash split: SHA-256 of ASCII '501:<TransactionDT>', first eight bytes interpreted
big-endian, remainder modulo 100 below 80 selects training; all other timestamp
groups are held out. The same model/optimizer traversed windows 1–6, using only
each window's training subset plus eligible earlier-window replay. Development
held-out records never became training examples. Class weight and impact scale
were fixed from window-1 training, with reference losses recorded only after
source-window training. Training assumed source-window labels were available
when that window completed. These labels are all strictly earlier than the
new validation/reserved prediction periods.

Replay remains 128 slots/class, min(64,available) retrieved/class/epoch,
without replacement/backfill, admission after a source window completes, alpha
.5, sum combination, zero random mixture. Class normalization is the only
difference between the two combined policies. Nothing is tuned here.

The saved sklearn preprocessor was fit only on window-1 training records:
max-64-category one-hot encoding and median imputation/standardization of dist1
and dist2. Load it by its verified hash and call transform only. Do not fit on
validation, reserved features, or development held-out records. IDs, absolute
time, labels and TransactionAmt remain metadata, never classifier predictors.
Impact scaling and frozen reference losses originated only in historical
training data. No memory is populated or retrieved during final inference.

## Prediction and scoring semantics

Inference receives only the transformed predictor matrix and model state.
All 25 models produce sigmoid scores for the entire authorized partition;
ID/score files and their hashes are persisted before any label-based metrics
are calculated. Labels and amounts are passed only to the scoring function.
Predictions depend only on frozen states and transformed predictors, with
the established batch size 256; row labels/amounts and metric results are not
inference inputs. Each model is frozen before any later timestamps are predicted.

Use a separate quota for validation and reserved: k=ceil(.05*N), ranking by
descending score then ascending TransactionID. Review is an offline batch
assessment using that partition's unlabeled scores, not an operational online
threshold or causal arrival-by-arrival review policy. No label or amount breaks
ranking ties. Do not calibrate a probability threshold on validation. Primary
metrics pool the reserved partition as one assessment batch; they are not the
macro-average of development windows and must not be equated with that measure.

AP uses the established sklearn average_precision_score definition. Fraud
recall divides fraud in the top k by all fraud in the partition. Amount-weighted
fraud recall divides fraud amount in the top k by all fraud amount, with no
recovered-money interpretation. AP/fraud recall are null if no fraud is present;
amount recall is null if total fraud amount is zero. Record unavailable metrics
rather than inventing zero performance. A missing primary value precludes the
primary comparison. Fail on duplicate IDs, invalid binary labels, negative or
nonfinite amounts, out-of-range/nonfinite scores, or inconsistent dimensions.

## Primary and secondary comparisons

Primary: paired class-normalized minus original combined reserved amount-weighted
fraud recall at the 5% quota. Keep its direction even though development was
negative; no replacement contrast. Report each seed's difference, arithmetic
mean difference, sample SD, positive/tie/negative counts and an exploratory
paired t 95% interval (df=4). Mean difference >0 alone is not superiority evidence.

Secondary: reserved AP and fraud recall at 5%; all five methods' mean/sample SD
for AP, fraud recall and amount recall; paired AP and fraud-recall differences
between the two combined methods. No-replay/random/forgetting comparisons are
secondary descriptive baselines. Validation results are diagnostic, never a
selection gate based on favorable metric values. All seeds, including 501,
remain included; no failures, negative results or undefined metrics are hidden.
No multiplicity-adjusted claims are made for secondary comparisons.

## Execution gates, provenance, and source access

The default/check command is metadata-only and never opens the raw transaction
CSV. Freeze writes only a new contract directory, containing checkpoint,
configuration, prepared-input, implementation/protocol and preparation-report
hashes plus the fixed method/seed list. Check every hash before inference.
Never overwrite an existing contract, receipt or results directory.

Validation consumes exactly the reported development+validation row prefix of
the existing shared source CSV and never requests/decodes a reserved logical
row. It hashes the consumed prefix, not the entire source. Ordinary filesystem
buffering can read ahead internally; no reserved row is parsed or used. Full
raw-source SHA verification is deferred to the explicitly authorized reserved
command, because hashing the shared CSV in validation would consume reserved
bytes. Source size and preparation metadata are checked in validation; this
is not a full source-content attestation until reserved verification succeeds.

Reserved requires a complete, intact validation receipt for the same contract,
code and model/preprocessor hashes. It verifies the validation prediction-file
hashes, the unchanged source prefix, and the complete source SHA from the prior
preparation report before producing reserved predictions. Check chronology,
exact row counts, timestamp ranges, rational boundary assignments and ID
uniqueness across the consumed source prefix. Any technical failure stops the
run; it never triggers policy changes or replacement seeds. A failed/partial
output stays preserved. Deliberate independent reproductions require new paths.
This receipt is an integrity gate, not a cryptographic access-control system.

New prediction files contain restricted IDs/scores and remain in ignored
results/. Version only code/tests/docs and future approved aggregate reports.
Hashing or reading saved development artifacts does not open reserved records.
All earlier code and results remain unchanged.

## Commands and remaining work

From the repository root (with the existing environment):

```sh
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/impact-mpl
export XDG_CACHE_HOME=/tmp/impact-cache
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/run_ieee_final_evaluation.py check
.venv/bin/python scripts/run_ieee_final_evaluation.py freeze --output results/ieee-final-contract-v1
# Future authorized execution only; neither command is run in this coding task:
.venv/bin/python scripts/run_ieee_final_evaluation.py validation --contract results/ieee-final-contract-v1/contract.json --output results/ieee-final-validation-v1
.venv/bin/python scripts/run_ieee_final_evaluation.py reserved --contract results/ieee-final-contract-v1/contract.json --validation results/ieee-final-validation-v1/receipt.json --output results/ieee-final-reserved-v1
```

After this coding task, freezing and validation execution remain, followed by
an explicit decision to authorize the one-time reserved evaluation. No model
training or development experiment needs rerunning. The runner produces aggregate
per-seed and mean/SD tables and paired contrasts, with prediction lineage.
Paper figures/interpretation and final paper integration remain after execution.

## Implementation verification in this coding task

All 35 tests passed, including full synthetic validation/reserved integration
with 25 dummy development checkpoints, metadata-only raw-access guards,
chronological boundaries, transform-only preprocessing, outcome-invariant
predictions, no optimizer/replay admission/retrieval, frozen policy/budget
checks, receipt/hash failures, primary paired aggregation, and exact metric
agreement with the established evaluator on valid two-class data.

The production metadata-only check successfully verified the 25 actual saved
development checkpoints and frozen preprocessor. Earlier protected outputs,
prepared inputs, and all 25 study results/configurations retain their hashes.
No real raw source CSV was opened; no real validation/reserved predictions,
contract output or final metrics were generated. This verifies implementation
readiness, while real partition I/O and metrics intentionally remain unexecuted.

## Limitations

Five training seeds on one dataset do not quantify uncertainty over future
populations. Seed 501 informed development of class normalization. Labels are
retrospective IEEE-CIS labels; the training study assumed immediate availability
at source-window completion, not realistic operational label delays. The
feature set is limited; TransactionDT is elapsed time with no calendar origin.
Frozen future transfer may deteriorate with drift and cannot establish
continual-adaptation performance. Batch top-5% recall is not an online review
capacity guarantee or measured financial recovery. Validation/reserved have
no final results yet, and technical readiness is not evidence of superiority.
