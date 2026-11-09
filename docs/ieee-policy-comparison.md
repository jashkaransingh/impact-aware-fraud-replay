# IEEE-CIS development policy comparison protocol

Preserve results/ieee-no-replay-baseline-501 and
results/ieee-class-balanced-replay-501 as read-only comparison anchors.
Full development runs and milestone pushes are now authorized. Preserve existing
outputs; never use the reserved evaluation partition for tuning.

Verification command (no full experiments):
`PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python -m unittest discover -s tests`

First run the synthetic verification in tests/test_ieee_policies.py. It must
exercise all policies, historical/held-out guards, quotas, reference losses,
amount isolation, and unchanged random selection without reading real data.

For later authorized runs, use run_ieee_balanced_replay.py --policy POLICY
with a fresh --output directory. Policies: random, forgetting, impact, combined.
Keep seed 501, six prepared chronological development windows, fitted
preprocessing (TransactionAmt excluded), imported baseline network, Adam,
positive weight from window-1 training, three epochs, batch size 256, and the
same current-data permutation stream. Do not regenerate windows or tune on
reserved evaluation outcomes. Development held-out metrics may inform only the
three recorded exploratory candidates; held-out rows never enter scores. Keep the existing no-replay runner unchanged.

Admission remains the original independent class-balanced Algorithm-R process:
128 slots per class; admit only after completing each window. Selection takes
min(64, available) per class per epoch, without replacement or backfill, shuffles
and distributes over the existing current batches. Preserve current exposures
and optimizer steps. Separate admission, retrieval, and current-order RNGs.

For scoring, store unweighted per-example BCE after the source window finishes
training; freeze that reference. Before each later epoch, compute unweighted
BCE only on validated historical reservoir rows. Pass current minus reference
loss to replay.score (positive clipping and 95th-percentile normalization).
Impact is fraud label times clipped log1p(amount), scaled by the 95th percentile
of log1p(amount) in window-1 training only. Combined uses the existing score
function with alpha=.5 and sum, fixed in advance. Legitimate impact is zero;
seeded ties still fill the legitimate quota. Amount never enters model features,
loss weights, admission, or forgetting-only ranking. No held-out values fit any
score, reference, or scale. Held-out IDs are read only to enforce exclusion.

Audit saved retrieval IDs against historical training IDs, memory snapshots,
class quotas, current exposures and steps; require exact window-1 baseline
losses and evaluations. Reservoir ID snapshots must match across policies.
A fresh random rerun must reproduce the saved random metrics and retrievals
before treating new comparisons as validated. Verify preparation provenance,
settings, and package versions against both saved anchors before interpreting
results; report mismatches rather than replacing anchors.

Report AP retention curves for all six held-out windows, initial/final/best AP,
best-to-final forgetting and mean final AP on windows 1–5, window-6 AP for
plasticity, fraud recall and amount-weighted fraud recall at the same fixed
5% review quota (ties by ID). Pair each policy with both saved anchors. Report
selection overhead separately; this is one development seed, not confirmatory
statistical evidence. Initial policies and alpha=.5 are fixed before the first comparison. Up to three
subsequent candidates may be investigated using development outcomes; record
every setting and failure. These are exploratory development comparisons, not
unbiased evaluation of the selected candidate. Reserved evaluation stays unused.
