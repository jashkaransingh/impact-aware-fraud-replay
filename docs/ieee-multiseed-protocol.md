# Paired IEEE-CIS development evaluation: seeds 501–505

Protocol recorded before any new-seed training. Parent study is frozen at
26dd87b (with 3f496e7 and 9483bd3 preserved). No completed experiment will be
rerun. No hyperparameter tuning or new candidate method is allowed.

Primary question: does class-normalized combined replay improve historical
amount-weighted fraud recall at the fixed 5% review quota over original
combined replay? Historical AP and best-to-final AP forgetting are explicit
tradeoffs, not secondary routes to claim success on the primary question.

Exactly five methods: no replay, random replay, forgetting-only, original
combined (alpha=.5, sum, global normalization, no random mixture), and combined
with within-class normalization (all other settings identical).

Predetermined training seeds: 501, 502, 503, 504, 505. The prepared data split
seed stays 501 for every run. Existing six chronological development windows,
row membership, features, fitted preprocessor, model (495→32→1 with ReLU),
positive class weight from window-1 training, Adam lr=.001, three epochs per
window, batch size 256, CPU/deterministic operations, and evaluation remain
unchanged. Each replay method keeps 128 memory slots per class and selects
min(64, available) per class once per epoch, without backfill. Admission occurs
after source-window training. Replay is distributed over current batches;
current exposure/order and optimizer steps match the same-seed no-replay run.
No-replay has zero replay exposure, as required by its definition.

Use the same training seed within each five-method block for initialization,
current-order generator, and separate reservoir-admission and retrieval RNGs.
Across seeds, those streams change; dataset membership/preprocessing do not.
The new adapter sets module training constants explicitly and restores them
after each call. Original runner source files stay unchanged. The adapter
finalizes accurate training/split seed fields and RNG descriptions in new run
configurations; this does not alter numerical results. Synthetic regression
checks must demonstrate seed-501 compatibility without repeating real runs.

Reuse seed-501 outputs: results/ieee-no-replay-baseline-501,
results/ieee-class-balanced-replay-501, results/ieee-policy-study-501/forgetting,
results/ieee-policy-study-501/combined, and
results/ieee-policy-study-501/candidates/combined-classnorm.
Audit configurations, source provenance, saved IDs/checkpoints, and the prior
reproduction report before accepting reuse. New seeds require 20 full runs,
using unused directories under results/ieee-multiseed-501-505. Do not rerun
seed 501 or completed new-seed runs. Retain failures in a registry; no replacement
seeds or settings. Reproducibility checks use synthetic data, saved artifacts,
source/input hashes and order hashes rather than repeating completed full runs.

Report these five per-run outcomes using existing summarize semantics:
- Historical AP: mean final AP on held-out development windows 1–5.
- Current AP: window-6 AP immediately after completing window 6.
- Forgetting: mean (best observed AP minus final AP) on windows 1–5; lower better.
- Historical amount-weighted recall: mean final amount-weighted fraud recall
  on windows 1–5 at ceil(5% of each window's rows), with score ties by ID.
- Current amount-weighted recall: same metric for window 6.

Report arithmetic mean and sample SD (ddof=1) across the five training seeds,
not across windows. Provide classnorm-minus-original paired differences for
every seed and outcome, mean/SD of differences, win/tie/loss counts, and a
paired t 95% interval (df=4) as a descriptive uncertainty summary. No unpaired
test, seed filtering, or multiple comparisons search. Because seed 501 informed
candidate development, also report the prespecified mean paired differences
for seeds 502–505. Five seeds measure training randomness on one fixed dataset,
not uncertainty over future temporal populations. Intervals are exploratory,
not confirmatory evidence after development selection.

Produce a mean±SD plot for all five metrics and a per-seed paired difference
plot for the three primary/tradeoff metrics. Version small tables, JSON audit
summaries and paper-ready PNG/PDF plots; raw records, IDs, matrices,
checkpoints and large artifacts stay ignored.

Before/after audits must hash all existing study outputs, all prepared window
files and fitted preprocessing; check per-run settings and package versions,
held-out exclusion from every replay candidate, historical source windows,
unique retrievals and class quotas, identical admission snapshots within each
seed, same current-order hashes and optimizer steps, exact window-1 results
and checkpoints across methods. Score references and amount scaling use only
training data. TransactionAmt stays excluded from classifier inputs. No
reserved evaluation partition may be opened, fitted, scored or used for tuning.

Readiness decision: freeze this bounded comparison and report negative/mixed
results honestly. A superiority claim requires independent final evaluation;
no further development search is authorized. Recommend a final evaluation
protocol only after interpreting primary paired results and AP/forgetting costs.
