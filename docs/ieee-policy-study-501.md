# IEEE-CIS seed-501 development study

Initial implementation milestone: 3f496e7. The original 15 tests passed before
commit and push. Random reproduction completed in a separate directory and
matched results, replay logs, smoke values, package versions and fixed settings
exactly. All six model/optimizer checkpoints also match tensor for tensor.

The initial five-method comparison is in
summaries/ieee-policy-study-501/base-comparison.md (full precision JSON alongside).
Historical means refer to final evaluations on windows 1–5, current refers to
window 6, and forgetting is mean best-to-final AP on windows 1–5. Amount recall
uses the unchanged 5% review quota. No replay wins historical amount recall;
forgetting wins both AP metrics; combined wins forgetting and current amount
recall. These are descriptive single-seed development results.

All three new policies passed metadata membership/exclusion checks, exact
window-1 checks, shared reservoir-ID snapshots, replay counts, current exposure,
optimizer-step and version checks. Runtime logs: forgetting 10.63s, impact 10.51s,
combined 10.54s (concurrent CPU processes; not controlled runtime benchmarks).

Only aggregate summaries are versioned. Transaction IDs, feature matrices,
metadata, checkpoints and raw per-run outputs remain under ignored results/.
The reserved evaluation partition has not been opened or used. The existing
no-replay and random folders remain unchanged.

## Three candidate settings fixed before candidate runs

Training-only diagnostics show combined replay covers 63.36% of the reservoir
across three epochs, versus 87.58% for random. Its adjacent-epoch Jaccard overlap
is 0.762 versus 0.327 for random. Selected fraud mean amount is 228.67 versus
141.22 in candidate memory (random selected 143.86). Impact-only selected 236.53,
but this did not translate into better held-out amount recall. Repeated exposure
to high-amount examples may reduce coverage; this is a hypothesis, not causality.
At the start of window 5 under combined, global positive loss-delta p95 is
0.986, legitimate p95 1.430 and fraud p95 0.148. Global normalization thus
compresses fraud forgetting relative to impact. Mean fraud forgetting score is
0.026 versus mean impact 0.728. Most fraud loss deltas are nonpositive.

Budget: exactly three complete candidate runs; no sequential weight search.
All retain the base combined policy except for one controlled change:

1. `combined-alpha075`: alpha=.75 instead of .5. Give more influence to loss
   increases; forgetting-only's strong AP motivates this direction.
2. `combined-mix050`: half of each class quota sampled uniformly, then fill
   by combined priority excluding those IDs. Test coverage without changing
   class budgets or total replay exposures.
3. `combined-classnorm`: normalize positive loss deltas separately within each
   class using the same p95 function. Remove cross-class scale interference.

Primary development improvement target: historical amount recall at 5%, while
reporting historical AP and current AP/amount recall and forgetting as separate
tradeoffs. No claim of universal improvement based on one favorable metric.
No extra candidates if these disappoint. Start with synthetic checks and an
isolated current/replay optimizer smoke for each candidate, then full six-window
runs only on passing candidates. The random implementation's defaults must
continue to reproduce the previous selection path. Reserved evaluation remains
unused. All development held-out results may inform these three candidates,
which makes subsequent reported performance exploratory rather than unbiased.

## Outcomes and interpretation

All three candidates passed smoke checks, full six-window training, and saved
run audits. There were no candidate training failures and no replacement
settings. The machine-readable attempted-settings ledger, including return
codes, commands, source hashes and runtime logs, is
summaries/ieee-policy-study-501/candidate-registry.json. Its base commit predates
the candidate options; exact executed source hashes identify the working tree
that was subsequently committed. Tests increased from 15 to 19, covering
aggregation, audit rejection, mixture uniqueness/quotas and normalization.

The final eight-method table, full-precision metrics, per-window trajectories
and audit results are in summaries/ieee-policy-study-501/final-comparison.{md,json}.
Relative to base combined:

- Alpha .75: historical AP -0.002808; historical amount recall -0.001969;
  current amount recall +0.000725; forgetting +0.001093 (worse). Increasing
  forgetting's scalar weight did not solve historical value retention.
- 50% random mixture: reservoir coverage rises from 63.36% to 76.64%; historical
  amount recall +0.002925 and current AP +0.001588, but historical AP -0.002425,
  current amount recall -0.014146 and forgetting +0.003299. Better coverage
  alone does not guarantee better high-value fraud ranking.
- Class normalization: historical amount recall +0.007155 and current amount
  recall +0.001979, with historical AP -0.001973 and forgetting +0.001458.
  It wins current amount recall among all eight methods, but no replay still
  wins historical amount recall. This is an amount-recall improvement over
  base combined on this development seed, with explicit AP/forgetting costs.

Among all eight, forgetting-only wins historical AP; the random mixture wins
window-6 AP; base combined wins forgetting; no replay wins historical amount
recall; class-normalized combined wins window-6 amount recall. Among the
original five alone, forgetting-only also wins window-6 AP and base combined
wins window-6 amount recall. Forgetting-only wins mean diagonal AP (0.215229),
which is recorded separately from the final window-6 plasticity metric.
Class normalization wins historical unweighted fraud recall at 5% (0.327865).

Interpretation is constrained by the fixed classifier and training objective:

- No replay's historical amount result shows that replay can disturb useful
  rankings; AP improvement does not imply improvement at the review quota or
  on dollar-weighted recall.
- Random preserves broad memory coverage and improves historical AP and
  forgetting versus no replay, but slightly lowers historical amount recall.
- Forgetting-only chooses historical loss increases, which is aligned with
  correcting classification deterioration. Its AP gains do not extend to
  historical fraud or amount recall at exactly 5%, a different ranking slice.
- Impact-only selects substantially larger transactions (mean 236.53 versus
  memory mean 141.22), but repeatedly choosing them ignores whether they need
  relearning. Amount is absent from classifier features and loss weighting;
  selection alone cannot guarantee that high-dollar held-out fraud ranks well.
- Combined balances these signals and preserves AP better than impact-only,
  but has the highest replay repetition. Global loss normalization sometimes
  suppresses fraud forgetting relative to impact because legitimate losses
  determine the scale. Class normalization addresses that measured mismatch;
  changed example choices plausibly explain its value/AP tradeoff.

These are mechanisms consistent with the diagnostics, not causal proof.
The experiments isolate policy settings, not the effect of each selected row.
Single-seed variance, small fixed memory and replay budgets, restricted feature
set, and repeated use of development metrics limit generalization. Clipped
log-amount scores compress the largest amounts. Do not interpret review recall
as recovered money or claim an overall winner. No candidate dominates all
metrics, and no method beats no replay on historical amount recall here.

Selection timers (all 18 retrieval calls together) were 0.01485s for alpha .75,
0.01361s for the mixture, and 0.01591s for class normalization, against
5.38–6.15s for their training-window loops including evaluation/I/O. The initial
base runs did not have selection timers; training-only retrieval microbenchmarks
are included in validated-diagnostics.json to compare policy overhead without
rerunning or changing the base experiments. Mean historical retrieval calls were
0.136ms random, 0.489ms forgetting, 0.231ms impact and 0.644ms combined;
class normalization was 0.600ms and mixture 0.665ms. These are wall-clock observations,
not controlled performance benchmarks.

The first candidate diagnostic export used global normalization for every
candidate's reconstructed forgetting values. For classnorm those particular
values described a counterfactual global scale, not its actual score scale.
This diagnostic-only defect was corrected in validated-diagnostics.json; no
training or selection code used that export and no training run was changed.
The earlier local export is retained for provenance.

## Artifacts and reproduction

All local paths below are relative to /Users/jas/Downloads/impact-replay/:

- Original anchors: results/ieee-no-replay-baseline-501/ and
  results/ieee-class-balanced-replay-501/ (preserved).
- First and final-code exact checks: results/ieee-policy-study-501/
  random-reproduction/ and random-final-code-reproduction/.
- Original new policies: results/ieee-policy-study-501/forgetting/,
  impact/, combined/.
- Candidates: results/ieee-policy-study-501/candidates/combined-alpha075/,
  combined-mix050/, combined-classnorm/. Each includes configuration, six
  checkpoints, results, replay log, verification, comparison, plot and timing.
- Attempt ledger and smoke/full logs: results/ieee-policy-study-501/candidates/.
- Final audited table, full-precision JSON, CSV and figure:
  results/ieee-policy-study-501/validated-comparison/.
- Training-only diagnosis: results/ieee-policy-study-501/validated-diagnostics.json.
- Versioned aggregate reports: summaries/ieee-policy-study-501/.

Run the tests with the command in ieee-policy-comparison.md. Reproduce all three
candidate settings with `.venv/bin/python scripts/run_ieee_policy_candidates.py
--output NEW_DIRECTORY`. Existing directories are rejected. Use
compare_ieee_policies.py with `--random-reproduction PATH --run NAME=PATH`
(repeated for each policy) and `--output NEW_DIRECTORY`. Run diagnosis with
`diagnose_ieee_replay.py --run NAME=PATH --output NEW_JSON`.

Final-code random verification again matches the original saved random metrics,
losses, replay logs and all six model/optimizer checkpoints exactly. Anchor
hashes and the final runner hash are retained in final-reproduction-audit.json.
No restricted datasets, IDs, matrices, checkpoints or large artifacts are
versioned. No reserved evaluation data were accessed. All selection references,
normalization and amounts derive only from training records. The imported
classifier and prepared feature manifest continue to exclude TransactionAmt.

Next highest-value experiment: paired multi-seed development replication of
random, base combined, and class-normalized combined, with historical amount
recall as primary and historical AP/forgetting reported as constraints. Keep
all other conditions fixed and declare seed set and decision rule before
running. This tests whether the observed amount improvement and AP cost are
stable before any further tuning or use of the reserved evaluation partition.
The three-candidate budget is exhausted; that follow-up has not been run.
