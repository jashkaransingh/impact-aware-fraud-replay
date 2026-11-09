# IEEE-CIS paired multi-seed development results

Protocol and seed adapter were committed before training as 33368d2. The
previous eight-method study and commits 3f496e7, 9483bd3, 26dd87b were preserved.
No original runner, model, preprocessing or replay implementation file changed.
The experiment followed docs/ieee-multiseed-protocol.md without setting changes.

## Completion and scope

Exactly five methods × five training seeds (501–505) were included. All five
seed-501 artifacts were reused, with 20 new full six-window development runs
for seeds 502–505. Every new run passed, with no replacements, dropped seeds,
extra methods or tuning. All 24 tests pass. Existing result artifacts and the
27 prepared-window/preprocessing files retain their pre-run hashes. Only
prepared development data were read; the reserved evaluation partition was
not accessed.

The data split seed remains 501; training seeds independently set model
initialization, current-order and the separate replay RNGs. The adapter scopes
those constants in the unchanged legacy runners and finalizes accurate seed
and provenance descriptions only in new output configurations. It records
observed current-order permutation hashes without changing their tensors.
Synthetic six-window regressions show exact legacy/adapter equality at 501,
including replay logs, losses and all six replay model/optimizer checkpoints.
They also show seed 502 changes training while leaving split files unchanged.
No completed real-data seed-501 experiment was repeated.

## Primary result

Historical amount-weighted fraud recall at 5% review did not improve reliably
under class normalization. The paired difference (class-normalized minus
original combined) is -0.001793 on average, sample SD 0.011945, with an
exploratory paired t 95% interval [-0.016624, +0.013039]. This corresponds to a
mean change of -0.179 percentage points, interval [-1.662, +1.304] points.
Only two of five seeds improve. The prespecified new-seeds-only sensitivity
(502–505) has mean difference -0.004030 (-0.403 percentage points).

Per-seed primary differences, retaining every seed:

| Training seed | Historical amount recall difference |
| --- | ---: |
| 501 (reused development seed) | +0.007155 |
| 502 | -0.000727 |
| 503 | -0.001533 |
| 504 | -0.021706 |
| 505 | +0.007847 |

Seed 504 has the largest negative value difference while its historical AP
improves by 0.001838. This illustrates the distinction between AP over the
whole ranked list and amount recall at a narrow review quota. It remains in
every summary. The original seed-501 amount improvement alone was insufficient
evidence that normalization would generalize across training randomness.

## Tradeoffs and five-method comparison

The full five-metric mean±sample-SD table and all paired per-seed differences
are in summaries/ieee-multiseed-501-505/summary.md. CSVs and full-precision JSON
are provided alongside the plots (paths below). SD is computed across training
seeds with ddof=1, not across temporal windows. Historical values average final
windows 1–5; current is window 6; forgetting is mean best-to-final AP on 1–5.

Class normalization minus original combined, averaged over all five seeds:

- Historical AP: +0.000666 (four seeds improve); paired 95% interval
  [-0.001336, +0.002668].
- Current AP: +0.000797 (four improve); interval [-0.000403, +0.001998].
- Forgetting: -0.000187 (three improve; negative is better); interval
  [-0.001814, +0.001440].
- Current amount recall: +0.005507 (three improve); interval
  [-0.009515, +0.020528].

None of these intervals excludes zero. These are descriptive intervals on
training randomness after development selection, not confirmatory tests.
The earlier single-seed AP cost of class normalization also does not persist as
a mean AP cost here; this reinforces the need to report paired variability.

The largest observed means belong to forgetting-only for historical AP
(0.192992) and lowest forgetting (0.031468), no replay for current AP (0.193014)
and historical amount recall (0.221232), and class normalization for current
amount recall (0.207033). The latter has sample SD 0.022726. Ranking the means
does not establish population superiority. No method dominates these outcomes.

Training-only diagnostics from seed 501 suggested that within-class
normalization would reduce interference from legitimate-example loss scales.
The multi-seed results show that correcting this score-scale mismatch is not
sufficient for a stable historical value-recall gain. Different training and
reservoir seeds change selected historical examples and the learned ranking;
amount is excluded from the model and the BCE training objective is unchanged.
Thus, a plausible selection mechanism does not ensure improvement on large
held-out fraud amounts at the quota. No additional causal experiment or
post-hoc setting search was performed.

## Audits and reproducibility

Before reporting, all 25 saved runs were checked again without retraining:

- Fixed model width, learning rate, epochs, class weight, batch size, package
  versions, preparation source, and expected method/normalization settings.
- Complete six-window evaluations and the same 5% review quota/membership.
- Training and held-out IDs disjoint; every memory/replay item is from the
  allowed training subset and source window, with valid label, unique ID,
  class cap and retrieval quota. Held-out IDs never enter selection candidates.
- Identical reservoir snapshots across replay methods within each seed;
  original weighted-BCE/current-batch training and replay budgets retained.
- Exact window-1 losses, evaluations, and model/optimizer checkpoints across
  the five methods within each seed; cumulative optimizer steps match at every
  checkpoint.
- For all new runs, observed current-order hashes match the declared seed,
  and saved configuration hashes identify the same prepared files and executed
  training sources. Model/optimizer/source/input provenance is retained.
- Seed-501 current-order hashes were not historically recorded. Reuse relies
  on the unchanged legacy source, prior exact reproduction audits, saved
  checkpoint comparisons and reconstructed expected orders. No new observation
  of those historical training steps is claimed.
- The prepared feature manifest and all 495 transformed feature names exclude
  TransactionAmt. Amount scaling is checked against window-1 training data;
  references and score normalization remain historical-training-only.
- Pre-run hashes for all existing parent-study files and prepared inputs
  match after completion. The protocol hash also matches the preregistered
  version; no reserved-evaluation path is part of the runner or reporter.

The first report used standard CSV CRLF line endings, which the repository
whitespace check rejected. The exporter was corrected to LF and the paper
report regenerated read-only in report-paper/. All numeric summaries, paired
differences, audits, provenance and parsed CSV cells match the initial export
exactly; no training was repeated and the initial export was preserved.

The registry records every reused and new run, commands, statuses, runtimes,
source/input hashes and seed audits. Saved artifacts and synthetic deterministic
regressions provide reproducibility checks; completed full runs were not
repeated to estimate numerical variability a second time.

## Artifact locations

All local paths are relative to /Users/jas/Downloads/impact-replay/.

- New full runs: results/ieee-multiseed-501-505/seed-{502,503,504,505}/METHOD/,
  where METHOD is no-replay, random, forgetting, combined, combined-classnorm.
- Reused seed-501 paths: listed explicitly in ieee-multiseed-protocol.md and
  in scripts/ieee_multiseed.py (REUSED); no copies or modifications were made.
- Logs, registry and audits: results/ieee-multiseed-501-505/.
- Local paper report, raw-precision tables and PNG/PDF plots:
  results/ieee-multiseed-501-505/report-paper/. Initial export remains in report/.
- Versioned report/audits/plots: summaries/ieee-multiseed-501-505/.
- Versioned mean/SD table: summaries/ieee-multiseed-501-505.csv.
- Versioned per-seed metrics: summaries/ieee-multiseed-501-505-per-seed.csv.
- Versioned paired differences: summaries/ieee-multiseed-501-505-paired.csv.

Plots: method-summary.{png,pdf} shows individual seeds and mean±sample SD for
all five metrics. paired-differences.{png,pdf} shows primary and tradeoff
per-seed differences with a zero line and the mean. PDFs embed TrueType fonts.
All versioned artifacts are small aggregate outputs; no IDs, raw records,
restricted datasets, feature matrices or checkpoints are committed.

Reproduction commands (use previously unused output directories):

```sh
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python scripts/run_ieee_multiseed.py --output NEW_STUDY_DIRECTORY
PYTHONPATH=src MPLCONFIGDIR=/tmp/impact-mpl XDG_CACHE_HOME=/tmp/impact-cache .venv/bin/python scripts/summarize_ieee_multiseed.py --study NEW_STUDY_DIRECTORY --output NEW_REPORT_DIRECTORY
```

The second command is for an independent reproduction, not an instruction to
repeat this completed study. The reporter can audit existing runs and regenerate
reports into a new directory without training. All paths refuse overwrite.

## Readiness for the paper and reserved evaluation

The implementation and development evidence are ready to freeze for a final
prespecified evaluation, including a negative primary finding. The evidence
does not support presenting class-normalized combined as superior. Retain all
five methods and declare the final primary contrast, tradeoff metrics, seed
aggregation and evaluation procedure before opening the reserved partition.
Any final evaluation should be a one-time assessment of that frozen protocol;
its outcomes must not drive another hyperparameter search.

Limitations include five training seeds on one fixed temporal development
split, seed-501 involvement in candidate selection, high variability of amount
recall, and no independent final-evaluation results. Training-seed SD does not
measure uncertainty over future transactions, users or temporal distributions.
These results can support an honest DS 340W methods/results section and a
bounded conclusion, without a claimed value-recall improvement.
