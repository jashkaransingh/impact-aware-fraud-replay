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
