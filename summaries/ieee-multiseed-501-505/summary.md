Values are mean ± sample SD over seeds 501–505. Historical windows 1–5; current window 6. Amount recall at fixed 5% review quota.

| Method | Historical AP | Current AP (window 6) | AP forgetting (lower better) | Historical amount recall at 5% | Current amount recall at 5% |
| --- | --- | --- | --- | --- | --- |
| no-replay | 0.188340 ± 0.003722 | 0.193014 ± 0.005020 | 0.035442 ± 0.001745 | 0.221232 ± 0.009014 | 0.206412 ± 0.014272 |
| random | 0.191100 ± 0.002377 | 0.190661 ± 0.005066 | 0.032488 ± 0.002814 | 0.217510 ± 0.016697 | 0.195097 ± 0.016315 |
| forgetting | 0.192992 ± 0.004305 | 0.192316 ± 0.004225 | 0.031468 ± 0.001513 | 0.217744 ± 0.016164 | 0.201124 ± 0.007959 |
| combined | 0.190716 ± 0.004094 | 0.191749 ± 0.003267 | 0.033186 ± 0.002758 | 0.220756 ± 0.014470 | 0.201526 ± 0.012952 |
| combined-classnorm | 0.191382 ± 0.003665 | 0.192546 ± 0.003984 | 0.032999 ± 0.001567 | 0.218963 ± 0.014477 | 0.207033 ± 0.022726 |

Paired differences: class normalization minus original combined. Positive forgetting is worse.

| Seed | Historical AP | Current AP (window 6) | AP forgetting (lower better) | Historical amount recall at 5% | Current amount recall at 5% |
| --- | --- | --- | --- | --- | --- |
| 501 | -0.001973 | +0.000149 | +0.001458 | +0.007155 | +0.001979 |
| 502 | +0.000223 | +0.000184 | -0.000780 | -0.000727 | -0.000493 |
| 503 | +0.001776 | -0.000049 | -0.001852 | -0.001533 | +0.019302 |
| 504 | +0.001838 | +0.001855 | +0.000771 | -0.021706 | -0.009619 |
| 505 | +0.001466 | +0.001849 | -0.000534 | +0.007847 | +0.016365 |

Exploratory paired t interval across training seeds on one fixed development dataset; seed 501 informed candidate selection. Not independent final evaluation.
