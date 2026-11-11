Validation: 105,088 rows; mean ± sample SD across seeds 501–505. Fixed 5% batch review quota.

| Method | AP | Fraud recall @5% | Amount-weighted fraud recall @5% |
| --- | --- | --- | --- |
| no-replay | 0.177740 ± 0.000978 | 0.329095 ± 0.005583 | 0.171268 ± 0.015302 |
| random | 0.177066 ± 0.002214 | 0.326797 ± 0.004563 | 0.164687 ± 0.019595 |
| forgetting | 0.176792 ± 0.001165 | 0.324645 ± 0.003218 | 0.159751 ± 0.019506 |
| combined | 0.176514 ± 0.002402 | 0.325183 ± 0.006041 | 0.163713 ± 0.015751 |
| combined-classnorm | 0.176796 ± 0.001466 | 0.327579 ± 0.004851 | 0.169214 ± 0.016962 |

Paired differences: class-normalized minus original combined.

| Seed | AP difference | Fraud recall difference | Amount recall difference |
| --- | --- | --- | --- |
| 501 | -0.001876 | -0.001711 | +0.002380 |
| 502 | +0.002207 | +0.004401 | -0.001395 |
| 503 | -0.000771 | +0.000000 | +0.003502 |
| 504 | -0.000364 | +0.003912 | +0.001724 |
| 505 | +0.002213 | +0.005379 | +0.021293 |
