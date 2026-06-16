# NanoSentry — Experiment Results

All results are means across **5 independent seeds** `{42, 7, 13, 99, 2025}`.
Standard deviations are reported alongside each mean.

---

## Parameter Counts (Domain-Specific)

NanoSentry's parameter count varies with input sensors (`n_sensors`) and
class count (`n_classes`), but all configurations remain below 48 KB:

| Domain | n_sensors | n_classes | Params | Size (KB) |
|---|---|---|---|---|
| C-MAPSS FD001–FD004 | 14 | 3 | 12,052 | 47.1 |
| NASA Battery | 4 | 3 | 11,572 | 45.3 |
| CWRU Bearing | 1 | 10 | 11,659 | 45.6 |

---

## C-MAPSS RUL Estimation (RMSE ↓, lower is better)

| Model | Params | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|---|
| **NanoSentry** | **12,052** | **14.75 ± 0.30** | **26.74 ± 0.55** | **13.95 ± 0.40** | **27.67 ± 0.58** |
| LSTM-64 | 53,825 | 16.28 | 27.95 | 16.02 | 27.15 |
| CNN-32 | 7,841 | 26.47 | 40.84 | 24.79 | 39.62 |
| Ridge | <100 | 44.14 | 54.44 | 42.43 | 55.07 |

NanoSentry outperforms LSTM-64 (4.5× larger) on FD001, FD002, FD003.

---

## Critical-Zone RMSE (RUL ≤ 30 cycles ↓)

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **NanoSentry** | **11.00 ± 0.90** | **20.08 ± 0.29** | **10.50 ± 0.29** | **21.29 ± 0.31** |

Critical-zone RMSE isolates end-of-life prediction quality where
maintenance decisions are most consequential (RUL ≤ 30 cycles).

---

## Battery State-of-Health (5 seeds)

| Metric | Mean | Std |
|---|---|---|
| RMSE | 1.53 | ± 0.34 |
| MAE | 0.34 | ± 0.06 |
| Accuracy | 97.25% | ± 0.87% |
| NASA Score | 8.18 | ± 2.17 |
| CZ-RMSE | 1.28 | ± 0.42 |

Params: **11,572** (45.3 KB)

---

## CWRU Bearing Fault Classification (5 seeds)

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 99.87% | ± 0.07% |
| NASA Score | 32.93 | ± 19.30 |

Params: **11,659** (45.6 KB). 10-class fault diagnosis across all
drive-end bearing fault conditions. NASA score variance is expected
for classification tasks where the RUL head output is not the
primary objective.

---

## Files

| File | Description |
|---|---|
| `five_seeds_raw.csv` | Per-seed results for all 6 tasks (30 rows) |
| `five_seeds_summary.csv` | Mean ± std summary table (6 rows) |
| `main_results.csv` | Full comparison including all baselines |
| `model_sizes.csv` | Parameter counts and KB sizes per domain |
| `ablation.csv` | Ablation study results |
| `critical_zone.csv` | Critical-zone RMSE analysis |
| `transfer.csv` | Cross-domain transfer learning results |
