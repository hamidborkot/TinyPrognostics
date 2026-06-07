# TinyPrognostics

> **A 47.1 KB unified prognostics architecture for edge deployment — RUL estimation and fault classification across three industrial domains.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-NASA%20C--MAPSS%20%7C%20Battery%20%7C%20CWRU-lightgrey)](#datasets)

---

## Overview

TinyPrognostics is a **12,052-parameter (47.1 KB)** neural network that jointly performs:
- **RUL regression** — predicting remaining useful life in cycles
- **Health-state classification** — degradation state (healthy / degrading / critical)

It is the first unified architecture under 50 KB demonstrated simultaneously on:
- Turbofan engine degradation (NASA C-MAPSS, 4 subsets)
- Lithium-ion battery state-of-health (NASA Battery Dataset)
- Rotating machinery fault classification (CWRU Bearing Dataset)

The architecture combines:
- Dilated causal 1D convolutions (TCN) with receptive fields 1→2→4
- A novel **CrossSensorGate** for adaptive inter-sensor weighting
- A residual skip connection from raw input
- A single-layer GRU with orthogonal initialisation

---

## Key Results

### C-MAPSS RUL Estimation (RMSE ↓, lower is better)

| Model | Size | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|---|
| Ridge Regression | — | 44.14 | 54.44 | 42.43 | 55.07 |
| CNN-32 | 30.6 KB | 26.47 | 40.84 | 24.79 | 39.62 |
| LSTM-64 | 210.3 KB | 16.28 | 27.95 | 16.02 | **27.15** |
| **TinyPrognostics** | **47.1 KB** | **14.75** | **26.74** | **13.95** | 27.67 |

TinyPrognostics outperforms LSTM-64 on FD001, FD002, and FD003 while being **4.5× smaller**.

### Critical-Zone RMSE (RUL ≤ 30 cycles, ↓ lower is better)

Accuracy in the final 30 cycles before failure is the most operationally relevant metric.

| Model | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **TinyPrognostics** | **4.63** | 5.52 | 3.14 | 6.62 |
| LSTM-64 | 5.27 | **4.42** | **2.66** | **5.94** |
| CNN-32 | 29.05 ⚠️ | 20.67 | 21.50 | 28.18 |

> ⚠️ CNN-32 critical-zone RMSE **exceeds** its overall RMSE on FD001 (29.05 vs 26.47), indicating systematic RUL over-estimation near end-of-life — a hazardous property for maintenance-critical applications. TinyPrognostics critical-zone RMSE is **3–7× lower** than CNN-32.

### NASA Score (↓ lower is better)

| Dataset | NASAScore |
|---|---|
| FD001 | 350.47 |
| FD002 | 7405.94 |
| FD003 | 315.86 |
| FD004 | 8677.78 |

### Multi-Domain Results

| Dataset | Task | Metric | Result |
|---|---|---|---|
| NASA Battery | RUL regression | RMSE (cycles) | **1.87** |
| NASA Battery | RUL regression | MAE | **0.42** |
| CWRU Bearing | 10-class fault classification | Accuracy | **99.96%** |

### Ablation Study (FD001 / FD003, RMSE)

All variants use the **same final TinyPrognostics checkpoint** (FD001 = 14.75, FD003 = 13.95) as the baseline. Each row removes one component and re-trains from scratch.

| Variant | FD001 | Δ FD001 | FD003 | Δ FD003 |
|---|---|---|---|---|
| **Full model** | **14.75** | — | **13.95** | — |
| w/o CrossSensorGate | 17.31 | +2.56 | 16.58 | +2.63 |
| w/o Skip connection | 15.94 | +1.19 | 14.87 | +0.92 |
| w/o Dilation | 16.28 | +1.53 | 15.41 | +1.46 |

The **CrossSensorGate is the dominant component** — removing it increases RMSE by +2.56 (FD001) and +2.63 (FD003), the largest degradation of any single component.

### Transfer Learning (FD001 pretrain → target)

| Target | 10% data | 25% | 50% | 100% | Scratch |
|---|---|---|---|---|---|
| FD003 (RMSE) | 15.08 | 15.35 | 14.64 | 15.31 | **13.95** |
| Battery (RMSE) | 3.84 | 3.83 | 2.48 | 3.75 | **1.87** |

> Transfer learning does not improve over training from scratch on either target domain, suggesting the backbone learns domain-specific rather than universal degradation priors — a direction for future work.

---

## Architecture

```
Input (B, T, C)
    │
    ├─ Linear embed → (B, T, d=24)
    │       │
    │   DilatedBlock(dilation=1)
    │   DilatedBlock(dilation=2)   ← TCN stack
    │   DilatedBlock(dilation=4)
    │       │
    │   CrossSensorGate            ← adaptive inter-sensor weighting
    │       │
    ├─ ────(+)──── Skip Linear(C→d)
    │
    GRU(d→d, batch_first)  [orthogonal init]
    │
    ├─ RUL head    → scalar  (regression)
    └─ State head  → n_classes (classification)
```

| Property | Value |
|---|---|
| Parameters | 12,052 |
| Size (FP32) | 47.1 KB |
| Sequence length | 64 timesteps |
| Hidden dim | 24 |
| Sensor inputs | Configurable (14 for C-MAPSS, 4 for Battery, 1 for CWRU) |

---

## Datasets

| Dataset | Source | Task |
|---|---|---|
| NASA C-MAPSS (FD001–FD004) | [Kaggle: behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) | RUL regression |
| NASA Battery | [Kaggle: patrickfleith/nasa-battery-dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) | SoH / RUL |
| CWRU Bearing | [Kaggle: sufian79/cwru-mat-full-dataset](https://www.kaggle.com/datasets/sufian79/cwru-mat-full-dataset) | 10-class fault classification |

**C-MAPSS preprocessing:** Per-condition K-Means normalization (6 clusters) for FD002/FD004; global z-score for FD001/FD003. Piecewise-linear RUL cap at 125 cycles. Sequence windows of length 64.

---

## Quickstart

```bash
git clone https://github.com/hamidborkot/TinyPrognostics
cd TinyPrognostics
pip install -r requirements.txt
```

Then open and run the Kaggle notebook, or:

```bash
python src/train.py --dataset fd001
python src/train.py --dataset battery
python src/train.py --dataset cwru
```

---

## Project Structure

```
TinyPrognostics/
├── src/
│   ├── model.py          # TinyPrognostics architecture
│   ├── data.py           # Data loaders (C-MAPSS, Battery, CWRU)
│   ├── train.py          # Training + evaluation CLI
│   └── transfer.py       # Transfer learning experiments
├── results/
│   ├── main_results.csv  # All RMSE / MAE / NASAScore numbers
│   ├── critical_zone.csv # Critical-zone RMSE (RUL ≤ 30)
│   ├── ablation.csv      # Ablation study results
│   └── transfer.csv      # Transfer learning results
├── paper/
│   └── tables.tex        # LaTeX tables ready for paper
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Citation

```bibtex
@misc{tulla2026tinyprognostics,
  title   = {TinyPrognostics: A Unified Sub-50KB Architecture for Multi-Domain Industrial Prognostics},
  author  = {Tulla, MD Hamid Borkot},
  year    = {2026},
  url     = {https://github.com/hamidborkot/TinyPrognostics}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
