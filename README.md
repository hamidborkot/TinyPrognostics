# TinyPrognostics

> **A 47 KB dual-task causal dilated CNN for Remaining Useful Life (RUL) estimation and fault classification — designed for edge deployment.**

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Datasets](https://img.shields.io/badge/Datasets-NASA%20C--MAPSS%20%7C%20Battery%20%7C%20CWRU-lightgrey)](#datasets)

---

## Overview

TinyPrognostics is a **12,052-parameter** (47.1 KB) neural network that jointly performs:
- **RUL regression** — predicting remaining useful life in cycles
- **Health-state classification** — 3-class degradation state (healthy / degrading / critical)

The architecture combines:
- Dilated causal 1D convolutions (TCN) with receptive fields 1→2→4
- A novel **Cross-Sensor Gate** for adaptive inter-sensor weighting
- A skip connection from raw input
- A single-layer GRU with orthogonal initialisation

---

## Key Results

### C-MAPSS RUL Estimation (RMSE ↓)

| Model | FD001 | FD002 | FD003 | FD004 | Size |
|---|---|---|---|---|---|
| Ridge Regression | 42.27 | 54.73 | 43.37 | 54.76 | — |
| 1D-CNN | 27.43 | 41.27 | 26.34 | 59.03 | 30.6 KB |
| LSTM-64 | 16.53 | 27.88 | 16.80 | 29.92 | 210.3 KB |
| **TinyPrognostics** | **16.51** | **26.97** | **13.77** | **29.35** | **47.1 KB** |

### Other Datasets

| Dataset | Metric | Result |
|---|---|---|
| NASA Battery | RMSE (cycles) | **2.00** |
| NASA Battery | Health-state Acc | **98.2%** |
| CWRU Bearing | Fault Classif. Acc | **99.79%** |

### Ablation Study (RMSE on FD001 / FD003)

| Variant | FD001 | ΔRMSE | FD003 | ΔRMSE |
|---|---|---|---|---|
| Full model | 15.86 | — | 14.39 | — |
| w/o Cross-sensor gate | 17.79 | +1.93 | 14.88 | +0.49 |
| w/o Skip connection | 14.53 | −1.33 | 14.77 | +0.38 |
| w/o Dilation | 14.63 | −1.22 | 14.52 | +0.13 |

### Transfer Learning (FD001 pretrain → target)

| Target | 10% data | 25% | 50% | 100% | Scratch |
|---|---|---|---|---|---|
| FD003 (RMSE) | 16.31 | 14.84 | 13.82 | 14.17 | 13.77 |
| Battery (RMSE) | 4.29 | 3.92 | 3.94 | 3.80 | 2.00 |

---

## Architecture

```
Input (B, T, C)
    │
    ├─ Linear embed → (B, T, d=24)
    │       │
    │   DilatedBlock(d=1)
    │   DilatedBlock(d=2)   ← TCN
    │   DilatedBlock(d=4)
    │       │
    │   CrossSensorGate    ← adaptive inter-sensor weighting
    │       │
    ├─ ────(+)──── Skip Linear(C→d)
    │
    GRU(d, d)  [orthogonal init]
    │
    ├─ RUL head    → scalar (regression)
    └─ State head  → n_classes (classification)
```

**Parameters:** 12,052 | **Size:** 47.1 KB | **Inference:** CPU-compatible

---

## Datasets

| Dataset | Source | Task |
|---|---|---|
| NASA C-MAPSS (FD001–FD004) | [Kaggle: behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) | RUL regression |
| NASA Battery | [Kaggle: patrickfleith/nasa-battery-dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) | SoH / RUL |
| CWRU Bearing | [Kaggle: sufian79/cwru-mat-full-dataset](https://www.kaggle.com/datasets/sufian79/cwru-mat-full-dataset) | 10-class fault classification |

---

## Quickstart

```bash
git clone https://github.com/hamidborkot/TinyPrognostics
cd TinyPrognostics
pip install -r requirements.txt
```

Then open `notebooks/tinyprognostics_full.ipynb` in Kaggle (GPU recommended) or run:

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
├── notebooks/
│   └── tinyprognostics_full.ipynb
├── results/
│   ├── main_results.csv
│   ├── ablation.csv
│   └── transfer.csv
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
  title   = {TinyPrognostics: Dual-Task Edge Prognostics with Dilated Causal Convolutions},
  author  = {Tulla, MD Hamid Borkot},
  year    = {2026},
  url     = {https://github.com/hamidborkot/TinyPrognostics}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
