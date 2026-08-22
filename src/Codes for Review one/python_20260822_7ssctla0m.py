# ============================================================
# NanoSentry Final Polish Experiments
# Fixes ECA, runs ablation, improves Battery LOCO, improves CWRU
# ============================================================

import os
import csv
import time
import warnings
import gc
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"Device: {DEVICE}")

# ============================================================
# Global config
# ============================================================

SEQ = 64
CWRU_SEQ = 1024
BATCH = 128
BATTERY_HORIZON = 200.0

BASE_SEEDS = [42, 7, 13]

# CWRU polish is heavy. Run it only if GPU is available.
RUN_CWRU_POLISH = DEVICE == "cuda"
CWRU_SEEDS = [42, 7, 13] if DEVICE == "cuda" else [42]

RESULTS_FILE = "/kaggle/working/polished_results.csv"
ERROR_FILE = "/kaggle/working/polished_errors.log"

FIELDNAMES = [
    "dataset",
    "subset",
    "model",
    "variant",
    "seed",
    "fold",
    "rmse",
    "mae",
    "score",
    "cz_rmse",
    "acc",
    "f1",
    "params",
    "kb"
]


# ============================================================
# Checkpoint system
# ============================================================

def is_done(dataset, subset, model, variant, seed, fold=""):
    if not os.path.exists(RESULTS_FILE):
        return False

    try:
        df = pd.read_csv(RESULTS_FILE)
        mask = (
            (df["dataset"] == dataset) &
            (df["subset"] == subset) &
            (df["model"] == model) &
            (df["variant"] == variant) &
            (df["seed"] == seed) &
            (df["fold"].astype(str) == str(fold))
        )
        return len(df[mask]) > 0
    except Exception:
        return False


def save_result(row):
    for k in FIELDNAMES:
        row.setdefault(k, "")

    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})

    print(
        f"Saved: {row['dataset']} | {row['subset']} | {row['model']} | "
        f"{row['variant']} | seed {row['seed']} | fold {row['fold']}"
    )


def log_error(message):
    with open(ERROR_FILE, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message + "\n")
    print(f"[ERROR] {message}")


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()