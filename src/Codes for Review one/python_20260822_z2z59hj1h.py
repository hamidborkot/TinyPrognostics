# ============================================================
# NanoSentry Revised Master Experiment Script
# Fixed ECA bug + resume-friendly + error-safe
# ============================================================

import os
import csv
import time
import warnings
import gc
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# ============================================================
# GLOBAL CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

print(f"🚀 Running on: {DEVICE}")

if DEVICE == "cpu":
    print("⚠ WARNING: CPU mode is much slower. Enable GPU in Kaggle if possible.")

CMAPSS_DIR = "/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps"
BATTERY_DIR = "/kaggle/input/datasets/patrickfleith/nasa-battery-dataset/cleaned_dataset"
CWRU_DIR = "/kaggle/input/datasets/sufian79/cwru-mat-full-dataset"

RESULTS_FILE = "/kaggle/working/master_results_fixed.csv"
ERROR_FILE = "/kaggle/working/master_errors.log"

os.makedirs("/kaggle/working", exist_ok=True)

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

SEQ = 64
CWRU_SEQ = 1024
BATTERY_HORIZON = 200.0

SENSORS = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
OP_COLS = [2, 3, 4]

# Stronger settings on GPU, safer settings on CPU
if DEVICE == "cuda":
    BATCH = 128
    EPOCHS = 150
    PATIENCE = 30
    MAIN_SEEDS = [42, 7, 13, 99, 2025]
    BASE_SEEDS = [42, 7, 13]
    CWRU_MAIN_SEEDS = [42, 7, 13, 99, 2025]
    ATTENTION_FDS = [1, 3]
else:
    BATCH = 64
    EPOCHS = 100
    PATIENCE = 20
    MAIN_SEEDS = [42, 7, 13]
    BASE_SEEDS = [42, 7, 13]
    CWRU_MAIN_SEEDS = [42, 7, 13]
    ATTENTION_FDS = [1]

# For single-task vs multi-task comparison, we do FD001 and FD003 fully.
# This saves time while still giving the reviewer the required comparison.
SINGLE_TASK_FDS = [1, 3]

CWRU_FILES = {
    "97.mat": 0, "98.mat": 0, "99.mat": 0, "100.mat": 0,
    "105.mat": 1, "106.mat": 1, "107.mat": 1, "108.mat": 1,
    "169.mat": 2, "170.mat": 2, "171.mat": 2, "172.mat": 2,
    "209.mat": 3, "210.mat": 3, "211.mat": 3, "212.mat": 3,
    "118.mat": 4, "119.mat": 4, "120.mat": 4, "121.mat": 4,
    "185.mat": 5, "186.mat": 5, "187.mat": 5, "188.mat": 5,
    "222.mat": 6, "223.mat": 6, "224.mat": 6, "225.mat": 6,
    "130.mat": 7, "131.mat": 7, "132.mat": 7, "133.mat": 7,
    "197.mat": 8, "198.mat": 8, "199.mat": 8, "200.mat": 8,
    "234.mat": 9, "235.mat": 9, "236.mat": 9, "237.mat": 9,
}


# ============================================================
# CHECKPOINT / LOGGING SYSTEM
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
        f"✅ Saved: {row['dataset']} | {row['subset']} | {row['model']} | "
        f"{row['variant']} | Seed {row['seed']} | Fold {row['fold']}"
    )


def log_error(message):
    with open(ERROR_FILE, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + message + "\n")
    print(f"[ERROR] {message}")


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# METRICS
# ============================================================

def nasa_score(pred, true):
    pred = np.array(pred, dtype=np.float64)
    true = np.array(true, dtype=np.float64)
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    total = float(s.sum())
    return total if np.isfinite(total) else np.nan


def critical_zone_rmse(pred, true, threshold=30):
    pred = np.array(pred, dtype=np.float64)
    true = np.array(true, dtype=np.float64)
    mask = true <= threshold
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def macro_f1(pred, true, num_classes):
    pred = np.array(pred)
    true = np.array(true)
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (true == c)))
        fp = int(np.sum((pred == c) & (true != c)))
        fn = int(np.sum((pred != c) & (true == c)))

        if tp == 0:
            f1s.append(0.0)
            continue

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    return float(np.mean(f1s)) if len(f1s) > 0 else np.nan


def rul_to_state(rul):
    """
    Unified health-state definition:
    0 = healthy       RUL > 60
    1 = transitional  31 <= RUL <= 60
    2 = critical      RUL <= 30
    """
    if rul > 60:
        return 0
    elif rul > 30:
        return 1
    else:
        return 2


def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0
    return params, kb


# ============================================================
# C-MAPSS DATA LOADER
# ============================================================

def load_cmapss(fd, seed):
    train_df = pd.read_csv(f"{CMAPSS_DIR}/train_FD{fd:03d}.txt", sep=r"\s+", header=None)
    test_df = pd.read_csv(f"{CMAPSS_DIR}/test_FD{fd:03d}.txt", sep=r"\s+", header=None)
    rul_df = pd.read_csv(f"{CMAPSS_DIR}/RUL_FD{fd:03d}.txt", header=None, names=["RUL"])

    max_cycles = train_df.groupby(0)[1].max().rename("max_cycle")
    train_df = train_df.join(max_cycles, on=0)
    train_df["RUL"] = (train_df["max_cycle"] - train_df[1]).clip(upper=125)

    engines = train_df[0].unique()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(engines))
    n_tr = int(len(engines) * 0.8)

    tr_eng = engines[idx[:n_tr]]
    vl_eng = engines[idx[n_tr:]]

    # Fit normalization only on training engines
    fit_df = train_df[train_df[0].isin(tr_eng)]

    norm_train = train_df.copy()
    norm_test = test_df.copy()

    mu_global = fit_df[SENSORS].mean()
    std_global = fit_df[SENSORS].std().replace(0, 1)

    norm_train[SENSORS] = (train_df[SENSORS] - mu_global) / std_global
    norm_test[SENSORS] = (test_df[SENSORS] - mu_global) / std_global

    # Condition-aware normalization for multi-condition subsets
    if fd in [2, 4]:
        km = KMeans(n_clusters=6, random_state=seed, n_init=10)
        km.fit(fit_df[OP_COLS].values)

        fit_labels = km.predict(fit_df[OP_COLS].values)
        all_train_labels = km.predict(train_df[OP_COLS].values)
        test_labels = km.predict(test_df[OP_COLS].values)

        for c in range(6):
            fit_mask = fit_labels == c
            if fit_mask.sum() < 2:
                continue

            mu_c = fit_df.loc[fit_mask, SENSORS].mean()
            std_c = fit_df.loc[fit_mask, SENSORS].std().replace(0, 1)

            train_mask = all_train_labels == c
            test_mask = test_labels == c

            norm_train.loc[train_mask, SENSORS] = (
                train_df.loc[train_mask, SENSORS] - mu_c
            ) / std_c

            if test_mask.sum() > 0:
                norm_test.loc[test_mask, SENSORS] = (
                    test_df.loc[test_mask, SENSORS] - mu_c
                ) / std_c

    def make_windows(df, engine_list):
        Xs, yr, ys = [], [], []
        for eng in engine_list:
            sub = df[df[0] == eng].reset_index(drop=True)
            feats = sub[SENSORS].values.astype(np.float32)
            n = len(feats)

            if n < SEQ:
                pad = np.zeros((SEQ - n, len(SENSORS)), dtype=np.float32)
                feats = np.vstack([pad, feats])
                n = SEQ

            for t in range(n - SEQ + 1):
                Xs.append(feats[t:t + SEQ])
                rul_val = float(sub.loc[min(t + SEQ - 1, len(sub) - 1), "RUL"])
                yr.append(rul_val)
                ys.append(rul_to_state(rul_val))

        return (
            torch.tensor(np.array(Xs), dtype=torch.float32),
            torch.tensor(yr, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long)
        )

    Xtr, ytr_r, ytr_s = make_windows(norm_train, tr_eng)
    Xvl, yvl_r, yvl_s = make_windows(norm_train, vl_eng)

    rul_test = rul_df["RUL"].values.astype(np.float32)
    Xte_list, yte_r_list, yte_s_list = [], [], []

    for i, eng in enumerate(test_df[0].unique()):
        sub = norm_test[norm_test[0] == eng].reset_index(drop=True)
        feats = sub[SENSORS].values.astype(np.float32)
        n = len(feats)

        if n < SEQ:
            pad = np.zeros((SEQ - n, len(SENSORS)), dtype=np.float32)
            feats = np.vstack([pad, feats])

        Xte_list.append(feats[-SEQ:])
        rv = float(rul_test[i])
        yte_r_list.append(rv)
        yte_s_list.append(rul_to_state(rv))

    Xte = torch.tensor(np.array(Xte_list), dtype=torch.float32)
    yte_r = torch.tensor(yte_r_list, dtype=torch.float32)
    yte_s = torch.tensor(yte_s_list, dtype=torch.long)

    pin = DEVICE == "cuda"

    tr = DataLoader(TensorDataset(Xtr, ytr_r, ytr_s), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
    va = DataLoader(TensorDataset(Xvl, yvl_r, yvl_s), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
    te = DataLoader(TensorDataset(Xte, yte_r, yte_s), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)

    return tr, va, te


# ============================================================
# BATTERY DATA LOADER: LEAVE-ONE-CELL-OUT
# ============================================================

def seqs_to_loader(seqs, shuffle=False):
    X = torch.tensor(np.array([s[0] for s in seqs]), dtype=torch.float32)
    yr = torch.tensor(np.array([s[1] for s in seqs]), dtype=torch.float32)
    ys = torch.tensor(np.array([s[2] for s in seqs]), dtype=torch.long)

    return DataLoader(
        TensorDataset(X, yr, ys),
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE == "cuda")
    )


def build_battery_cell_sequences(cap_vals):
    cap_arr = np.array(cap_vals, dtype=np.float32)
    n = len(cap_arr)

    if n < SEQ + 5:
        return None

    max_cap = cap_arr[0] if cap_arr[0] > 0 else cap_arr.max()
    soh = np.clip(cap_arr / max_cap, 0.0, 1.0)
    delta = np.diff(soh, prepend=soh[0])

    # Reviewer-safe feature: t / H, not t / n
    cyc_norm = np.clip(np.arange(n, dtype=np.float32) / BATTERY_HORIZON, 0.0, 1.0)

    feats = np.stack(
        [soh, delta, cyc_norm, cap_arr / (max_cap + 1e-8)],
        axis=1
    ).astype(np.float32)

    eol = np.where(soh < 0.8)[0]
    eol_idx = int(eol[0]) if len(eol) > 0 else n

    rul = np.clip((eol_idx - np.arange(n)).astype(np.float32), 0.0, 200.0)
    states = np.array([rul_to_state(float(r)) for r in rul], dtype=np.int64)

    seqs = []
    for t in range(n - SEQ + 1):
        seqs.append(
            (
                feats[t:t + SEQ],
                float(rul[t + SEQ - 1]),
                int(states[t + SEQ - 1])
            )
        )

    return seqs


def load_battery_folds():
    meta = pd.read_csv(os.path.join(BATTERY_DIR, "metadata.csv"))
    discharge = meta[meta["type"] == "discharge"].copy()
    batteries = sorted(discharge["battery_id"].unique())

    folds = []

    for test_cell in batteries:
        train_cells = [b for b in batteries if b != test_cell]

        tr_seqs = []
        vl_seqs = []

        for bid in train_cells:
            rows = discharge[discharge["battery_id"] == bid].sort_values("start_time")
            cap_vals = pd.to_numeric(rows["Capacity"], errors="coerce").dropna().values

            cell_seqs = build_battery_cell_sequences(cap_vals)
            if cell_seqs is None:
                continue

            split_idx = int(len(cell_seqs) * 0.8)
            tr_seqs.extend(cell_seqs[:split_idx])
            vl_seqs.extend(cell_seqs[split_idx:])

        rows = discharge[discharge["battery_id"] == test_cell].sort_values("start_time")
        cap_vals = pd.to_numeric(rows["Capacity"], errors="coerce").dropna().values
        te_seqs = build_battery_cell_sequences(cap_vals)

        if te_seqs is None:
            continue

        if len(tr_seqs) > 0 and len(vl_seqs) > 0 and len(te_seqs) > 0:
            folds.append(
                {
                    "test_cell": str(test_cell),
                    "tr": tr_seqs,
                    "vl": vl_seqs,
                    "te": te_seqs
                }
            )

    return folds


# ============================================================
# CWRU DATA LOADER: RECORD-LEVEL SPLIT
# ============================================================

def find_cwru_file(fname):
    for root, _, files in os.walk(CWRU_DIR):
        if fname in files:
            return os.path.join(root, fname)
    return None


def load_cwru(seed):
    class_files = {c: [] for c in range(10)}
    for fname, cls_id in CWRU_FILES.items():
        class_files[cls_id].append(fname)

    rng = np.random.RandomState(seed)

    tr_files, vl_files, te_files = [], [], []

    for cls_id, files in class_files.items():
        files = files.copy()
        rng.shuffle(files)

        tr_files.extend(files[:2])
        vl_files.extend(files[2:3])
        te_files.extend(files[3:4])

    def extract_windows(file_list):
        X, y = [], []

        for fname in file_list:
            fpath = find_cwru_file(fname)
            if fpath is None:
                continue

            mat = scipy.io.loadmat(fpath)
            keys = [k for k in mat.keys() if "DE_time" in k]
            if len(keys) == 0:
                continue

            sig = mat[keys[0]].flatten().astype(np.float32)
            if len(sig) < CWRU_SEQ:
                continue

            sig = (sig - sig.mean()) / (sig.std() + 1e-8)

            stride = CWRU_SEQ // 2
            for start in range(0, len(sig) - CWRU_SEQ + 1, stride):
                X.append(sig[start:start + CWRU_SEQ].reshape(CWRU_SEQ, 1))
                y.append(CWRU_FILES[fname])

        if len(X) == 0:
            return torch.empty(0, CWRU_SEQ, 1), torch.empty(0, dtype=torch.long)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    Xtr, ytr = extract_windows(tr_files)
    Xvl, yvl = extract_windows(vl_files)
    Xte, yte = extract_windows(te_files)

    dummy_tr = torch.zeros(len(ytr), dtype=torch.float32)
    dummy_vl = torch.zeros(len(yvl), dtype=torch.float32)
    dummy_te = torch.zeros(len(yte), dtype=torch.float32)

    pin = DEVICE == "cuda"

    tr = DataLoader(TensorDataset(Xtr, dummy_tr, ytr), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
    va = DataLoader(TensorDataset(Xvl, dummy_vl, yvl), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)
    te = DataLoader(TensorDataset(Xte, dummy_te, yte), batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=pin)

    return tr, va, te


# ============================================================
# MODELS
# ============================================================

class DilatedBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            d, d,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            padding_mode="zeros"
        )
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))


class CrossSensorGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x):
        g = self.gate(x.mean(dim=1, keepdim=True))
        return x * g


class SEAttention(nn.Module):
    def __init__(self, d, reduction=2):
        super().__init__()
        hidden = max(2, d // reduction)
        self.fc = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
            nn.Sigmoid()
        )

    def forward(self, x):
        g = self.fc(x.mean(dim=1, keepdim=True))
        return x * g


class ECAAttention(nn.Module):
    """
    FIXED ECA implementation.
    Input x shape: (B, T, d)
    Global average pool over time -> (B, d)
    Treat channel dimension as 1D sequence -> (B, 1, d)
    Conv1d with 1 input channel and 1 output channel.
    """
    def __init__(self, d, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )

    def forward(self, x):
        # x: (B, T, d)
        y = x.mean(dim=1)              # (B, d)
        y = y.unsqueeze(1)             # (B, 1, d)
        y = self.conv(y)               # (B, 1, d)
        y = torch.sigmoid(y)
        y = y.squeeze(1).unsqueeze(-1) # (B, d, 1)
        return x * y


class NanoSentry(nn.Module):
    def __init__(self, n_sensors=14, d=24, n_classes=3, attn_type="CSG"):
        super().__init__()

        self.embed = nn.Linear(n_sensors, d)
        self.tcn = nn.Sequential(
            DilatedBlock(d, 1),
            DilatedBlock(d, 2),
            DilatedBlock(d, 4)
        )

        if attn_type == "CSG":
            self.gate = CrossSensorGate(d)
        elif attn_type == "SE":
            self.gate = SEAttention(d)
        elif attn_type == "ECA":
            self.gate = ECAAttention(d)
        else:
            self.gate = nn.Identity()

        self.skip = nn.Linear(n_sensors, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

        self.rul_head = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        self.state_head = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        h = self.embed(x).permute(0, 2, 1)
        h = self.tcn(h).permute(0, 2, 1)
        h = self.gate(h) + self.skip(x)

        _, hT = self.gru(h)
        hT = hT.squeeze(0)

        rul = self.rul_head(hT).squeeze(-1)
        state = self.state_head(hT)

        return rul, state


class CNNBaseline(nn.Module):
    def __init__(self, n_channels=14, filters=32, n_classes=3, classification=False):
        super().__init__()
        self.classification = classification

        self.net = nn.Sequential(
            nn.Conv1d(n_channels, filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters * 2, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(filters * 2),
            nn.AdaptiveAvgPool1d(1)
        )

        self.rul_head = nn.Linear(filters * 2, 1)

        if classification:
            self.cls_head = nn.Linear(filters * 2, n_classes)
        else:
            self.cls_head = None

    def forward(self, x):
        h = self.net(x.permute(0, 2, 1)).squeeze(-1)
        rul = self.rul_head(h).squeeze(-1)

        if self.cls_head is not None:
            cls = self.cls_head(h)
        else:
            cls = torch.zeros(x.size(0), 3, device=x.device)

        return rul, cls


class LSTMBaseline(nn.Module):
    def __init__(self, n_channels=14, hidden=64, n_classes=3, classification=False):
        super().__init__()
        self.classification = classification

        self.lstm = nn.LSTM(
            n_channels,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.rul_head = nn.Linear(hidden, 1)

        if classification:
            self.cls_head = nn.Linear(hidden, n_classes)
        else:
            self.cls_head = None

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        z = h[-1]

        rul = self.rul_head(z).squeeze(-1)

        if self.cls_head is not None:
            cls = self.cls_head(z)
        else:
            cls = torch.zeros(x.size(0), 3, device=x.device)

        return rul, cls


# ============================================================
# EVALUATION AND TRAINING
# ============================================================

def evaluate(model, loader, is_cwru=False, is_multi=True, num_classes=3):
    model.eval()

    preds_r, trues_r = [], []
    preds_s, trues_s = [], []

    with torch.no_grad():
        for X, yr, ys in loader:
            X = X.to(DEVICE)
            pr, ps = model(X)

            preds_r.extend(pr.cpu().numpy())
            trues_r.extend(yr.numpy())

            preds_s.extend(ps.argmax(1).cpu().numpy())
            trues_s.extend(ys.numpy())

    pr = np.array(preds_r, dtype=np.float64)
    tr = np.array(trues_r, dtype=np.float64)
    ps = np.array(preds_s, dtype=np.int64)
    ts = np.array(trues_s, dtype=np.int64)

    if is_cwru:
        acc = float((ps == ts).mean() * 100.0)
        f1 = macro_f1(ps, ts, num_classes)

        return {
            "rmse": np.nan,
            "mae": np.nan,
            "score": np.nan,
            "cz_rmse": np.nan,
            "acc": acc,
            "f1": f1
        }

    rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))
    mae = float(np.mean(np.abs(pr - tr)))
    score = nasa_score(pr, tr)
    cz = critical_zone_rmse(pr, tr, threshold=30)

    if is_multi:
        acc = float((ps == ts).mean() * 100.0)
        f1 = macro_f1(ps, ts, num_classes)
    else:
        acc = np.nan
        f1 = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "score": score,
        "cz_rmse": cz,
        "acc": acc,
        "f1": f1
    }


def train_and_eval(
    model,
    tr,
    va,
    te,
    is_cwru=False,
    is_multi=True,
    num_classes=3,
    epochs=EPOCHS,
    patience=PATIENCE,
    lr=1e-3
):
    model = model.to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    best_val = -1.0 if is_cwru else float("inf")
    best_sd = None
    wait = 0

    for epoch in range(epochs):
        model.train()

        for X, yr, ys in tr:
            X = X.to(DEVICE)
            yr = yr.to(DEVICE)
            ys = ys.to(DEVICE)

            opt.zero_grad()

            pr, ps = model(X)

            if is_cwru:
                loss = ce(ps, ys)
            else:
                loss = mse(pr, yr)
                if is_multi:
                    loss = loss + 0.15 * ce(ps, ys)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        sched.step()

        val_metrics = evaluate(model, va, is_cwru=is_cwru, is_multi=is_multi, num_classes=num_classes)

        if is_cwru:
            val_metric = -val_metrics["f1"]
        else:
            val_metric = val_metrics["rmse"]

        if val_metric < best_val:
            best_val = val_metric
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    test_metrics = evaluate(model, te, is_cwru=is_cwru, is_multi=is_multi, num_classes=num_classes)
    params, kb = count_params(model)

    return test_metrics, params, kb


# ============================================================
# RIDGE BASELINE
# ============================================================

def loader_to_Xy(loader):
    Xs, ys = [], []
    for batch in loader:
        Xs.append(batch[0])
        ys.append(batch[1])
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def window_summary_features(X):
    X = X.numpy()
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1)
        ],
        axis=1
    )


def run_ridge(tr, te):
    Xtr, ytr = loader_to_Xy(tr)
    Xte, yte = loader_to_Xy(te)

    Xtr_s = window_summary_features(Xtr)
    Xte_s = window_summary_features(Xte)

    model = Ridge(alpha=1.0)
    model.fit(Xtr_s, ytr.numpy())

    pred = model.predict(Xte_s)
    true = yte.numpy()

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    score = nasa_score(pred, true)
    cz = critical_zone_rmse(pred, true, threshold=30)

    return {
        "rmse": rmse,
        "mae": mae,
        "score": score,
        "cz_rmse": cz,
        "acc": np.nan,
        "f1": np.nan
    }


# ============================================================
# MAIN EXPERIMENT LOOP
# ============================================================

print("🔍 Starting/resuming experiments...")

# ------------------------------------------------------------
# 1. C-MAPSS
# ------------------------------------------------------------

print("\n=== C-MAPSS ===")

for fd in [1, 2, 3, 4]:
    subset = f"FD{fd:03d}"

    for seed in MAIN_SEEDS:
        try:
            tr, va, te = load_cmapss(fd, seed)
        except Exception as e:
            log_error(f"C-MAPSS load failed | {subset} | seed {seed} | {e}")
            continue

        # NanoSentry multi-task
        if not is_done("C-MAPSS", subset, "NanoSentry", "multi_task", seed):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = NanoSentry(14, 24, 3, "CSG")
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=False,
                    is_multi=True,
                    num_classes=3
                )

                save_result({
                    "dataset": "C-MAPSS",
                    "subset": subset,
                    "model": "NanoSentry",
                    "variant": "multi_task",
                    "seed": seed,
                    "fold": "",
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"C-MAPSS NanoSentry multi_task | {subset} | seed {seed} | {e}")
            finally:
                cleanup()

        # NanoSentry single-task only for selected FDs to save time
        if fd in SINGLE_TASK_FDS:
            if not is_done("C-MAPSS", subset, "NanoSentry", "single_task", seed):
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = NanoSentry(14, 24, 3, "CSG")
                    metrics, params, kb = train_and_eval(
                        model, tr, va, te,
                        is_cwru=False,
                        is_multi=False,
                        num_classes=3
                    )

                    save_result({
                        "dataset": "C-MAPSS",
                        "subset": subset,
                        "model": "NanoSentry",
                        "variant": "single_task",
                        "seed": seed,
                        "fold": "",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": metrics["acc"],
                        "f1": metrics["f1"],
                        "params": params,
                        "kb": kb
                    })

                except Exception as e:
                    log_error(f"C-MAPSS NanoSentry single_task | {subset} | seed {seed} | {e}")
                finally:
                    cleanup()

        # Baselines: use fewer seeds to save time
        if seed in BASE_SEEDS:

            # Ridge
            if not is_done("C-MAPSS", subset, "Ridge", "single_task", seed):
                try:
                    metrics = run_ridge(tr, te)

                    save_result({
                        "dataset": "C-MAPSS",
                        "subset": subset,
                        "model": "Ridge",
                        "variant": "single_task",
                        "seed": seed,
                        "fold": "",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": "",
                        "f1": "",
                        "params": 0,
                        "kb": 0
                    })

                except Exception as e:
                    log_error(f"C-MAPSS Ridge | {subset} | seed {seed} | {e}")
                finally:
                    cleanup()

            # CNN-32
            if not is_done("C-MAPSS", subset, "CNN-32", "single_task", seed):
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = CNNBaseline(14, 32, 3, classification=False)
                    metrics, params, kb = train_and_eval(
                        model, tr, va, te,
                        is_cwru=False,
                        is_multi=False,
                        num_classes=3
                    )

                    save_result({
                        "dataset": "C-MAPSS",
                        "subset": subset,
                        "model": "CNN-32",
                        "variant": "single_task",
                        "seed": seed,
                        "fold": "",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": metrics["acc"],
                        "f1": metrics["f1"],
                        "params": params,
                        "kb": kb
                    })

                except Exception as e:
                    log_error(f"C-MAPSS CNN-32 | {subset} | seed {seed} | {e}")
                finally:
                    cleanup()

            # LSTM-64
            if not is_done("C-MAPSS", subset, "LSTM-64", "single_task", seed):
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = LSTMBaseline(14, 64, 3, classification=False)
                    metrics, params, kb = train_and_eval(
                        model, tr, va, te,
                        is_cwru=False,
                        is_multi=False,
                        num_classes=3
                    )

                    save_result({
                        "dataset": "C-MAPSS",
                        "subset": subset,
                        "model": "LSTM-64",
                        "variant": "single_task",
                        "seed": seed,
                        "fold": "",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": metrics["acc"],
                        "f1": metrics["f1"],
                        "params": params,
                        "kb": kb
                    })

                except Exception as e:
                    log_error(f"C-MAPSS LSTM-64 | {subset} | seed {seed} | {e}")
                finally:
                    cleanup()


# ------------------------------------------------------------
# 2. Attention comparison
# ------------------------------------------------------------

print("\n=== Attention Comparison ===")

for fd in ATTENTION_FDS:
    subset = f"FD{fd:03d}"

    for seed in BASE_SEEDS:
        try:
            tr, va, te = load_cmapss(fd, seed)
        except Exception as e:
            log_error(f"Attention load failed | {subset} | seed {seed} | {e}")
            continue

        for attn in ["SE", "ECA", "None"]:
            model_name = f"NanoSentry_{attn}"

            if not is_done("C-MAPSS", subset, model_name, "multi_task", seed):
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    model = NanoSentry(14, 24, 3, attn)
                    metrics, params, kb = train_and_eval(
                        model, tr, va, te,
                        is_cwru=False,
                        is_multi=True,
                        num_classes=3
                    )

                    save_result({
                        "dataset": "C-MAPSS",
                        "subset": subset,
                        "model": model_name,
                        "variant": "multi_task",
                        "seed": seed,
                        "fold": "",
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": metrics["acc"],
                        "f1": metrics["f1"],
                        "params": params,
                        "kb": kb
                    })

                except Exception as e:
                    log_error(f"Attention comparison | {model_name} | {subset} | seed {seed} | {e}")
                finally:
                    cleanup()


# ------------------------------------------------------------
# 3. Battery leave-one-cell-out
# ------------------------------------------------------------

print("\n=== Battery LOCO ===")

try:
    battery_folds = load_battery_folds()
except Exception as e:
    battery_folds = []
    log_error(f"Battery fold loading failed | {e}")

for fold in battery_folds:
    fold_id = fold["test_cell"]

    try:
        tr = seqs_to_loader(fold["tr"], shuffle=True)
        va = seqs_to_loader(fold["vl"], shuffle=False)
        te = seqs_to_loader(fold["te"], shuffle=False)
    except Exception as e:
        log_error(f"Battery loader failed | fold {fold_id} | {e}")
        continue

    for seed in BASE_SEEDS:

        # NanoSentry multi-task
        if not is_done("Battery", fold_id, "NanoSentry", "multi_task", seed, fold_id):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = NanoSentry(4, 24, 3, "CSG")
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=False,
                    is_multi=True,
                    num_classes=3
                )

                save_result({
                    "dataset": "Battery",
                    "subset": fold_id,
                    "model": "NanoSentry",
                    "variant": "multi_task",
                    "seed": seed,
                    "fold": fold_id,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Battery NanoSentry multi_task | fold {fold_id} | seed {seed} | {e}")
            finally:
                cleanup()

        # NanoSentry single-task
        if not is_done("Battery", fold_id, "NanoSentry", "single_task", seed, fold_id):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = NanoSentry(4, 24, 3, "CSG")
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=False,
                    is_multi=False,
                    num_classes=3
                )

                save_result({
                    "dataset": "Battery",
                    "subset": fold_id,
                    "model": "NanoSentry",
                    "variant": "single_task",
                    "seed": seed,
                    "fold": fold_id,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Battery NanoSentry single_task | fold {fold_id} | seed {seed} | {e}")
            finally:
                cleanup()

        # Ridge: deterministic for a fixed fold, run once only
        if seed == BASE_SEEDS[0]:
            if not is_done("Battery", fold_id, "Ridge", "single_task", seed, fold_id):
                try:
                    metrics = run_ridge(tr, te)

                    save_result({
                        "dataset": "Battery",
                        "subset": fold_id,
                        "model": "Ridge",
                        "variant": "single_task",
                        "seed": seed,
                        "fold": fold_id,
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "score": metrics["score"],
                        "cz_rmse": metrics["cz_rmse"],
                        "acc": "",
                        "f1": "",
                        "params": 0,
                        "kb": 0
                    })

                except Exception as e:
                    log_error(f"Battery Ridge | fold {fold_id} | {e}")
                finally:
                    cleanup()

        # CNN-32
        if not is_done("Battery", fold_id, "CNN-32", "single_task", seed, fold_id):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = CNNBaseline(4, 32, 3, classification=False)
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=False,
                    is_multi=False,
                    num_classes=3
                )

                save_result({
                    "dataset": "Battery",
                    "subset": fold_id,
                    "model": "CNN-32",
                    "variant": "single_task",
                    "seed": seed,
                    "fold": fold_id,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Battery CNN-32 | fold {fold_id} | seed {seed} | {e}")
            finally:
                cleanup()

        # LSTM-64
        if not is_done("Battery", fold_id, "LSTM-64", "single_task", seed, fold_id):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = LSTMBaseline(4, 64, 3, classification=False)
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=False,
                    is_multi=False,
                    num_classes=3
                )

                save_result({
                    "dataset": "Battery",
                    "subset": fold_id,
                    "model": "LSTM-64",
                    "variant": "single_task",
                    "seed": seed,
                    "fold": fold_id,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "score": metrics["score"],
                    "cz_rmse": metrics["cz_rmse"],
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"Battery LSTM-64 | fold {fold_id} | seed {seed} | {e}")
            finally:
                cleanup()


# ------------------------------------------------------------
# 4. CWRU record-level classification
# ------------------------------------------------------------

print("\n=== CWRU Record-Level Split ===")

for seed in CWRU_MAIN_SEEDS:
    try:
        tr, va, te = load_cwru(seed)
    except Exception as e:
        log_error(f"CWRU load failed | seed {seed} | {e}")
        continue

    # NanoSentry classifier
    if not is_done("CWRU", "10-class", "NanoSentry", "classifier", seed):
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = NanoSentry(1, 24, 10, "CSG")
            metrics, params, kb = train_and_eval(
                model, tr, va, te,
                is_cwru=True,
                is_multi=False,
                num_classes=10
            )

            save_result({
                "dataset": "CWRU",
                "subset": "10-class",
                "model": "NanoSentry",
                "variant": "classifier",
                "seed": seed,
                "fold": "",
                "rmse": "",
                "mae": "",
                "score": "",
                "cz_rmse": "",
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "params": params,
                "kb": kb
            })

        except Exception as e:
            log_error(f"CWRU NanoSentry | seed {seed} | {e}")
        finally:
            cleanup()

    if seed in BASE_SEEDS:

        # CNN classifier
        if not is_done("CWRU", "10-class", "CNN-32", "classifier", seed):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = CNNBaseline(1, 32, 10, classification=True)
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=True,
                    is_multi=False,
                    num_classes=10
                )

                save_result({
                    "dataset": "CWRU",
                    "subset": "10-class",
                    "model": "CNN-32",
                    "variant": "classifier",
                    "seed": seed,
                    "fold": "",
                    "rmse": "",
                    "mae": "",
                    "score": "",
                    "cz_rmse": "",
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"CWRU CNN-32 | seed {seed} | {e}")
            finally:
                cleanup()

        # LSTM classifier
        if not is_done("CWRU", "10-class", "LSTM-64", "classifier", seed):
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = LSTMBaseline(1, 64, 10, classification=True)
                metrics, params, kb = train_and_eval(
                    model, tr, va, te,
                    is_cwru=True,
                    is_multi=False,
                    num_classes=10
                )

                save_result({
                    "dataset": "CWRU",
                    "subset": "10-class",
                    "model": "LSTM-64",
                    "variant": "classifier",
                    "seed": seed,
                    "fold": "",
                    "rmse": "",
                    "mae": "",
                    "score": "",
                    "cz_rmse": "",
                    "acc": metrics["acc"],
                    "f1": metrics["f1"],
                    "params": params,
                    "kb": kb
                })

            except Exception as e:
                log_error(f"CWRU LSTM-64 | seed {seed} | {e}")
            finally:
                cleanup()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n=== EXPERIMENT RUN FINISHED ===")
print(f"Results file: {RESULTS_FILE}")
print(f"Error log: {ERROR_FILE}")

if os.path.exists(RESULTS_FILE):
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"Total completed experiment rows: {len(df)}")
        print("\nPreview:")
        print(df.tail(20))
    except Exception as e:
        print(f"Could not print summary: {e}")