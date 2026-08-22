# ============================================================
# Battery Final Rescue — Safe LOCO Protocol
# ============================================================

import os
import csv
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Battery rescue device: {DEVICE}")

# ============================================================
# Config
# ============================================================

BATTERY_DIR = "/kaggle/input/datasets/patrickfleith/nasa-battery-dataset/cleaned_dataset"

SEQ = 64
BATCH = 64
H = 200.0
SEED = 42
EPOCHS = 150
PATIENCE = 30

RESULTS_FILE = "/kaggle/working/battery_final_results.csv"

FIELDNAMES = [
    "fold",
    "model",
    "variant",
    "rmse",
    "mae",
    "acc",
    "f1",
    "params",
    "kb"
]


# ============================================================
# Metrics
# ============================================================

def macro_f1(pred, true, num_classes=3):
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


def count_params(model):
    params = sum(p.numel() for p in model.parameters())
    kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0
    return params, kb


# ============================================================
# Models
# ============================================================

class DilatedBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()

        self.conv = nn.Conv1d(
            d,
            d,
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


class NanoSentry(nn.Module):
    def __init__(self, n_sensors=4, d=24, n_classes=3):
        super().__init__()

        self.embed = nn.Linear(n_sensors, d)

        self.tcn = nn.Sequential(
            DilatedBlock(d, 1),
            DilatedBlock(d, 2),
            DilatedBlock(d, 4)
        )

        self.gate = CrossSensorGate(d)
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
    def __init__(self, n_channels=4, filters=32, n_classes=3):
        super().__init__()

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

        self.state_head = nn.Sequential(
            nn.Linear(filters * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        h = self.net(x.permute(0, 2, 1)).squeeze(-1)

        rul = self.rul_head(h).squeeze(-1)
        state = self.state_head(h)

        return rul, state


class LSTMBaseline(nn.Module):
    def __init__(self, n_channels=4, hidden=64, n_classes=3):
        super().__init__()

        self.lstm = nn.LSTM(
            n_channels,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.rul_head = nn.Linear(hidden, 1)

        self.state_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)

        z = h[-1]

        rul = self.rul_head(z).squeeze(-1)
        state = self.state_head(z)

        return rul, state


# ============================================================
# Build battery cell sequences
# ============================================================

def build_cells():
    meta = pd.read_csv(os.path.join(BATTERY_DIR, "metadata.csv"))
    discharge = meta[meta["type"] == "discharge"].copy()

    batteries = sorted(discharge["battery_id"].unique())
    cells = {}

    for bid in batteries:
        rows = discharge[discharge["battery_id"] == bid].sort_values("start_time")

        cap_vals = pd.to_numeric(rows["Capacity"], errors="coerce").dropna().values
        cap_vals = cap_vals.astype(np.float32)

        n = len(cap_vals)

        if n < SEQ + 10:
            continue

        # Light smoothing only
        cap_smooth = (
            pd.Series(cap_vals)
            .rolling(window=3, center=True, min_periods=1)
            .mean()
            .values
        )

        max_cap = float(cap_smooth.max())

        if max_cap <= 0:
            continue

        soh = np.clip(cap_smooth / max_cap, 0.0, 1.0)

        # Safer EOL detection using a smoothed SOH curve
        soh_eval = (
            pd.Series(soh)
            .rolling(window=5, center=True, min_periods=1)
            .mean()
            .values
        )

        below = np.where(soh_eval < 0.8)[0]

        has_eol = len(below) > 0

        if has_eol:
            eol_idx = int(below[0])
        else:
            eol_idx = n

        rul = np.clip(eol_idx - np.arange(n), 0, H).astype(np.float32)

        delta = np.diff(soh, prepend=soh[0])
        cyc = np.clip(np.arange(n, dtype=np.float32) / H, 0.0, 1.0)

        feats = np.stack(
            [
                soh,
                delta,
                cyc,
                cap_smooth / max_cap
            ],
            axis=1
        ).astype(np.float32)

        seqs = []

        for t in range(n - SEQ + 1):
            idx = t + SEQ - 1

            rul_val = float(rul[idx])

            if rul_val > 60:
                state = 0
            elif rul_val > 30:
                state = 1
            else:
                state = 2

            seqs.append(
                (
                    feats[t:t + SEQ],
                    rul_val,
                    int(state)
                )
            )

        if len(seqs) > 0:
            cells[str(bid)] = {
                "seqs": seqs,
                "has_eol": has_eol
            }

    return cells


cells = build_cells()

all_cell_ids = list(cells.keys())
eol_cell_ids = [c for c in all_cell_ids if cells[c]["has_eol"]]

# If enough cells reached EOL, use only those.
# Otherwise, use all cells.
if len(eol_cell_ids) >= 8:
    use_cells = eol_cell_ids
    print(f"Using only cells with observed EOL: {len(use_cells)} cells")
else:
    use_cells = all_cell_ids
    print(f"Not enough EOL cells. Using all available cells: {len(use_cells)} cells")

print(f"Total cells found: {len(all_cell_ids)}")
print(f"Cells with EOL: {len(eol_cell_ids)}")
print(f"Cells used: {use_cells}")


# ============================================================
# Fold loaders
# ============================================================

def make_fold_loaders(test_cell):
    train_cells = [c for c in use_cells if c != test_cell]

    tr_seqs = []
    vl_seqs = []

    for c in train_cells:
        seqs = cells[c]["seqs"]
        split_idx = int(len(seqs) * 0.8)

        tr_seqs.extend(seqs[:split_idx])
        vl_seqs.extend(seqs[split_idx:])

    te_seqs = cells[test_cell]["seqs"]

    if len(tr_seqs) < 20 or len(vl_seqs) < 5 or len(te_seqs) < 5:
        return None

    # Feature scaler fitted only on training sequences
    Xtr_raw = np.array([s[0] for s in tr_seqs], dtype=np.float32)

    feat_scaler = StandardScaler()
    feat_scaler.fit(Xtr_raw.reshape(-1, Xtr_raw.shape[-1]))

    def transform(seqs):
        X = np.array([s[0] for s in seqs], dtype=np.float32)
        X = feat_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        yr = np.array([s[1] for s in seqs], dtype=np.float32)
        ys = np.array([s[2] for s in seqs], dtype=np.int64)

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(yr, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long)
        )

    Xtr, ytr, ys_tr = transform(tr_seqs)
    Xvl, yvl, ys_vl = transform(vl_seqs)
    Xte, yte, ys_te = transform(te_seqs)

    y_mean = float(ytr.mean())
    y_std = float(ytr.std() + 1e-8)

    counts = np.bincount(ys_tr.numpy(), minlength=3).astype(np.float32)
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * 3.0

    class_weight = torch.tensor(weights, dtype=torch.float32)

    def dl(X, yr, ys, shuffle):
        return DataLoader(
            TensorDataset(X, yr, ys),
            batch_size=min(BATCH, len(X)),
            shuffle=shuffle,
            num_workers=0
        )

    return {
        "tr": dl(Xtr, ytr, ys_tr, True),
        "vl": dl(Xvl, yvl, ys_vl, False),
        "te": dl(Xte, yte, ys_te, False),
        "y_mean": y_mean,
        "y_std": y_std,
        "class_weight": class_weight
    }


# ============================================================
# Evaluation and training
# ============================================================

def evaluate_net(model, loader, y_mean, y_std):
    model.eval()

    preds = []
    trues = []
    preds_s = []
    trues_s = []

    with torch.no_grad():
        for X, yr, ys in loader:
            X = X.to(DEVICE)

            pr, ps = model(X)

            pr = pr.cpu().numpy() * y_std + y_mean

            preds.extend(pr)
            trues.extend(yr.numpy())

            preds_s.extend(ps.argmax(1).cpu().numpy())
            trues_s.extend(ys.numpy())

    pred = np.array(preds, dtype=np.float64)
    true = np.array(trues, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))

    pred_s = np.array(preds_s)
    true_s = np.array(trues_s)

    acc = float((pred_s == true_s).mean() * 100.0)
    f1 = macro_f1(pred_s, true_s, num_classes=3)

    return {
        "rmse": rmse,
        "mae": mae,
        "acc": acc,
        "f1": f1
    }


def train_net(model, fold, multi=False):
    model = model.to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    mse = nn.MSELoss()

    if multi:
        ce = nn.CrossEntropyLoss(weight=fold["class_weight"].to(DEVICE))
    else:
        ce = None

    y_mean = fold["y_mean"]
    y_std = fold["y_std"]

    best_val = float("inf")
    best_sd = None
    wait = 0

    for epoch in range(EPOCHS):
        model.train()

        for X, yr, ys in fold["tr"]:
            X = X.to(DEVICE)
            yr = yr.to(DEVICE)
            ys = ys.to(DEVICE)

            yr_scaled = (yr - y_mean) / y_std

            opt.zero_grad()

            pr, ps = model(X)

            loss = mse(pr, yr_scaled)

            if multi:
                loss = loss + 0.15 * ce(ps, ys)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()

        sched.step()

        val_metrics = evaluate_net(model, fold["vl"], y_mean, y_std)
        val_rmse = val_metrics["rmse"]

        if not np.isfinite(val_rmse):
            val_rmse = float("inf")

        if val_rmse < best_val:
            best_val = val_rmse
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)

    test_metrics = evaluate_net(model, fold["te"], y_mean, y_std)

    return test_metrics


def loader_to_Xy(loader):
    Xs = []
    ys = []

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


def run_ridge(fold):
    Xtr, ytr = loader_to_Xy(fold["tr"])
    Xte, yte = loader_to_Xy(fold["te"])

    Xtr_s = window_summary_features(Xtr)
    Xte_s = window_summary_features(Xte)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_s)
    Xte_s = scaler.transform(Xte_s)

    y_mean = fold["y_mean"]
    y_std = fold["y_std"]

    ytr_scaled = (ytr.numpy() - y_mean) / y_std

    model = Ridge(alpha=1.0)
    model.fit(Xtr_s, ytr_scaled)

    pred_scaled = model.predict(Xte_s)
    pred = pred_scaled * y_std + y_mean
    true = yte.numpy()

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))

    return {
        "rmse": rmse,
        "mae": mae,
        "acc": np.nan,
        "f1": np.nan
    }


# ============================================================
# Run all battery folds
# ============================================================

rows = []

for test_cell in use_cells:
    print(f"\nBattery fold: {test_cell}")

    fold = make_fold_loaders(test_cell)

    if fold is None:
        print(f"Skipping {test_cell}: not enough sequences.")
        continue

    # NanoSentry multi-task
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = NanoSentry(n_sensors=4, d=24, n_classes=3)
    metrics = train_net(model, fold, multi=True)
    params, kb = count_params(model)

    rows.append({
        "fold": test_cell,
        "model": "NanoSentry",
        "variant": "multi_task",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "params": params,
        "kb": kb
    })

    print(
        f"  NanoSentry multi_task | "
        f"RMSE {metrics['rmse']:.4f} | "
        f"MAE {metrics['mae']:.4f} | "
        f"Acc {metrics['acc']:.2f}%"
    )

    # NanoSentry single-task
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = NanoSentry(n_sensors=4, d=24, n_classes=3)
    metrics = train_net(model, fold, multi=False)
    params, kb = count_params(model)

    rows.append({
        "fold": test_cell,
        "model": "NanoSentry",
        "variant": "single_task",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "acc": "",
        "f1": "",
        "params": params,
        "kb": kb
    })

    print(
        f"  NanoSentry single_task | "
        f"RMSE {metrics['rmse']:.4f} | "
        f"MAE {metrics['mae']:.4f}"
    )

    # Ridge
    metrics = run_ridge(fold)

    rows.append({
        "fold": test_cell,
        "model": "Ridge",
        "variant": "single_task",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "acc": "",
        "f1": "",
        "params": 0,
        "kb": 0
    })

    print(
        f"  Ridge | "
        f"RMSE {metrics['rmse']:.4f} | "
        f"MAE {metrics['mae']:.4f}"
    )

    # CNN-32
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = CNNBaseline(n_channels=4, filters=32, n_classes=3)
    metrics = train_net(model, fold, multi=False)
    params, kb = count_params(model)

    rows.append({
        "fold": test_cell,
        "model": "CNN-32",
        "variant": "single_task",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "acc": "",
        "f1": "",
        "params": params,
        "kb": kb
    })

    print(
        f"  CNN-32 | "
        f"RMSE {metrics['rmse']:.4f} | "
        f"MAE {metrics['mae']:.4f}"
    )

    # LSTM-64
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = LSTMBaseline(n_channels=4, hidden=64, n_classes=3)
    metrics = train_net(model, fold, multi=False)
    params, kb = count_params(model)

    rows.append({
        "fold": test_cell,
        "model": "LSTM-64",
        "variant": "single_task",
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "acc": "",
        "f1": "",
        "params": params,
        "kb": kb
    })

    print(
        f"  LSTM-64 | "
        f"RMSE {metrics['rmse']:.4f} | "
        f"MAE {metrics['mae']:.4f}"
    )


# ============================================================
# Save and summarize
# ============================================================

with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

df = pd.DataFrame(rows)

for c in ["rmse", "mae", "acc", "f1"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n" + "=" * 70)
print("BATTERY FINAL RESCUE SUMMARY")
print("=" * 70)

summary = (
    df.groupby("model")
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        rmse_median=("rmse", "median"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std")
    )
    .reset_index()
)

print(summary)

print("\nPer-fold NanoSentry RMSE:")
nano_pivot = df[df["model"] == "NanoSentry"].pivot_table(
    index="fold",
    columns="variant",
    values="rmse"
)

print(nano_pivot)

print(f"\nSaved battery results to: {RESULTS_FILE}")