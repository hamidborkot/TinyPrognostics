"""
run_five_seeds.py — 5-seed reproducibility evaluation for NanoSentry
=====================================================================
Usage (Kaggle):
    Set the three DATA_DIR paths below, then run all cells.

Usage (local):
    pip install torch numpy pandas scipy
    python experiments/run_five_seeds.py

Outputs:
    results/five_seeds_raw.csv     — per-seed results (30 rows)
    results/five_seeds_summary.csv — mean ± std per task (6 rows)
"""

# ── SET YOUR DATA PATHS ───────────────────────────────────────────
CMAPSS_DIR  = "/kaggle/input/nasa-cmaps/CMaps"
BATTERY_DIR = "/kaggle/input/nasa-battery-dataset/cleaned_dataset"
CWRU_DIR    = "/kaggle/input/cwru-mat-full-dataset"
# ─────────────────────────────────────────────────────────────────

SEEDS   = [42, 7, 13, 99, 2025]
BATCH   = 128
SEQ     = 64
SENSORS = [6,7,8,11,12,13,15,16,17,18,19,21,24,25]

import os, csv, warnings
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
class DilatedBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=dilation,
                              dilation=dilation, padding_mode="zeros")
        self.bn  = nn.BatchNorm1d(d)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.bn(self.conv(x)))

class CrossSensorGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x.mean(dim=1, keepdim=True))

class NanoSentry(nn.Module):
    def __init__(self, n_sensors=14, d=24, n_classes=3):
        super().__init__()
        self.embed = nn.Linear(n_sensors, d)
        self.tcn   = nn.Sequential(DilatedBlock(d,1), DilatedBlock(d,2), DilatedBlock(d,4))
        self.gate  = CrossSensorGate(d)
        self.skip  = nn.Linear(n_sensors, d)
        self.gru   = nn.GRU(d, d, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        self.rul_head   = nn.Sequential(nn.Linear(d,32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32,1))
        self.state_head = nn.Sequential(nn.Linear(d,32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32,n_classes))
    def forward(self, x):
        h = self.embed(x).permute(0,2,1)
        h = self.tcn(h).permute(0,2,1)
        h = self.gate(h) + self.skip(x)
        _, hT = self.gru(h)
        hT = hT.squeeze(0)
        return self.rul_head(hT).squeeze(-1), self.state_head(hT)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    def size_kb(self):
        return self.param_count() * 4 / 1024


# ══════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════
def load_cmapss(fd, seed):
    train_df = pd.read_csv(f"{CMAPSS_DIR}/train_FD{fd:03d}.txt", sep=r'\s+', header=None)
    test_df  = pd.read_csv(f"{CMAPSS_DIR}/test_FD{fd:03d}.txt",  sep=r'\s+', header=None)
    rul_df   = pd.read_csv(f"{CMAPSS_DIR}/RUL_FD{fd:03d}.txt",   header=None, names=["RUL"])
    max_cycles = train_df.groupby(0)[1].max().rename("max_cycle")
    train_df   = train_df.join(max_cycles, on=0)
    train_df["RUL"]   = (train_df["max_cycle"] - train_df[1]).clip(upper=125)
    train_df["state"] = pd.cut(train_df["RUL"], bins=[-1,25,50,200], labels=[0,1,2]).astype(int)
    feat_mean = train_df[SENSORS].mean()
    feat_std  = train_df[SENSORS].std().replace(0, 1)

    def make_windows(df, eng_list):
        Xs, yr, ys = [], [], []
        for eng in eng_list:
            sub   = df[df[0]==eng].reset_index(drop=True)
            feats = ((sub[SENSORS] - feat_mean) / feat_std).values.astype(np.float32)
            n = len(feats)
            if n < SEQ:
                feats = np.vstack([np.zeros((SEQ-n,14),dtype=np.float32), feats])
                n = SEQ
            for t in range(n-SEQ+1):
                Xs.append(feats[t:t+SEQ])
                yr.append(float(sub.loc[min(t+SEQ-1,len(sub)-1),"RUL"]))
                ys.append(int(sub.loc[min(t+SEQ-1,len(sub)-1),"state"]))
        return (torch.tensor(np.array(Xs)),
                torch.tensor(yr, dtype=torch.float32),
                torch.tensor(ys, dtype=torch.long))

    rng = np.random.RandomState(seed)
    engines = train_df[0].unique()
    idx = rng.permutation(len(engines))
    n_tr = int(len(engines)*0.8)
    Xtr,ytr_r,ytr_s = make_windows(train_df, engines[idx[:n_tr]])
    Xvl,yvl_r,yvl_s = make_windows(train_df, engines[idx[n_tr:]])

    rul_test = rul_df["RUL"].values.astype(np.float32)
    Xte_list,yte_r_list,yte_s_list = [],[],[]
    for i, eng in enumerate(test_df[0].unique()):
        sub   = test_df[test_df[0]==eng].reset_index(drop=True)
        feats = ((sub[SENSORS] - feat_mean) / feat_std).values.astype(np.float32)
        n = len(feats)
        if n < SEQ:
            feats = np.vstack([np.zeros((SEQ-n,14),dtype=np.float32), feats])
        Xte_list.append(feats[-SEQ:])
        rv = rul_test[i]
        yte_r_list.append(float(rv))
        yte_s_list.append(0 if rv>50 else (1 if rv>25 else 2))

    def ml(X,yr,ys,sh=False):
        return DataLoader(TensorDataset(X,yr,ys), batch_size=BATCH,
                          shuffle=sh, num_workers=2, pin_memory=True)
    return (ml(Xtr,ytr_r,ytr_s,True),
            ml(Xvl,yvl_r,yvl_s),
            ml(torch.tensor(np.array(Xte_list)),
               torch.tensor(yte_r_list,dtype=torch.float32),
               torch.tensor(yte_s_list,dtype=torch.long)))


def load_battery(seed):
    meta      = pd.read_csv(os.path.join(BATTERY_DIR,"metadata.csv"))
    discharge = meta[meta["type"]=="discharge"].copy()
    rng = np.random.RandomState(seed)
    all_seqs = []
    for bid in discharge["battery_id"].unique():
        rows = discharge[discharge["battery_id"]==bid].sort_values("start_time").reset_index(drop=True)
        cap_vals = []
        for _,r in rows.iterrows():
            try: cap_vals.append(float(r["Capacity"]))
            except: continue
        n = len(cap_vals)
        if n < SEQ+5: continue
        cap_arr = np.array(cap_vals, dtype=np.float32)
        max_cap = cap_arr[0] if cap_arr[0]>0 else cap_arr.max()
        soh     = np.clip(cap_arr/max_cap, 0.0, 1.0)
        delta   = np.diff(soh, prepend=soh[0])
        cyc_norm= np.arange(n, dtype=np.float32)/n
        feats   = np.stack([soh, delta, cyc_norm, cap_arr/(max_cap+1e-8)], axis=1)
        eol_idx = int(np.where(soh<0.8)[0][0]) if np.any(soh<0.8) else n
        rul     = np.clip((eol_idx-np.arange(n)).astype(np.float32), 0, 200)
        state   = (rul<=50).astype(np.int64)+(rul<=20).astype(np.int64)
        for t in range(n-SEQ+1):
            all_seqs.append((feats[t:t+SEQ], rul[t+SEQ-1], state[t+SEQ-1]))
    idx = rng.permutation(len(all_seqs))
    all_seqs = [all_seqs[i] for i in idx]
    n_total=len(all_seqs); n_tr=int(n_total*0.6); n_vl=int(n_total*0.2)
    def to_loader(seqs, sh=False):
        X  = torch.tensor(np.array([s[0] for s in seqs]))
        yr = torch.tensor([s[1] for s in seqs], dtype=torch.float32)
        ys = torch.tensor([s[2] for s in seqs], dtype=torch.long)
        return DataLoader(TensorDataset(X,yr,ys), batch_size=BATCH,
                          shuffle=sh, num_workers=2, pin_memory=True)
    return (to_loader(all_seqs[:n_tr],True),
            to_loader(all_seqs[n_tr:n_tr+n_vl]),
            to_loader(all_seqs[n_tr+n_vl:]), 4)


CWRU_FILES = {
    "97.mat":0,"98.mat":0,"99.mat":0,"100.mat":0,
    "105.mat":1,"106.mat":1,"107.mat":1,"108.mat":1,
    "169.mat":2,"170.mat":2,"171.mat":2,"172.mat":2,
    "209.mat":3,"210.mat":3,"211.mat":3,"212.mat":3,
    "118.mat":4,"119.mat":4,"120.mat":4,"121.mat":4,
    "185.mat":5,"186.mat":5,"187.mat":5,"188.mat":5,
    "222.mat":6,"223.mat":6,"224.mat":6,"225.mat":6,
    "130.mat":7,"131.mat":7,"132.mat":7,"133.mat":7,
    "197.mat":8,"198.mat":8,"199.mat":8,"200.mat":8,
    "234.mat":9,"235.mat":9,"236.mat":9,"237.mat":9,
}

def load_cwru(seed):
    CWRU_SEQ = 1024
    all_X, all_y = [], []
    for fname, cls_id in CWRU_FILES.items():
        fpath = None
        for root, dirs, files in os.walk(CWRU_DIR):
            if fname in files:
                fpath = os.path.join(root, fname)
                break
        if fpath is None: continue
        mat = scipy.io.loadmat(fpath)
        key = [k for k in mat.keys() if "DE_time" in k]
        if not key: continue
        sig = mat[key[0]].flatten().astype(np.float32)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        step = CWRU_SEQ // 2
        for start in range(0, len(sig)-CWRU_SEQ+1, step):
            all_X.append(sig[start:start+CWRU_SEQ].reshape(CWRU_SEQ,1))
            all_y.append(cls_id)
    idx   = np.random.RandomState(seed).permutation(len(all_X))
    all_X = np.array(all_X)[idx]; all_y = np.array(all_y)[idx]
    n_total=len(all_X); n_tr=int(n_total*0.6); n_vl=int(n_total*0.2)
    def to_loader(X, y, sh=False):
        return DataLoader(
            TensorDataset(torch.tensor(X),
                          torch.zeros(len(y),dtype=torch.float32),
                          torch.tensor(y,dtype=torch.long)),
            batch_size=BATCH, shuffle=sh, num_workers=2, pin_memory=True)
    return (to_loader(all_X[:n_tr],all_y[:n_tr],True),
            to_loader(all_X[n_tr:n_tr+n_vl],all_y[n_tr:n_tr+n_vl]),
            to_loader(all_X[n_tr+n_vl:],all_y[n_tr+n_vl:]))


# ══════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════
def nasa_score(pred, true):
    d = pred - true
    s = np.where(d<0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return float(s.sum()) if np.isfinite(s.sum()) else float("nan")

def critical_zone_rmse(pred, true, threshold=30):
    mask = true <= threshold
    if mask.sum()==0: return float("nan")
    return float(np.sqrt(np.mean((pred[mask]-true[mask])**2)))

def train_epoch(model, loader, optimizer, is_cwru=False):
    model.train()
    mse_fn=nn.MSELoss(); ce_fn=nn.CrossEntropyLoss()
    for X,yr,ys in loader:
        X,yr,ys = X.to(DEVICE),yr.to(DEVICE),ys.to(DEVICE)
        optimizer.zero_grad()
        pred_r,pred_s = model(X)
        loss = ce_fn(pred_s,ys) if is_cwru else mse_fn(pred_r,yr)+0.15*ce_fn(pred_s,ys)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

def evaluate(model, loader, is_cwru=False):
    model.eval()
    preds_r,trues_r,preds_s,trues_s=[],[],[],[]
    with torch.no_grad():
        for X,yr,ys in loader:
            pred_r,pred_s = model(X.to(DEVICE))
            preds_r.extend(pred_r.cpu().numpy())
            trues_r.extend(yr.numpy())
            preds_s.extend(pred_s.argmax(1).cpu().numpy())
            trues_s.extend(ys.numpy())
    pr,tr_=np.array(preds_r),np.array(trues_r)
    ps,ts=np.array(preds_s),np.array(trues_s)
    return {"rmse":float(np.sqrt(np.mean((pr-tr_)**2))),
            "mae":float(np.mean(np.abs(pr-tr_))),
            "acc":float((ps==ts).mean()*100),
            "score":nasa_score(pr,tr_),
            "cz_rmse":critical_zone_rmse(pr,tr_)}

def run_one(model, tr, va, te, epochs=100, lr=1e-3, is_cwru=False, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    best_val,best_state,wait = float("inf"),None,0
    for epoch in range(1,epochs+1):
        train_epoch(model, tr, optimizer, is_cwru=is_cwru)
        m = evaluate(model, va, is_cwru=is_cwru)
        val_metric = -m["acc"] if is_cwru else m["rmse"]
        scheduler.step()
        if val_metric < best_val:
            best_val=val_metric
            best_state={k:v.clone() for k,v in model.state_dict().items()}
            wait=0
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(best_state)
    return evaluate(model, te, is_cwru=is_cwru)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
TASKS = [
    ("FD001","cmapss",1),("FD002","cmapss",2),
    ("FD003","cmapss",3),("FD004","cmapss",4),
    ("Battery","battery",None),("CWRU","cwru",None),
]

all_rows = []

for task_name, task_type, fd in TASKS:
    print(f"\n{'='*55}\nTASK: {task_name}\n{'='*55}")
    task_results = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            if task_type=="cmapss":
                tr,va,te = load_cmapss(fd, seed)
                model = NanoSentry(14,24,3).to(DEVICE); is_cwru=False
            elif task_type=="battery":
                tr,va,te,nf = load_battery(seed)
                model = NanoSentry(nf,24,3).to(DEVICE); is_cwru=False
            else:
                tr,va,te = load_cwru(seed)
                model = NanoSentry(1,24,10).to(DEVICE); is_cwru=True
        except Exception as e:
            print(f"  [ERROR] Seed {seed}: {e}"); continue

        print(f"  Seed {seed:4d} | params={model.param_count():,} | size={model.size_kb():.1f}KB",
              end=" | ", flush=True)
        m = run_one(model, tr, va, te, is_cwru=is_cwru)
        print(f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"Acc={m['acc']:.2f}%  Score={m['score']:.1f}  CZ={m['cz_rmse']:.4f}")
        all_rows.append({"task":task_name,"seed":seed,
                         "params":model.param_count(),**m})
        task_results.append(m)

    if len(task_results)>1:
        for key in ["rmse","mae","acc","score","cz_rmse"]:
            vals=[r[key] for r in task_results
                  if not np.isnan(r.get(key,float("nan")))]
            if vals:
                print(f"  >>> {key:8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# Save
os.makedirs("results", exist_ok=True)
with open("results/five_seeds_raw.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["task","seed","params","rmse","mae","acc","score","cz_rmse"])
    w.writeheader(); w.writerows(all_rows)

tasks_seen = list(dict.fromkeys(r["task"] for r in all_rows))
with open("results/five_seeds_summary.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["task","params","rmse_mean","rmse_std","mae_mean","mae_std",
                "nasa_mean","nasa_std","cz_rmse_mean","cz_rmse_std",
                "acc_mean","acc_std"])
    print("\n"+"="*70)
    print(f"{'Task':<10} {'RMSE':>14} {'MAE':>12} {'CZ-RMSE':>12} {'Acc%':>10} {'Score':>12}")
    print("-"*70)
    for tname in tasks_seen:
        rows=[r for r in all_rows if r["task"]==tname]
        params=rows[0]["params"] if rows else ""
        def ms(key):
            vals=[r[key] for r in rows
                  if not np.isnan(r.get(key,float("nan")))]
            return (np.mean(vals),np.std(vals)) if vals else (float("nan"),float("nan"))
        r_m,r_s=ms("rmse"); a_m,a_s=ms("mae")
        c_m,c_s=ms("cz_rmse"); ac_m,ac_s=ms("acc"); sc_m,sc_s=ms("score")
        w.writerow([tname,params,f"{r_m:.4f}",f"{r_s:.4f}",f"{a_m:.4f}",f"{a_s:.4f}",
                    f"{sc_m:.1f}",f"{sc_s:.1f}",f"{c_m:.4f}",f"{c_s:.4f}",
                    f"{ac_m:.2f}",f"{ac_s:.2f}"])
        print(f"{tname:<10} {r_m:.2f}±{r_s:.2f}   {a_m:.2f}±{a_s:.2f}   "
              f"{c_m:.2f}±{c_s:.2f}   {ac_m:.2f}±{ac_s:.2f}   {sc_m:.0f}±{sc_s:.0f}")

print("\n✓ Saved: results/five_seeds_raw.csv")
print("✓ Saved: results/five_seeds_summary.csv")
