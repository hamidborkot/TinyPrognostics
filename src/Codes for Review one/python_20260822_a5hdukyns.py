import os, csv, time, warnings, scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running on: {DEVICE}")

# ==========================================================
# 1. CHECKPOINT SYSTEM (Disconnect-Proof)
# ==========================================================
RESULTS_FILE = '/kaggle/working/master_results.csv'
FIELDNAMES = ['dataset', 'subset', 'model', 'variant', 'seed', 'rmse', 'mae', 'score', 'cz_rmse', 'acc', 'params', 'kb']

def is_done(dataset, subset, model, variant, seed):
    if not os.path.exists(RESULTS_FILE): return False
    df = pd.read_csv(RESULTS_FILE)
    return len(df[(df['dataset']==dataset) & (df['subset']==subset) & 
                  (df['model']==model) & (df['variant']==variant) & (df['seed']==seed)]) > 0

def save_result(row):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists: writer.writeheader()
        writer.writerow(row)
    print(f"✅ Saved: {row['dataset']} | {row['subset']} | {row['model']} | {row['variant']} | Seed {row['seed']}")

# ==========================================================
# 2. LEAKAGE-FREE DATA LOADERS
# ==========================================================
CMAPSS_DIR = "/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps"
BATTERY_DIR = "/kaggle/input/datasets/patrickfleith/nasa-battery-dataset/cleaned_dataset"
CWRU_DIR = "/kaggle/input/datasets/sufian79/cwru-mat-full-dataset"

SENSORS = [6,7,8,11,12,13,15,16,17,18,19,21,24,25]
SEQ = 64
BATCH = 256

def load_cmapss(fd, seed):
    train_df = pd.read_csv(f"{CMAPSS_DIR}/train_FD{fd:03d}.txt", sep=r'\s+', header=None)
    test_df  = pd.read_csv(f"{CMAPSS_DIR}/test_FD{fd:03d}.txt",  sep=r'\s+', header=None)
    rul_df   = pd.read_csv(f"{CMAPSS_DIR}/RUL_FD{fd:03d}.txt",   header=None, names=["RUL"])
    
    max_cycles = train_df.groupby(0)[1].max().rename("max_cycle")
    train_df = train_df.join(max_cycles, on=0)
    train_df["RUL"] = (train_df["max_cycle"] - train_df[1]).clip(upper=125)
    
    # Condition-aware normalization for FD002/004
    if fd in [1, 3]:
        mu, std = train_df[SENSORS].mean(), train_df[SENSORS].std().replace(0, 1)
        train_df[SENSORS] = (train_df[SENSORS] - mu) / std
        test_df[SENSORS] = (test_df[SENSORS] - mu) / std
    else:
        km = KMeans(n_clusters=6, random_state=seed, n_init=10)
        km.fit(train_df[[2,3,4]].values)
        tr_lbls, te_lbls = km.predict(train_df[[2,3,4]].values), km.predict(test_df[[2,3,4]].values)
        for c in range(6):
            mask = tr_lbls == c
            if mask.sum() < 2: continue
            mu, std = train_df.loc[mask, SENSORS].mean(), train_df.loc[mask, SENSORS].std().replace(0, 1)
            train_df.loc[mask, SENSORS] = (train_df.loc[mask, SENSORS] - mu) / std
            te_mask = te_lbls == c
            if te_mask.sum() > 0: test_df.loc[te_mask, SENSORS] = (test_df.loc[te_mask, SENSORS] - mu) / std

    def make_windows(df, eng_list):
        Xs, yr, ys = [], [], []
        for eng in eng_list:
            sub = df[df[0]==eng].reset_index(drop=True)
            feats = sub[SENSORS].values.astype(np.float32)
            if len(feats) < SEQ: feats = np.vstack([np.zeros((SEQ-len(feats),14),dtype=np.float32), feats])
            for t in range(len(feats) - SEQ + 1):
                Xs.append(feats[t:t+SEQ])
                rul_val = sub.loc[min(t+SEQ-1, len(sub)-1), "RUL"]
                yr.append(float(rul_val))
                ys.append(0 if rul_val>50 else (1 if rul_val>25 else 2))
        return torch.tensor(np.array(Xs)), torch.tensor(yr,dtype=torch.float32), torch.tensor(ys,dtype=torch.long)

    rng = np.random.RandomState(seed)
    engines = train_df[0].unique()
    idx = rng.permutation(len(engines))
    n_tr = int(len(engines)*0.8)
    Xtr, ytr_r, ytr_s = make_windows(train_df, engines[idx[:n_tr]])
    Xvl, yvl_r, yvl_s = make_windows(train_df, engines[idx[n_tr:]])
    
    test_engines = test_df[0].unique()
    Xte_list, yte_r_list, yte_s_list = [], [], []
    for i, eng in enumerate(test_engines):
        sub = test_df[test_df[0]==eng].reset_index(drop=True)
        feats = sub[SENSORS].values.astype(np.float32)
        if len(feats) < SEQ: feats = np.vstack([np.zeros((SEQ-len(feats),14),dtype=np.float32), feats])
        Xte_list.append(feats[-SEQ:])
        rv = rul_df["RUL"].values.astype(np.float32)[i]
        yte_r_list.append(float(rv))
        yte_s_list.append(0 if rv>50 else (1 if rv>25 else 2))
        
    Xte = torch.tensor(np.array(Xte_list))
    yte_r = torch.tensor(yte_r_list, dtype=torch.float32)
    yte_s = torch.tensor(yte_s_list, dtype=torch.long)
    
    dl = lambda X,yr,ys,sh: DataLoader(TensorDataset(X,yr,ys), batch_size=BATCH, shuffle=sh, num_workers=0)
    return dl(Xtr,ytr_r,ytr_s,True), dl(Xvl,yvl_r,yvl_s,False), dl(Xte,yte_r,yte_s,False)

def load_battery(seed):
    meta = pd.read_csv(os.path.join(BATTERY_DIR, "metadata.csv"))
    discharge = meta[meta["type"]=="discharge"].copy()
    batteries = discharge["battery_id"].unique()
    
    # Leave-One-Cell-Out (LOCO) Folds
    folds = []
    for test_cell in batteries:
        train_cells = [b for b in batteries if b != test_cell]
        tr_seqs, vl_seqs, te_seqs = [], [], []
        
        # Process Train/Val Cells
        for bid in train_cells:
            rows = discharge[discharge["battery_id"]==bid].sort_values("start_time").reset_index(drop=True)
            cap_vals = [float(r['Capacity']) for _, r in rows.iterrows() if 'Capacity' in r and pd.notnull(r['Capacity'])]
            if len(cap_vals) < SEQ + 5: continue
            cap_arr = np.array(cap_vals, dtype=np.float32)
            max_cap = cap_arr[0] if cap_arr[0]>0 else cap_arr.max()
            soh = np.clip(cap_arr/max_cap, 0.0, 1.0)
            delta = np.diff(soh, prepend=soh[0])
            # REVIEWER FIX: t/H instead of t/n
            cyc_norm = np.arange(len(cap_vals), dtype=np.float32) / 200.0 
            feats = np.stack([soh, delta, cyc_norm, cap_arr/(max_cap+1e-8)], axis=1)
            eol_idx = int(np.where(soh<0.8)[0][0]) if np.any(soh<0.8) else len(cap_vals)
            rul = np.clip((eol_idx-np.arange(len(cap_vals))).astype(np.float32),0,200)
            state = (rul<=50).astype(np.int64)+(rul<=20).astype(np.int64)
            
            cell_seqs = [(feats[t:t+SEQ], rul[t+SEQ-1], state[t+SEQ-1]) for t in range(len(feats)-SEQ+1)]
            split_idx = int(len(cell_seqs) * 0.8)
            tr_seqs.extend(cell_seqs[:split_idx])
            vl_seqs.extend(cell_seqs[split_idx:])
            
        # Process Test Cell
        rows = discharge[discharge["battery_id"]==test_cell].sort_values("start_time").reset_index(drop=True)
        cap_vals = [float(r['Capacity']) for _, r in rows.iterrows() if 'Capacity' in r and pd.notnull(r['Capacity'])]
        if len(cap_vals) >= SEQ + 5:
            cap_arr = np.array(cap_vals, dtype=np.float32)
            max_cap = cap_arr[0] if cap_arr[0]>0 else cap_arr.max()
            soh = np.clip(cap_arr/max_cap, 0.0, 1.0)
            delta = np.diff(soh, prepend=soh[0])
            cyc_norm = np.arange(len(cap_vals), dtype=np.float32) / 200.0
            feats = np.stack([soh, delta, cyc_norm, cap_arr/(max_cap+1e-8)], axis=1)
            eol_idx = int(np.where(soh<0.8)[0][0]) if np.any(soh<0.8) else len(cap_vals)
            rul = np.clip((eol_idx-np.arange(len(cap_vals))).astype(np.float32),0,200)
            state = (rul<=50).astype(np.int64)+(rul<=20).astype(np.int64)
            te_seqs = [(feats[t:t+SEQ], rul[t+SEQ-1], state[t+SEQ-1]) for t in range(len(feats)-SEQ+1)]
        else:
            te_seqs = []
            
        if len(tr_seqs) > 0 and len(te_seqs) > 0:
            folds.append((test_cell, tr_seqs, vl_seqs, te_seqs))
            
    # Select fold based on seed
    fold_idx = seed % len(folds)
    test_cell, tr_seqs, vl_seqs, te_seqs = folds[fold_idx]
    
    rng = np.random.RandomState(seed)
    rng.shuffle(tr_seqs)
    
    dl = lambda seqs, sh: DataLoader(TensorDataset(
        torch.tensor(np.array([s[0] for s in seqs])),
        torch.tensor([s[1] for s in seqs], dtype=torch.float32),
        torch.tensor([s[2] for s in seqs], dtype=torch.long)
    ), batch_size=128, shuffle=sh, num_workers=0)
    
    return dl(tr_seqs, True), dl(vl_seqs, False), dl(te_seqs, False), 4, test_cell

def load_cwru(seed):
    CWRU_FILES = {
        '97.mat':0,'98.mat':0,'99.mat':0,'100.mat':0, '105.mat':1,'106.mat':1,'107.mat':1,'108.mat':1,
        '169.mat':2,'170.mat':2,'171.mat':2,'172.mat':2, '209.mat':3,'210.mat':3,'211.mat':3,'212.mat':3,
        '118.mat':4,'119.mat':4,'120.mat':4,'121.mat':4, '185.mat':5,'186.mat':5,'187.mat':5,'188.mat':5,
        '222.mat':6,'223.mat':6,'224.mat':6,'225.mat':6, '130.mat':7,'131.mat':7,'132.mat':7,'133.mat':7,
        '197.mat':8,'198.mat':8,'199.mat':8,'200.mat':8, '234.mat':9,'235.mat':9,'236.mat':9,'237.mat':9,
    }
    
    # REVIEWER FIX: Record-Level Split (No overlapping window leakage)
    classes = {}
    for f, c in CWRU_FILES.items():
        if c not in classes: classes[c] = []
        classes[c].append(f)
        
    tr_files, vl_files, te_files = [], [], []
    rng = np.random.RandomState(seed)
    for c, files in classes.items():
        rng.shuffle(files)
        tr_files.extend(files[:2])
        vl_files.extend(files[2:3])
        te_files.extend(files[3:])
        
    def extract_windows(file_list):
        X, y = [], []
        for fname in file_list:
            fpath = os.path.join(CWRU_DIR, fname)
            if not os.path.exists(fpath): continue
            mat = scipy.io.loadmat(fpath)
            key = [k for k in mat.keys() if "DE_time" in k][0]
            sig = mat[key].flatten().astype(np.float32)
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
            for start in range(0, len(sig)-1024+1, 512):
                X.append(sig[start:start+1024].reshape(1024, 1))
                y.append(CWRU_FILES[fname])
        return torch.tensor(np.array(X)), torch.tensor(y, dtype=torch.long)
        
    Xtr, ytr = extract_windows(tr_files)
    Xvl, yvl = extract_windows(vl_files)
    Xte, yte = extract_windows(te_files)
    
    dummy_rul = torch.zeros(len(ytr), dtype=torch.float32)
    dl = lambda X, y, sh: DataLoader(TensorDataset(X, dummy_rul[:len(X)], y), batch_size=128, shuffle=sh, num_workers=0)
    return dl(Xtr, ytr, True), dl(Xvl, yvl, False), dl(Xte, yte, False)

# ==========================================================
# 3. MODELS (NanoSentry, SE, ECA, CNN, LSTM)
# ==========================================================
class DilatedBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=dilation, dilation=dilation, padding_mode="zeros")
        self.bn, self.act = nn.BatchNorm1d(d), nn.GELU()
    def forward(self, x): return x + self.act(self.bn(self.conv(x)))

class CrossSensorGate(nn.Module):
    def __init__(self, d): super().__init__(); self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
    def forward(self, x): return x * self.gate(x.mean(dim=1, keepdim=True))

class SEAttention(nn.Module):
    def __init__(self, d): 
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, d), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x.mean(dim=1, keepdim=True))

class ECAAttention(nn.Module):
    def __init__(self, d): 
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    def forward(self, x): 
        y = self.pool(x.permute(0,2,1)).permute(0,2,1)
        return x * torch.sigmoid(self.conv(y.permute(0,2,1)).permute(0,2,1))

class NanoSentry(nn.Module):
    def __init__(self, n_sensors=14, d=24, n_classes=3, attn_type='CSG'):
        super().__init__()
        self.embed = nn.Linear(n_sensors, d)
        self.tcn = nn.Sequential(DilatedBlock(d,1), DilatedBlock(d,2), DilatedBlock(d,4))
        if attn_type == 'CSG': self.gate = CrossSensorGate(d)
        elif attn_type == 'SE': self.gate = SEAttention(d)
        elif attn_type == 'ECA': self.gate = ECAAttention(d)
        else: self.gate = nn.Identity()
        self.skip = nn.Linear(n_sensors, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        self.rul_head = nn.Sequential(nn.Linear(d,32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32,1))
        self.state_head = nn.Sequential(nn.Linear(d,32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32,n_classes))
    def forward(self, x):
        h = self.tcn(self.embed(x).permute(0,2,1)).permute(0,2,1)
        h = self.gate(h) + self.skip(x)
        _, hT = self.gru(h)
        hT = hT.squeeze(0)
        return self.rul_head(hT).squeeze(-1), self.state_head(hT)

class BaselineCNN(nn.Module):
    def __init__(self, n_sensors=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_sensors, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(64, 1)
    def forward(self, x): return self.head(self.net(x.permute(0,2,1)).squeeze(-1)).squeeze(-1), torch.zeros(x.size(0), 3).to(x.device)

class BaselineLSTM(nn.Module):
    def __init__(self, n_sensors=14):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, 64, 2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(64, 1)
    def forward(self, x): _, (h, _) = self.lstm(x); return self.head(h[-1]).squeeze(-1), torch.zeros(x.size(0), 3).to(x.device)

# ==========================================================
# 4. TRAINING & EVALUATION ENGINE
# ==========================================================
def nasa_score(p, t):
    d = p - t
    s = np.where(d<0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return float(s.sum()) if np.isfinite(s.sum()) else 0.0

def train_and_eval(model, tr, va, te, is_cwru=False, is_multi=True, lr=1e-3, epochs=100, patience=20):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    best_val, best_sd, wait = float('inf') if not is_cwru else -1.0, None, 0
    
    for _ in range(epochs):
        model.train()
        for X, yr, ys in tr:
            X, yr, ys = X.to(DEVICE), yr.to(DEVICE), ys.to(DEVICE)
            opt.zero_grad()
            pr, ps = model(X)
            if is_cwru: loss = ce(ps, ys)
            else: loss = mse(pr, yr) + (0.15 * ce(ps, ys) if is_multi else 0.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        preds_r, trues_r, preds_s, trues_s = [], [], [], []
        with torch.no_grad():
            for X, yr, ys in va:
                pr, ps = model(X.to(DEVICE))
                if not is_cwru: preds_r.extend(pr.cpu().numpy()); trues_r.extend(yr.numpy())
                preds_s.extend(ps.argmax(1).cpu().numpy()); trues_s.extend(ys.numpy())
        
        if is_cwru:
            val_metric = float((np.array(preds_s)==np.array(trues_s)).mean())
            if val_metric > best_val: best_val, best_sd, wait = val_metric, {k:v.clone() for k,v in model.state_dict().items()}, 0
            else: wait += 1
        else:
            val_metric = float(np.sqrt(np.mean((np.array(preds_r)-np.array(trues_r))**2)))
            if val_metric < best_val: best_val, best_sd, wait = val_metric, {k:v.clone() for k,v in model.state_dict().items()}, 0
            else: wait += 1
        if wait >= patience: break
        
    model.load_state_dict(best_sd)
    model.eval()
    preds_r, trues_r, preds_s, trues_s = [], [], [], []
    with torch.no_grad():
        for X, yr, ys in te:
            pr, ps = model(X.to(DEVICE))
            if not is_cwru: preds_r.extend(pr.cpu().numpy()); trues_r.extend(yr.numpy())
            preds_s.extend(ps.argmax(1).cpu().numpy()); trues_s.extend(ys.numpy())
            
    pr, tr_ = np.array(preds_r), np.array(trues_r)
    ps, ts = np.array(preds_s), np.array(trues_s)
    
    rmse = float(np.sqrt(np.mean((pr-tr_)**2))) if not is_cwru else 0.0
    mae = float(np.mean(np.abs(pr-tr_))) if not is_cwru else 0.0
    score = nasa_score(pr, tr_) if not is_cwru else 0.0
    cz_mask = tr_ <= 30
    cz_rmse = float(np.sqrt(np.mean((pr[cz_mask]-tr_[cz_mask])**2))) if cz_mask.sum()>0 and not is_cwru else 0.0
    acc = float((ps==ts).mean()*100)
    
    params = sum(p.numel() for p in model.parameters())
    kb = sum(p.numel()*p.element_size() for p in model.parameters())/1024
    return rmse, mae, score, cz_rmse, acc, params, kb

# ==========================================================
# 5. MASTER EXECUTION LOOP
# ==========================================================
SEEDS = [42, 7, 13, 99, 2025]

print("🔍 Scanning for already completed experiments...")
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    
    # --- C-MAPSS ---
    for fd in [1, 2, 3, 4]:
        tr, va, te = load_cmapss(fd, seed)
        subset = f"FD00{fd}"
        
        # NanoSentry Multi-Task
        if not is_done("C-MAPSS", subset, "NanoSentry", "multi_task", seed):
            m = NanoSentry(14, 24, 3, 'CSG').to(DEVICE)
            rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=True)
            save_result({'dataset':'C-MAPSS', 'subset':subset, 'model':'NanoSentry', 'variant':'multi_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})
            
        # NanoSentry Single-Task
        if not is_done("C-MAPSS", subset, "NanoSentry", "single_task", seed):
            m = NanoSentry(14, 24, 3, 'CSG').to(DEVICE)
            rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=False)
            save_result({'dataset':'C-MAPSS', 'subset':subset, 'model':'NanoSentry', 'variant':'single_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})

        # Baselines (3 seeds to save time)
        if seed in [42, 7, 13]:
            if not is_done("C-MAPSS", subset, "CNN-32", "single_task", seed):
                m = BaselineCNN(14).to(DEVICE)
                rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=False)
                save_result({'dataset':'C-MAPSS', 'subset':subset, 'model':'CNN-32', 'variant':'single_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})
                
            if not is_done("C-MAPSS", subset, "LSTM-64", "single_task", seed):
                m = BaselineLSTM(14).to(DEVICE)
                rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=False)
                save_result({'dataset':'C-MAPSS', 'subset':subset, 'model':'LSTM-64', 'variant':'single_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})

        # Attention Variants (FD001 only, 3 seeds)
        if fd == 1 and seed in [42, 7, 13]:
            for attn in ['SE', 'ECA', 'None']:
                if not is_done("C-MAPSS", subset, f"NanoSentry_{attn}", "multi_task", seed):
                    m = NanoSentry(14, 24, 3, attn).to(DEVICE)
                    rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=True)
                    save_result({'dataset':'C-MAPSS', 'subset':subset, 'model':f'NanoSentry_{attn}', 'variant':'multi_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})

    # --- BATTERY (LOCO) ---
    tr, va, te, nf, test_cell = load_battery(seed)
    subset = f"LOCO_{test_cell}"
    if not is_done("Battery", subset, "NanoSentry", "multi_task", seed):
        m = NanoSentry(nf, 24, 3, 'CSG').to(DEVICE)
        rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_multi=True)
        save_result({'dataset':'Battery', 'subset':subset, 'model':'NanoSentry', 'variant':'multi_task', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})

    # --- CWRU (Record-Level) ---
    tr, va, te = load_cwru(seed)
    if not is_done("CWRU", "10-class", "NanoSentry", "classifier", seed):
        m = NanoSentry(1, 24, 10, 'CSG').to(DEVICE)
        rmse, mae, score, cz, acc, p, kb = train_and_eval(m, tr, va, te, is_cwru=True, is_multi=False)
        save_result({'dataset':'CWRU', 'subset':'10-class', 'model':'NanoSentry', 'variant':'classifier', 'seed':seed, 'rmse':rmse, 'mae':mae, 'score':score, 'cz_rmse':cz, 'acc':acc, 'params':p, 'kb':kb})

print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
print(f"📊 Download your results from: {RESULTS_FILE}")