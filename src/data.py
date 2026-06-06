"""
Data loaders for C-MAPSS, NASA Battery, and CWRU.
All loaders return (train_loader, val_loader, test_loader).
Battery loader additionally returns n_feat (int).
"""
import os
import numpy as np
import pandas as pd
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

SEQ   = 64
BATCH = 128
SENSORS = [6,7,8,11,12,13,15,16,17,18,19,21,24,25]  # 14 informative C-MAPSS sensors


# ─────────────────────────────────────────────
# C-MAPSS
# ─────────────────────────────────────────────
def load_cmapss(fd: int, data_dir: str):
    """
    Load NASA C-MAPSS FDxxx.
    Returns: train_loader, val_loader, test_loader
    """
    train_df = pd.read_csv(f'{data_dir}/train_FD{fd:03d}.txt', sep=r'\s+', header=None)
    test_df  = pd.read_csv(f'{data_dir}/test_FD{fd:03d}.txt',  sep=r'\s+', header=None)
    rul_df   = pd.read_csv(f'{data_dir}/RUL_FD{fd:03d}.txt',   header=None, names=['RUL'])

    max_cycles = train_df.groupby(0)[1].max().rename('max_cycle')
    train_df   = train_df.join(max_cycles, on=0)
    train_df['RUL']   = (train_df['max_cycle'] - train_df[1]).clip(upper=125)
    train_df['state'] = pd.cut(train_df['RUL'], bins=[-1,25,50,200],
                                labels=[0,1,2]).astype(int)

    feat_mean = train_df[SENSORS].mean()
    feat_std  = train_df[SENSORS].std().replace(0, 1)

    def make_windows_for_engines(df, engine_list):
        Xs, yr, ys = [], [], []
        for eng in engine_list:
            sub   = df[df[0]==eng].reset_index(drop=True)
            feats = ((sub[SENSORS] - feat_mean) / feat_std).values.astype(np.float32)
            n     = len(feats)
            if n < SEQ:
                pad   = np.zeros((SEQ - n, 14), dtype=np.float32)
                feats = np.vstack([pad, feats])
                n     = SEQ
            for t in range(n - SEQ + 1):
                Xs.append(feats[t:t+SEQ])
                yr.append(float(sub.loc[min(t+SEQ-1, len(sub)-1), 'RUL']))
                ys.append(int(sub.loc[min(t+SEQ-1, len(sub)-1), 'state']))
        return (torch.tensor(np.array(Xs)),
                torch.tensor(yr, dtype=torch.float32),
                torch.tensor(ys, dtype=torch.long))

    engines  = train_df[0].unique()
    n_tr_eng = int(len(engines) * 0.8)
    Xtr, ytr_r, ytr_s = make_windows_for_engines(train_df, engines[:n_tr_eng])
    Xvl, yvl_r, yvl_s = make_windows_for_engines(train_df, engines[n_tr_eng:])

    rul_test = rul_df['RUL'].values.astype(np.float32)
    Xte_list, yte_r_list, yte_s_list = [], [], []
    for i, eng in enumerate(test_df[0].unique()):
        sub   = test_df[test_df[0]==eng].reset_index(drop=True)
        feats = ((sub[SENSORS] - feat_mean) / feat_std).values.astype(np.float32)
        n     = len(feats)
        if n < SEQ:
            pad   = np.zeros((SEQ - n, 14), dtype=np.float32)
            feats = np.vstack([pad, feats])
        Xte_list.append(feats[-SEQ:])
        rul_val = rul_test[i]
        yte_r_list.append(float(rul_val))
        yte_s_list.append(0 if rul_val > 50 else (1 if rul_val > 25 else 2))

    Xte  = torch.tensor(np.array(Xte_list))
    yte_r = torch.tensor(yte_r_list, dtype=torch.float32)
    yte_s = torch.tensor(yte_s_list, dtype=torch.long)

    def make_loader(X, yr, ys, shuffle=False):
        return DataLoader(TensorDataset(X, yr, ys), batch_size=BATCH,
                          shuffle=shuffle, num_workers=2, pin_memory=True)
    return (make_loader(Xtr, ytr_r, ytr_s, shuffle=True),
            make_loader(Xvl, yvl_r, yvl_s),
            make_loader(Xte, yte_r, yte_s))


# ─────────────────────────────────────────────
# NASA Battery
# ─────────────────────────────────────────────
def load_battery(data_dir: str):
    """
    Load NASA Battery dataset from cleaned_dataset structure.
    Uses pre-computed Capacity column in metadata.csv.
    Returns: train_loader, val_loader, test_loader, n_feat (=4)
    """
    meta      = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    discharge = meta[meta['type'] == 'discharge'].copy()
    batteries = discharge['battery_id'].unique()
    all_seqs  = []

    for bid in batteries:
        rows = (discharge[discharge['battery_id'] == bid]
                .sort_values('start_time').reset_index(drop=True))
        cap_vals = []
        for _, r in rows.iterrows():
            try:
                cap_vals.append(float(r['Capacity']))
            except Exception:
                continue

        n = len(cap_vals)
        if n < SEQ + 5:
            continue

        cap_arr  = np.array(cap_vals, dtype=np.float32)
        max_cap  = cap_arr[0] if cap_arr[0] > 0 else cap_arr.max()
        soh      = np.clip(cap_arr / max_cap, 0.0, 1.0)
        delta    = np.diff(soh, prepend=soh[0])
        cyc_norm = np.arange(n, dtype=np.float32) / n
        feats    = np.stack([soh, delta, cyc_norm, cap_arr / (max_cap + 1e-8)], axis=1)

        eol_idx  = int(np.where(soh < 0.8)[0][0]) if np.any(soh < 0.8) else n
        rul      = np.clip((eol_idx - np.arange(n)).astype(np.float32), 0, 200)
        state    = (rul <= 50).astype(np.int64) + (rul <= 20).astype(np.int64)

        for t in range(n - SEQ + 1):
            all_seqs.append((feats[t:t+SEQ], rul[t+SEQ-1], state[t+SEQ-1]))

    np.random.shuffle(all_seqs)
    n_total = len(all_seqs)
    n_tr    = int(n_total * 0.6)
    n_vl    = int(n_total * 0.2)

    def to_loader(seqs, shuffle=False):
        X  = torch.tensor(np.array([s[0] for s in seqs]))
        yr = torch.tensor([s[1] for s in seqs], dtype=torch.float32)
        ys = torch.tensor([s[2] for s in seqs], dtype=torch.long)
        return DataLoader(TensorDataset(X, yr, ys), batch_size=BATCH,
                          shuffle=shuffle, num_workers=2, pin_memory=True)

    return (to_loader(all_seqs[:n_tr], shuffle=True),
            to_loader(all_seqs[n_tr:n_tr+n_vl]),
            to_loader(all_seqs[n_tr+n_vl:]),
            4)


# ─────────────────────────────────────────────
# CWRU Bearing
# ─────────────────────────────────────────────
CWRU_FILES = {
    '97.mat':0,'98.mat':0,'99.mat':0,'100.mat':0,
    '105.mat':1,'106.mat':1,'107.mat':1,'108.mat':1,
    '169.mat':2,'170.mat':2,'171.mat':2,'172.mat':2,
    '209.mat':3,'210.mat':3,'211.mat':3,'212.mat':3,
    '118.mat':4,'119.mat':4,'120.mat':4,'121.mat':4,
    '185.mat':5,'186.mat':5,'187.mat':5,'188.mat':5,
    '222.mat':6,'223.mat':6,'224.mat':6,'225.mat':6,
    '130.mat':7,'131.mat':7,'132.mat':7,'133.mat':7,
    '197.mat':8,'198.mat':8,'199.mat':8,'200.mat':8,
    '234.mat':9,'235.mat':9,'236.mat':9,'237.mat':9,
}
CWRU_SEQ = 1024
N_CWRU_CLASSES = 10


def load_cwru(data_dir: str, seed: int = 42):
    """
    Load CWRU bearing dataset.
    Returns: train_loader, val_loader, test_loader
    """
    all_X, all_y = [], []
    for fname, cls_id in CWRU_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        mat = scipy.io.loadmat(fpath)
        key = [k for k in mat.keys() if 'DE_time' in k]
        if not key:
            continue
        sig  = mat[key[0]].flatten().astype(np.float32)
        sig  = (sig - sig.mean()) / (sig.std() + 1e-8)
        step = CWRU_SEQ // 2
        for start in range(0, len(sig) - CWRU_SEQ + 1, step):
            all_X.append(sig[start:start+CWRU_SEQ].reshape(CWRU_SEQ, 1))
            all_y.append(cls_id)

    idx   = np.random.RandomState(seed).permutation(len(all_X))
    all_X = np.array(all_X)[idx]
    all_y = np.array(all_y)[idx]
    n_total = len(all_X)
    n_tr    = int(n_total * 0.6)
    n_vl    = int(n_total * 0.2)

    def to_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.tensor(X),
                          torch.zeros(len(y), dtype=torch.float32),
                          torch.tensor(y, dtype=torch.long)),
            batch_size=BATCH, shuffle=shuffle, num_workers=2, pin_memory=True)

    return (to_loader(all_X[:n_tr],          all_y[:n_tr],          shuffle=True),
            to_loader(all_X[n_tr:n_tr+n_vl], all_y[n_tr:n_tr+n_vl]),
            to_loader(all_X[n_tr+n_vl:],     all_y[n_tr+n_vl:]))
