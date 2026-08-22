# ============================================================
# Dataset paths
# ============================================================

CMAPSS_DIR = "/kaggle/input/datasets/behrad3d/nasa-cmaps/CMaps"
BATTERY_DIR = "/kaggle/input/datasets/patrickfleith/nasa-battery-dataset/cleaned_dataset"
CWRU_DIR = "/kaggle/input/datasets/sufian79/cwru-mat-full-dataset"

SENSORS = [6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25]
OP_COLS = [2, 3, 4]
RUL_CAP = 125


# ============================================================
# C-MAPSS loader
# ============================================================

def normalize_cmapss(train_df, test_df, fd):
    """
    FD001/FD003: global normalization.
    FD002/FD004: condition-aware normalization using KMeans on operating settings.
    """

    if fd in [1, 3]:
        mu = train_df[SENSORS].mean()
        std = train_df[SENSORS].std().replace(0, 1)

        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[SENSORS] = (train_df[SENSORS] - mu) / std
        test_df[SENSORS] = (test_df[SENSORS] - mu) / std

        return train_df, test_df

    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    km.fit(train_df[OP_COLS].values)

    train_df = train_df.copy()
    test_df = test_df.copy()

    tr_labels = km.predict(train_df[OP_COLS].values)
    te_labels = km.predict(test_df[OP_COLS].values)

    stats = {}

    for c in range(6):
        mask = tr_labels == c

        if mask.sum() < 2:
            continue

        mu = train_df.loc[mask, SENSORS].mean()
        std = train_df.loc[mask, SENSORS].std().replace(0, 1)

        stats[c] = (mu, std)
        train_df.loc[mask, SENSORS] = (train_df.loc[mask, SENSORS] - mu) / std

    for c in range(6):
        mask = te_labels == c

        if mask.sum() == 0 or c not in stats:
            continue

        mu, std = stats[c]
        test_df.loc[mask, SENSORS] = (test_df.loc[mask, SENSORS] - mu) / std

    return train_df, test_df


def load_cmapss(fd, seed):
    train_df = pd.read_csv(
        f"{CMAPSS_DIR}/train_FD{fd:03d}.txt",
        sep=r"\s+",
        header=None
    )

    test_df = pd.read_csv(
        f"{CMAPSS_DIR}/test_FD{fd:03d}.txt",
        sep=r"\s+",
        header=None
    )

    rul_df = pd.read_csv(
        f"{CMAPSS_DIR}/RUL_FD{fd:03d}.txt",
        header=None,
        names=["RUL"]
    )

    max_cycles = train_df.groupby(0)[1].max().rename("max_cycle")
    train_df = train_df.join(max_cycles, on=0)
    train_df["RUL"] = (train_df["max_cycle"] - train_df[1]).clip(upper=RUL_CAP)
    train_df["state"] = train_df["RUL"].apply(rul_to_state)

    train_df, test_df = normalize_cmapss(train_df, test_df, fd)

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
                yr.append(float(sub.loc[min(t + SEQ - 1, len(sub) - 1), "RUL"]))
                ys.append(int(sub.loc[min(t + SEQ - 1, len(sub) - 1), "state"]))

        return (
            torch.tensor(np.array(Xs), dtype=torch.float32),
            torch.tensor(yr, dtype=torch.float32),
            torch.tensor(ys, dtype=torch.long)
        )

    engines = train_df[0].unique()
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(engines))

    n_tr = int(len(engines) * 0.8)

    tr_eng = engines[idx[:n_tr]]
    vl_eng = engines[idx[n_tr:]]

    Xtr, ytr_r, ytr_s = make_windows(train_df, tr_eng)
    Xvl, yvl_r, yvl_s = make_windows(train_df, vl_eng)

    rul_test = rul_df["RUL"].values.astype(np.float32)

    Xte_list = []
    yte_r_list = []
    yte_s_list = []

    for i, eng in enumerate(test_df[0].unique()):
        sub = test_df[test_df[0] == eng].reset_index(drop=True)
        feats = sub[SENSORS].values.astype(np.float32)

        if len(feats) < SEQ:
            pad = np.zeros((SEQ - len(feats), len(SENSORS)), dtype=np.float32)
            feats = np.vstack([pad, feats])

        Xte_list.append(feats[-SEQ:])

        rv = float(rul_test[i])
        yte_r_list.append(rv)
        yte_s_list.append(rul_to_state(rv))

    Xte = torch.tensor(np.array(Xte_list), dtype=torch.float32)
    yte_r = torch.tensor(yte_r_list, dtype=torch.float32)
    yte_s = torch.tensor(yte_s_list, dtype=torch.long)

    pin = DEVICE == "cuda"

    tr = DataLoader(
        TensorDataset(Xtr, ytr_r, ytr_s),
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=pin
    )

    va = DataLoader(
        TensorDataset(Xvl, yvl_r, yvl_s),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=pin
    )

    te = DataLoader(
        TensorDataset(Xte, yte_r, yte_s),
        batch_size=BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=pin
    )

    return tr, va, te


# ============================================================
# Battery LOCO loader with improved robust preprocessing
# ============================================================

def build_battery_cell_sequences():
    meta = pd.read_csv(os.path.join(BATTERY_DIR, "metadata.csv"))
    discharge = meta[meta["type"] == "discharge"].copy()

    batteries = sorted(discharge["battery_id"].unique())
    all_cells = {}

    for bid in batteries:
        rows = discharge[discharge["battery_id"] == bid].sort_values("start_time")

        cap_vals = pd.to_numeric(rows["Capacity"], errors="coerce").dropna().values
        cap_vals = cap_vals.astype(np.float32)

        n = len(cap_vals)

        if n < SEQ + 10:
            continue

        # Robust smoothing to reduce capacity-regeneration noise
        cap_smooth = (
            pd.Series(cap_vals)
            .rolling(window=5, center=True, min_periods=1)
            .median()
            .values
        )

        # Use early-cycle median as rated capacity estimate
        max_cap = float(np.median(cap_smooth[:10]))

        if max_cap <= 0:
            max_cap = float(cap_smooth.max())

        soh = np.clip(cap_smooth / max_cap, 0.0, 1.0)

        # Robust EOL detection: require 3 consecutive cycles below 0.8
        below = soh < 0.8
        eol_idx = n

        for i in range(len(below) - 2):
            if below[i] and below[i + 1] and below[i + 2]:
                eol_idx = i
                break

        # Fallback: first below-threshold cycle
        if eol_idx == n and below.any():
            eol_idx = int(np.where(below)[0][0])

        rul = np.clip(eol_idx - np.arange(n), 0, BATTERY_HORIZON).astype(np.float32)

        delta = np.diff(soh, prepend=soh[0])
        cyc_norm = np.clip(np.arange(n, dtype=np.float32) / BATTERY_HORIZON, 0.0, 1.0)

        feats = np.stack(
            [
                soh,
                delta,
                cyc_norm,
                cap_smooth / (max_cap + 1e-8)
            ],
            axis=1
        ).astype(np.float32)

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

        if len(seqs) > 0:
            all_cells[str(bid)] = seqs

    return all_cells


def build_battery_folds():
    all_cells = build_battery_cell_sequences()
    folds = []

    for test_cell in all_cells.keys():
        tr_seqs = []
        vl_seqs = []

        for cell_id, seqs in all_cells.items():
            if cell_id == test_cell:
                continue

            split_idx = int(len(seqs) * 0.8)

            tr_seqs.extend(seqs[:split_idx])
            vl_seqs.extend(seqs[split_idx:])

        te_seqs = all_cells[test_cell]

        if len(tr_seqs) < 10 or len(vl_seqs) < 5 or len(te_seqs) < 5:
            continue

        rng = np.random.RandomState(42)
        rng.shuffle(tr_seqs)

        y_train = np.array([s[1] for s in tr_seqs], dtype=np.float32)

        y_mean = float(y_train.mean())
        y_std = float(y_train.std() + 1e-8)

        state_train = np.array([s[2] for s in tr_seqs], dtype=np.int64)
        counts = np.bincount(state_train, minlength=3).astype(np.float32)

        weights = 1.0 / (counts + 1.0)
        weights = weights / weights.sum() * 3.0

        class_weight = torch.tensor(weights, dtype=torch.float32)

        def to_loader(seqs, shuffle):
            X = torch.tensor(np.array([s[0] for s in seqs]), dtype=torch.float32)
            yr = torch.tensor([s[1] for s in seqs], dtype=torch.float32)
            ys = torch.tensor([s[2] for s in seqs], dtype=torch.long)

            bs = min(BATCH, len(X))

            return DataLoader(
                TensorDataset(X, yr, ys),
                batch_size=bs,
                shuffle=shuffle,
                num_workers=0
            )

        folds.append(
            {
                "test_cell": test_cell,
                "tr": to_loader(tr_seqs, True),
                "vl": to_loader(vl_seqs, False),
                "te": to_loader(te_seqs, False),
                "y_mean": y_mean,
                "y_std": y_std,
                "class_weight": class_weight
            }
        )

    return folds


# ============================================================
# CWRU record-level loader
# ============================================================

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


def find_cwru_file(fname):
    for root, _, files in os.walk(CWRU_DIR):
        if fname in files:
            return os.path.join(root, fname)
    return None


def load_cwru_split(seed):
    class_files = {c: [] for c in range(10)}

    for fname, cls_id in CWRU_FILES.items():
        class_files[cls_id].append(fname)

    rng = np.random.RandomState(seed)

    tr_files = []
    vl_files = []
    te_files = []

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

        return (
            torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

    Xtr, ytr = extract_windows(tr_files)
    Xvl, yvl = extract_windows(vl_files)
    Xte, yte = extract_windows(te_files)

    if len(Xtr) == 0:
        raise RuntimeError("CWRU training set is empty. Check dataset path.")

    bs_tr = min(BATCH, len(Xtr))
    bs_vl = min(BATCH, len(Xvl))
    bs_te = min(BATCH, len(Xte))

    tr = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=bs_tr,
        shuffle=True,
        num_workers=0
    )

    va = DataLoader(
        TensorDataset(Xvl, yvl),
        batch_size=bs_vl,
        shuffle=False,
        num_workers=0
    )

    te = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=bs_te,
        shuffle=False,
        num_workers=0
    )

    return tr, va, te