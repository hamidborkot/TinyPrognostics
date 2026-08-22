# ============================================================
# Metrics
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