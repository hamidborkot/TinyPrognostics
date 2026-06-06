"""
CLI training script for TinyPrognostics.

Usage:
    python src/train.py --dataset fd001 --epochs 100
    python src/train.py --dataset battery --epochs 100
    python src/train.py --dataset cwru --epochs 80

Dataset path env vars (set these to your local data directories):
    CMAPSS_DIR   : path to NASA C-MAPSS CMaps directory
    BATTERY_DIR  : path to cleaned_dataset directory
    CWRU_DIR     : path to CWRU .mat files directory
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from model import TinyPrognostics
from data  import load_cmapss, load_battery, load_cwru, N_CWRU_CLASSES

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED   = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def nasa_score(pred, true):
    d = pred - true
    s = np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return float(s.sum()) if np.isfinite(s.sum()) else float('nan')


def train_epoch(model, loader, optimizer, is_cwru=False):
    model.train()
    mse_fn = nn.MSELoss()
    ce_fn  = nn.CrossEntropyLoss()
    for X, yr, ys in loader:
        X, yr, ys = X.to(DEVICE), yr.to(DEVICE), ys.to(DEVICE)
        optimizer.zero_grad()
        pred_r, pred_s = model(X)
        loss = ce_fn(pred_s, ys) if is_cwru else \
               mse_fn(pred_r, yr) + 0.15 * ce_fn(pred_s, ys)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate(model, loader, is_cwru=False):
    model.eval()
    preds_r, trues_r, preds_s, trues_s = [], [], [], []
    with torch.no_grad():
        for X, yr, ys in loader:
            pred_r, pred_s = model(X.to(DEVICE))
            preds_r.extend(pred_r.cpu().numpy())
            trues_r.extend(yr.numpy())
            preds_s.extend(pred_s.argmax(1).cpu().numpy())
            trues_s.extend(ys.numpy())
    pr, tr_ = np.array(preds_r), np.array(trues_r)
    ps, ts  = np.array(preds_s), np.array(trues_s)
    return {
        'rmse':  float(np.sqrt(np.mean((pr - tr_)**2))),
        'mae':   float(np.mean(np.abs(pr - tr_))),
        'acc':   float((ps == ts).mean() * 100),
        'score': nasa_score(pr, tr_),
    }


def train(model, tr, va, te, epochs=100, lr=1e-3,
          is_cwru=False, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(1, epochs+1):
        train_epoch(model, tr, optimizer, is_cwru=is_cwru)
        m = evaluate(model, va, is_cwru=is_cwru)
        val_metric = -m['acc'] if is_cwru else m['rmse']
        scheduler.step()
        if val_metric < best_val:
            best_val   = val_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'  Early stop at epoch {epoch}')
                break

    model.load_state_dict(best_state)
    return evaluate(model, te, is_cwru=is_cwru)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',  required=True,
                        choices=['fd001','fd002','fd003','fd004','battery','cwru'])
    parser.add_argument('--epochs',   type=int, default=100)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save',     default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)

    if args.dataset.startswith('fd'):
        fd  = int(args.dataset[2:])
        tr, va, te = load_cmapss(fd, os.environ.get('CMAPSS_DIR', 'data/CMaps'))
        model = TinyPrognostics(n_sensors=14, d=24, n_classes=3).to(DEVICE)
        is_cwru = False
    elif args.dataset == 'battery':
        tr, va, te, n_feat = load_battery(
            os.environ.get('BATTERY_DIR', 'data/battery/cleaned_dataset'))
        model = TinyPrognostics(n_sensors=n_feat, d=24, n_classes=3).to(DEVICE)
        is_cwru = False
    else:
        tr, va, te = load_cwru(os.environ.get('CWRU_DIR', 'data/cwru'))
        model = TinyPrognostics(n_sensors=1, d=24, n_classes=N_CWRU_CLASSES).to(DEVICE)
        is_cwru = True

    print(f'Training on {args.dataset.upper()} | device={DEVICE} '
          f'| params={sum(p.numel() for p in model.parameters()):,}')
    metrics = train(model, tr, va, te,
                    epochs=args.epochs, lr=args.lr,
                    is_cwru=is_cwru, patience=args.patience)

    print(f"\nTest results:")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v:.4f}")

    ckpt = os.path.join(args.save, f'tiny_{args.dataset}.pt')
    torch.save(model.state_dict(), ckpt)
    print(f"Checkpoint saved → {ckpt}")


if __name__ == '__main__':
    main()
