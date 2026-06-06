"""
Transfer learning: load a pretrained TinyPrognostics checkpoint,
freeze TCN + GRU, fine-tune embed + skip + heads on a new domain.

Usage:
    python src/transfer.py \
        --source checkpoints/tiny_fd001.pt \
        --source_sensors 14 \
        --target fd003 \
        --frac 0.5
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
from train import evaluate, train_epoch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED   = 42


def partial_load(source_pt, source_sensors, target_sensors, d=24, n_classes=3):
    """Load source weights into target model, skipping shape-mismatched layers."""
    src_model = TinyPrognostics(n_sensors=source_sensors, d=d, n_classes=n_classes)
    src_model.load_state_dict(torch.load(source_pt, map_location='cpu'))
    tgt_model = TinyPrognostics(n_sensors=target_sensors, d=d, n_classes=n_classes)
    src_sd, tgt_sd = src_model.state_dict(), tgt_model.state_dict()
    for k in tgt_sd:
        if k in src_sd and src_sd[k].shape == tgt_sd[k].shape:
            tgt_sd[k] = src_sd[k].clone()
    tgt_model.load_state_dict(tgt_sd)
    return tgt_model


def finetune(model, tr, va, te, data_frac=1.0,
             epochs=50, lr=5e-4, patience=12):
    for p in model.tcn.parameters(): p.requires_grad = False
    for p in model.gru.parameters(): p.requires_grad = False

    if data_frac < 1.0:
        ds     = tr.dataset
        n_keep = max(64, int(len(ds) * data_frac))
        idx    = torch.randperm(len(ds))[:n_keep]
        tr     = DataLoader(Subset(ds, idx), batch_size=128,
                            shuffle=True, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(1, epochs+1):
        train_epoch(model, tr, optimizer)
        val_rmse = evaluate(model, va)['rmse']
        scheduler.step()
        if val_rmse < best_val:
            best_val   = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return evaluate(model, te)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',         required=True)
    parser.add_argument('--source_sensors', type=int, default=14)
    parser.add_argument('--target',         required=True,
                        choices=['fd001','fd002','fd003','fd004','battery'])
    parser.add_argument('--frac',           type=float, default=1.0)
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--lr',             type=float, default=5e-4)
    args = parser.parse_args()

    if args.target.startswith('fd'):
        fd = int(args.target[2:])
        tr, va, te = load_cmapss(fd, os.environ.get('CMAPSS_DIR','data/CMaps'))
        n_tgt = 14
    else:
        tr, va, te, n_tgt = load_battery(
            os.environ.get('BATTERY_DIR','data/battery/cleaned_dataset'))

    model = partial_load(args.source, args.source_sensors, n_tgt).to(DEVICE)
    metrics = finetune(model, tr, va, te, data_frac=args.frac,
                       epochs=args.epochs, lr=args.lr)
    print(f"Transfer {args.source} → {args.target} @ {args.frac*100:.0f}% data")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == '__main__':
    main()
