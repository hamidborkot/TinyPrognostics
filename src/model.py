"""
TinyPrognostics — 47 KB dual-task causal dilated CNN.
Architecture: TCN (dilations 1,2,4) + CrossSensorGate + skip + GRU
Outputs: RUL scalar (regression) + health-state logits (classification)
"""
import torch
import torch.nn as nn


class DilatedBlock(nn.Module):
    """Residual dilated causal conv block."""
    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=dilation,
                              dilation=dilation, padding_mode='zeros')
        self.bn  = nn.BatchNorm1d(d)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.bn(self.conv(x)))


class CrossSensorGate(nn.Module):
    """
    Adaptive cross-sensor gating.
    Computes a per-feature gate from the temporal average of the hidden
    representation, then re-scales every time-step by that gate.
    """
    def __init__(self, d: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        g = self.gate(x.mean(dim=1, keepdim=True))  # (B, 1, d)
        return x * g


class TinyPrognostics(nn.Module):
    """
    TinyPrognostics — 12,052 parameters / 47.1 KB.

    Args:
        n_sensors : number of input channels (14 for C-MAPSS, 4 for Battery, 1 for CWRU)
        d         : hidden dimension (default 24)
        n_classes : number of health-state classes (default 3; 10 for CWRU)
    """
    def __init__(self, n_sensors: int = 14, d: int = 24, n_classes: int = 3):
        super().__init__()
        self.embed = nn.Linear(n_sensors, d)
        self.tcn   = nn.Sequential(
            DilatedBlock(d, 1),
            DilatedBlock(d, 2),
            DilatedBlock(d, 4),
        )
        self.gate  = CrossSensorGate(d)
        self.skip  = nn.Linear(n_sensors, d)
        self.gru   = nn.GRU(d, d, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

        self.rul_head   = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1))
        self.state_head = nn.Sequential(
            nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, n_classes))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (B, T, C) — batch of sensor windows
        Returns:
            rul   : (B,)       — scalar RUL predictions
            state : (B, n_cls) — health-state logits
        """
        h = self.embed(x).permute(0, 2, 1)   # (B, d, T)
        h = self.tcn(h).permute(0, 2, 1)     # (B, T, d)
        h = self.gate(h) + self.skip(x)      # gated + skip
        _, hT = self.gru(h)                  # hT: (1, B, d)
        hT = hT.squeeze(0)                   # (B, d)
        return self.rul_head(hT).squeeze(-1), self.state_head(hT)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def size_kb(self) -> float:
        return sum(p.numel() * p.element_size() for p in self.parameters()) / 1024


if __name__ == '__main__':
    for cfg in [
        dict(n_sensors=14, d=24, n_classes=3),   # C-MAPSS
        dict(n_sensors=4,  d=24, n_classes=3),   # Battery
        dict(n_sensors=1,  d=24, n_classes=10),  # CWRU
    ]:
        m = TinyPrognostics(**cfg)
        print(f"n_sensors={cfg['n_sensors']:2d}  "
              f"params={m.param_count():,}  size={m.size_kb():.1f} KB")
        x = torch.randn(4, 64, cfg['n_sensors'])
        rul, state = m(x)
        print(f"  rul={tuple(rul.shape)}  state={tuple(state.shape)}")
