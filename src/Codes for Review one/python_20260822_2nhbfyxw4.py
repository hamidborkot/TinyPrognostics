# ============================================================
# Model blocks
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
    Fixed ECA implementation.
    Input x shape: (B, T, d)
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
        y = torch.sigmoid(y)           # (B, 1, d)

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


class NanoSentryAblation(nn.Module):
    def __init__(
        self,
        n_sensors=14,
        d=24,
        n_classes=3,
        no_gate=False,
        no_skip=False,
        no_dilation=False
    ):
        super().__init__()

        self.no_gate = no_gate
        self.no_skip = no_skip

        self.embed = nn.Linear(n_sensors, d)

        dilations = [1, 1, 1] if no_dilation else [1, 2, 4]

        self.tcn = nn.Sequential(
            *[DilatedBlock(d, dil) for dil in dilations]
        )

        if not no_gate:
            self.gate = CrossSensorGate(d)

        if not no_skip:
            self.skip = nn.Linear(n_sensors, d)

        self.gru = nn.GRU(d, d, batch_first=True)
        nn.init.orthogonal_(self.gru.weight_hh_l0)

        self.rul_head = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.state_head = nn.Sequential(
            nn.Linear(d, 32),
            nn.GELU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        h = self.embed(x).permute(0, 2, 1)
        h = self.tcn(h).permute(0, 2, 1)

        if not self.no_gate:
            h = self.gate(h)

        if not self.no_skip:
            h = h + self.skip(x)

        _, hT = self.gru(h)
        hT = hT.squeeze(0)

        rul = self.rul_head(hT).squeeze(-1)
        state = self.state_head(hT)

        return rul, state


class CNNBaseline(nn.Module):
    def __init__(
        self,
        n_channels=14,
        filters=32,
        n_classes=3,
        classification=False
    ):
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
    def __init__(
        self,
        n_channels=14,
        hidden=64,
        n_classes=3,
        classification=False
    ):
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