# ============================================================
# hw_export_onnx.py
# Export trained NanoSentry variants to ONNX for STM32Cube.AI
# ============================================================
import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# ---- NanoSentry architecture (must match your trained model) ----
class DilatedBlock(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=dilation,
                              dilation=dilation, padding_mode="zeros")
        self.bn   = nn.BatchNorm1d(d)
        self.act  = nn.GELU()
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
        self.tcn   = nn.Sequential(DilatedBlock(d,1),
                                   DilatedBlock(d,2),
                                   DilatedBlock(d,4))
        self.gate  = CrossSensorGate(d)
        self.skip  = nn.Linear(n_sensors, d)
        self.gru   = nn.GRU(d, d, batch_first=True)
        # ---- heads ----
        self.rul_head   = nn.Sequential(nn.Linear(d,32), nn.GELU(),
                                        nn.Linear(32,1))
        self.state_head = nn.Sequential(nn.Linear(d,32), nn.GELU(),
                                        nn.Linear(d if False else 32, n_classes))
    def forward(self, x):
        h = self.tcn(self.embed(x).permute(0,2,1)).permute(0,2,1)
        h = self.gate(h) + self.skip(x)
        _, hT = self.gru(h)
        hT = hT.squeeze(0)
        return self.rul_head(hT).squeeze(-1), self.state_head(hT)


# ---- ONNX-friendly wrapper: return ONLY the task output ----
# STM32Cube.AI handles single-output graphs more reliably.
class NanoSentryONNX(nn.Module):
    def __init__(self, n_sensors, d, n_classes, task):
        super().__init__()
        self.base = NanoSentry(n_sensors, d, n_classes)
        self.task = task
    def forward(self, x):
        rul, state = self.base(x)
        return rul if self.task == "reg" else state

os.makedirs("onnx_models", exist_ok=True)

configs = [
    # name,          n_sensors, d,  n_classes, task, seq_len, checkpoint
    ("cmapss_reg",   14,         24, 3,         "reg", 64,   "tiny_fd001.pt"),
    ("battery_reg",  4,          24, 3,         "reg", 64,   "tiny_battery.pt"),
    ("cwru_cls",     1,          24, 10,        "cls", 1024, "tiny_cwru.pt"),
]

for name, C, d, K, task, T, ckpt in configs:
    model = NanoSentryONNX(C, d, K, task)

    # Load only if the checkpoint exists; otherwise export random-init
    # (size/MACs/latency are identical regardless of weight values).
    full = NanoSentry(C, d, K)
    if os.path.exists(ckpt):
        full.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.base.load_state_dict(full.state_dict())
        print(f"[{name}] loaded trained weights from {ckpt}")
    else:
        print(f"[{name}] WARNING: {ckpt} not found — exporting random weights "
              f"(memory/latency are unaffected)")

    model.eval()
    dummy = torch.randn(1, T, C)
    out_path = f"onnx_models/{name}.onnx"

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["output"],
        opset_version=11,
        dynamic_axes=None,          # fixed shape → easier for Cube.AI
    )
    print(f"[{name}] exported -> {out_path}  shape=[1,{T},{C}]")

print("\nDone. Next: open these .onnx files in STM32Cube.AI.")