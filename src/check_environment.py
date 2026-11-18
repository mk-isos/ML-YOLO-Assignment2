#!/usr/bin/env python3
"""Capture reproducible runtime and MPS availability information."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch
import ultralytics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results/metrics/environment.json"


def main() -> None:
    information: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if information["mps_available"]:
        device = torch.device("mps")
        x = torch.randn(8, 3, 224, 224, device=device, requires_grad=True)
        layer = torch.nn.Conv2d(3, 16, 3, padding=1).to(device)
        start = time.perf_counter()
        loss = layer(x).square().mean()
        loss.backward()
        torch.mps.synchronize()
        information["mps_backward_loss"] = float(loss.detach().cpu())
        information["mps_backward_seconds"] = time.perf_counter() - start
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(information, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(information, indent=2))
    if not information["mps_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
