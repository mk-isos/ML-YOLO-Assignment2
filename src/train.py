#!/usr/bin/env python3
"""Run reproducible YOLOv8 smoke, benchmark, or full fine-tuning."""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/processed/kitti3"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
BASE_MODEL = PROJECT_ROOT / "models/yolov8n.pt"
RUNS_ROOT = PROJECT_ROOT / "results/runs"
RUNTIME_ROOT = PROJECT_ROOT / "results/runtime"
TRAIN_CONFIG = PROJECT_ROOT / "configs/train.yaml"


def balanced_subset(split: str, limit: int) -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = [row for row in manifest["samples"] if row["split"] == split]
    selected: list[dict[str, object]] = []
    selected_stems: set[str] = set()
    per_class = {class_id: [row for row in rows if class_id in row["classes"]] for class_id in range(3)}
    cursors = {class_id: 0 for class_id in range(3)}
    while len(selected) < limit:
        made_progress = False
        for class_id in (2, 1, 0):  # rare classes first
            candidates = per_class[class_id]
            while cursors[class_id] < len(candidates):
                row = candidates[cursors[class_id]]
                cursors[class_id] += 1
                stem = str(row["stem"])
                if stem not in selected_stems:
                    selected.append(row)
                    selected_stems.add(stem)
                    made_progress = True
                    break
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return [str((DATASET_ROOT / "images" / split / f"{row['stem']}.png").resolve()) for row in selected]


def write_runtime_dataset(mode: str) -> Path:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if mode == "smoke":
        train_paths = balanced_subset("train", 72)
        val_paths = balanced_subset("val", 30)
        train_list = RUNTIME_ROOT / "smoke_train.txt"
        val_list = RUNTIME_ROOT / "smoke_val.txt"
        train_list.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
        val_list.write_text("\n".join(val_paths) + "\n", encoding="utf-8")
        train_value = str(train_list.resolve())
        val_value = str(val_list.resolve())
    else:
        train_value = str((DATASET_ROOT / "images/train").resolve())
        val_value = str((DATASET_ROOT / "images/val").resolve())

    runtime_yaml = RUNTIME_ROOT / f"{mode}_data.yaml"
    runtime_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(DATASET_ROOT.resolve()),
                "train": train_value,
                "val": val_value,
                "test": str((DATASET_ROOT / "images/test").resolve()),
                "names": {0: "car", 1: "pedestrian", 2: "cyclist"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def load_training_config() -> dict[str, object]:
    return yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "benchmark", "full"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--name")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently fall back to CPU")
    dataset_yaml = write_runtime_dataset(args.mode)
    config = load_training_config()

    if args.mode == "smoke":
        config.update(
            {
                "epochs": 1,
                "batch": args.batch or 4,
                "imgsz": args.imgsz or 320,
                "patience": 0,
                "workers": 2,
                "name": args.name or "smoke_mps",
                "plots": False,
            }
        )
    elif args.mode == "benchmark":
        config.update(
            {
                "epochs": args.epochs or 1,
                "batch": args.batch or int(config["batch"]),
                "imgsz": args.imgsz or int(config["imgsz"]),
                "patience": 0,
                "name": args.name or "benchmark_mps",
                "plots": False,
            }
        )
    else:
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.batch is not None:
            config["batch"] = args.batch
        if args.imgsz is not None:
            config["imgsz"] = args.imgsz
        config["name"] = args.name or "yolov8n_kitti3"
        config["plots"] = True

    config["data"] = str(dataset_yaml.resolve())
    config["project"] = str(RUNS_ROOT.resolve())
    config["exist_ok"] = True
    config["device"] = "mps"

    if args.resume:
        model_path = args.resume.resolve()
    else:
        model_path = BASE_MODEL

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    model = YOLO(str(model_path))
    results = model.train(**config)
    duration = time.perf_counter() - start
    ended_at = datetime.now(timezone.utc)

    run_directory = Path(results.save_dir).resolve()
    metadata = {
        "mode": args.mode,
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "duration_seconds": duration,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "model": str(model_path),
        "dataset_yaml": str(dataset_yaml),
        "arguments": config,
        "save_dir": str(run_directory),
    }
    (run_directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
