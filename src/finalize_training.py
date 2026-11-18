#!/usr/bin/env python3
"""Promote the completed training run into compact, submission-ready artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = PROJECT_ROOT / "results/runs/yolov8n_kitti3_30e"
EXPECTED_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    source_weight = run / "weights/best.pt"
    source_results = run / "results.csv"
    source_metadata = run / "run_metadata.json"
    for required in (source_weight, source_results, source_metadata):
        if not required.exists():
            raise FileNotFoundError(required)

    models = PROJECT_ROOT / "models"
    metrics = PROJECT_ROOT / "results/metrics"
    plots = PROJECT_ROOT / "results/plots"
    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    destination_weight = models / "best.pt"
    shutil.copy2(source_weight, destination_weight)
    model = YOLO(str(destination_weight))
    if model.names != EXPECTED_NAMES:
        raise ValueError(f"Unexpected model classes: {model.names}")

    destination_results = metrics / "training_results.csv"
    destination_metadata = metrics / "training_metadata.json"
    shutil.copy2(source_results, destination_results)
    shutil.copy2(source_metadata, destination_metadata)

    copied_plots = []
    for source_name, destination_name in (
        ("results.png", "training_curves.png"),
        ("labels.jpg", "training_labels.jpg"),
        ("confusion_matrix.png", "validation_confusion_matrix.png"),
        ("confusion_matrix_normalized.png", "validation_confusion_matrix_normalized.png"),
        ("PR_curve.png", "validation_PR_curve.png"),
        ("F1_curve.png", "validation_F1_curve.png"),
        ("val_batch0_pred.jpg", "validation_predictions.jpg"),
    ):
        source = run / source_name
        if source.exists():
            target = plots / destination_name
            shutil.copy2(source, target)
            copied_plots.append(str(target.relative_to(PROJECT_ROOT)))

    with destination_results.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Training CSV has no epochs")
    for row in rows:
        row.update({key: value.strip() for key, value in row.items()})
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    metadata = json.loads(destination_metadata.read_text(encoding="utf-8"))
    summary = {
        "run": str(run),
        "completed_epochs": len(rows),
        "configured_epochs": metadata["arguments"]["epochs"],
        "duration_seconds": metadata["duration_seconds"],
        "best_epoch": int(float(best["epoch"])),
        "best_validation": {
            "precision": float(best["metrics/precision(B)"]),
            "recall": float(best["metrics/recall(B)"]),
            "map50": float(best["metrics/mAP50(B)"]),
            "map50_95": float(best["metrics/mAP50-95(B)"]),
        },
        "final_losses": {
            "train_box": float(rows[-1]["train/box_loss"]),
            "train_cls": float(rows[-1]["train/cls_loss"]),
            "train_dfl": float(rows[-1]["train/dfl_loss"]),
            "val_box": float(rows[-1]["val/box_loss"]),
            "val_cls": float(rows[-1]["val/cls_loss"]),
            "val_dfl": float(rows[-1]["val/dfl_loss"]),
        },
        "weight": str(destination_weight.relative_to(PROJECT_ROOT)),
        "weight_bytes": destination_weight.stat().st_size,
        "weight_sha256": sha256_file(destination_weight),
        "model_names": model.names,
        "plots": copied_plots,
    }
    output = metrics / "training_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
