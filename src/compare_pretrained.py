#!/usr/bin/env python3
"""Compare COCO-pretrained and fine-tuned YOLOv8n on safely mappable KITTI classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import ultralytics
from ultralytics import YOLO

from analyze_diagnostics import DATASET_ROOT, counts_to_metrics, load_ground_truth, match_detections


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_WEIGHT = PROJECT_ROOT / "models/yolov8n.pt"
FINE_TUNED_WEIGHT = PROJECT_ROOT / "models/best.pt"
ANALYSIS = PROJECT_ROOT / "results/metrics/additional_analysis.json"
OUTPUT = PROJECT_ROOT / "results/metrics/pretrained_comparison.json"
PLOT = PROJECT_ROOT / "results/plots/pretrained_finetuned_comparison.png"
COCO_TO_KITTI = {2: 0, 0: 1}  # car -> car, person -> pedestrian
KITTI_NAMES = {0: "car", 1: "pedestrian"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_pretrained(model: YOLO, device: str, threshold: float) -> dict[str, object]:
    counts = {name: Counter() for name in KITTI_NAMES.values()}
    stream = model.predict(
        source=str(DATASET_ROOT / "images/test"),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        classes=sorted(COCO_TO_KITTI),
        device=device,
        stream=True,
        verbose=False,
    )
    images = 0
    for result in stream:
        images += 1
        ground_truth_all = load_ground_truth("test", Path(result.path))
        predictions = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for class_id, confidence, box in zip(classes, confidences, boxes, strict=True):
                if int(class_id) in COCO_TO_KITTI and float(confidence) >= threshold:
                    predictions.append(
                        {
                            "class_id": COCO_TO_KITTI[int(class_id)],
                            "confidence": float(confidence),
                            "box": box,
                        }
                    )
        for target_id, class_name in KITTI_NAMES.items():
            ground_truth = [
                item for item in ground_truth_all if int(item["class_id"]) == target_id
            ]
            class_predictions = [
                item for item in predictions if int(item["class_id"]) == target_id
            ]
            matched = match_detections(ground_truth, class_predictions, iou_threshold=0.5)
            counts[class_name].update(
                {key: int(matched[key]) for key in ("tp", "fp", "fn")}
            )
    return {
        "images": images,
        "per_class": {
            name: counts_to_metrics(values["tp"], values["fp"], values["fn"])
            for name, values in counts.items()
        },
    }


def plot_comparison(comparison: dict[str, object]) -> None:
    classes = list(KITTI_NAMES.values())
    metrics = ("precision", "recall", "f1")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), constrained_layout=True)
    x_values = range(len(classes))
    for axis, metric in zip(axes, metrics, strict=True):
        pretrained = [comparison["pretrained"]["per_class"][name][metric] for name in classes]
        fine_tuned = [comparison["fine_tuned"]["per_class"][name][metric] for name in classes]
        axis.bar([x - 0.18 for x in x_values], pretrained, width=0.36, label="COCO pretrained")
        axis.bar([x + 0.18 for x in x_values], fine_tuned, width=0.36, label="Fine-tuned")
        axis.set_xticks(list(x_values), classes)
        axis.set_ylim(0, 1.05)
        axis.set_title(metric.capitalize())
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PLOT, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently use CPU")

    source_hash_before = sha256_file(PRETRAINED_WEIGHT)
    final_hash_before = sha256_file(FINE_TUNED_WEIGHT)
    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    model = YOLO(str(PRETRAINED_WEIGHT))
    pretrained = evaluate_pretrained(model, args.device, args.confidence)

    diagnostics = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    fine_tuned_threshold = next(
        item
        for item in diagnostics["confidence_thresholds"]
        if abs(float(item["threshold"]) - args.confidence) < 1e-9
    )
    fine_tuned = {
        "images": diagnostics["split_distribution"]["test"]["images"],
        "per_class": {
            name: fine_tuned_threshold["per_class"][name] for name in KITTI_NAMES.values()
        },
    }
    result = {
        "evaluated_at_utc": started.isoformat(),
        "duration_seconds": time.perf_counter() - start_time,
        "operation": "inference-only comparison; no training or weight update",
        "environment": {"ultralytics": ultralytics.__version__, "device": args.device},
        "settings": {
            "test_images": pretrained["images"],
            "imgsz": 640,
            "confidence": args.confidence,
            "nms_iou": 0.7,
            "matching_iou": 0.5,
            "metric_scope": "fixed-threshold per-class P/R/F1; not official mAP",
        },
        "mapping": {
            "COCO car (id 2)": "KITTI car (id 0)",
            "COCO person (id 0)": "KITTI pedestrian (id 1)",
            "cyclist": "excluded: KITTI cyclist is a rider+bicycle object while COCO predicts person and bicycle separately",
        },
        "pretrained": {
            "weight": str(PRETRAINED_WEIGHT.relative_to(PROJECT_ROOT)),
            "sha256": source_hash_before,
            **pretrained,
        },
        "fine_tuned": {
            "weight": str(FINE_TUNED_WEIGHT.relative_to(PROJECT_ROOT)),
            "sha256": final_hash_before,
            **fine_tuned,
        },
        "limitations": [
            "This is not an official mAP comparison; it is a fixed-confidence diagnostic.",
            "COCO person and KITTI pedestrian annotations are similar but not identical.",
            "KITTI excludes Van/Truck from the target car class, while a COCO model may emit car predictions on them.",
            "Cyclist is intentionally excluded because the label and bounding-box semantics are incompatible.",
            "The fine-tuned operating point was reused from the previously saved diagnostic to avoid redundant inference.",
        ],
        "artifact": str(PLOT.relative_to(PROJECT_ROOT)),
        "weight_integrity": {
            "pretrained_unchanged": source_hash_before == sha256_file(PRETRAINED_WEIGHT),
            "fine_tuned_unchanged": final_hash_before == sha256_file(FINE_TUNED_WEIGHT),
        },
    }
    plot_comparison(result)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
