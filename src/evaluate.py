#!/usr/bin/env python3
"""Evaluate the fine-tuned detector on the held-out KITTI3 test split."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/processed/kitti3"
DATA_YAML = DATASET_ROOT / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models/best.pt"
RUNS_ROOT = PROJECT_ROOT / "results/runs"
METRICS_ROOT = PROJECT_ROOT / "results/metrics"
PLOTS_ROOT = PROJECT_ROOT / "results/plots"
PREDICTIONS_ROOT = PROJECT_ROOT / "results/predictions"
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def xywhn_to_xyxy(row: list[float], width: int, height: int) -> np.ndarray:
    _, xc, yc, bw, bh = row
    return np.array(
        [
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ],
        dtype=np.float32,
    )


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def load_ground_truth(image_path: Path) -> list[tuple[int, np.ndarray]]:
    label_path = DATASET_ROOT / "labels/test" / f"{image_path.stem}.txt"
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    ground_truth: list[tuple[int, np.ndarray]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        ground_truth.append((int(values[0]), xywhn_to_xyxy(values, width, height)))
    return ground_truth


def match_detections(
    ground_truth: list[tuple[int, np.ndarray]],
    predictions: list[tuple[int, float, np.ndarray]],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    unmatched_gt = set(range(len(ground_truth)))
    true_positives: list[tuple[int, int, float]] = []
    false_positives: list[int] = []
    for prediction_index in sorted(range(len(predictions)), key=lambda i: -predictions[i][1]):
        class_id, _, predicted_box = predictions[prediction_index]
        candidates = [
            (gt_index, box_iou(predicted_box, gt_box))
            for gt_index, (gt_class, gt_box) in enumerate(ground_truth)
            if gt_index in unmatched_gt and gt_class == class_id
        ]
        best = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best[1] >= iou_threshold:
            unmatched_gt.remove(best[0])
            true_positives.append((prediction_index, best[0], best[1]))
        else:
            false_positives.append(prediction_index)
    return {
        "tp": len(true_positives),
        "fp": len(false_positives),
        "fn": len(unmatched_gt),
        "matches": true_positives,
        "unmatched_gt": sorted(unmatched_gt),
        "false_predictions": false_positives,
    }


def draw_case(
    image_path: Path,
    result: object,
    ground_truth: list[tuple[int, np.ndarray]],
    match: dict[str, object],
    output_path: Path,
) -> None:
    annotated = result.plot(line_width=2, font_size=0.6)
    for gt_index in match["unmatched_gt"]:
        class_id, box = ground_truth[int(gt_index)]
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"MISSED {CLASS_NAMES[class_id]}",
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(f"Failed to write {output_path}")


def metric_list(values: object) -> list[float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(value) for value in array]


def extract_metrics(metrics: object) -> dict[str, object]:
    box = metrics.box
    per_class_precision = metric_list(box.p)
    per_class_recall = metric_list(box.r)
    per_class_map50 = metric_list(box.ap50)
    per_class_map = metric_list(box.maps)
    per_class = {}
    for class_id, name in CLASS_NAMES.items():
        per_class[name] = {
            "precision": per_class_precision[class_id],
            "recall": per_class_recall[class_id],
            "map50": per_class_map50[class_id],
            "map50_95": per_class_map[class_id],
        }
    return {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }


def select_cases(model: YOLO, device: str) -> dict[str, object]:
    records: list[dict[str, object]] = []
    aggregate = Counter()
    stream = model.predict(
        source=str(DATASET_ROOT / "images/test"),
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=device,
        stream=True,
        verbose=False,
    )
    for result in stream:
        image_path = Path(result.path)
        gt = load_ground_truth(image_path)
        predictions: list[tuple[int, float, np.ndarray]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            predictions = [
                (int(class_id), float(confidence), box.astype(np.float32))
                for class_id, confidence, box in zip(classes, confidences, boxes, strict=True)
            ]
        matched = match_detections(gt, predictions)
        aggregate.update({key: int(matched[key]) for key in ("tp", "fp", "fn")})
        records.append(
            {
                "image": image_path.name,
                "ground_truth_count": len(gt),
                "prediction_count": len(predictions),
                "tp": matched["tp"],
                "fp": matched["fp"],
                "fn": matched["fn"],
                "score": int(matched["tp"]) - int(matched["fp"]) - 2 * int(matched["fn"]),
            }
        )

    successful = sorted(
        [record for record in records if record["tp"] > 0 and record["fp"] == 0 and record["fn"] == 0],
        key=lambda record: (-int(record["tp"]), str(record["image"])),
    )[:3]
    failures = sorted(
        [record for record in records if record["fp"] > 0 or record["fn"] > 0],
        key=lambda record: (int(record["score"]), -int(record["fn"]), -int(record["fp"]), str(record["image"])),
    )[:3]

    PREDICTIONS_ROOT.mkdir(parents=True, exist_ok=True)
    selected_public: list[dict[str, object]] = []
    for category, selected in (("success", successful), ("failure", failures)):
        for index, record in enumerate(selected, 1):
            image_path = DATASET_ROOT / "images/test" / str(record["image"])
            result = model.predict(
                source=str(image_path),
                imgsz=640,
                conf=0.25,
                iou=0.7,
                device=device,
                verbose=False,
            )[0]
            boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else []
            classes = (
                result.boxes.cls.detach().cpu().numpy().astype(int)
                if result.boxes is not None
                else []
            )
            confidences = (
                result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else []
            )
            predictions = [
                (int(class_id), float(confidence), box.astype(np.float32))
                for class_id, confidence, box in zip(classes, confidences, boxes, strict=True)
            ]
            ground_truth = load_ground_truth(image_path)
            matched = match_detections(ground_truth, predictions)
            output_path = PREDICTIONS_ROOT / f"{category}_{index}_{image_path.stem}.jpg"
            draw_case(image_path, result, ground_truth, matched, output_path)
            selected_public.append(
                record
                | {"category": category, "output": str(output_path.relative_to(PROJECT_ROOT))}
            )

    total_tp, total_fp, total_fn = aggregate["tp"], aggregate["fp"], aggregate["fn"]
    return {
        "thresholds": {"confidence": 0.25, "iou_match": 0.5, "nms_iou": 0.7},
        "aggregate": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
            "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        },
        "selected_cases": selected_public,
    }


def copy_plots(run_directory: Path) -> list[str]:
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxPR_curve.png",
        "BoxR_curve.png",
    ):
        source = run_directory / name
        if source.exists():
            target = PLOTS_ROOT / f"test_{name.removeprefix('Box')}"
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(PROJECT_ROOT)))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--name", default="test_evaluation")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently use CPU")
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model.resolve()))
    if model.names != CLASS_NAMES:
        raise ValueError(f"Unexpected model classes: {model.names}")

    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    metrics = model.val(
        data=str(DATA_YAML.resolve()),
        split="test",
        imgsz=640,
        batch=args.batch,
        device=args.device,
        plots=True,
        project=str(RUNS_ROOT.resolve()),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    evaluation_seconds = time.perf_counter() - start
    summary = extract_metrics(metrics)
    run_directory = Path(metrics.save_dir).resolve()
    summary.update(
        {
            "evaluated_at_utc": started.isoformat(),
            "evaluation_seconds": evaluation_seconds,
            "model": str(args.model.resolve()),
            "dataset": str(DATA_YAML.resolve()),
            "split": "test",
            "test_images": len(list((DATASET_ROOT / "images/test").glob("*.png"))),
            "device": args.device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "run_directory": str(run_directory),
            "plots": copy_plots(run_directory),
        }
    )
    summary["case_analysis"] = select_cases(model, args.device)
    output_path = METRICS_ROOT / "test_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
