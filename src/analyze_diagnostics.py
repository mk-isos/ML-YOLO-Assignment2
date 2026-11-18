#!/usr/bin/env python3
"""Run post-hoc confidence, object-size, and split-distribution diagnostics.

This script never trains the model. It evaluates the existing ``models/best.pt``
and writes measured results used by the final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/processed/kitti3"
DATA_YAML = DATASET_ROOT / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models/best.pt"
METRICS_ROOT = PROJECT_ROOT / "results/metrics"
PLOTS_ROOT = PROJECT_ROOT / "results/plots"
PREDICTIONS_ROOT = PROJECT_ROOT / "results/predictions"
RUNS_ROOT = PROJECT_ROOT / "results/runs"
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}
SIZE_NAMES = ("small", "medium", "large")
DEFAULT_THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.75)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywhn_to_xyxy(values: list[float], width: int, height: int) -> np.ndarray:
    _, xc, yc, box_width, box_height = values
    return np.array(
        [
            (xc - box_width / 2) * width,
            (yc - box_height / 2) * height,
            (xc + box_width / 2) * width,
            (yc + box_height / 2) * height,
        ],
        dtype=np.float32,
    )


def box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def size_bin(area_pixels: float) -> str:
    """Return the COCO-style diagnostic size bin for original-image pixels."""
    if area_pixels < 32**2:
        return "small"
    if area_pixels < 96**2:
        return "medium"
    return "large"


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    intersection = max(0.0, float(bottom_right[0] - top_left[0])) * max(
        0.0, float(bottom_right[1] - top_left[1])
    )
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Greedily match class-consistent predictions in descending confidence order."""
    unmatched_ground_truth = set(range(len(ground_truth)))
    matches: list[tuple[int, int, float]] = []
    false_predictions: list[int] = []
    prediction_order = sorted(
        range(len(predictions)), key=lambda index: -float(predictions[index]["confidence"])
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        candidates = [
            (ground_truth_index, box_iou(prediction["box"], target["box"]))
            for ground_truth_index, target in enumerate(ground_truth)
            if ground_truth_index in unmatched_ground_truth
            and int(target["class_id"]) == int(prediction["class_id"])
        ]
        best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best_iou >= iou_threshold:
            unmatched_ground_truth.remove(best_index)
            matches.append((prediction_index, best_index, best_iou))
        else:
            false_predictions.append(prediction_index)
    return {
        "tp": len(matches),
        "fp": len(false_predictions),
        "fn": len(unmatched_ground_truth),
        "matches": matches,
        "unmatched_gt": sorted(unmatched_ground_truth),
        "false_predictions": false_predictions,
    }


def load_ground_truth(split: str, image_path: Path) -> list[dict[str, object]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]
    label_path = DATASET_ROOT / "labels" / split / f"{image_path.stem}.txt"
    targets: list[dict[str, object]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.split()]
        box = xywhn_to_xyxy(values, width, height)
        area = box_area(box)
        targets.append(
            {
                "class_id": int(values[0]),
                "box": box,
                "area_pixels": area,
                "size": size_bin(area),
            }
        )
    return targets


def safe_metric(tp: int, denominator: int) -> float | None:
    return tp / denominator if denominator else None


def counts_to_metrics(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    precision = safe_metric(tp, tp + fp)
    recall = safe_metric(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_threshold(
    records: list[dict[str, object]], threshold: float, iou_threshold: float
) -> dict[str, object]:
    overall = Counter()
    per_class = {name: Counter() for name in CLASS_NAMES.values()}
    for record in records:
        ground_truth = record["ground_truth"]
        predictions = [
            prediction
            for prediction in record["predictions"]
            if float(prediction["confidence"]) >= threshold
        ]
        matched = match_detections(ground_truth, predictions, iou_threshold)
        overall.update({key: int(matched[key]) for key in ("tp", "fp", "fn")})
        for class_id, class_name in CLASS_NAMES.items():
            class_ground_truth = [
                target for target in ground_truth if int(target["class_id"]) == class_id
            ]
            class_predictions = [
                prediction
                for prediction in predictions
                if int(prediction["class_id"]) == class_id
            ]
            class_matched = match_detections(class_ground_truth, class_predictions, iou_threshold)
            per_class[class_name].update(
                {key: int(class_matched[key]) for key in ("tp", "fp", "fn")}
            )
    return {
        "threshold": threshold,
        "micro": counts_to_metrics(overall["tp"], overall["fp"], overall["fn"]),
        "per_class": {
            name: counts_to_metrics(counts["tp"], counts["fp"], counts["fn"])
            for name, counts in per_class.items()
        },
    }


def evaluate_size_recall(
    records: list[dict[str, object]], threshold: float, iou_threshold: float
) -> dict[str, object]:
    counts = {
        class_name: {size: Counter() for size in SIZE_NAMES}
        for class_name in CLASS_NAMES.values()
    }
    for record in records:
        ground_truth = record["ground_truth"]
        predictions = [
            prediction
            for prediction in record["predictions"]
            if float(prediction["confidence"]) >= threshold
        ]
        matched = match_detections(ground_truth, predictions, iou_threshold)
        matched_ground_truth = {ground_truth_index for _, ground_truth_index, _ in matched["matches"]}
        for ground_truth_index, target in enumerate(ground_truth):
            class_name = CLASS_NAMES[int(target["class_id"])]
            size = str(target["size"])
            counts[class_name][size]["ground_truth"] += 1
            if ground_truth_index in matched_ground_truth:
                counts[class_name][size]["tp"] += 1
            else:
                counts[class_name][size]["fn"] += 1
    output: dict[str, object] = {}
    for class_name, size_counts in counts.items():
        output[class_name] = {}
        for size, values in size_counts.items():
            ground_truth_count = values["ground_truth"]
            output[class_name][size] = {
                "ground_truth": ground_truth_count,
                "tp": values["tp"],
                "fn": values["fn"],
                "recall": safe_metric(values["tp"], ground_truth_count),
            }
    return {
        "confidence": threshold,
        "iou": iou_threshold,
        "definition": {
            "small": "area < 32^2 pixels",
            "medium": "32^2 <= area < 96^2 pixels",
            "large": "area >= 96^2 pixels",
            "note": "COCO-style diagnostic bins on original KITTI image pixels; not an official KITTI metric.",
        },
        "per_class": output,
    }


def summarize_split(split: str) -> dict[str, object]:
    image_paths = sorted((DATASET_ROOT / "images" / split).glob("*.png"))
    class_counts = Counter()
    size_counts = Counter()
    class_sizes = {name: Counter() for name in CLASS_NAMES.values()}
    areas = {name: [] for name in CLASS_NAMES.values()}
    objects_per_image: list[int] = []
    for image_path in image_paths:
        targets = load_ground_truth(split, image_path)
        objects_per_image.append(len(targets))
        for target in targets:
            class_name = CLASS_NAMES[int(target["class_id"])]
            size = str(target["size"])
            class_counts[class_name] += 1
            size_counts[size] += 1
            class_sizes[class_name][size] += 1
            areas[class_name].append(float(target["area_pixels"]))
    total_instances = sum(class_counts.values())
    return {
        "images": len(image_paths),
        "instances": total_instances,
        "instances_per_image": total_instances / len(image_paths) if image_paths else 0.0,
        "objects_per_image_median": float(np.median(objects_per_image)) if objects_per_image else 0.0,
        "objects_per_image_p90": float(np.percentile(objects_per_image, 90)) if objects_per_image else 0.0,
        "class_instances": dict(class_counts),
        "class_share": {
            name: class_counts[name] / total_instances if total_instances else 0.0
            for name in CLASS_NAMES.values()
        },
        "size_instances": dict(size_counts),
        "size_share": {
            size: size_counts[size] / total_instances if total_instances else 0.0
            for size in SIZE_NAMES
        },
        "per_class_size_instances": {
            name: {size: class_sizes[name][size] for size in SIZE_NAMES}
            for name in CLASS_NAMES.values()
        },
        "per_class_median_area_pixels": {
            name: float(np.median(values)) if values else None for name, values in areas.items()
        },
    }


def metric_list(values: object) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def summarize_ultralytics_metrics(metrics: object) -> dict[str, object]:
    box = metrics.box
    precision = metric_list(box.p)
    recall = metric_list(box.r)
    map50 = metric_list(box.ap50)
    map50_95 = metric_list(box.maps)
    return {
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": {
            name: {
                "precision": precision[class_id],
                "recall": recall[class_id],
                "map50": map50[class_id],
                "map50_95": map50_95[class_id],
            }
            for class_id, name in CLASS_NAMES.items()
        },
    }


def collect_test_predictions(model: YOLO, device: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    stream = model.predict(
        source=str(DATASET_ROOT / "images/test"),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=device,
        stream=True,
        verbose=False,
    )
    for result in stream:
        image_path = Path(result.path)
        predictions: list[dict[str, object]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            predictions = [
                {"class_id": int(class_id), "confidence": float(confidence), "box": box.astype(np.float32)}
                for class_id, confidence, box in zip(classes, confidences, boxes, strict=True)
            ]
        records.append(
            {
                "image_path": image_path,
                "ground_truth": load_ground_truth("test", image_path),
                "predictions": predictions,
            }
        )
    return records


def plot_thresholds(threshold_results: list[dict[str, object]], output: Path) -> None:
    thresholds = [float(result["threshold"]) for result in threshold_results]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for metric, color in (("precision", "#2F5597"), ("recall", "#C55A11"), ("f1", "#548235")):
        axes[0].plot(
            thresholds,
            [float(result["micro"][metric]) for result in threshold_results],
            marker="o",
            linewidth=2,
            label=metric.capitalize(),
            color=color,
        )
    axes[0].set(title="Test micro metrics by confidence", xlabel="Confidence threshold", ylabel="Score")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    colors = {"car": "#4472C4", "pedestrian": "#ED7D31", "cyclist": "#70AD47"}
    for class_name in CLASS_NAMES.values():
        axes[1].plot(
            thresholds,
            [float(result["per_class"][class_name]["recall"]) for result in threshold_results],
            marker="o",
            linewidth=2,
            label=class_name,
            color=colors[class_name],
        )
    axes[1].set(title="Class recall by confidence", xlabel="Confidence threshold", ylabel="Recall")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_size_recall(size_analysis: dict[str, object], output: Path) -> None:
    x_values = np.arange(len(SIZE_NAMES))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    for offset, class_name in enumerate(CLASS_NAMES.values()):
        recalls = []
        labels = []
        for size in SIZE_NAMES:
            values = size_analysis["per_class"][class_name][size]
            recall = values["recall"]
            recalls.append(float(recall) if recall is not None else 0.0)
            labels.append(f"n={values['ground_truth']}")
        positions = x_values + (offset - 1) * width
        bars = axis.bar(positions, recalls, width=width, label=class_name)
        for bar, label in zip(bars, labels, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_xticks(x_values, ["Small (<32²)", "Medium (32²-96²)", "Large (≥96²)"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Recall at conf=0.25, IoU=0.5")
    axis.set_title("Test recall by ground-truth object size")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def select_comparison_record(records: list[dict[str, object]]) -> dict[str, object]:
    def score(record: dict[str, object]) -> tuple[int, int, int]:
        classes = {int(target["class_id"]) for target in record["ground_truth"]}
        predictions = record["predictions"]
        low_count = sum(float(prediction["confidence"]) >= 0.10 for prediction in predictions)
        high_count = sum(float(prediction["confidence"]) >= 0.50 for prediction in predictions)
        return (len(classes), low_count - high_count, len(record["ground_truth"]))

    return max(records, key=score)


def draw_confidence_comparison(record: dict[str, object], output: Path) -> dict[str, object]:
    original = cv2.imread(str(record["image_path"]))
    if original is None:
        raise ValueError(f"Cannot read {record['image_path']}")
    panels: list[np.ndarray] = []
    summaries: list[dict[str, object]] = []
    colors = {0: (196, 114, 68), 1: (49, 125, 237), 2: (71, 173, 112)}
    for threshold in (0.10, 0.25, 0.50):
        panel = original.copy()
        predictions = [
            prediction
            for prediction in record["predictions"]
            if float(prediction["confidence"]) >= threshold
        ]
        matched = match_detections(record["ground_truth"], predictions, 0.5)
        for prediction in predictions:
            class_id = int(prediction["class_id"])
            x1, y1, x2, y2 = (int(value) for value in prediction["box"])
            cv2.rectangle(panel, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(
                panel,
                f"{CLASS_NAMES[class_id]} {float(prediction['confidence']):.2f}",
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colors[class_id],
                1,
                cv2.LINE_AA,
            )
        for ground_truth_index in matched["unmatched_gt"]:
            target = record["ground_truth"][ground_truth_index]
            x1, y1, x2, y2 = (int(value) for value in target["box"])
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 0, 230), 2)
        title = (
            f"conf={threshold:.2f}  TP={matched['tp']}  FP={matched['fp']}  FN={matched['fn']}"
        )
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 35), (255, 255, 255), -1)
        cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)
        panels.append(panel)
        summaries.append({"threshold": threshold, "tp": matched["tp"], "fp": matched["fp"], "fn": matched["fn"]})
    comparison = np.vstack(panels)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), comparison):
        raise RuntimeError(f"Failed to write {output}")
    return {
        "image": Path(record["image_path"]).name,
        "ground_truth_classes": sorted(
            {CLASS_NAMES[int(target["class_id"])] for target in record["ground_truth"]}
        ),
        "panels": summaries,
        "output": str(output.relative_to(PROJECT_ROOT)),
        "note": "Prediction boxes are class-colored; unmatched ground-truth boxes are red.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently use CPU")
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not all(0 < threshold < 1 for threshold in args.thresholds):
        raise ValueError("All confidence thresholds must be between 0 and 1")

    model = YOLO(str(args.model.resolve()))
    if model.names != CLASS_NAMES:
        raise ValueError(f"Unexpected model classes: {model.names}")

    started = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    validation_metrics = model.val(
        data=str(DATA_YAML.resolve()),
        split="val",
        imgsz=640,
        batch=args.batch,
        device=args.device,
        plots=False,
        project=str(RUNS_ROOT.resolve()),
        name="validation_recheck",
        exist_ok=True,
        verbose=False,
    )
    validation = summarize_ultralytics_metrics(validation_metrics)
    records = collect_test_predictions(model, args.device)
    threshold_results = [
        evaluate_threshold(records, threshold, iou_threshold=0.5)
        for threshold in sorted(set(args.thresholds))
    ]
    size_analysis = evaluate_size_recall(records, threshold=0.25, iou_threshold=0.5)
    split_distribution = {split: summarize_split(split) for split in ("val", "test")}

    threshold_plot = PLOTS_ROOT / "confidence_threshold_analysis.png"
    size_plot = PLOTS_ROOT / "size_recall_analysis.png"
    comparison_plot = PREDICTIONS_ROOT / "confidence_threshold_comparison.jpg"
    plot_thresholds(threshold_results, threshold_plot)
    plot_size_recall(size_analysis, size_plot)
    comparison = draw_confidence_comparison(select_comparison_record(records), comparison_plot)

    test_metrics = json.loads((METRICS_ROOT / "test_metrics.json").read_text(encoding="utf-8"))
    validation_test_gap = {
        metric: float(validation["overall"][metric]) - float(test_metrics["overall"][metric])
        for metric in ("precision", "recall", "map50", "map50_95")
    }
    per_class_gap = {
        class_name: {
            metric: float(validation["per_class"][class_name][metric])
            - float(test_metrics["per_class"][class_name][metric])
            for metric in ("precision", "recall", "map50", "map50_95")
        }
        for class_name in CLASS_NAMES.values()
    }
    output = {
        "analysis_type": "post-hoc evaluation only; no model training or weight update",
        "evaluated_at_utc": started.isoformat(),
        "duration_seconds": time.perf_counter() - start_time,
        "model": str(args.model.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256_file(args.model),
        "device": args.device,
        "settings": {
            "imgsz": 640,
            "batch": args.batch,
            "prediction_min_confidence": 0.001,
            "nms_iou": 0.7,
            "matching_iou": 0.5,
            "thresholds": sorted(set(args.thresholds)),
        },
        "validation_recheck": validation,
        "test_reference": {
            "overall": test_metrics["overall"],
            "per_class": test_metrics["per_class"],
        },
        "validation_minus_test": {
            "overall": validation_test_gap,
            "per_class": per_class_gap,
        },
        "confidence_thresholds": threshold_results,
        "size_recall": size_analysis,
        "split_distribution": split_distribution,
        "confidence_comparison": comparison,
        "artifacts": {
            "confidence_plot": str(threshold_plot.relative_to(PROJECT_ROOT)),
            "size_plot": str(size_plot.relative_to(PROJECT_ROOT)),
            "comparison_image": str(comparison_plot.relative_to(PROJECT_ROOT)),
        },
    }
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_ROOT / "additional_analysis.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
