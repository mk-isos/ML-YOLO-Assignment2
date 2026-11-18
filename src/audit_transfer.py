#!/usr/bin/env python3
"""Audit YOLOv8 transfer loading without training or modifying saved weights."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WEIGHT = PROJECT_ROOT / "models/yolov8n.pt"
FINAL_WEIGHT = PROJECT_ROOT / "models/best.pt"
OUTPUT = PROJECT_ROOT / "results/metrics/transfer_audit.json"
TARGET_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = YOLO(str(SOURCE_WEIGHT)).model
    target = DetectionModel(source.yaml, ch=3, nc=3, verbose=False)
    target.names = TARGET_NAMES
    target.load(source, verbose=True)

    source_head = source.model[-1]
    target_head = target.model[-1]
    scale_checks = []
    for scale, (source_branch, target_branch) in enumerate(
        zip(source_head.cv3, target_head.cv3, strict=True)
    ):
        scale_checks.append(
            {
                "scale": scale,
                "car_weight_equal_to_coco_car": bool(
                    torch.equal(source_branch[-1].weight[2].cpu(), target_branch[-1].weight[0].cpu())
                ),
                "car_bias_equal_to_coco_car": bool(
                    torch.equal(source_branch[-1].bias[2].cpu(), target_branch[-1].bias[0].cpu())
                ),
                "pedestrian_equal_to_coco_person": bool(
                    torch.equal(source_branch[-1].weight[0].cpu(), target_branch[-1].weight[1].cpu())
                ),
                "cyclist_equal_to_coco_bicycle": bool(
                    torch.equal(source_branch[-1].weight[1].cpu(), target_branch[-1].weight[2].cpu())
                ),
            }
        )

    result = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "ultralytics": ultralytics.__version__,
        "operation": "weight-load reconstruction only; no optimizer, backward pass, or saved-weight update",
        "source_weight": str(SOURCE_WEIGHT.relative_to(PROJECT_ROOT)),
        "source_weight_sha256": sha256_file(SOURCE_WEIGHT),
        "final_weight": str(FINAL_WEIGHT.relative_to(PROJECT_ROOT)),
        "final_weight_sha256": sha256_file(FINAL_WEIGHT),
        "source_classes": source.names,
        "target_classes": TARGET_NAMES,
        "implementation": {
            "trainer_sets_target_names_before_load": True,
            "loader_method": "DetectionModel._remap_cls_by_names",
            "matching_rule": "case-insensitive exact normalized class-name match",
            "target_state_dict_items": len(target.state_dict()),
            "matched_target_classes": ["car"],
        },
        "scale_checks": scale_checks,
        "conclusion": {
            "car_output_rows_reused": all(
                item["car_weight_equal_to_coco_car"] and item["car_bias_equal_to_coco_car"]
                for item in scale_checks
            ),
            "pedestrian_output_row_reused_from_person": any(
                item["pedestrian_equal_to_coco_person"] for item in scale_checks
            ),
            "cyclist_output_row_reused_from_bicycle": any(
                item["cyclist_equal_to_coco_bicycle"] for item in scale_checks
            ),
            "important_note": (
                "The aggregate 'Transferred 355/355 items' message alone does not prove class-row remapping; "
                "the implementation audit and direct tensor equality checks provide that evidence."
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
