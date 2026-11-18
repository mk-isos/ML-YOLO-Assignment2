#!/usr/bin/env python3
"""Validate the generated three-class dataset and persist an audit report."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data/processed/kitti3"
REPORT_PATH = PROJECT_ROOT / "results/metrics/dataset_validation.json"
CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(dataset_root: Path = DATASET_ROOT) -> dict[str, object]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report: dict[str, object] = {"valid": True, "errors": [], "splits": {}}
    hashes_by_split: dict[str, set[str]] = {}
    groups_by_split: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    sample_manifest = {row["stem"]: row for row in manifest["samples"]}

    for split in ("train", "val", "test"):
        images = sorted((dataset_root / "images" / split).glob("*.png"))
        labels = sorted((dataset_root / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            report["errors"].append(f"{split}: image/label stems differ")
        image_counts = Counter()
        instance_counts = Counter()
        invalid_rows = 0
        for label_path in labels:
            present = set()
            for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("column count")
                    class_id = int(parts[0])
                    values = list(map(float, parts[1:]))
                    if class_id not in CLASS_NAMES:
                        raise ValueError("class ID")
                    if any(value < 0.0 or value > 1.0 for value in values):
                        raise ValueError("coordinate range")
                    if values[2] <= 0.0 or values[3] <= 0.0:
                        raise ValueError("box size")
                except ValueError:
                    invalid_rows += 1
                    continue
                present.add(class_id)
                instance_counts[class_id] += 1
            for class_id in present:
                image_counts[class_id] += 1
            groups_by_split[split].add(int(sample_manifest[label_path.stem]["scene_group"]))
        hashes_by_split[split] = {sha256_file(path) for path in images}
        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "invalid_rows": invalid_rows,
            "class_images": {CLASS_NAMES[cid]: image_counts[cid] for cid in CLASS_NAMES},
            "class_instances": {
                CLASS_NAMES[cid]: instance_counts[cid] for cid in CLASS_NAMES
            },
        }
        if invalid_rows:
            report["errors"].append(f"{split}: {invalid_rows} invalid label rows")
        for class_id in CLASS_NAMES:
            if image_counts[class_id] < 100:
                report["errors"].append(
                    f"{split}: {CLASS_NAMES[class_id]} appears in fewer than 100 images"
                )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        duplicate_hashes = hashes_by_split[left] & hashes_by_split[right]
        overlapping_groups = groups_by_split[left] & groups_by_split[right]
        if duplicate_hashes:
            report["errors"].append(f"{left}/{right}: exact duplicate images found")
        if overlapping_groups:
            report["errors"].append(f"{left}/{right}: scene groups overlap")

    report["valid"] = not report["errors"]
    report["known_limitation"] = manifest["limitation"]
    return report


def main() -> None:
    report = validate()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
