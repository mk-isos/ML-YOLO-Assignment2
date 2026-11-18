#!/usr/bin/env python3
"""Validate, remap, scene-group, and split the Ultralytics KITTI archive.

The source archive exposes eight YOLO class IDs.  This script retains and
densely remaps exactly three:

    source 0 car        -> target 0 car
    source 3 pedestrian -> target 1 pedestrian
    source 5 cyclist    -> target 2 cyclist

KITTI's packaged 2D detection split does not expose recording-sequence IDs.
To reduce scene leakage, images are grouped using two perceptual hashes plus a
low-resolution pixel similarity check.  This is an auditable approximation,
not a claim that the original recording sequence is perfectly reconstructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "data/raw/kitti-source"
ARCHIVE_PATH = PROJECT_ROOT / "data/raw/kitti.zip"
OUTPUT_ROOT = PROJECT_ROOT / "data/processed/kitti3"
CACHE_ROOT = PROJECT_ROOT / "data/cache"

SOURCE_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}
CLASS_REMAP = {0: 0, 3: 1, 5: 2}
TARGET_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


@dataclass(frozen=True)
class Sample:
    source_split: str
    stem: str
    image: Path
    label: Path
    labels: tuple[tuple[int, float, float, float, float], ...]

    @property
    def class_presence(self) -> np.ndarray:
        result = np.zeros(len(TARGET_NAMES), dtype=np.int64)
        for row in self.labels:
            result[row[0]] = 1
        return result


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1
        return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_source_label(path: Path) -> tuple[tuple[int, float, float, float, float], ...]:
    remapped: list[tuple[int, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(parts)}")
        source_id = int(parts[0])
        if source_id not in SOURCE_NAMES:
            raise ValueError(f"{path}:{line_number}: unknown source class {source_id}")
        x, y, width, height = map(float, parts[1:])
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            raise ValueError(f"{path}:{line_number}: coordinate outside [0, 1]")
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        if source_id in CLASS_REMAP:
            remapped.append((CLASS_REMAP[source_id], x, y, width, height))
    if not remapped:
        raise ValueError(f"{path}: no target-class boxes after remapping")
    return tuple(remapped)


def discover_samples(source_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for split in ("train", "val"):
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels" / split
        images = sorted(image_dir.glob("*.png"))
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != len(labels):
            raise ValueError(f"{split}: image/label count mismatch {len(images)} != {len(labels)}")
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                raise FileNotFoundError(f"Missing label for {image}")
            samples.append(
                Sample(
                    source_split=split,
                    stem=image.stem,
                    image=image,
                    label=label,
                    labels=parse_source_label(label),
                )
            )
    if len({sample.stem for sample in samples}) != len(samples):
        raise ValueError("Duplicate image stems found across source splits")
    return samples


def fingerprint_samples(samples: list[Sample], cache_path: Path) -> tuple[np.ndarray, ...]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stems = np.array([sample.stem for sample in samples])
    if cache_path.exists():
        cached = np.load(cache_path)
        if np.array_equal(cached["stems"], stems):
            return cached["phash"], cached["dhash"], cached["signature"]

    phash = np.zeros(len(samples), dtype=np.uint64)
    dhash = np.zeros(len(samples), dtype=np.uint64)
    signature = np.zeros((len(samples), 20 * 64 * 3), dtype=np.uint8)
    for index, sample in enumerate(samples):
        with Image.open(sample.image) as image:
            rgb = image.convert("RGB")
            phash[index] = int(str(imagehash.phash(rgb, hash_size=8)), 16)
            dhash[index] = int(str(imagehash.dhash(rgb, hash_size=8)), 16)
            reduced = rgb.resize((64, 20), Image.Resampling.BILINEAR)
            signature[index] = np.asarray(reduced, dtype=np.uint8).reshape(-1)
    np.savez_compressed(cache_path, stems=stems, phash=phash, dhash=dhash, signature=signature)
    return phash, dhash, signature


def build_scene_groups(
    samples: list[Sample],
    phash: np.ndarray,
    dhash: np.ndarray,
    signature: np.ndarray,
    phash_threshold: int = 20,
    dhash_threshold: int = 12,
    mae_threshold: float = 0.065,
    correlation_threshold: float = 0.90,
) -> tuple[list[list[int]], dict[str, int | float]]:
    """Connect images only when both hash and pixel checks indicate the same scene."""
    union_find = UnionFind(len(samples))
    linked_pairs = 0
    cross_original_split_pairs = 0
    signature_float = signature.astype(np.float32) / 255.0
    centered = signature_float - signature_float.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)

    for left in range(len(samples) - 1):
        ph_dist = np.bitwise_count(np.bitwise_xor(phash[left + 1 :], phash[left]))
        dh_dist = np.bitwise_count(np.bitwise_xor(dhash[left + 1 :], dhash[left]))
        candidate_offsets = np.flatnonzero(
            (ph_dist <= phash_threshold) & (dh_dist <= dhash_threshold)
        )
        for offset in candidate_offsets:
            right = left + 1 + int(offset)
            mae = float(np.abs(signature_float[left] - signature_float[right]).mean())
            if mae > mae_threshold:
                continue
            denominator = float(norms[left] * norms[right])
            correlation = (
                float(np.dot(centered[left], centered[right]) / denominator)
                if denominator > 0.0
                else 0.0
            )
            if correlation < correlation_threshold:
                continue
            if union_find.union(left, right):
                linked_pairs += 1
            if samples[left].source_split != samples[right].source_split:
                cross_original_split_pairs += 1

    by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        by_root[union_find.find(index)].append(index)
    groups = sorted(by_root.values(), key=lambda group: min(samples[index].stem for index in group))
    audit: dict[str, int | float] = {
        "scene_groups": len(groups),
        "multi_image_groups": sum(len(group) > 1 for group in groups),
        "largest_group": max(map(len, groups)),
        "linked_pairs": linked_pairs,
        "cross_original_split_similar_pairs": cross_original_split_pairs,
        "phash_threshold": phash_threshold,
        "dhash_threshold": dhash_threshold,
        "mae_threshold": mae_threshold,
        "correlation_threshold": correlation_threshold,
    }
    return groups, audit


def assign_groups(
    samples: list[Sample], groups: list[list[int]], seed: int = 42
) -> dict[str, list[int]]:
    split_names = ("train", "val", "test")
    ratios = np.array([0.70, 0.15, 0.15], dtype=np.float64)
    presence = np.stack([sample.class_presence for sample in samples])
    target_size = ratios * len(samples)
    target_class = ratios[:, None] * presence.sum(axis=0)[None, :]
    current_size = np.zeros(3, dtype=np.float64)
    current_class = np.zeros((3, len(TARGET_NAMES)), dtype=np.float64)

    randomizer = random.Random(seed)
    ordered_groups = groups.copy()
    randomizer.shuffle(ordered_groups)
    class_totals = np.maximum(presence.sum(axis=0), 1)

    def priority(group: list[int]) -> tuple[float, int]:
        class_presence = presence[group].sum(axis=0)
        rarity = float((class_presence / class_totals).sum())
        return rarity, len(group)

    ordered_groups.sort(key=priority, reverse=True)
    assigned: dict[str, list[int]] = {name: [] for name in split_names}

    for group in ordered_groups:
        group_class = presence[group].sum(axis=0)
        scores: list[float] = []
        for split_index in range(3):
            next_size = current_size.copy()
            next_class = current_class.copy()
            next_size[split_index] += len(group)
            next_class[split_index] += group_class
            size_error = np.square((next_size - target_size) / np.maximum(target_size, 1)).sum()
            class_error = np.square(
                (next_class - target_class) / np.maximum(target_class, 1)
            ).sum()
            overfill = np.square(
                np.maximum(0.0, next_size - target_size * 1.03)
                / np.maximum(target_size, 1)
            ).sum()
            scores.append(float(size_error + 1.5 * class_error + 20.0 * overfill))
        chosen = min(range(3), key=lambda index: (scores[index], index))
        current_size[chosen] += len(group)
        current_class[chosen] += group_class
        assigned[split_names[chosen]].extend(group)

    for split in split_names:
        assigned[split].sort(key=lambda index: samples[index].stem)
    return assigned


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_dataset(
    samples: list[Sample],
    groups: list[list[int]],
    assigned: dict[str, list[int]],
    output_root: Path,
    archive_sha256: str,
    scene_audit: dict[str, int | float],
) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    group_id_by_index = {
        sample_index: group_id
        for group_id, group in enumerate(groups)
        for sample_index in group
    }
    manifest_rows: list[dict[str, object]] = []
    split_summary: dict[str, object] = {}

    for split, indexes in assigned.items():
        image_counts = Counter()
        instance_counts = Counter()
        for index in indexes:
            sample = samples[index]
            image_destination = output_root / "images" / split / f"{sample.stem}.png"
            label_destination = output_root / "labels" / split / f"{sample.stem}.txt"
            link_or_copy(sample.image, image_destination)
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.write_text(
                "".join(
                    f"{class_id} {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n"
                    for class_id, x, y, width, height in sample.labels
                ),
                encoding="utf-8",
            )
            present = {row[0] for row in sample.labels}
            for class_id in present:
                image_counts[class_id] += 1
            for class_id, *_ in sample.labels:
                instance_counts[class_id] += 1
            manifest_rows.append(
                {
                    "stem": sample.stem,
                    "split": split,
                    "source_split": sample.source_split,
                    "scene_group": group_id_by_index[index],
                    "classes": sorted(present),
                    "instances": len(sample.labels),
                }
            )
        split_summary[split] = {
            "images": len(indexes),
            "class_images": {TARGET_NAMES[cid]: image_counts[cid] for cid in TARGET_NAMES},
            "class_instances": {
                TARGET_NAMES[cid]: instance_counts[cid] for cid in TARGET_NAMES
            },
        }

    dataset_yaml = """# Generated by src/prepare_dataset.py\npath: {path}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: car\n  1: pedestrian\n  2: cyclist\n""".format(path=output_root.resolve().as_posix())
    (output_root / "data.yaml").write_text(dataset_yaml, encoding="utf-8")

    manifest: dict[str, object] = {
        "source": {
            "name": "KITTI 2D object detection, Ultralytics YOLO conversion",
            "origin_url": "https://www.cvlibs.net/datasets/kitti/",
            "download_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip",
            "license_note": "KITTI states that the dataset is available for academic use only.",
            "archive_sha256": archive_sha256,
            "source_class_names": SOURCE_NAMES,
        },
        "class_remap": {
            "0 car": "0 car",
            "3 pedestrian": "1 pedestrian",
            "5 cyclist": "2 cyclist",
        },
        "split_seed": 42,
        "split_method": "scene-group-aware greedy multilabel balance, 70/15/15",
        "scene_grouping": scene_audit,
        "limitation": (
            "The packaged KITTI detection data provides no recording-sequence ID. "
            "Scene groups are inferred from perceptual hashes and low-resolution pixel similarity; "
            "undetected same-drive or similar-scene leakage may remain."
        ),
        "splits": split_summary,
        "samples": sorted(manifest_rows, key=lambda row: str(row["stem"])),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    if not args.archive.exists():
        raise FileNotFoundError(f"Archive not found: {args.archive}")
    samples = discover_samples(args.source)
    phash, dhash, signature = fingerprint_samples(
        samples, CACHE_ROOT / "kitti_fingerprints_v1.npz"
    )
    groups, scene_audit = build_scene_groups(samples, phash, dhash, signature)
    assigned = assign_groups(samples, groups)
    manifest = write_dataset(
        samples,
        groups,
        assigned,
        args.output,
        sha256_file(args.archive),
        scene_audit,
    )
    print(json.dumps({"scene_grouping": scene_audit, "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
