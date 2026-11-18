#!/usr/bin/env python3
"""Measure track-ID continuity from the saved detection CSV without claiming GT tracking accuracy."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results/metrics/tracking_detections.csv"
METADATA_PATH = PROJECT_ROOT / "results/metrics/tracking_metadata.json"
OUTPUT_PATH = PROJECT_ROOT / "results/metrics/tracking_continuity.json"
CLASSES = ("car", "pedestrian", "cyclist")


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        track_id = int(row["track_id"])
        if track_id >= 0:
            tracks[(row["class_name"], track_id)].append(int(row["frame"]))

    per_class: dict[str, object] = {}
    for class_name in CLASSES:
        track_records = []
        for (name, track_id), frames in tracks.items():
            if name != class_name:
                continue
            unique_frames = sorted(set(frames))
            gaps = [right - left for left, right in zip(unique_frames, unique_frames[1:])]
            track_records.append(
                {
                    "track_id": track_id,
                    "detections": len(unique_frames),
                    "first_frame": unique_frames[0],
                    "last_frame": unique_frames[-1],
                    "span_frames": unique_frames[-1] - unique_frames[0] + 1,
                    "consecutive_pairs": sum(gap == 1 for gap in gaps),
                    "all_frame_pairs": len(gaps),
                    "max_gap": max(gaps, default=0),
                }
            )
        all_pairs = sum(item["all_frame_pairs"] for item in track_records)
        consecutive_pairs = sum(item["consecutive_pairs"] for item in track_records)
        per_class[class_name] = {
            "unique_ids": len(track_records),
            "ids_seen_in_multiple_frames": sum(item["detections"] >= 2 for item in track_records),
            "ids_seen_in_10plus_frames": sum(item["detections"] >= 10 for item in track_records),
            "longest_detection_count": max(item["detections"] for item in track_records),
            "longest_span_frames": max(item["span_frames"] for item in track_records),
            "consecutive_frame_pairs": consecutive_pairs,
            "all_observed_frame_pairs": all_pairs,
            "consecutive_pair_ratio": consecutive_pairs / all_pairs if all_pairs else None,
            "longest_tracks": sorted(
                track_records, key=lambda item: item["detections"], reverse=True
            )[:3],
        }

    result = {
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "video": metadata["output"],
        "model_sha256": metadata["model_sha256"],
        "detections": len(rows),
        "detections_with_id": sum(int(row["track_id"]) >= 0 for row in rows),
        "per_class": per_class,
        "interpretation": {
            "supported": (
                "The same tracker IDs occur across adjacent frames, so the output is frame-to-frame tracking "
                "rather than independent per-frame detection only."
            ),
            "not_supported": (
                "No ground-truth track IDs are available for this clip; IDF1, HOTA, identity-switch counts, "
                "and claims of stable or accurate tracking are not measured."
            ),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
