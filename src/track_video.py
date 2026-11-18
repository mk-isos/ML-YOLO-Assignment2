#!/usr/bin/env python3
"""Create and verify a five-second ByteTrack result using the fine-tuned model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/tracking.yaml"
VIDEOS_ROOT = PROJECT_ROOT / "videos"
METRICS_ROOT = PROJECT_ROOT / "results/metrics"
PLOTS_ROOT = PROJECT_ROOT / "results/plots"
EXPECTED_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, object]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def create_clip(source: Path, destination: Path, start: float, duration: float, fps: int) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-vf",
            f"fps={fps}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )


def verify_decode(path: Path) -> int:
    result = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    if result.stderr.strip():
        raise RuntimeError(f"ffmpeg decode reported errors: {result.stderr}")
    capture = cv2.VideoCapture(str(path))
    frame_count = 0
    while True:
        okay, _ = capture.read()
        if not okay:
            break
        frame_count += 1
    capture.release()
    if frame_count == 0:
        raise RuntimeError("Output video contains no decodable frames")
    return frame_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing to silently use CPU")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source = PROJECT_ROOT / config["source"]["file"]
    if not source.exists():
        raise FileNotFoundError(source)
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    model = YOLO(str(args.model.resolve()))
    if model.names != EXPECTED_NAMES:
        raise ValueError(f"Unexpected model classes: {model.names}")

    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = VIDEOS_ROOT / "kitti3_tracking_5s.mp4"
    csv_path = METRICS_ROOT / "tracking_detections.csv"
    metadata_path = METRICS_ROOT / "tracking_metadata.json"
    representative_frame = PLOTS_ROOT / "tracking_frame.jpg"

    clip_config = config["clip"]
    tracking_config = config["tracking"]
    with tempfile.TemporaryDirectory(prefix="yolo_track_") as temporary_directory:
        temporary_root = Path(temporary_directory)
        clip_path = temporary_root / "input_5s.mp4"
        intermediate_path = temporary_root / "annotated.mp4"
        create_clip(
            source,
            clip_path,
            float(clip_config["start_seconds"]),
            float(clip_config["duration_seconds"]),
            int(clip_config["output_fps"]),
        )

        capture = cv2.VideoCapture(str(clip_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not open the intermediate video writer")

        rows: list[dict[str, object]] = []
        per_class_detections = Counter()
        per_class_frames: dict[str, set[int]] = defaultdict(set)
        per_class_track_ids: dict[str, set[int]] = defaultdict(set)
        tracked_detections = 0
        frame_index = 0
        started = time.perf_counter()
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            result = model.track(
                frame,
                persist=True,
                tracker=str(tracking_config["tracker"]),
                conf=float(tracking_config["confidence"]),
                iou=float(tracking_config["iou"]),
                imgsz=int(tracking_config["image_size"]),
                device=args.device,
                verbose=False,
            )[0]
            writer.write(result.plot(line_width=3, font_size=0.7))
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                confidences = result.boxes.conf.detach().cpu().numpy()
                track_ids = (
                    result.boxes.id.detach().cpu().numpy().astype(int)
                    if result.boxes.id is not None
                    else [-1] * len(boxes)
                )
                for box, class_id, confidence, track_id in zip(
                    boxes, classes, confidences, track_ids, strict=True
                ):
                    class_name = EXPECTED_NAMES[int(class_id)]
                    per_class_detections[class_name] += 1
                    per_class_frames[class_name].add(frame_index)
                    if int(track_id) >= 0:
                        tracked_detections += 1
                        per_class_track_ids[class_name].add(int(track_id))
                    rows.append(
                        {
                            "frame": frame_index,
                            "time_seconds": frame_index / fps,
                            "track_id": int(track_id),
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "confidence": float(confidence),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )
            frame_index += 1
        inference_seconds = time.perf_counter() - started
        capture.release()
        writer.release()

        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(intermediate_path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                "2.5",
                "-i",
                str(output_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(representative_frame),
            ]
        )

    fieldnames = [
        "frame",
        "time_seconds",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_probe = probe(output_path)
    decoded_frames = verify_decode(output_path)
    duration = float(output_probe["format"]["duration"])
    expected_duration = float(clip_config["duration_seconds"])
    if abs(duration - expected_duration) > 0.10:
        raise RuntimeError(f"Unexpected duration {duration:.3f}s; expected {expected_duration:.3f}s")
    if tracked_detections == 0:
        raise RuntimeError("No tracking IDs were produced")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "source": config["source"],
        "clip": clip_config,
        "tracking": tracking_config,
        "device": args.device,
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "representative_frame": str(representative_frame.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_path),
        "ffprobe": output_probe,
        "decoded_frames": decoded_frames,
        "processed_frames": frame_index,
        "inference_seconds": inference_seconds,
        "detections": len(rows),
        "detections_with_track_id": tracked_detections,
        "per_class_detection_count": dict(per_class_detections),
        "per_class_frame_count": {
            name: len(per_class_frames[name]) for name in EXPECTED_NAMES.values()
        },
        "per_class_unique_track_ids": {
            name: sorted(per_class_track_ids[name]) for name in EXPECTED_NAMES.values()
        },
        "csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "verification": {
            "duration_tolerance_seconds": 0.10,
            "duration_valid": abs(duration - expected_duration) <= 0.10,
            "decode_valid": decoded_frames > 0,
            "tracking_ids_present": tracked_detections > 0,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
