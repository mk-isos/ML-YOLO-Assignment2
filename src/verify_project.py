#!/usr/bin/env python3
"""Run the final assignment checklist against produced artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = PROJECT_ROOT / "results/metrics"
EXPECTED_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    dataset = read_json(METRICS / "dataset_validation.json")
    checks["dataset_valid"] = {"passed": bool(dataset["valid"]), "details": dataset["errors"]}
    class_counts = {
        split: dataset["splits"][split]["class_images"] for split in ("train", "val", "test")
    }
    checks["three_classes_and_counts"] = {
        "passed": all(
            set(counts) == set(EXPECTED_NAMES.values()) and min(counts.values()) >= 100
            for counts in class_counts.values()
        ),
        "details": class_counts,
    }

    weight = PROJECT_ROOT / "models/best.pt"
    model = YOLO(str(weight))
    checks["model_loadable_exact_three_classes"] = {
        "passed": weight.exists() and model.names == EXPECTED_NAMES,
        "details": {
            "path": str(weight.relative_to(PROJECT_ROOT)),
            "names": model.names,
            "bytes": weight.stat().st_size,
            "sha256": sha256_file(weight),
        },
    }
    training = read_json(METRICS / "training_summary.json")
    checks["fine_tuning_executed"] = {
        "passed": int(training["completed_epochs"]) > 1 and float(training["duration_seconds"]) > 0,
        "details": {
            "completed_epochs": training["completed_epochs"],
            "duration_seconds": training["duration_seconds"],
            "best_epoch": training["best_epoch"],
        },
    }

    test = read_json(METRICS / "test_metrics.json")
    checks["held_out_test_metrics"] = {
        "passed": test["split"] == "test"
        and int(test["test_images"]) == int(dataset["splits"]["test"]["images"])
        and all(0.0 <= float(value) <= 1.0 for value in test["overall"].values()),
        "details": {"images": test["test_images"], **test["overall"]},
    }

    video = PROJECT_ROOT / "videos/kitti3_tracking_5s.mp4"
    tracking = read_json(METRICS / "tracking_metadata.json")
    duration = float(tracking["ffprobe"]["format"]["duration"])
    class_frames = tracking["per_class_frame_count"]
    checks["tracking_video"] = {
        "passed": video.exists()
        and abs(duration - 5.0) <= 0.10
        and bool(tracking["verification"]["decode_valid"])
        and bool(tracking["verification"]["tracking_ids_present"])
        and all(int(class_frames[name]) > 0 for name in EXPECTED_NAMES.values()),
        "details": {
            "duration_seconds": duration,
            "decoded_frames": tracking["decoded_frames"],
            "per_class_frame_count": class_frames,
            "detections_with_track_id": tracking["detections_with_track_id"],
            "sha256": sha256_file(video),
        },
    }
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        text=True,
        capture_output=True,
    )
    checks["video_full_decode"] = {
        "passed": decode.returncode == 0 and not decode.stderr.strip(),
        "details": decode.stderr.strip(),
    }

    report = PROJECT_ROOT / "reports/final_report.pdf"
    reader = PdfReader(str(report))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    metric_tokens = [
        f"{float(test['overall']['map50']):.4f}",
        f"{float(test['overall']['map50_95']):.4f}",
        f"{duration:.3f}",
    ]
    checks["pdf_report"] = {
        "passed": report.exists()
        and len(reader.pages) >= 8
        and all(token in extracted_text for token in metric_tokens),
        "details": {
            "pages": len(reader.pages),
            "bytes": report.stat().st_size,
            "metric_tokens": metric_tokens,
            "sha256": sha256_file(report),
        },
    }

    checks["github_public_access"] = {
        "passed": False,
        "pending_user_approval": True,
        "details": "Repository creation and public push intentionally deferred until explicit approval.",
    }
    core_checks = [check for name, check in checks.items() if name != "github_public_access"]
    report_data = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "core_passed": all(bool(check["passed"]) for check in core_checks),
        "all_passed_including_github": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }
    output = METRICS / "final_verification.json"
    output.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))
    if not report_data["core_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
