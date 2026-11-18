#!/usr/bin/env python3
"""Run the final assignment checklist against produced artifacts."""

from __future__ import annotations

import argparse
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
DEFAULT_GITHUB_URL = "https://github.com/mk-isos/ML-YOLO-Assignment2"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-url", default=DEFAULT_GITHUB_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    analysis = read_json(METRICS / "additional_analysis.json")
    thresholds = analysis["confidence_thresholds"]
    size_recall = analysis["size_recall"]["per_class"]
    analysis_artifacts = [PROJECT_ROOT / path for path in analysis["artifacts"].values()]
    checks["posthoc_diagnostics"] = {
        "passed": analysis["analysis_type"].startswith("post-hoc evaluation only")
        and analysis["model_sha256"] == sha256_file(weight)
        and [round(float(item["threshold"]), 2) for item in thresholds]
        == [0.05, 0.10, 0.25, 0.50, 0.75]
        and all(
            0.0 <= float(item["micro"][metric]) <= 1.0
            for item in thresholds
            for metric in ("precision", "recall", "f1")
        )
        and all(path.exists() and path.stat().st_size > 0 for path in analysis_artifacts)
        and int(analysis["split_distribution"]["val"]["images"])
        == int(dataset["splits"]["val"]["images"])
        and int(analysis["split_distribution"]["test"]["images"])
        == int(dataset["splits"]["test"]["images"]),
        "details": {
            "analysis_type": analysis["analysis_type"],
            "duration_seconds": analysis["duration_seconds"],
            "model_sha256": analysis["model_sha256"],
            "thresholds": [item["threshold"] for item in thresholds],
            "pedestrian_size_recall": {
                size: size_recall["pedestrian"][size]["recall"]
                for size in ("small", "medium", "large")
            },
            "cyclist_size_recall": {
                size: size_recall["cyclist"][size]["recall"]
                for size in ("small", "medium", "large")
            },
            "artifacts": [str(path.relative_to(PROJECT_ROOT)) for path in analysis_artifacts],
        },
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
    environment = read_json(METRICS / "environment.json")
    metric_tokens = [
        f"{float(test['overall']['map50']):.4f}",
        f"{float(test['overall']['map50_95']):.4f}",
        f"{duration:.3f}",
    ]
    consistency_tokens = {
        "author": "김문기",
        "student_id": "2021041023",
        "python_torch": f"{environment['python']} / {environment['torch']}",
        "ultralytics": str(environment["ultralytics"]),
        "completed_epochs": f"{training['completed_epochs']} Epoch",
        "best_epoch": f"Epoch는 {training['best_epoch']}",
        "train_images": str(dataset["splits"]["train"]["images"]),
        "train_car": (
            f"{dataset['splits']['train']['class_images']['car']} / "
            f"{dataset['splits']['train']['class_instances']['car']}"
        ),
        "train_pedestrian": (
            f"{dataset['splits']['train']['class_images']['pedestrian']} / "
            f"{dataset['splits']['train']['class_instances']['pedestrian']}"
        ),
        "train_cyclist": (
            f"{dataset['splits']['train']['class_images']['cyclist']} / "
            f"{dataset['splits']['train']['class_instances']['cyclist']}"
        ),
        "test_precision": f"{float(test['overall']['precision']):.4f}",
        "test_recall": f"{float(test['overall']['recall']):.4f}",
        "test_map50": f"{float(test['overall']['map50']):.4f}",
        "test_map50_95": f"{float(test['overall']['map50_95']):.4f}",
        "car_map50_95": f"{float(test['per_class']['car']['map50_95']):.4f}",
        "pedestrian_map50_95": f"{float(test['per_class']['pedestrian']['map50_95']):.4f}",
        "cyclist_map50_95": f"{float(test['per_class']['cyclist']['map50_95']):.4f}",
        "video_duration_frames": f"{duration:.3f}초 / {tracking['decoded_frames']} 프레임",
        "threshold_f1": f"{float(thresholds[2]['micro']['f1']):.4f}",
        "pedestrian_small_recall": f"{float(size_recall['pedestrian']['small']['recall']):.4f}",
        "cyclist_small_recall": f"{float(size_recall['cyclist']['small']['recall']):.4f}",
        "validation_recheck_map50_95": f"{float(analysis['validation_recheck']['overall']['map50_95']):.4f}",
        "validation_test_gap": f"{float(analysis['validation_minus_test']['overall']['map50_95']):.4f}",
        "test_instances_per_image": f"{float(analysis['split_distribution']['test']['instances_per_image']):.3f}",
    }
    consistency = {
        key: {"expected": token, "present": token in extracted_text}
        for key, token in consistency_tokens.items()
    }
    checks["pdf_report"] = {
        "passed": report.exists()
        and len(reader.pages) >= 12
        and all(token in extracted_text for token in metric_tokens)
        and all(item["present"] for item in consistency.values()),
        "details": {
            "pages": len(reader.pages),
            "bytes": report.stat().st_size,
            "metric_tokens": metric_tokens,
            "log_report_consistency": consistency,
            "sha256": sha256_file(report),
        },
    }

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    remote_url = remote.stdout.strip().removesuffix(".git")
    repository = subprocess.run(
        ["gh", "repo", "view", args.github_url, "--json", "url,visibility"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    repository_data: dict[str, object] = {}
    if repository.returncode == 0:
        repository_data = json.loads(repository.stdout)
    github_checks = {
        "origin_matches": remote.returncode == 0 and remote_url == args.github_url,
        "readme_contains_url": args.github_url in readme_text,
        "report_contains_url": args.github_url in extracted_text,
        "repository_reachable": repository.returncode == 0,
        "repository_public": repository_data.get("visibility") == "PUBLIC",
    }
    checks["github_public_access"] = {
        "passed": all(github_checks.values()),
        "details": {
            "url": args.github_url,
            "origin": remote_url,
            "repository": repository_data,
            "checks": github_checks,
            "gh_error": repository.stderr.strip(),
        },
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
