#!/usr/bin/env python3
"""Run the final assignment checklist against produced artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = PROJECT_ROOT / "results/metrics"
EXPECTED_NAMES = {0: "car", 1: "pedestrian", 2: "cyclist"}
DEFAULT_GITHUB_URL = "https://github.com/mk-isos/ML-YOLO-Assignment2"
FINAL_REPORT_NAME = "머신러닝_과제2_2021041023_김문기.pdf"


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
    training_metadata = read_json(METRICS / "training_metadata.json")
    with (METRICS / "training_results.csv").open(encoding="utf-8", newline="") as handle:
        training_rows = list(csv.DictReader(handle))
    validation_scores = [float(row["metrics/mAP50-95(B)"]) for row in training_rows]
    logged_best_epoch = max(range(1, len(validation_scores) + 1), key=lambda epoch: validation_scores[epoch - 1])
    early_stopping_evidence = {
        "configured_epochs": int(training_metadata["arguments"]["epochs"]),
        "patience": int(training_metadata["arguments"]["patience"]),
        "completed_epochs": len(training_rows),
        "best_epoch": logged_best_epoch,
        "best_map50_95": validation_scores[logged_best_epoch - 1],
        "epochs_since_best": len(training_rows) - logged_best_epoch,
    }
    checks["fine_tuning_executed"] = {
        "passed": int(training["completed_epochs"]) > 1
        and float(training["duration_seconds"]) > 0
        and early_stopping_evidence["configured_epochs"] == int(training["configured_epochs"]) == 30
        and early_stopping_evidence["patience"] == 8
        and early_stopping_evidence["completed_epochs"] == int(training["completed_epochs"]) == 16
        and early_stopping_evidence["best_epoch"] == int(training["best_epoch"]) == 8
        and early_stopping_evidence["epochs_since_best"] == early_stopping_evidence["patience"]
        and training_metadata["arguments"]["model"] == "yolov8n.pt",
        "details": {
            "duration_seconds": training["duration_seconds"],
            "early_stopping": early_stopping_evidence,
            "selection_metric": "validation mAP@0.5:0.95 (Ultralytics detection fitness weight 1.0)",
            "base_model": training_metadata["arguments"]["model"],
        },
    }

    transfer = read_json(METRICS / "transfer_audit.json")
    checks["transfer_learning_audit"] = {
        "passed": transfer["ultralytics"] == training_metadata["ultralytics"]
        and transfer["final_weight_sha256"] == sha256_file(weight)
        and bool(transfer["conclusion"]["car_output_rows_reused"])
        and not bool(transfer["conclusion"]["pedestrian_output_row_reused_from_person"])
        and not bool(transfer["conclusion"]["cyclist_output_row_reused_from_bicycle"])
        and all(
            item["car_weight_equal_to_coco_car"] and item["car_bias_equal_to_coco_car"]
            for item in transfer["scale_checks"]
        ),
        "details": {
            "ultralytics": transfer["ultralytics"],
            "loader_method": transfer["implementation"]["loader_method"],
            "target_state_dict_items": transfer["implementation"]["target_state_dict_items"],
            "matched_target_classes": transfer["implementation"]["matched_target_classes"],
            "conclusion": transfer["conclusion"],
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

    pretrained = read_json(METRICS / "pretrained_comparison.json")
    comparable_classes = ("car", "pedestrian")
    checks["pretrained_finetuned_comparison"] = {
        "passed": pretrained["operation"].startswith("inference-only")
        and pretrained["settings"]["metric_scope"].startswith("fixed-threshold")
        and int(pretrained["settings"]["test_images"]) == int(test["test_images"])
        and float(pretrained["settings"]["confidence"]) == 0.25
        and pretrained["fine_tuned"]["sha256"] == sha256_file(weight)
        and bool(pretrained["weight_integrity"]["pretrained_unchanged"])
        and bool(pretrained["weight_integrity"]["fine_tuned_unchanged"])
        and set(pretrained["pretrained"]["per_class"]) == set(comparable_classes)
        and "excluded" in pretrained["mapping"]["cyclist"]
        and (PROJECT_ROOT / pretrained["artifact"]).exists(),
        "details": {
            "scope": pretrained["settings"]["metric_scope"],
            "mapping": pretrained["mapping"],
            "pretrained": pretrained["pretrained"]["per_class"],
            "fine_tuned": pretrained["fine_tuned"]["per_class"],
            "limitations": pretrained["limitations"],
        },
    }

    video = PROJECT_ROOT / "videos/kitti3_tracking_5s.mp4"
    tracking = read_json(METRICS / "tracking_metadata.json")
    continuity = read_json(METRICS / "tracking_continuity.json")
    duration = float(tracking["ffprobe"]["format"]["duration"])
    class_frames = tracking["per_class_frame_count"]
    checks["tracking_video"] = {
        "passed": video.exists()
        and abs(duration - 5.0) <= 0.10
        and int(tracking["decoded_frames"]) == 150
        and tracking["ffprobe"]["streams"][0]["codec_name"] == "h264"
        and tracking["model_sha256"] == sha256_file(weight)
        and bool(tracking["verification"]["decode_valid"])
        and bool(tracking["verification"]["tracking_ids_present"])
        and int(tracking["detections"]) == int(tracking["detections_with_track_id"])
        and int(continuity["detections"]) == int(continuity["detections_with_id"])
        and continuity["model_sha256"] == sha256_file(weight)
        and all(
            int(continuity["per_class"][name]["consecutive_frame_pairs"]) > 0
            and int(continuity["per_class"][name]["ids_seen_in_multiple_frames"]) > 0
            for name in EXPECTED_NAMES.values()
        )
        and all(int(class_frames[name]) > 0 for name in EXPECTED_NAMES.values()),
        "details": {
            "duration_seconds": duration,
            "decoded_frames": tracking["decoded_frames"],
            "codec": tracking["ffprobe"]["streams"][0]["codec_name"],
            "model_sha256": tracking["model_sha256"],
            "per_class_frame_count": class_frames,
            "detections_with_track_id": tracking["detections_with_track_id"],
            "continuity": {
                name: {
                    "unique_ids": continuity["per_class"][name]["unique_ids"],
                    "consecutive_frame_pairs": continuity["per_class"][name]["consecutive_frame_pairs"],
                    "all_observed_frame_pairs": continuity["per_class"][name]["all_observed_frame_pairs"],
                }
                for name in EXPECTED_NAMES.values()
            },
            "tracking_accuracy_limitation": continuity["interpretation"]["not_supported"],
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

    report = PROJECT_ROOT / "reports" / FINAL_REPORT_NAME
    old_report = PROJECT_ROOT / "reports/final_report.pdf"
    reader = PdfReader(str(report))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    pdf_metadata = reader.metadata or {}
    pdftoppm = shutil.which("pdftoppm")
    render_result = None
    rendered_pages: list[Path] = []
    if pdftoppm:
        with tempfile.TemporaryDirectory(prefix="yolo_report_verify_") as directory:
            prefix = Path(directory) / "page"
            render_result = subprocess.run(
                [pdftoppm, "-png", "-r", "90", str(report), str(prefix)],
                text=True,
                capture_output=True,
            )
            rendered_pages = sorted(Path(directory).glob("page-*.png"))
            rendered_page_sizes = [path.stat().st_size for path in rendered_pages]
    else:
        rendered_page_sizes = []
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
        "configured_epochs": f"최대 설정은 {training['configured_epochs']} Epoch",
        "early_stopping_patience": "patience는 8",
        "best_epoch": f"Epoch는 {training['best_epoch']}",
        "transfer_remap": "Remapped 1/3",
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
        "pretrained_car_f1": f"{float(pretrained['pretrained']['per_class']['car']['f1']):.4f}",
        "finetuned_car_f1": f"{float(pretrained['fine_tuned']['per_class']['car']['f1']):.4f}",
        "pretrained_pedestrian_recall": f"{float(pretrained['pretrained']['per_class']['pedestrian']['recall']):.4f}",
        "finetuned_pedestrian_recall": f"{float(pretrained['fine_tuned']['per_class']['pedestrian']['recall']):.4f}",
        "tracking_continuity_car": (
            f"car {continuity['per_class']['car']['consecutive_frame_pairs']}/"
            f"{continuity['per_class']['car']['all_observed_frame_pairs']}"
        ),
        "final_report_name": FINAL_REPORT_NAME,
    }
    consistency = {
        key: {"expected": token, "present": token in extracted_text}
        for key, token in consistency_tokens.items()
    }
    checks["pdf_report"] = {
        "passed": report.exists()
        and report.name == FINAL_REPORT_NAME
        and not old_report.exists()
        and len(reader.pages) >= 12
        and str(pdf_metadata.get("/Author", "")) == "김문기"
        and "머신러닝 과제 2" in str(pdf_metadata.get("/Title", ""))
        and pdftoppm is not None
        and render_result is not None
        and render_result.returncode == 0
        and len(rendered_pages) == len(reader.pages)
        and all(size > 10_000 for size in rendered_page_sizes)
        and all(token in extracted_text for token in metric_tokens)
        and all(item["present"] for item in consistency.values()),
        "details": {
            "filename": report.name,
            "old_report_absent": not old_report.exists(),
            "pages": len(reader.pages),
            "bytes": report.stat().st_size,
            "metadata": {
                "title": pdf_metadata.get("/Title"),
                "author": pdf_metadata.get("/Author"),
                "subject": pdf_metadata.get("/Subject"),
            },
            "rendering": {
                "renderer": pdftoppm,
                "returncode": None if render_result is None else render_result.returncode,
                "rendered_pages": len(rendered_pages),
                "minimum_png_bytes": min(rendered_page_sizes, default=0),
                "stderr_tail": "" if render_result is None else render_result.stderr[-1000:],
            },
            "metric_tokens": metric_tokens,
            "log_report_consistency": consistency,
            "sha256": sha256_file(report),
        },
    }

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    readme_tokens = {
        "report_link": f"reports/{FINAL_REPORT_NAME}",
        "model_link": "models/best.pt",
        "video_link": "videos/kitti3_tracking_5s.mp4",
        "classes": "`car`, `pedestrian`, `cyclist`",
        "test_precision": f"{float(test['overall']['precision']):.4f}",
        "test_recall": f"{float(test['overall']['recall']):.4f}",
        "test_map50": f"{float(test['overall']['map50']):.4f}",
        "test_map50_95": f"{float(test['overall']['map50_95']):.4f}",
    }
    checks["readme_consistency"] = {
        "passed": all(token in readme_text for token in readme_tokens.values())
        and "reports/final_report.pdf" not in readme_text,
        "details": {
            key: {"expected": token, "present": token in readme_text}
            for key, token in readme_tokens.items()
        }
        | {"old_report_reference_absent": "reports/final_report.pdf" not in readme_text},
    }

    tracked_result = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    tracked_files = tracked_result.stdout.splitlines()
    forbidden_prefixes = (
        "references/",
        "data/raw/",
        "data/processed/",
        ".venv/",
        ".cache/",
    )
    forbidden_tracked = [
        path for path in tracked_files if path.startswith(forbidden_prefixes)
    ]
    suspicious_names = [
        path
        for path in tracked_files
        if Path(path).name.lower() in {".env", "credentials", "credentials.json", "id_rsa"}
    ]
    secret_scan = subprocess.run(
        [
            "git",
            "grep",
            "-I",
            "-n",
            "-E",
            r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----)",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    checks["repository_exclusions_and_secrets"] = {
        "passed": tracked_result.returncode == 0
        and not forbidden_tracked
        and not suspicious_names
        and secret_scan.returncode == 1,
        "details": {
            "tracked_files": len(tracked_files),
            "forbidden_tracked": forbidden_tracked,
            "suspicious_names": suspicious_names,
            "secret_pattern_match_count": len(secret_scan.stdout.splitlines()),
        },
    }

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
    repo_slug = args.github_url.removeprefix("https://github.com/").strip("/")
    public_artifacts = {
        "model": PROJECT_ROOT / "models/best.pt",
        "video": video,
        "report": report,
        "test_metrics": METRICS / "test_metrics.json",
        "additional_analysis": METRICS / "additional_analysis.json",
        "confidence_plot": PROJECT_ROOT / "results/plots/confidence_threshold_analysis.png",
    }
    remote_artifacts: dict[str, object] = {}
    artifact_checks: dict[str, bool] = {}
    for name, local_path in public_artifacts.items():
        relative_path = local_path.relative_to(PROJECT_ROOT).as_posix()
        result = subprocess.run(
            ["gh", "api", f"repos/{repo_slug}/contents/{relative_path}"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        data = json.loads(result.stdout) if result.returncode == 0 else {}
        remote_artifacts[name] = {
            "path": relative_path,
            "local_bytes": local_path.stat().st_size,
            "remote_bytes": data.get("size"),
            "download_url": data.get("download_url"),
            "error": result.stderr.strip(),
        }
        artifact_checks[name] = (
            result.returncode == 0
            and int(data.get("size", -1)) == local_path.stat().st_size
            and bool(data.get("download_url"))
        )
    github_checks = {
        "origin_matches": remote.returncode == 0 and remote_url == args.github_url,
        "readme_contains_url": args.github_url in readme_text,
        "report_contains_url": args.github_url in extracted_text,
        "repository_reachable": repository.returncode == 0,
        "repository_public": repository_data.get("visibility") == "PUBLIC",
        "public_artifacts_match": all(artifact_checks.values()),
    }
    checks["github_public_access"] = {
        "passed": all(github_checks.values()),
        "details": {
            "url": args.github_url,
            "origin": remote_url,
            "repository": repository_data,
            "checks": github_checks,
            "artifact_checks": artifact_checks,
            "artifacts": remote_artifacts,
            "gh_error": repository.stderr.strip(),
        },
    }
    core_checks = [check for name, check in checks.items() if not name.startswith("github_")]
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
