#!/usr/bin/env python3
"""Download the exact dataset archive and licensed tracking source used here."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data/raw"
ASSETS = {
    "kitti": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip",
        "path": RAW_ROOT / "kitti.zip",
        "sha256": "1ab901eec94c6bb53182330aa8b3eaed92bc1df4c55a06b3abd8b8a2eca3b8f7",
    },
    "tracking": {
        "url": "https://videos.pexels.com/video-files/4772894/4772894-hd_1920_1080_30fps.mp4",
        "path": RAW_ROOT / "tracking_source_pexels_4772894.mp4",
        "sha256": "6fc37f043670d2ac185341f6e09a1bd151fab51e046d1a2804e84867e22ccb70",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, force: bool = False) -> Path:
    asset = ASSETS[name]
    destination = Path(asset["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        actual = sha256_file(destination)
        if actual == asset["sha256"]:
            print(f"{name}: verified existing {destination}")
            return destination
        raise ValueError(
            f"Existing {destination} has SHA-256 {actual}; expected {asset['sha256']}. "
            "Inspect it or rerun with --force."
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(str(asset["url"]), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    actual = sha256_file(temporary)
    if actual != asset["sha256"]:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"{name}: SHA-256 mismatch {actual} != {asset['sha256']}")
    temporary.replace(destination)
    print(f"{name}: downloaded and verified {destination}")
    return destination


def extract_kitti(archive: Path, force: bool = False) -> Path:
    destination = RAW_ROOT / "kitti-source"
    sentinel = destination / "images/train"
    if sentinel.exists() and not force:
        print(f"kitti: using existing extraction {destination}")
        return destination
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        bundle.extractall(destination)
    print(f"kitti: extracted to {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=("all", "kitti", "tracking"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = ASSETS if args.asset == "all" else {args.asset: ASSETS[args.asset]}
    for name in selected:
        path = download(name, force=args.force)
        if name == "kitti":
            extract_kitti(path, force=args.force)


if __name__ == "__main__":
    main()
