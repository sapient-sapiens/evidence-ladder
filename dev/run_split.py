#!/usr/bin/env python3
"""Materialize, run, time, validate, and score one sanctioned DEV split."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"
RUNS = REPO / "dev" / "runs"


def resolve_manifest(name: str) -> Path:
    path = MANIFESTS / f"{name}.txt"
    if not path.is_file():
        choices = ", ".join(sorted(p.stem for p in MANIFESTS.glob("*.txt") if p.stem != "hold200"))
        raise SystemExit(f"unknown DEV split {name!r}; choose one of: {choices}")
    ids = path.read_text().split()
    dev = set((MANIFESTS / "dev800.txt").read_text().split())
    if not ids or not set(ids) <= dev:
        raise SystemExit("run_split.py accepts DEV800 subsets only")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="Manifest stem, e.g. tune100 or fit_fold_0")
    parser.add_argument("--tag", required=True, help="Short unique experiment/run label")
    parser.add_argument(
        "--truth",
        type=Path,
        required=True,
        help="Explicit DEV800 truth CSV (never inferred from combined labels)",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/mib-core-ocr-cache"))
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--build", action="store_true", help="Build the Docker image before running")
    parser.add_argument("--image", default="mib-core:dev")
    args = parser.parse_args()
    if args.build and not args.docker:
        raise SystemExit("--build requires --docker")
    if not args.tag.replace("-", "").replace("_", "").isalnum():
        raise SystemExit("tag must contain only letters, digits, hyphens, and underscores")
    if not args.truth.is_file():
        raise SystemExit(f"truth CSV does not exist: {args.truth}")
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass explicit DEV800 truth")

    manifest = resolve_manifest(args.split)
    ids = manifest.read_text().split()
    run_dir = RUNS / args.tag
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions = run_dir / "predictions.jsonl"
    metadata_path = run_dir / "run.json"

    with tempfile.TemporaryDirectory(prefix=f"mib-{args.split}-") as tmp:
        input_dir = Path(tmp) / "input"
        input_dir.mkdir()
        for case_id in ids:
            source = CHALLENGE / "data" / "train" / f"{case_id}.pdf"
            target = input_dir / source.name
            if args.docker:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source)

        if args.docker:
            if args.build:
                subprocess.check_call(["docker", "build", "-t", args.image, str(REPO)])
            command = [
                "docker", "run", "--rm", "--network", "none", "--read-only",
                "--cpus", "4", "--memory", "8g", "--pids-limit", "512",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
                "-e", "MIB_WORKERS=4",
                "-v", f"{input_dir}:/input:ro",
                "-v", f"{run_dir.resolve()}:/output",
                args.image, "/input", "/output/predictions.jsonl",
            ]
            environment = None
        else:
            command = [sys.executable, str(REPO / "solution.py"), str(input_dir), str(predictions)]
            environment = os.environ.copy()
            environment["MIB_WORKERS"] = str(max(1, args.workers))
            environment["MIB_OCR_CACHE"] = str(args.cache)

        started = time.perf_counter()
        subprocess.check_call(command, cwd=REPO, env=environment)
        elapsed = time.perf_counter() - started

    subprocess.check_call([
        sys.executable, str(CHALLENGE / "scripts" / "validate_submission.py"),
        "--submission", str(predictions), "--pdf-dir", str(CHALLENGE / "data" / "train"),
    ], stdout=subprocess.DEVNULL)
    metadata = {
        "tag": args.tag,
        "split": args.split,
        "n": len(ids),
        "mode": "docker" if args.docker else "local",
        "workers": 4 if args.docker else args.workers,
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_pdf": round(elapsed / len(ids), 4),
        "cache": "container_tmp_cold" if args.docker else str(args.cache),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    subprocess.check_call([
        sys.executable, str(REPO / "dev" / "evaluate_split.py"),
        "--predictions", str(predictions), "--manifest", str(manifest),
        "--truth", str(args.truth),
    ])


if __name__ == "__main__":
    main()
