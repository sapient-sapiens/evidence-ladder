#!/usr/bin/env python3
"""Run a frozen candidate on HOLD200 and emit aggregate metrics only."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFEST = REPO / "dev" / "manifests" / "hold200.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/mib-core-hold-cache"))
    parser.add_argument("--docker", action="store_true")
    parser.add_argument("--image", default="mib-core:dev")
    args = parser.parse_args()
    ids = MANIFEST.read_text().split()

    with tempfile.TemporaryDirectory(prefix="mib-hold-") as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        predictions = tmp_path / "predictions.jsonl"
        for case_id in ids:
            source = CHALLENGE / "data" / "train" / f"{case_id}.pdf"
            target = input_dir / source.name
            if args.docker:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source)
        if args.docker:
            command = [
                "docker", "run", "--rm", "--network", "none", "--read-only",
                "--cpus", "4", "--memory", "8g", "--pids-limit", "512",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "-e", "MIB_WORKERS=4",
                "-v", f"{input_dir}:/input:ro", "-v", f"{tmp_path}:/output",
                args.image, "/input", "/output/predictions.jsonl",
            ]
            environment = None
        else:
            command = [sys.executable, str(REPO / "solution.py"), str(input_dir), str(predictions)]
            environment = os.environ.copy()
            environment["MIB_WORKERS"] = str(max(1, args.workers))
            environment["MIB_OCR_CACHE"] = str(args.cache)
        subprocess.check_call(command, cwd=REPO, env=environment)
        subprocess.check_call([
            sys.executable, str(REPO / "dev" / "score_holdout.py"),
            "--predictions", str(predictions), "--candidate", args.candidate,
        ])


if __name__ == "__main__":
    main()
