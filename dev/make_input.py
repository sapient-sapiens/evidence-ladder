#!/usr/bin/env python3
"""Materialize a manifest as a temporary/local PDF input directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files for Docker mounts")
    args = parser.parse_args()

    ids = args.manifest.read_text(encoding="utf-8").split()
    dev = set((REPO / "dev" / "manifests" / "dev800.txt").read_text().split())
    if not ids or not set(ids) <= dev:
        raise SystemExit("routine input manifests must be non-empty DEV800 subsets")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output}")
    for case_id in ids:
        source = CHALLENGE / "data" / "train" / f"{case_id}.pdf"
        target = args.output / source.name
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)
    print(f"materialized {len(ids)} DEV PDFs at {args.output}")


if __name__ == "__main__":
    main()
