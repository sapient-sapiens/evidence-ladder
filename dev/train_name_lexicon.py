#!/usr/bin/env python3
"""Build the applicant-name token lexicon from frozen FIT600 labels only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"


def validate_fit_manifest(path: Path) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    dev = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= dev:
        raise SystemExit("name lexicon training is restricted to a unique DEV800 subset")
    return ids


def build(ids: list[str], manifest: Path, truth_path: Path) -> dict:
    if truth_path.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    allowed = set(ids)
    first: Counter[str] = Counter()
    last: Counter[str] = Counter()
    skipped = 0
    for row in csv.DictReader(
        truth_path.open(encoding="utf-8")
    ):
        if row["case_id"] not in allowed:
            continue
        parts = row["applicant_name"].split()
        if len(parts) != 2:
            skipped += 1
            continue
        first[parts[0]] += 1
        last[parts[1]] += 1
    if sum(first.values()) + skipped != len(ids):
        raise SystemExit("FIT labels did not cover the manifest exactly")
    tokens = first + last
    return {
        "version": 2,
        "source": manifest.name,
        "fit_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "n_names": sum(first.values()),
        "n_skipped": skipped,
        "n_unique_first": len(first),
        "n_unique_last": len(last),
        "n_unique_tokens": len(tokens),
        "first_tokens": dict(sorted(first.items())),
        "last_tokens": dict(sorted(last.items())),
        "tokens": dict(sorted(tokens.items())),
        "anti_leakage": {
            "contains_case_ids": False,
            "contains_full_names": False,
            "shipped": "token_counts_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFESTS / "fit600.txt"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    args = parser.parse_args()
    ids = validate_fit_manifest(args.manifest)
    artifact = build(ids, args.manifest, args.truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: artifact[k] for k in (
        "source", "fit_manifest_sha256", "n_names", "n_skipped",
        "n_unique_first", "n_unique_last", "n_unique_tokens",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
