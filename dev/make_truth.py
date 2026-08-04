#!/usr/bin/env python3
"""Write dev/truth/dev800_labels.csv from the challenge's public train labels.

The organizer's labels are not vendored here.  This regenerates the DEV800-only
truth file that `dev/evaluate_split.py` requires (it refuses the combined
train_labels.csv on purpose, so a split can never be scored against labels it
was fitted on).
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent


def main() -> None:
    ids = set((REPO / "dev" / "manifests" / "dev800.txt").read_text().split())
    source = CHALLENGE / "data" / "train_labels.csv"
    if not source.is_file():
        raise SystemExit(f"challenge labels not found at {source}")
    rows = [r for r in csv.DictReader(source.open()) if r["case_id"] in ids]
    if len(rows) != len(ids):
        raise SystemExit(f"expected {len(ids)} rows, matched {len(rows)}")
    out = REPO / "dev" / "truth" / "dev800_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
