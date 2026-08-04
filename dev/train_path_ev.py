#!/usr/bin/env python3
"""Fit per-path empirical EV tables from dumped production state."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dev"))

from src.arbiter import LABELS
from path_ev import DEFAULT_M, fit_path_tables, save_path_ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "dev" / "runs" / "path_ev.json")
    parser.add_argument("--m", type=float, default=DEFAULT_M)
    parser.add_argument(
        "--ids",
        type=Path,
        help="Optional manifest restricting which state rows are used for fitting",
    )
    args = parser.parse_args()
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")

    truth = {r["case_id"]: r for r in csv.DictReader(args.truth.open())}
    allow = set(args.ids.read_text().split()) if args.ids else None
    states = []
    labels = []
    for line in args.state.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["case_id"]
        if allow is not None and cid not in allow:
            continue
        states.append(row)
        labels.append(truth[cid]["adjudication"])
    blob = fit_path_tables(states, labels, m=args.m)
    out = save_path_ev(blob, args.out)
    print(
        json.dumps(
            {
                "out": str(out),
                "n_train": blob["n_train"],
                "n_paths": blob["n_paths"],
                "m": blob["m"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
