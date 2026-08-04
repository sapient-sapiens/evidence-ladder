#!/usr/bin/env python3
"""Evaluate a DEV-only manifest; case detail is never available for HOLD200."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    parser.add_argument("--case-details", type=Path)
    args = parser.parse_args()

    ids = args.manifest.read_text(encoding="utf-8").split()
    dev = set((MANIFESTS / "dev800.txt").read_text().split())
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    if not ids or not set(ids) <= dev:
        raise SystemExit("evaluate_split.py accepts DEV800 subsets only")

    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    predictions = {row["case_id"]: row for row in rows}
    if len(predictions) != len(rows):
        raise SystemExit("duplicate prediction IDs")
    missing = set(ids) - set(predictions)
    if missing:
        raise SystemExit(f"missing {len(missing)} requested predictions")

    all_truth = list(csv.DictReader(args.truth.open()))
    selected = [row for row in all_truth if row["case_id"] in set(ids)]
    with tempfile.TemporaryDirectory(prefix="mib-dev-score-") as tmp:
        tmp_path = Path(tmp)
        truth_path = tmp_path / "truth.csv"
        pred_path = tmp_path / "predictions.jsonl"
        result_path = tmp_path / "evaluation.json"
        with truth_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
            writer.writeheader()
            writer.writerows(selected)
        with pred_path.open("w", encoding="utf-8") as handle:
            for case_id in ids:
                handle.write(json.dumps(predictions[case_id], sort_keys=True) + "\n")
        command = [
            sys.executable,
            str(CHALLENGE / "scripts" / "evaluate.py"),
            "--truth", str(truth_path),
            "--submission", str(pred_path),
            "--output-json", str(result_path),
        ]
        if args.case_details:
            command.extend(["--case-scores-jsonl", str(args.case_details)])
        subprocess.check_call(command, stdout=subprocess.DEVNULL)
        result = json.loads(result_path.read_text())
    scores = result["scores"]
    public = {
        "split": args.manifest.stem,
        "n": len(ids),
        "total": scores["total_score"],
        "extraction": scores["extraction_score"],
        "classification": scores["classification_score"],
        "calibration": scores["calibration_score"],
        "false_approvals": result["raw"]["catastrophic_false_approvals"],
    }
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
