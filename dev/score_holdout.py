#!/usr/bin/env python3
"""Score HOLD200 while exposing only aggregate, non-diagnostic metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    ids = (MANIFESTS / "hold200.txt").read_text().split()
    raw = args.predictions.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    case_ids = [row.get("case_id") for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise SystemExit("duplicate prediction IDs")
    predictions = {row["case_id"]: row for row in rows}
    missing = set(ids) - set(predictions)
    if missing:
        raise SystemExit(f"missing {len(missing)} HOLD200 predictions")

    all_truth = list(csv.DictReader((CHALLENGE / "data" / "train_labels.csv").open()))
    selected = [row for row in all_truth if row["case_id"] in set(ids)]
    with tempfile.TemporaryDirectory(prefix="mib-hold-score-") as tmp:
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
        subprocess.check_call(
            [sys.executable, str(CHALLENGE / "scripts" / "evaluate.py"),
             "--truth", str(truth_path), "--submission", str(pred_path),
             "--output-json", str(result_path)],
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(result_path.read_text())

    scores = result["scores"]
    public = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "predictions_sha256": hashlib.sha256(raw).hexdigest(),
        "n": len(ids),
        "total": round(scores["total_score"], 4),
        "extraction": round(scores["extraction_score"], 4),
        "classification": round(scores["classification_score"], 4),
        "calibration": round(scores["calibration_score"], 4),
        "false_approvals": result["raw"]["catastrophic_false_approvals"],
        "missing": result["counts"]["missing_cases"],
    }
    log = REPO / "dev" / "holdout_scores.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(public, sort_keys=True) + "\n")
    print(json.dumps(public, sort_keys=True))


if __name__ == "__main__":
    main()
