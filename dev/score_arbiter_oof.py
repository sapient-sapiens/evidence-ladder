#!/usr/bin/env python3
"""Score the arbiter out of fold using the shipped decision rule.

Replays ``src.arbiter.apply_arbiter`` against cross-fitted probabilities so the
measured split score reflects exactly what the runtime would emit if the model
had never seen the packet.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from src.arbiter import LABELS, UTILITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--proba", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--margin", type=float, default=2.0)
    args = parser.parse_args()

    import csv

    truth = {
        r["case_id"]: r["adjudication"]
        for r in csv.DictReader(open(CHALLENGE / "data/train_labels.csv"))
    }
    states = {
        json.loads(l)["case_id"]: json.loads(l)
        for l in (REPO / args.state).read_text().splitlines() if l
    }
    proba = {
        json.loads(l)["case_id"]: json.loads(l)["proba"]
        for l in (REPO / args.proba).read_text().splitlines() if l
    }
    changes = Counter()
    out = []
    for cid, state in states.items():
        pred = dict(state["prediction"])
        p = proba[cid]
        current = pred["adjudication"]
        eligible = (
            current == "NEEDS_REVIEW"
            and state.get("adjudicator_finding") != "NEEDS_REVIEW"
        )
        action = current
        if eligible:
            eu = {
                a: sum(UTILITY[a][LABELS[k]] * p[k] for k in range(3)) for a in LABELS
            }
            if eu["APPROVED"] - eu["NEEDS_REVIEW"] > args.margin:
                action = "APPROVED"
        if action != current:
            pred["adjudication"] = action
            pred["confidence"] = round(min(max(p[LABELS.index(action)], 0.01), 0.99), 4)
            changes[(current, action, truth[cid])] += 1
        else:
            blended = 0.5 * p[LABELS.index(current)] + 0.5 * float(
                state.get("runtime_confidence") or pred["confidence"]
            )
            pred["confidence"] = round(min(max(blended, 0.01), 0.99), 4)
        out.append(pred)

    run_dir = REPO / "dev/runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for pred in out:
            f.write(json.dumps(pred, sort_keys=True) + "\n")
    for k, v in sorted(changes.items(), key=lambda i: -i[1]):
        print("  change", k, v)
    subprocess.run(
        [sys.executable, str(REPO / "dev/evaluate_split.py"),
         "--predictions", str(path), "--manifest", str(REPO / args.manifest)],
        check=True,
    )


if __name__ == "__main__":
    main()
