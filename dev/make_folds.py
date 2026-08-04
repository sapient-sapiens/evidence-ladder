#!/usr/bin/env python3
"""Create six deterministic, stratified working folds inside FIT600."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"
SEED = 80902026


def main() -> None:
    fit = (MANIFESTS / "fit600.txt").read_text(encoding="utf-8").split()
    dev = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    hold = set((MANIFESTS / "hold200.txt").read_text(encoding="utf-8").split())
    if len(fit) != 600 or not set(fit) <= dev or set(fit) & hold:
        raise SystemExit("invalid FIT600 boundary")

    truth = {
        row["case_id"]: row["adjudication"]
        for row in csv.DictReader((CHALLENGE / "data" / "train_labels.csv").open())
    }
    y = np.asarray([truth[case_id] for case_id in fit])
    splitter = StratifiedKFold(n_splits=6, shuffle=True, random_state=SEED)
    metadata = {"seed": SEED, "source": "fit600.txt", "folds": {}}
    for index, (_, test_index) in enumerate(splitter.split(np.zeros(len(fit)), y)):
        ids = [fit[i] for i in test_index]
        path = MANIFESTS / f"fit_fold_{index}.txt"
        path.write_text("\n".join(ids) + "\n", encoding="utf-8")
        metadata["folds"][str(index)] = {
            "n": len(ids),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (MANIFESTS / "fit_folds.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

