#!/usr/bin/env python3
"""Fit confidence to OOF correctness of the exact emitted decision pathway."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dev.train_arbiter import SEEDS, make_model  # noqa: E402
from src.arbiter import (  # noqa: E402
    LABELS, UTILITY, build_features, correctness_features, runtime_majority_action,
)


def load(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def predict(models, x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), len(LABELS)))
    for model in models:
        raw = model.predict_proba(x)
        for column, cls in enumerate(model.classes_):
            out[:, int(cls)] += raw[:, column]
    return out / len(models)


def decide(row: dict, p: np.ndarray) -> str:
    if row.get("adjudicator_finding") in LABELS:
        return str(row["adjudicator_finding"])
    if row.get("hard") == "DENIED":
        return "DENIED"
    expected = [sum(UTILITY[a][LABELS[k]] * p[k] for k in range(3)) for a in LABELS]
    return LABELS[int(np.argmax(expected))]


def points(gold: str, action: str) -> tuple[int, bool]:
    if gold == action: return 8, False
    if gold == "DENIED" and action == "APPROVED": return -4, True
    if action == "NEEDS_REVIEW": return 2, False
    if gold == "NEEDS_REVIEW": return 1, False
    return 0, False


def make_calibrator():
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
        C=0.3, max_iter=2000, random_state=80902026))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, action="append", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, action="append", required=True)
    parser.add_argument("--arbiter", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pathway", choices=("replace", "runtime", "runtime_majority"),
                        default="replace")
    args = parser.parse_args()
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    truth = {row["case_id"]: row["adjudication"] for row in csv.DictReader(args.truth.open())}
    rows = load(args.state)
    fold_of = {}
    for fold_index, manifest in enumerate(args.fold_manifest):
        for case_id in manifest.read_text().split():
            if case_id in fold_of: raise SystemExit("fold manifests overlap")
            fold_of[case_id] = fold_index
    if set(fold_of) != {row["case_id"] for row in rows}:
        raise SystemExit("fold manifests do not exactly partition state rows")
    blob = joblib.load(args.arbiter)
    names = list(blob["features"])
    x = np.asarray([[build_features(row).get(name, 0.0) for name in names] for row in rows])
    y = np.asarray([LABELS.index(truth[row["case_id"]]) for row in rows])
    fold = np.asarray([fold_of[row["case_id"]] for row in rows])
    cache = args.report.with_suffix(".probabilities.npz")
    if cache.exists():
        oof = np.load(cache)["oof"]
    else:
        oof = np.zeros((len(rows), len(LABELS)))
        for held_fold in sorted(set(fold.tolist())):
            fit, held = fold != held_fold, fold == held_fold
            oof[held] = predict([make_model(seed).fit(x[fit], y[fit]) for seed in SEEDS], x[held])
        np.savez_compressed(cache, oof=oof)
    if args.pathway == "runtime":
        actions = [str(row.get("final_decision")) for row in rows]
    elif args.pathway == "runtime_majority":
        actions = [runtime_majority_action(row, str(row.get("final_decision"))) for row in rows]
    else:
        actions = [decide(row, p) for row, p in zip(rows, oof)]
    correct = np.asarray([action == truth[row["case_id"]]
                          for row, action in zip(rows, actions)], dtype=float)
    feature_rows = [correctness_features(row, p.tolist(), action)
                    for row, p, action in zip(rows, oof, actions)]
    cal_names = sorted(feature_rows[0])
    cal_x = np.asarray([[feat[name] for name in cal_names] for feat in feature_rows])
    cal_oof = np.zeros(len(rows))
    for held_fold in sorted(set(fold.tolist())):
        fit, held = fold != held_fold, fold == held_fold
        cal_oof[held] = make_calibrator().fit(cal_x[fit], correct[fit]).predict_proba(cal_x[held])[:, 1]
    calibrator = make_calibrator().fit(cal_x, correct)
    for key in ("pathway_gate", "pathway_gate_threshold", "pathway_gate_training_rows"):
        blob.pop(key, None)
    blob.update({"correctness_calibrator": calibrator,
                 "correctness_features": cal_names,
                 "correctness_training_path": f"cross_fitted_exact_{args.pathway}_action",
                 "decision_pathway": args.pathway,
                 "stacking_folds": len(args.fold_manifest)})
    joblib.dump(blob, args.arbiter, compress=3)
    raw = false_approvals = 0
    for row, action in zip(rows, actions):
        score, bad = points(truth[row["case_id"]], action)
        raw += score; false_approvals += int(bad)
    brier = float(np.mean((cal_oof - correct) ** 2))
    report = {"n": len(rows), "folds": len(args.fold_manifest),
              "decision_pathway": args.pathway,
              "classification": 10.0 * raw / len(rows), "accuracy": float(correct.mean()),
              "false_approvals": false_approvals, "correctness_brier": brier,
              "correctness_score": 20.0 * max(0.0, 1.0 - 2.0 * brier)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
