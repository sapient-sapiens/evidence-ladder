#!/usr/bin/env python3
"""Matched-pair: learned arbiter vs path-EV, both fit on non_iter200, score holdouts.

Rewrites adjudication/confidence on a baseline predictions.jsonl using dumped
states so OCR is not re-run.  Prints iter200 and tune100 evaluate_split JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dev"))

from src.arbiter import (  # noqa: E402
    DECISION_FEATURES,
    LABELS,
    UTILITY,
    build_features,
    is_rule_reason,
)
from path_ev import decide_path_ev, fit_path_tables  # noqa: E402
from train_arbiter import SEEDS, make_model  # noqa: E402

MANIFESTS = REPO / "dev" / "manifests"


def load_states(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["case_id"]] = row
    return out


def load_preds(path: Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {r["case_id"]: r for r in rows}


def design_replace(rows: list[dict]):
    feats = [build_features(r) for r in rows]
    names = sorted(
        n
        for n in {k for f in feats for k in f}
        if n not in DECISION_FEATURES and not is_rule_reason(n)
    )
    x = np.asarray([[f.get(n, 0.0) for n in names] for f in feats], dtype=float)
    return x, names


def fit_hgb(train_states: list[dict], y: np.ndarray):
    x, names = design_replace(train_states)
    models = [make_model(seed).fit(x, y) for seed in SEEDS]
    return models, names


def hgb_proba(models, names, state: dict) -> list[float]:
    feat = build_features(state)
    x = np.asarray([[feat.get(n, 0.0) for n in names]], dtype=float)
    acc = np.zeros(3)
    for model in models:
        p = model.predict_proba(x)[0]
        for j, cls in enumerate(model.classes_):
            acc[int(cls)] += p[j]
    return (acc / len(models)).tolist()


def decide_hgb(models, names, state: dict) -> tuple[str, list[float]]:
    finding = state.get("adjudicator_finding")
    if finding in LABELS:
        return str(finding), hgb_proba(models, names, state)
    if state.get("hard") == "DENIED":
        return "DENIED", hgb_proba(models, names, state)
    proba = hgb_proba(models, names, state)
    expected = {
        action: sum(UTILITY[action][LABELS[k]] * proba[k] for k in range(3))
        for action in LABELS
    }
    return max(LABELS, key=lambda a: expected[a]), proba


def score_manifest(preds: dict[str, dict], manifest: Path, truth: Path) -> dict:
    ids = manifest.read_text().split()
    with tempfile.TemporaryDirectory(prefix="mib-match-") as tmp:
        tmp_path = Path(tmp)
        selected = [r for r in csv.DictReader(truth.open()) if r["case_id"] in set(ids)]
        truth_path = tmp_path / "truth.csv"
        pred_path = tmp_path / "predictions.jsonl"
        result_path = tmp_path / "evaluation.json"
        with truth_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
            writer.writeheader()
            writer.writerows(selected)
        with pred_path.open("w", encoding="utf-8") as handle:
            for case_id in ids:
                handle.write(json.dumps(preds[case_id], sort_keys=True) + "\n")
        subprocess.check_call(
            [
                sys.executable,
                str(CHALLENGE / "scripts" / "evaluate.py"),
                "--truth",
                str(truth_path),
                "--submission",
                str(pred_path),
                "--output-json",
                str(result_path),
            ],
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(result_path.read_text())
    scores = result["scores"]
    return {
        "split": manifest.stem,
        "n": len(ids),
        "total": scores["total_score"],
        "extraction": scores["extraction_score"],
        "classification": scores["classification_score"],
        "calibration": scores["calibration_score"],
        "false_approvals": result["raw"]["catastrophic_false_approvals"],
    }


def patch_preds(
    base: dict[str, dict],
    states: dict[str, dict],
    ids: list[str],
    decide_fn,
) -> dict[str, dict]:
    out = {cid: dict(row) for cid, row in base.items()}
    for cid in ids:
        chosen, proba = decide_fn(states[cid])
        out[cid]["adjudication"] = chosen
        out[cid]["confidence"] = round(
            min(max(float(proba[LABELS.index(chosen)]), 0.01), 0.99), 4
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "dev" / "runs" / "baseline-dev800" / "predictions.jsonl",
    )
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--m", type=float, default=10.0)
    parser.add_argument("--write-path-ev", type=Path)
    args = parser.parse_args()

    truth = {r["case_id"]: r for r in csv.DictReader(args.truth.open())}
    states = load_states(args.state)
    base = load_preds(args.predictions)
    non_ids = (MANIFESTS / "non_iter200.txt").read_text().split()
    iter_ids = (MANIFESTS / "iter200.txt").read_text().split()
    tune_ids = (MANIFESTS / "tune100.txt").read_text().split()

    train_states = [states[c] for c in non_ids]
    y = np.asarray([LABELS.index(truth[c]["adjudication"]) for c in non_ids])
    models, names = fit_hgb(train_states, y)
    path_blob = fit_path_tables(
        train_states,
        [truth[c]["adjudication"] for c in non_ids],
        m=args.m,
    )
    if args.write_path_ev:
        args.write_path_ev.parent.mkdir(parents=True, exist_ok=True)
        args.write_path_ev.write_text(
            json.dumps(path_blob, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    variants = {
        "hgb600": lambda s: decide_hgb(models, names, s),
        "path_ev600": lambda s: decide_path_ev(path_blob, s),
    }
    report = {"path_ev": {"n_paths": path_blob["n_paths"], "m": path_blob["m"]}}
    for name, decide in variants.items():
        report[name] = {}
        for split, ids in (("iter200", iter_ids), ("tune100", tune_ids)):
            patched = patch_preds(base, states, ids, decide)
            report[name][split] = score_manifest(
                patched, MANIFESTS / f"{split}.txt", args.truth
            )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
