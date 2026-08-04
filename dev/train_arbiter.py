#!/usr/bin/env python3
"""Fit the shipped three-class arbiter from dumped production state.

The artifact is a small seed ensemble of histogram gradient boosting models over
evidence-only features.  Two modes:

``--out``      fit every supplied state row and write ``models/arbiter.joblib``.
``--oof``      cross-fit over the six FIT folds and write out-of-fold class
               probabilities for honest evaluation; no artifact is written.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from src.arbiter import (
    DECISION_FEATURES,
    DEFAULT_MARGIN,
    LABELS,
    build_features,
    is_rule_reason,
)

SEEDS = (0, 1, 2)


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=seed,
    )


def load_states(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in (REPO / path).read_text().splitlines()
            if line.strip()
        )
    return rows


def design(rows: list[dict], names: list[str] | None = None):
    feats = [build_features(r) for r in rows]
    if names is None:
        names = sorted({k for f in feats for k in f})
    return np.asarray([[f.get(n, 0.0) for n in names] for f in feats]), names


def promoted_precision(rows, ids, x, y, margin: float, fold_of: dict[str, int]):
    """Out-of-fold precision of the promotion rule itself.

    Cases are assigned to the stable FIT folds where they belong and round-robin
    otherwise, so this works for a FIT600 fit and for the frozen DEV800 refit.
    """
    from src.arbiter import UTILITY

    fold = np.asarray([fold_of[cid] for cid in ids])
    proba = np.zeros((len(ids), 3))
    for f in sorted(set(fold.tolist())):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        acc = np.zeros((int(te.sum()), 3))
        for seed in SEEDS:
            model = make_model(seed).fit(x[tr], y[tr])
            p = model.predict_proba(x[te])
            for j, cls in enumerate(model.classes_):
                acc[:, int(cls)] += p[:, j]
        proba[te] = acc / len(SEEDS)

    # Per-class out-of-fold reliability, used by replace mode to cap confidence.
    chosen_hits, chosen_n = {lab: 0 for lab in LABELS}, {lab: 0 for lab in LABELS}
    for i, row in enumerate(rows):
        finding = row.get("adjudicator_finding")
        if finding in LABELS:
            action = str(finding)
        elif row.get("hard") == "DENIED":
            action = "DENIED"
        else:
            eu = {
                a: sum(UTILITY[a][LABELS[k]] * proba[i, k] for k in range(3))
                for a in LABELS
            }
            action = max(LABELS, key=lambda a: eu[a])
        chosen_n[action] += 1
        chosen_hits[action] += int(y[i] == LABELS.index(action))
    class_precision = {
        lab: (chosen_hits[lab] / chosen_n[lab] if chosen_n[lab] >= 20 else 1.0)
        for lab in LABELS
    }

    promoted, correct = 0, 0
    for i, row in enumerate(rows):
        if row.get("final_decision") != "NEEDS_REVIEW":
            continue
        if row.get("adjudicator_finding") == "NEEDS_REVIEW":
            continue
        expected = {
            action: sum(UTILITY[action][LABELS[k]] * proba[i, k] for k in range(3))
            for action in LABELS
        }
        if expected["APPROVED"] - expected["NEEDS_REVIEW"] > margin:
            promoted += 1
            correct += int(y[i] == LABELS.index("APPROVED"))
    if promoted < 20:
        return 1.0, promoted, class_precision
    return correct / promoted, promoted, class_precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--oof", type=Path)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    parser.add_argument("--fold-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mode", default="promote", choices=["promote", "replace"],
                        help="replace drops the runtime-verdict features and decides directly")
    parser.add_argument("--skip-promotion-oof", action="store_true",
                        help="skip obsolete promotion calibration when an exact-path calibrator follows")
    args = parser.parse_args()

    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    truth = {r["case_id"]: r for r in csv.DictReader(args.truth.open())}
    rows = load_states(args.state)
    ids = [r["case_id"] for r in rows]
    fold_of = {}
    for fold_index, manifest in enumerate(args.fold_manifest):
        for case_id in manifest.read_text().split():
            if case_id in fold_of:
                raise SystemExit("fold manifests overlap")
            fold_of[case_id] = fold_index
    if set(fold_of) != set(ids):
        raise SystemExit("fold manifests do not exactly partition state rows")
    y = np.asarray([LABELS.index(truth[c]["adjudication"]) for c in ids])
    x, names = design(rows)
    if args.mode == "replace":
        keep = [
            i for i, n in enumerate(names)
            if n not in DECISION_FEATURES and not is_rule_reason(n)
        ]
        names = [names[i] for i in keep]
        x = x[:, keep]

    if args.oof:
        fold = np.asarray([fold_of[c] for c in ids])
        proba = np.zeros((len(ids), 3))
        for f in sorted(set(fold.tolist())):
            tr, te = fold != f, fold == f
            acc = np.zeros((int(te.sum()), 3))
            for seed in SEEDS:
                model = make_model(seed)
                model.fit(x[tr], y[tr])
                p = model.predict_proba(x[te])
                for j, cls in enumerate(model.classes_):
                    acc[:, int(cls)] += p[:, j]
            proba[te] = acc / len(SEEDS)
        args.oof.parent.mkdir(parents=True, exist_ok=True)
        with args.oof.open("w", encoding="utf-8") as f:
            for cid, row in zip(ids, proba.tolist()):
                f.write(json.dumps({"case_id": cid, "proba": row}) + "\n")
        print(f"wrote out-of-fold probabilities for {len(ids)} cases to {args.oof}")
        return

    # Measure how often the promotion rule is actually right, out of fold, so the
    # shipped confidence for a promoted approval cannot exceed its reliability.
    if args.skip_promotion_oof:
        precision, promoted = 1.0, 0
        class_precision = {label: 1.0 for label in LABELS}
    else:
        precision, promoted, class_precision = promoted_precision(
            rows, ids, x, y, args.margin, fold_of
        )
    print(f"out-of-fold promoted approvals: {promoted}, precision {precision:.4f}")
    print("out-of-fold per-class precision: " + ", ".join(
        f"{k}={v:.4f}" for k, v in class_precision.items()
    ))

    models = [make_model(seed).fit(x, y) for seed in SEEDS]
    blob = {
        "models": models,
        "features": names,
        "labels": list(LABELS),
        "margin": float(args.margin),
        "mode": args.mode,
        "promoted_precision": float(precision),
        "class_precision": {k: float(v) for k, v in class_precision.items()},
        "n_train": len(ids),
        "sources": list(args.state),
    }
    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(blob, args.out, compress=3)
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB) trained on {len(ids)} cases")


if __name__ == "__main__":
    main()
