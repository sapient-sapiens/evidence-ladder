#!/usr/bin/env python3
"""Fit evidence-only adjudication, DQ, and pre-arbiter calibration on a DEV subset.

This is evaluation plumbing for nested outer folds.  Hyperparameters and runtime
thresholds are selected using only inner out-of-fold predictions from the supplied
training manifest.  The scored outer fold is never read by this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "dev" / "manifests"
sys.path.insert(0, str(REPO))

from dev.fit_evidence_models import (  # noqa: E402
    MODEL_EXCLUDED_FEATURES,
    SPECS,
    false_approval_details,
    feature_importance,
    final_select,
    fit_calibrator,
    fit_pair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fixed-spec", choices=tuple(spec.name for spec in SPECS))
    args = parser.parse_args()

    train_ids = args.manifest.read_text(encoding="utf-8").split()
    dev_ids = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    if not train_ids or len(set(train_ids)) != len(train_ids) or not set(train_ids) <= dev_ids:
        raise SystemExit("training manifest must be a unique DEV800 subset")
    wanted = set(train_ids)
    all_rows = [
        json.loads(line) for line in args.cache.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows_by_id = {row["case_id"]: row for row in all_rows if row.get("case_id") in wanted}
    if set(rows_by_id) != wanted:
        raise SystemExit("raw evidence cache does not cover the training manifest")
    rows = [rows_by_id[case_id] for case_id in train_ids]

    mapping: dict[str, int] = {}
    for fold, path in enumerate(args.fold_manifest):
        ids = path.read_text(encoding="utf-8").split()
        if not ids or not set(ids) <= wanted or set(ids) & set(mapping):
            raise SystemExit(f"invalid or overlapping inner fold: {path}")
        mapping.update({case_id: fold for case_id in ids})
    if set(mapping) != wanted:
        raise SystemExit("inner folds must exactly partition the training manifest")

    all_names = sorted(rows[0]["features"])
    feature_names = [name for name in all_names if name not in MODEL_EXCLUDED_FEATURES]
    candidate_specs = (tuple(spec for spec in SPECS if spec.name == args.fixed_spec)
                       if args.fixed_spec else SPECS)
    spec, config, oof_metrics, oof_outputs = final_select(
        rows, mapping, feature_names, candidate_specs)
    models = fit_pair(rows, spec, feature_names)
    calibrator, calibration = fit_calibrator(rows, mapping, oof_outputs, config)

    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metadata = {
        "version": "nested-dev800-v1",
        "train_n": len(rows),
        "train_manifest_sha256": manifest_sha,
        "selection": f"inner_{len(set(mapping.values()))}_fold_oof",
        "excluded_features": sorted(MODEL_EXCLUDED_FEATURES),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": models[0], "features": feature_names,
            "runtime_config": config,
            "metadata": metadata | {"task": "adjudication", "spec": spec.name},
        },
        args.out_dir / "adjudication.joblib", compress=3,
    )
    joblib.dump(
        {
            "model": models[1], "features": feature_names,
            "metadata": metadata | {"task": "has_dq", "spec": "logreg_c03"},
        },
        args.out_dir / "has_dq.joblib", compress=3,
    )
    calibrator["train"] = "nested_dev_outer_train_oof"
    calibrator["train_n"] = len(rows)
    calibrator["train_manifest_sha256"] = manifest_sha
    calibrator["fit_manifest_sha256"] = manifest_sha
    (args.out_dir / "confidence_calibrator.json").write_text(
        json.dumps(calibrator, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "train_n": len(rows), "inner_folds": len(set(mapping.values())),
        "fixed_spec": args.fixed_spec,
        "selected_spec": spec.name, "runtime_config": config,
        "oof_metrics": oof_metrics, "calibration": calibration,
        "false_approval_details": false_approval_details(rows, oof_outputs, config),
        "adjudication_importance": feature_importance(models[0], feature_names),
        "dq_importance": feature_importance(models[1], feature_names),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "train_n", "inner_folds", "selected_spec", "runtime_config",
        "oof_metrics", "calibration",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
