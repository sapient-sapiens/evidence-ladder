#!/usr/bin/env python3
"""Fit missing-only extraction models from a frozen FIT600 feature cache."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "dev" / "manifests"
TARGET_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "fee_status",
)
MIN_LEAVES = {
    "species_code": 4,
    "home_world": 2,
    "visa_class": 3,
    "declared_purpose": 4,
    "fee_status": 2,
}
MISSING = frozenset({"", "unknown", "none", "n/a", "null"})


def is_missing(value: object) -> bool:
    return str(value or "").strip().casefold() in MISSING


def model_features(row: dict) -> dict[str, object]:
    features = dict(row["features"])
    fields = row["fields"]
    for field in TARGET_FIELDS:
        features[f"field_{field}"] = str(fields.get(field, "unknown"))
    return features


def fit_bundle(rows: list[dict], field: str) -> dict:
    cohort = [row for row in rows if is_missing(row["fields"].get(field))]
    if not cohort:
        raise SystemExit(f"no missing-field training cohort for {field}")
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform([model_features(row) for row in cohort])
    labels = [row["truth"]["fields"][field] for row in cohort]
    estimator = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=MIN_LEAVES[field],
        max_features=1.0,
        class_weight="balanced",
        random_state=19,
        n_jobs=-1,
    ).fit(matrix, labels)
    estimator.n_jobs = 1
    return {
        "vectorizer": vectorizer,
        "estimator": estimator,
        "training_examples": len(cohort),
    }


def fit_archive_specialist(rows: list[dict]) -> dict:
    cohort = [
        row for row in rows if is_missing(row["fields"].get("declared_purpose"))
    ]
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform([model_features(row) for row in cohort])
    labels = [row["truth"]["fields"]["declared_purpose"] for row in cohort]
    estimator = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=19,
        n_jobs=-1,
    ).fit(matrix, labels)
    estimator.n_jobs = 1
    return {
        "vectorizer": vectorizer,
        "estimator": estimator,
        "training_examples": len(cohort),
        "assertion": "archive audit only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFESTS / "fit600.txt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "models" / "extraction_imputer.joblib",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.features_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fit_ids = set(args.manifest.read_text(encoding="utf-8").split())
    dev_ids = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    row_ids = {str(row.get("case_id")) for row in rows}
    if not rows or row_ids != fit_ids or not row_ids <= dev_ids:
        raise SystemExit("training cache must exactly match a DEV800-subset manifest")
    if any("features" not in row or "fields" not in row or "truth" not in row for row in rows):
        raise SystemExit("training cache is missing features, extracted fields, or truth")

    artifact = {
        "version": "e030-v1-missing-only",
        "training": {
            "manifest": args.manifest.name,
            "manifest_sha256": hashlib.sha256(
                args.manifest.read_bytes()
            ).hexdigest(),
            "feature_cache_sha256": hashlib.sha256(
                args.features_jsonl.read_bytes()
            ).hexdigest(),
            "target_contract": "truth label only when E020 extraction is sentinel",
        },
        "anti_leakage": {
            "contains_case_ids": False,
            "contains_filenames": False,
            "contains_document_hashes": False,
        },
        "target_fields": list(TARGET_FIELDS),
        "models": {field: fit_bundle(rows, field) for field in TARGET_FIELDS},
        "purpose_archive_specialist": fit_archive_specialist(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report = {
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "training_examples": {
            field: artifact["models"][field]["training_examples"]
            for field in TARGET_FIELDS
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
