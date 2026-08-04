#!/usr/bin/env python3
"""Fit a compact revoked-sponsor policy artifact from frozen FIT600 labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"
PUBLIC_REVOKED = {"SPN-0007", "SPN-0139", "SPN-4040"}
RECEIPT_DATE = date.fromisoformat("2026-07-07")
MIN_CLEAN_NON_DIP = 3
MIN_CAUSAL_NON_DIP = 3


def is_clean_sponsor_probe(row: dict[str, str]) -> bool:
    """True when other public hard-denial causes are absent."""
    if row["visa_class"] in {"DIP-1", "TRANSIT-7"}:
        return False
    if row["fee_status"] != "paid" or row["risk_flags"] != "none":
        return False
    try:
        arrival = date.fromisoformat(row["arrival_date"])
    except ValueError:
        return False
    return (RECEIPT_DATE - arrival).days <= 180


def infer_policy(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    by_sponsor: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sponsor[row["sponsor_id"]].append(row)

    learned: dict[str, dict[str, int]] = {}
    for sponsor, sponsor_rows in sorted(by_sponsor.items()):
        if sponsor in PUBLIC_REVOKED:
            continue
        clean_non_dip = [row for row in sponsor_rows if is_clean_sponsor_probe(row)]
        non_dip = [
            row for row in sponsor_rows
            if row["visa_class"] not in {"DIP-1", "TRANSIT-7"}
        ]
        dip = [row for row in sponsor_rows if row["visa_class"] == "DIP-1"]
        clean_pattern = (
            len(clean_non_dip) >= MIN_CLEAN_NON_DIP
            and all(row["adjudication"] == "DENIED" for row in clean_non_dip)
        )
        causal_pattern = (
            len(non_dip) >= MIN_CAUSAL_NON_DIP
            and all(row["adjudication"] == "DENIED" for row in non_dip)
        )
        # DIP-1 exempts sponsor validity. A review on a diplomatic packet is
        # compatible with that exception because it represents uncertainty in
        # some other evidence; only a diplomatic denial contradicts it.
        dip_exception = dip and all(
            row["adjudication"] != "DENIED" for row in dip
        )
        if not dip_exception or not (clean_pattern or causal_pattern):
            continue
        learned[sponsor] = {
            "clean_non_dip_denied": len(clean_non_dip),
            "all_non_dip_denied": len(non_dip),
            "dip_non_denied": len(dip),
            "total_examples": len(sponsor_rows),
            "selection_mode": "clean" if clean_pattern else "causal_exception",
        }
    return learned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFESTS / "fit600.txt"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "models" / "sponsor_policy.json"
    )
    parser.add_argument(
        "--fold-manifest", type=Path, action="append",
        help="Inner validation folds partitioning --manifest (repeatable)",
    )
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    args = parser.parse_args()
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")

    ids = args.manifest.read_text(encoding="utf-8").split()
    dev = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= dev:
        raise SystemExit("sponsor policy fitting is restricted to a unique DEV800 subset")

    wanted = set(ids)
    rows = [
        row
        for row in csv.DictReader(
            args.truth.open(encoding="utf-8")
        )
        if row["case_id"] in wanted
    ]
    if len(rows) != len(ids):
        raise SystemExit("FIT600 truth rows are incomplete")

    full_learned = infer_policy(rows)
    crossfit_counts = {sponsor: 0 for sponsor in full_learned}
    rows_by_id = {row["case_id"]: row for row in rows}
    fold_manifests = args.fold_manifest or [
        MANIFESTS / f"fit_fold_{index}.txt" for index in range(6)
    ]
    seen_fold_ids: set[str] = set()
    for index, fold_manifest in enumerate(fold_manifests):
        held = set(fold_manifest.read_text(encoding="utf-8").split())
        if not held or not held <= wanted:
            raise SystemExit(f"invalid FIT fold {index}")
        if seen_fold_ids & held:
            raise SystemExit("inner sponsor-policy folds overlap")
        seen_fold_ids |= held
        selected = infer_policy(
            [rows_by_id[case_id] for case_id in ids if case_id not in held]
        )
        for sponsor in crossfit_counts:
            crossfit_counts[sponsor] += int(sponsor in selected)
    if seen_fold_ids != wanted:
        raise SystemExit("inner sponsor-policy folds must partition the training manifest")
    learned = {}
    for sponsor, evidence in full_learned.items():
        # A sponsor represented by one diplomatic exception necessarily drops
        # out when that witness's fold is held, and a grouped fold can contain
        # multiple denial witnesses. Survival in all but two inner folds plus
        # independent full-training non-DIP denials identifies the causal
        # exception without storing case identities.
        required = (
            max(1, len(fold_manifests) - 2)
            if evidence["selection_mode"] == "causal_exception"
            else len(fold_manifests)
        )
        if crossfit_counts.get(sponsor, 0) >= required:
            learned[sponsor] = evidence | {
                "required_leave_one_fold_selections": required,
            }
    artifact = {
        "version": "e010-v1",
        "fit_manifest": args.manifest.name,
        "fit_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "criteria": {
            "minimum_clean_non_dip_examples": MIN_CLEAN_NON_DIP,
            "minimum_causal_non_dip_examples": MIN_CAUSAL_NON_DIP,
            "clean_probe": "non-DIP, non-TRANSIT, paid, risk_flags=none, non-stale",
            "clean_non_dip_denial_rate": 1.0,
            "minimum_dip_examples": 1,
            "dip_denial_rate": 0.0,
            "required_leave_one_fold_selections": len(fold_manifests),
            "single_exception_required_leave_one_fold_selections": max(
                1, len(fold_manifests) - 2
            ),
        },
        "crossfit_selection_counts": {
            sponsor: crossfit_counts[sponsor] for sponsor in sorted(learned)
        },
        "learned_revoked_sponsors": learned,
        "anti_leakage": {
            "contains_case_ids": False,
            "contains_filenames": False,
            "contains_document_hashes": False,
            "contains_sponsor_field_values": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
