#!/usr/bin/env python3
"""Extract exact-production FIT600 state for evidence-only model fitting."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
MANIFESTS = REPO / "dev" / "manifests"
sys.path.insert(0, str(REPO))

from solution import prepare_packet  # noqa: E402
from src.adjudicate import (  # noqa: E402
    _has_trusted_biometric_evidence,
    _missing_critical,
    _rule_adjudicate,
)
from src.constants import DISQUALIFYING_FLAGS  # noqa: E402
from src.features import feature_dict  # noqa: E402
from src.fee_absent import fee_is_absent_imputed  # noqa: E402

FORBIDDEN_FEATURES = frozenset({"pages", "pdf_bytes"})
OUTPUT_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)


def featurize(case_id: str, truth: dict[str, str]) -> dict:
    pdf = CHALLENGE / "data" / "train" / f"{case_id}.pdf"
    packet, _, _ = prepare_packet(pdf)
    rule_fields, reasons, hard = _rule_adjudicate(packet)
    # Hard policy exits deliberately do not populate ``adjudication`` because
    # the production caller returns ``hard`` directly before model inference.
    rule_decision = hard or rule_fields["adjudication"]
    features = feature_dict(packet, pdf, rule_decision)
    features = {
        key: value for key, value in features.items() if key not in FORBIDDEN_FEATURES
    }
    truth_flags = set((truth.get("risk_flags") or "none").split("|"))
    return {
        "case_id": case_id,
        "features": features,
        "fields": {key: rule_fields[key] for key in OUTPUT_FIELDS},
        "state": {
            "rule_decision": rule_decision,
            "rule_reasons": reasons,
            "hard_decision": hard,
            "visa_class": rule_fields["visa_class"],
            "declared_purpose": rule_fields["declared_purpose"],
            "fee_status": rule_fields["fee_status"],
            "missing": _missing_critical(rule_fields),
            "has_trusted_biometric": _has_trusted_biometric_evidence(packet),
            "fee_absent_imputed": fee_is_absent_imputed(packet),
            "positive_waiver_seen": packet.positive_waiver_seen,
            "has_untrusted_conflicts": bool(packet.untrusted_conflicts),
            "field_sources": dict(sorted(packet.field_sources.items())),
            "conflicts": sorted(packet.conflicts),
            "untrusted_conflicts": sorted(packet.untrusted_conflicts),
            "r35_repair_evidence": packet.r35_repair_evidence,
        },
        "truth": {
            "adjudication": truth["adjudication"],
            "has_dq": int(bool(truth_flags & DISQUALIFYING_FLAGS)),
            "fields": {key: truth[key] for key in OUTPUT_FIELDS},
        },
    }


def worker(payload: tuple[str, dict[str, str]]) -> dict:
    return featurize(*payload)


def load_truth(ids: list[str], truth_path: Path) -> dict[str, dict[str, str]]:
    if truth_path.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    allowed = set(ids)
    selected: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(
        truth_path.open(encoding="utf-8")
    ):
        if row["case_id"] in allowed:
            selected[row["case_id"]] = row
    if set(selected) != allowed:
        raise SystemExit("FIT manifest contains unknown cases")
    return selected


def validate_fit_manifest(path: Path, *, allow_subset: bool = False) -> list[str]:
    ids = path.read_text(encoding="utf-8").split()
    dev = set((MANIFESTS / "dev800.txt").read_text(encoding="utf-8").split())
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not set(ids) <= dev
    ):
        raise SystemExit(
            "evidence extraction is restricted to a unique DEV800 subset"
        )
    return ids


def extract_rows(
    ids: list[str],
    truth: dict[str, dict[str, str]],
    *,
    workers: int,
) -> list[dict]:
    rows_by_id: dict[str, dict] = {}
    payloads = [(case_id, truth[case_id]) for case_id in ids]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(worker, payload): payload[0] for payload in payloads}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows_by_id[row["case_id"]] = row
            if index % 25 == 0:
                print(f"featurized {index}/{len(ids)}", flush=True)
    return [rows_by_id[case_id] for case_id in ids]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFESTS / "fit600.txt"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    ids = validate_fit_manifest(args.manifest, allow_subset=args.extract_only)
    if args.cache.exists() and not args.rebuild:
        rows = [
            json.loads(line)
            for line in args.cache.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if {row["case_id"] for row in rows} != set(ids):
            raise SystemExit("feature cache does not match FIT600")
    else:
        rows = extract_rows(ids, load_truth(ids, args.truth), workers=args.workers)
        write_rows(args.cache, rows)

    feature_names = sorted(rows[0]["features"])
    if FORBIDDEN_FEATURES & set(feature_names):
        raise SystemExit("forbidden envelope feature leaked into training")
    summary = {
        "n": len(rows),
        "manifest": args.manifest.name,
        "cache": str(args.cache),
        "feature_names": feature_names,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not args.extract_only:
        raise SystemExit("model fitting is not implemented until feature audit passes")


if __name__ == "__main__":
    main()
