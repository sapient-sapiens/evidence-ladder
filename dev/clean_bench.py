#!/usr/bin/env python3
"""Honest 200-case bench: fit everything on FIT600, score TUNE100 + PROBE100.

Every learned component in this repository was fitted on data that includes the
splits it is scored on, so a FIT600 "out of fold" number overstates by roughly
nine points of accuracy (E107, E108).  This bench removes that: the inherited
adjudication/DQ artifacts and the arbiter are all fitted on FIT600 alone, so
TUNE100 and PROBE100 are genuinely untouched and can be iterated against without
spending HOLD200.

Compares arbiter feature-set and mode variants under that regime.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from src.arbiter import DECISION_FEATURES, LABELS, UTILITY, build_features, is_rule_reason
from dev.train_arbiter import SEEDS, make_model

WEIGHTS = {
    "applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
    "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3, "risk_flags": 8,
    "fee_status": 4,
}


def points(t: str, p: str) -> int:
    if t == p:
        return 8
    if t == "DENIED" and p == "APPROVED":
        return -4
    if p == "NEEDS_REVIEW":
        return 2
    if t == "NEEDS_REVIEW":
        return 1
    return 0


def load(name: str) -> list[dict]:
    return [
        json.loads(l)
        for l in (REPO / f"dev/runs/{name}.jsonl").read_text().splitlines() if l
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="cleanA-fit600")
    parser.add_argument("--test", action="append", default=None)
    args = parser.parse_args()
    tests = args.test or ["cleanA-tune100", "cleanA-probe100"]

    truth = {
        r["case_id"]: r for r in csv.DictReader(open(CHALLENGE / "data/train_labels.csv"))
    }
    train = load(args.train)
    held = [row for name in tests for row in load(name)]

    tr_feats = [build_features(r) for r in train]
    te_feats = [build_features(r) for r in held]
    allnames = sorted({k for f in tr_feats for k in f})
    y = np.asarray([LABELS.index(truth[r["case_id"]]["adjudication"]) for r in train])

    def run(names, mode, label):
        xtr = np.asarray([[f.get(n, 0.0) for n in names] for f in tr_feats])
        xte = np.asarray([[f.get(n, 0.0) for n in names] for f in te_feats])
        acc = np.zeros((len(held), 3))
        for seed in SEEDS:
            m = make_model(seed).fit(xtr, y)
            p = m.predict_proba(xte)
            for j, cls in enumerate(m.classes_):
                acc[:, int(cls)] += p[:, j]
        proba = acc / len(SEEDS)
        raw, fa, briers = 0, 0, []
        for i, row in enumerate(held):
            cid = row["case_id"]
            cur = row["final_decision"]
            eu = {a: sum(UTILITY[a][LABELS[k]] * proba[i, k] for k in range(3)) for a in LABELS}
            if mode == "replace":
                finding = row.get("adjudicator_finding")
                if finding in LABELS:
                    action = str(finding)
                elif row.get("hard") == "DENIED":
                    action = "DENIED"
                else:
                    action = max(LABELS, key=lambda a: eu[a])
            else:
                action = cur
                if cur == "NEEDS_REVIEW" and row.get("adjudicator_finding") != "NEEDS_REVIEW":
                    if eu["APPROVED"] - eu["NEEDS_REVIEW"] > 2.0:
                        action = "APPROVED"
            gold = truth[cid]["adjudication"]
            raw += points(gold, action)
            fa += int(action == "APPROVED" and gold == "DENIED")
            conf = float(proba[i, LABELS.index(action)])
            if action != cur:
                conf = min(conf, 0.68)
            briers.append((conf - (1.0 if action == gold else 0.0)) ** 2)
        cls = 80 * raw / (8 * len(held))
        cal = 20 * max(0.0, 1 - 2 * float(np.mean(briers)))
        print(f"  {label:46s} cls={cls:7.3f} cal={cal:6.3f} sum={cls + cal:7.3f} FA={fa}")

    base = [n for n in allnames if n not in DECISION_FEATURES and not is_rule_reason(n)]
    print(f"honest bench: fit on {len(train)}, scored on {len(held)} untouched packets")
    run(allnames, "promote", "promote mode (sees runtime verdict)")
    run(base, "replace", "replace mode (shipped)")
    run([n for n in allnames if n not in DECISION_FEATURES], "replace", "replace + narrow rule reasons")
    run([n for n in base if not n.startswith("why_")], "replace", "replace minus all rule reasons")
    run([n for n in base if n not in {"meta_conf", "meta_approved", "meta_denied", "meta_review", "dq_prob"}],
        "replace", "replace minus inherited model outputs")
    nowhy = [n for n in base if not n.startswith("why_")]
    POLICY = {
        "why_disqualifying_flags", "why_transit_visa", "why_stale_arrival",
        "why_unpaid_fee", "why_revoked_sponsor", "why_adjudicator_note_denied",
        "why_adjudicator_note_review", "why_adjudicator_finding_approved",
    }
    run(nowhy + [n for n in base if n in POLICY], "replace", "replace, only public-manual policy reasons")
    run(nowhy, "promote", "promote minus all rule reasons")
    run([n for n in nowhy if n not in {"meta_conf", "meta_approved", "meta_denied", "meta_review", "dq_prob"}],
        "replace", "replace minus rule reasons and inherited outputs")


if __name__ == "__main__":
    main()
