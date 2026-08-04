#!/usr/bin/env python3
"""Fit/score the final replace arbiter and exact-path correctness calibrator.

Training states must be produced out of fold for every upstream learned
component.  Held states must be produced by components fitted only on the outer
training partition.  The script emits aggregate metrics only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from dev.train_arbiter import SEEDS, make_model  # noqa: E402
from src.arbiter import (  # noqa: E402
    DECISION_FEATURES, LABELS, UTILITY, build_features, is_rule_reason,
    runtime_majority_action,
)

FIELD_WEIGHTS = {
    "applicant_name": 5, "species_code": 6, "home_world": 5, "visa_class": 5,
    "sponsor_id": 5, "arrival_date": 4, "declared_purpose": 3,
    "risk_flags": 8, "fee_status": 4,
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def models_predict(models: list, x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), len(LABELS)))
    for model in models:
        p = model.predict_proba(x)
        for j, cls in enumerate(model.classes_):
            out[:, int(cls)] += p[:, j]
    return out / len(models)


def decide(row: dict, proba: np.ndarray) -> str:
    finding = row.get("adjudicator_finding")
    if finding in LABELS:
        return str(finding)
    if row.get("hard") == "DENIED":
        return "DENIED"
    expected = {
        action: sum(UTILITY[action][LABELS[k]] * proba[k] for k in range(3))
        for action in LABELS
    }
    return max(LABELS, key=lambda action: expected[action])


def legacy_promotion_precision(rows: list[dict], proba: np.ndarray, truth: dict) -> tuple[float, int]:
    promoted = correct = 0
    for row, p in zip(rows, proba):
        if row.get("final_decision") != "NEEDS_REVIEW":
            continue
        if row.get("adjudicator_finding") == "NEEDS_REVIEW":
            continue
        expected = {
            action: sum(UTILITY[action][LABELS[k]] * p[k] for k in range(3))
            for action in LABELS
        }
        if expected["APPROVED"] - expected["NEEDS_REVIEW"] > 2.0:
            promoted += 1
            correct += int(truth[row["case_id"]]["adjudication"] == "APPROVED")
    return ((correct / promoted) if promoted >= 20 else 1.0), promoted


def calibration_row(row: dict, p: np.ndarray, action: str) -> dict[str, float]:
    evidence = build_features(row)
    current = str(row.get("final_decision") or "")
    chosen = float(p[LABELS.index(action)])
    runtime = float(row.get("runtime_confidence") or 0.0)
    feat = {
        "p_approved": float(p[0]), "p_denied": float(p[1]), "p_review": float(p[2]),
        "chosen_p": chosen, "chosen_logit": math.log(max(chosen, 1e-4) / max(1 - chosen, 1e-4)),
        "runtime_confidence": runtime, "changed": float(action != current),
        "hard_denied": float(row.get("hard") == "DENIED"),
        "finding_present": float(row.get("adjudicator_finding") in LABELS),
        "n_missing": evidence["n_missing"], "n_conflicts": evidence["n_conflicts"],
        "n_untrusted_conflicts": evidence["n_untrusted_conflicts"],
        "n_flags": evidence["n_flags"], "observed_flags_seen": evidence["observed_flags_seen"],
        "has_dq_flag": evidence["has_dq_flag"], "has_review_flag": evidence["has_review_flag"],
        "src_intake": evidence["src_intake"], "src_fee": evidence["src_fee"],
        "src_biometric": evidence["src_biometric"],
        "risk_source_present": evidence["risk_source_present"],
        "trusted_chars_lt_120": evidence["trusted_chars_lt_120"],
        "trusted_chars_lt_400": evidence["trusted_chars_lt_400"],
    }
    for label in LABELS:
        feat[f"action_{label}"] = float(action == label)
        feat[f"current_{label}"] = float(current == label)
    return feat


def make_calibrator() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(C=0.3, max_iter=2000, random_state=80902026)),
    ])


def gate_features(row: dict, p: np.ndarray, base: str, runtime: str) -> list[float]:
    expected = np.asarray([
        sum(UTILITY[action][LABELS[k]] * p[k] for k in range(3))
        for action in LABELS
    ])
    ordered = np.sort(expected)
    evidence = build_features(row)
    result = list(p) + list(expected) + [
        float(ordered[-1] - ordered[-2]),
        float(p[LABELS.index(base)] - p[LABELS.index(runtime)]),
        evidence["n_missing"], evidence["n_conflicts"], evidence["n_untrusted_conflicts"],
        evidence["n_flags"], evidence["meta_conf"], evidence["dq_prob"],
        evidence["observed_flags_seen"], evidence["risk_source_present"],
    ]
    result.extend(float(base == label) for label in LABELS)
    result.extend(float(runtime == label) for label in LABELS)
    return result


def fit_gate(rows: list[dict], proba: np.ndarray, truth: dict, indexes: np.ndarray):
    x, y, weights = [], [], []
    for i in indexes:
        row, p = rows[int(i)], proba[int(i)]
        base, runtime = decide(row, p), str(row.get("final_decision"))
        if base == runtime:
            continue
        gold = truth[row["case_id"]]["adjudication"]
        base_score = classification_points(gold, base)[0]
        runtime_score = classification_points(gold, runtime)[0]
        if base_score == runtime_score:
            continue
        x.append(gate_features(row, p, base, runtime))
        y.append(int(base_score > runtime_score))
        weights.append(abs(base_score - runtime_score))
    gate = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(
        C=0.05, max_iter=2000, random_state=0))])
    gate.fit(np.asarray(x), np.asarray(y), model__sample_weight=np.asarray(weights))
    return gate


def gate_actions(rows: list[dict], proba: np.ndarray, gate, threshold: float) -> list[str]:
    actions = []
    for row, p in zip(rows, proba):
        base, runtime = decide(row, p), str(row.get("final_decision"))
        if base == runtime:
            actions.append(base)
            continue
        use_base = gate.predict_proba([gate_features(row, p, base, runtime)])[0, 1] >= threshold
        actions.append(base if use_base else runtime)
    return actions


def select_gate_threshold(rows: list[dict], proba: np.ndarray, truth: dict, gate,
                          indexes: np.ndarray) -> float:
    def key(threshold: float):
        selected_rows = [rows[int(i)] for i in indexes]
        selected_p = proba[indexes]
        actions = gate_actions(selected_rows, selected_p, gate, threshold)
        raw = false_approvals = 0.0
        for row, action in zip(selected_rows, actions):
            score, bad = classification_points(truth[row["case_id"]]["adjudication"], action)
            raw += score
            false_approvals += int(bad)
        return raw, -false_approvals, threshold
    return max((0.35, 0.45, 0.55, 0.65, 0.75, 0.85), key=key)


def classification_points(truth: str, predicted: str) -> tuple[float, bool]:
    if truth == predicted:
        return 8.0, False
    if truth == "DENIED" and predicted == "APPROVED":
        return -4.0, True
    if predicted == "NEEDS_REVIEW":
        return 2.0, False
    if truth == "NEEDS_REVIEW":
        return 1.0, False
    return 0.0, False


def norm(value: object) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def norm_flags(value: object) -> str:
    raw = norm(value)
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(part.strip() for part in raw.split("|") if part.strip()))


def reliability(conf: np.ndarray, y: np.ndarray) -> list[dict]:
    result = []
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf <= hi) if hi == 1 else (conf < hi))
        if mask.any():
            result.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                           "mean_confidence": float(conf[mask].mean()),
                           "accuracy": float(y[mask].mean())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-state", type=Path, required=True)
    parser.add_argument("--held-state", type=Path, required=True)
    parser.add_argument("--train-fold-manifest", type=Path, action="append", required=True)
    parser.add_argument("--held-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    parser.add_argument("--pathway-gate", action="store_true",
                        help="cross-fit and evaluate the learned arbiter/runtime gate")
    parser.add_argument("--runtime-pathway", action="store_true",
                        help="use the existing runtime decision on every pathway disagreement")
    parser.add_argument("--runtime-majority-pathway", action="store_true",
                        help="use two-of-three runtime/rule/meta agreement")
    parser.add_argument("--probability-cache", type=Path,
                        help="reuse an exact-shape arbiter probability cache")
    args = parser.parse_args()
    if sum((args.pathway_gate, args.runtime_pathway, args.runtime_majority_pathway)) > 1:
        raise SystemExit("choose at most one pathway policy")
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")

    truth = {row["case_id"]: row for row in csv.DictReader(
        args.truth.open(encoding="utf-8"))}
    train, held = load(args.train_state), load(args.held_state)
    held_ids = args.held_manifest.read_text().split()
    if {r["case_id"] for r in held} != set(held_ids):
        raise SystemExit("held states do not match held manifest")

    fold_of: dict[str, int] = {}
    for fold, path in enumerate(args.train_fold_manifest):
        for case_id in path.read_text().split():
            if case_id in fold_of:
                raise SystemExit("training folds overlap")
            fold_of[case_id] = fold
    if set(fold_of) != {r["case_id"] for r in train}:
        raise SystemExit("training folds do not partition training states")

    built_train = [build_features(row) for row in train]
    built_held = [build_features(row) for row in held]
    names = sorted({key for feat in built_train for key in feat}
                   - set(DECISION_FEATURES))
    names = [name for name in names if not is_rule_reason(name)]
    x_train = np.asarray([[feat.get(name, 0.0) for name in names] for feat in built_train])
    x_held = np.asarray([[feat.get(name, 0.0) for name in names] for feat in built_held])
    y_train_class = np.asarray([LABELS.index(truth[r["case_id"]]["adjudication"]) for r in train])
    fold = np.asarray([fold_of[r["case_id"]] for r in train])

    probability_cache = args.probability_cache or args.output.with_suffix(".probabilities.npz")
    if probability_cache.exists():
        cached = np.load(probability_cache)
        oof, held_p = cached["oof"], cached["held"]
        if oof.shape != (len(train), 3) or held_p.shape != (len(held), 3):
            raise SystemExit("cached arbiter probability dimensions do not match states")
    else:
        oof = np.zeros((len(train), 3))
        for f in sorted(set(fold.tolist())):
            tr, te = fold != f, fold == f
            models = [make_model(seed).fit(x_train[tr], y_train_class[tr]) for seed in SEEDS]
            oof[te] = models_predict(models, x_train[te])
        full_models = [make_model(seed).fit(x_train, y_train_class) for seed in SEEDS]
        held_p = models_predict(full_models, x_held)
        np.savez_compressed(probability_cache, oof=oof, held=held_p)
    gate_train_rows = 0
    gate_threshold = None
    if args.pathway_gate:
        train_actions: list[str | None] = [None] * len(train)
        for f in sorted(set(fold.tolist())):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            gate = fit_gate(train, oof, truth, tr)
            threshold = select_gate_threshold(train, oof, truth, gate, tr)
            actions = gate_actions([train[int(i)] for i in te], oof[te], gate, threshold)
            for i, action in zip(te, actions):
                train_actions[int(i)] = action
        train_actions = [str(action) for action in train_actions]
        all_indexes = np.arange(len(train))
        full_gate = fit_gate(train, oof, truth, all_indexes)
        gate_threshold = select_gate_threshold(train, oof, truth, full_gate, all_indexes)
        gate_train_rows = sum(decide(row, p) != str(row.get("final_decision"))
                              for row, p in zip(train, oof))
    elif args.runtime_pathway:
        train_actions = [str(row.get("final_decision")) for row in train]
    elif args.runtime_majority_pathway:
        train_actions = [runtime_majority_action(row, str(row.get("final_decision")))
                         for row in train]
    else:
        train_actions = [decide(row, p) for row, p in zip(train, oof)]
    y_correct = np.asarray([
        float(action == truth[row["case_id"]]["adjudication"])
        for row, action in zip(train, train_actions)
    ])
    cal_rows = [calibration_row(row, p, action)
                for row, p, action in zip(train, oof, train_actions)]
    cal_names = sorted(cal_rows[0])
    x_cal = np.asarray([[feat[name] for name in cal_names] for feat in cal_rows])
    cal_oof = np.zeros(len(train))
    for f in sorted(set(fold.tolist())):
        tr, te = fold != f, fold == f
        cal_oof[te] = make_calibrator().fit(x_cal[tr], y_correct[tr]).predict_proba(x_cal[te])[:, 1]
    calibrator = make_calibrator().fit(x_cal, y_correct)

    promotion_precision, promotion_n = legacy_promotion_precision(train, oof, truth)
    if args.pathway_gate:
        held_actions = gate_actions(held, held_p, full_gate, gate_threshold)
    elif args.runtime_pathway:
        held_actions = [str(row.get("final_decision")) for row in held]
    elif args.runtime_majority_pathway:
        held_actions = [runtime_majority_action(row, str(row.get("final_decision")))
                        for row in held]
    else:
        held_actions = [decide(row, p) for row, p in zip(held, held_p)]
    held_cal_rows = [calibration_row(row, p, action)
                     for row, p, action in zip(held, held_p, held_actions)]
    held_x_cal = np.asarray([[feat[name] for name in cal_names] for feat in held_cal_rows])
    exact_conf = np.clip(calibrator.predict_proba(held_x_cal)[:, 1], 0.01, 0.99)
    legacy_conf = np.asarray([
        min(float(p[LABELS.index(action)]), promotion_precision)
        if action != str(row.get("final_decision") or "")
        else float(p[LABELS.index(action)])
        for row, p, action in zip(held, held_p, held_actions)
    ])
    legacy_conf = np.clip(legacy_conf, 0.01, 0.99)
    held_correct = np.asarray([
        float(action == truth[row["case_id"]]["adjudication"])
        for row, action in zip(held, held_actions)
    ])

    class_raw = 0.0
    false_approvals = 0
    per_action = defaultdict(lambda: {"n": 0, "correct": 0})
    for row, action in zip(held, held_actions):
        gold = truth[row["case_id"]]["adjudication"]
        points, fa = classification_points(gold, action)
        class_raw += points
        false_approvals += int(fa)
        per_action[action]["n"] += 1
        per_action[action]["correct"] += int(action == gold)

    extraction_raw = 0.0
    maximum = len(held) * sum(FIELD_WEIGHTS.values())
    misses = {field: 0 for field in FIELD_WEIGHTS}
    for row in held:
        pred = row["prediction"]
        gold = truth[row["case_id"]]
        for field, weight in FIELD_WEIGHTS.items():
            matched = (norm_flags(pred.get(field)) == norm_flags(gold.get(field))) \
                if field == "risk_flags" else (norm(pred.get(field)) == norm(gold.get(field)))
            extraction_raw += weight * int(matched)
            misses[field] += int(not matched)
    extraction = 50.0 * extraction_raw / maximum
    classification = 80.0 * class_raw / (8.0 * len(held))

    def cal_report(conf: np.ndarray) -> dict:
        brier = float(np.mean((conf - held_correct) ** 2))
        return {"brier": brier, "score": 20.0 * max(0.0, 1.0 - 2.0 * brier),
                "reliability": reliability(conf, held_correct)}

    legacy = cal_report(legacy_conf)
    exact = cal_report(exact_conf)
    result = {
        "n": len(held), "extraction": extraction, "extraction_misses": misses,
        "classification": classification, "accuracy": float(held_correct.mean()),
        "false_approvals": false_approvals,
        "per_action": dict(per_action),
        "legacy_calibration": legacy,
        "exact_path_calibration": exact,
        "legacy_total": extraction + classification + legacy["score"],
        "exact_path_total": extraction + classification + exact["score"],
        "calibration_delta": exact["score"] - legacy["score"],
        "legacy_promotion_precision": promotion_precision,
        "legacy_promotion_n": promotion_n,
        "train_final_path_accuracy_oof": float(y_correct.mean()),
        "train_exact_calibrator_crossfit_brier": float(np.mean((cal_oof - y_correct) ** 2)),
        "feature_count": len(names), "calibration_feature_count": len(cal_names),
        "pathway_gate": bool(args.pathway_gate), "gate_threshold": gate_threshold,
        "runtime_pathway": bool(args.runtime_pathway),
        "runtime_majority_pathway": bool(args.runtime_majority_pathway),
        "gate_disagreement_rows": gate_train_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in (
        "n", "extraction", "classification", "accuracy", "false_approvals",
        "legacy_calibration", "exact_path_calibration", "legacy_total",
        "exact_path_total", "calibration_delta",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
