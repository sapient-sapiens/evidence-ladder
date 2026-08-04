#!/usr/bin/env python3
"""Nested-FIT selection and fitting for evidence-only adjudication models."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "dev" / "manifests"
sys.path.insert(0, str(REPO))

from src.confidence_calibrate import LOGREG_FEATURES, build_cal_features  # noqa: E402
from src.meta_model import DEFAULT_RUNTIME_CONFIG  # noqa: E402
from src.parse_fields import ParsedPacket  # noqa: E402

warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

MODEL_EXCLUDED_FEATURES = frozenset({"pages", "pdf_bytes", "trusted_chars"})
CACHE_FORBIDDEN_FEATURES = frozenset({"pages", "pdf_bytes"})
FINDING_FEATURES = ("finding_approved", "finding_denied", "finding_review")
SEED = 80902026
FIELD_WEIGHTS = {
    "applicant_name": 5,
    "species_code": 6,
    "home_world": 5,
    "visa_class": 5,
    "sponsor_id": 5,
    "arrival_date": 4,
    "declared_purpose": 3,
    "risk_flags": 8,
    "fee_status": 4,
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    params: dict[str, Any]


SPECS = (
    ModelSpec(
        "gbc_d1",
        "gbc",
        {
            "max_depth": 1,
            "n_estimators": 180,
            "learning_rate": 0.04,
            "min_samples_leaf": 8,
        },
    ),
    ModelSpec(
        "gbc_d2",
        "gbc",
        {
            "max_depth": 2,
            "n_estimators": 160,
            "learning_rate": 0.04,
            "min_samples_leaf": 8,
        },
    ),
    ModelSpec(
        "gbc_d3",
        "gbc",
        {
            "max_depth": 3,
            "n_estimators": 120,
            "learning_rate": 0.04,
            "min_samples_leaf": 10,
        },
    ),
    ModelSpec(
        "logreg_c03",
        "logreg",
        {"C": 0.3, "class_weight": "balanced", "max_iter": 2000},
    ),
    ModelSpec(
        "extra_leaf5",
        "extra",
        {
            "n_estimators": 240,
            "min_samples_leaf": 5,
            "max_features": 0.7,
            "class_weight": "balanced",
        },
    ),
)


def make_adjudication_model(spec: ModelSpec) -> Pipeline:
    if spec.kind == "gbc":
        estimator = GradientBoostingClassifier(random_state=SEED, **spec.params)
        return Pipeline([("gb", estimator)])
    if spec.kind == "logreg":
        estimator = LogisticRegression(random_state=SEED, **spec.params)
        return Pipeline([("scaler", StandardScaler()), ("gb", estimator)])
    if spec.kind == "extra":
        estimator = ExtraTreesClassifier(
            random_state=SEED, n_jobs=1, **spec.params
        )
        return Pipeline([("gb", estimator)])
    raise ValueError(f"unknown model kind: {spec.kind}")


def make_dq_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gb",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 600 or len({row["case_id"] for row in rows}) != 600:
        raise SystemExit("feature cache must contain exactly 600 unique FIT rows")
    schemas = {tuple(sorted(row["features"])) for row in rows}
    if len(schemas) != 1:
        raise SystemExit("feature cache has inconsistent schemas")
    schema = set(next(iter(schemas)))
    leaked = CACHE_FORBIDDEN_FEATURES & schema
    if leaked:
        raise SystemExit(f"envelope features leaked into extraction cache: {leaked}")
    # Retain raw trusted character count in the audit cache, but exclude it
    # from fitting in favor of coarse evidence-quality buckets.
    if "trusted_chars" not in schema:
        raise SystemExit("audit cache is missing raw trusted_chars")
    return rows


def fold_map(rows: list[dict]) -> dict[str, int]:
    fit_ids = {row["case_id"] for row in rows}
    mapping: dict[str, int] = {}
    for fold in range(6):
        ids = (MANIFESTS / f"fit_fold_{fold}.txt").read_text().split()
        if len(ids) != 100:
            raise SystemExit(f"fit fold {fold} is not frozen at 100 cases")
        for case_id in ids:
            if case_id in mapping:
                raise SystemExit("FIT folds overlap")
            mapping[case_id] = fold
    if set(mapping) != fit_ids:
        raise SystemExit("FIT folds do not partition the feature cache")
    return mapping


def model_reachable(row: dict) -> bool:
    if row["state"]["hard_decision"] is not None:
        return False
    return not any(row["features"].get(key, 0) for key in FINDING_FEATURES)


def dq_reachable(row: dict) -> bool:
    return (
        model_reachable(row)
        and not row["features"].get("observed_flags_seen", 0)
        and not row["features"].get("n_flags", 0)
    )


def vector(row: dict, names: list[str]) -> list[float]:
    return [float(row["features"].get(name, 0.0)) for name in names]


def fit_pair(
    rows: list[dict], spec: ModelSpec, feature_names: list[str]
) -> tuple[Pipeline, Pipeline]:
    adj_rows = [row for row in rows if model_reachable(row)]
    dq_rows = [row for row in rows if dq_reachable(row)]
    adj_y = np.asarray([row["truth"]["adjudication"] for row in adj_rows])
    dq_y = np.asarray([row["truth"]["has_dq"] for row in dq_rows], dtype=int)
    if len(set(adj_y)) < 3:
        raise RuntimeError("model-reachable training cohort lacks an adjudication class")
    if len(set(dq_y)) < 2:
        raise RuntimeError("DQ-reachable training cohort lacks a binary class")
    adj = make_adjudication_model(spec)
    dq = make_dq_model()
    adj.fit(np.asarray([vector(row, feature_names) for row in adj_rows]), adj_y)
    dq.fit(np.asarray([vector(row, feature_names) for row in dq_rows]), dq_y)
    return adj, dq


def probability_maps(model: Pipeline, x: np.ndarray) -> list[dict[str, float]]:
    probabilities = model.predict_proba(x)
    estimator = model.steps[-1][1]
    classes = [str(value) for value in estimator.classes_]
    return [
        {label: float(prob) for label, prob in zip(classes, row)}
        for row in probabilities
    ]


def predict_pair(
    models: tuple[Pipeline, Pipeline],
    rows: list[dict],
    feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    adj, dq = models
    x = np.asarray([vector(row, feature_names) for row in rows])
    adj_maps = probability_maps(adj, x)
    dq_maps = probability_maps(dq, x)
    out: dict[str, dict[str, Any]] = {}
    for row, adj_probs, dq_probs in zip(rows, adj_maps, dq_maps):
        out[row["case_id"]] = {
            "adj": adj_probs,
            "dq": float(dq_probs.get("1", 0.0)),
        }
    return out


def crossfit_outputs(
    rows: list[dict],
    mapping: dict[str, int],
    allowed_folds: set[int],
    spec: ModelSpec,
    feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for held in sorted(allowed_folds):
        train = [
            row
            for row in rows
            if mapping[row["case_id"]] in allowed_folds - {held}
        ]
        valid = [row for row in rows if mapping[row["case_id"]] == held]
        outputs.update(predict_pair(fit_pair(train, spec, feature_names), valid, feature_names))
    expected = {
        row["case_id"]
        for row in rows
        if mapping[row["case_id"]] in allowed_folds
    }
    if set(outputs) != expected:
        raise RuntimeError("crossfit predictions do not cover the requested folds")
    return outputs


def base_confidence(decision: str, row: dict, reasons: list[str]) -> float:
    missing = row["state"]["missing"]
    feat = row["features"]
    rs = set(reasons)
    policy_review = bool(
        {"review_flags", "adjudicator_note_review", "unreadable_arrival"} & rs
    )
    if "injection_only" in rs:
        return 0.15
    if decision == "NEEDS_REVIEW":
        if "xw2_medical_no_biometric" in rs and not missing:
            return 0.72
        if policy_review and len(missing) <= 1:
            return 0.85
        if policy_review:
            return 0.7
        if "suspicious_sponsor" in rs and not missing:
            return 0.55
        return 0.28
    if decision == "DENIED":
        if {"disqualifying_flags", "transit_visa"} & rs:
            return 0.9 if not missing else 0.55
        if {"stale_arrival", "unpaid_fee"} & rs:
            return 0.88 if not missing else 0.55
        if {"revoked_sponsor", "meta_deny"} & rs:
            return 0.8 if not missing else 0.5
        if "adjudicator_note_denied" in rs:
            return 0.88
        return 0.55
    if "adjudicator_finding_approved" in rs:
        return 0.88
    if feat.get("observed_flags_seen", 0) and not missing:
        return 0.84
    if not missing and feat.get("src_intake", 0):
        return 0.76
    return 0.45


def simulate(
    row: dict, output: dict[str, Any], config: dict[str, float]
) -> dict[str, Any]:
    state = row["state"]
    feat = row["features"]
    reasons = list(state["rule_reasons"])
    rule = state["rule_decision"]
    hard = state["hard_decision"]
    if hard is not None:
        return {
            "decision": hard,
            "raw_confidence": base_confidence(hard, row, reasons),
            "reasons": reasons,
        }
    if any(feat.get(key, 0) for key in FINDING_FEATURES):
        return {
            "decision": rule,
            "raw_confidence": base_confidence(rule, row, reasons),
            "reasons": reasons,
        }

    decision = rule
    confidence = base_confidence(decision, row, reasons)
    dq_prob = 0.0
    if dq_reachable(row):
        dq_prob = float(output["dq"])
        if dq_prob >= config["dq_deny_threshold"]:
            reasons.append("meta_has_dq")
            return {
                "decision": "DENIED",
                "raw_confidence": max(0.65, dq_prob),
                "reasons": reasons,
            }

    adj_probs = output["adj"]
    meta_label = max(adj_probs, key=adj_probs.get)
    meta_conf = float(adj_probs[meta_label])
    complete = (
        not state["missing"]
        and not feat.get("n_flags", 0)
        and not feat.get("n_conflicts", 0)
        and not state["has_untrusted_conflicts"]
    )
    if meta_label == "APPROVED":
        if (
            feat.get("observed_flags_seen", 0)
            and complete
            and meta_conf
            >= config.get("observed_approve_threshold", config["approve_threshold"])
        ):
            decision = "APPROVED"
            confidence = max(meta_conf, 0.75)
            reasons.append("meta_approve_with_flags")
        elif (
            not feat.get("observed_flags_seen", 0)
            and complete
            and meta_conf >= config["approve_threshold"]
            and feat.get("src_fee", 0)
            and dq_prob < config["dq_approve_ceiling"]
        ):
            decision = "APPROVED"
            confidence = meta_conf
            reasons.append("meta_approve_complete")
        elif rule == "APPROVED":
            decision = "APPROVED"
            confidence = max(confidence, meta_conf)
    elif meta_label == "DENIED":
        if meta_conf >= config["deny_threshold"]:
            decision = "DENIED"
            confidence = meta_conf
            reasons.append("meta_deny")
    elif meta_label == "NEEDS_REVIEW":
        if rule != "APPROVED" or meta_conf >= config["review_threshold"]:
            decision = "NEEDS_REVIEW"
            confidence = max(0.4, min(0.88, meta_conf))
            reasons.append("meta_review")

    if (
        decision == "APPROVED"
        and state["visa_class"] == "XW-2"
        and state["declared_purpose"] == "medical consult"
        and not state["has_trusted_biometric"]
    ):
        decision = "NEEDS_REVIEW"
        reasons.append("xw2_medical_no_biometric")
        confidence = base_confidence(decision, row, reasons)
    if (
        decision == "APPROVED"
        and state["fee_status"] == "waived"
        and state["visa_class"] != "DIP-1"
        and not state["positive_waiver_seen"]
    ):
        decision = "NEEDS_REVIEW"
        if "waived_non_dip" not in reasons:
            reasons.append("waived_non_dip")
        confidence = base_confidence(decision, row, reasons)
    if decision == "APPROVED" and state["fee_absent_imputed"]:
        decision = "NEEDS_REVIEW"
        reasons.append("fee_absent_imputed")
        confidence = base_confidence(decision, row, reasons)
    return {
        "decision": decision,
        "raw_confidence": confidence,
        "reasons": reasons,
    }


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


def normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().split()).casefold()


def normalized_flags(value: Any) -> str:
    raw = normalized(value)
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(part.strip() for part in raw.split("|") if part.strip()))


def extraction_metrics(rows: list[dict]) -> dict[str, Any]:
    raw = 0
    maximum = len(rows) * sum(FIELD_WEIGHTS.values())
    misses = {key: 0 for key in FIELD_WEIGHTS}
    for row in rows:
        for key, weight in FIELD_WEIGHTS.items():
            predicted = row["fields"][key]
            truth = row["truth"]["fields"][key]
            matched = (
                normalized_flags(predicted) == normalized_flags(truth)
                if key == "risk_flags"
                else normalized(predicted) == normalized(truth)
            )
            raw += weight * int(matched)
            misses[key] += int(not matched)
    return {
        "raw": raw,
        "maximum": maximum,
        "score": 50.0 * raw / maximum,
        "misses": misses,
    }


def score_config(
    rows: list[dict],
    outputs: dict[str, dict[str, Any]],
    config: dict[str, float],
) -> dict[str, Any]:
    raw = 0.0
    false_approvals = 0
    correct = 0
    approvals = 0
    for row in rows:
        predicted = simulate(row, outputs[row["case_id"]], config)["decision"]
        truth = row["truth"]["adjudication"]
        points, false_approval = classification_points(truth, predicted)
        raw += points
        false_approvals += int(false_approval)
        correct += int(predicted == truth)
        approvals += int(predicted == "APPROVED")
    return {
        "n": len(rows),
        "classification_raw": raw,
        "classification_score": 80.0 * raw / (8.0 * len(rows)),
        "false_approvals": false_approvals,
        "correct": correct,
        "accuracy": correct / len(rows),
        "approvals": approvals,
    }


def false_approval_details(
    rows: list[dict],
    outputs: dict[str, dict[str, Any]],
    config: dict[str, float],
) -> list[dict[str, Any]]:
    details = []
    for row in rows:
        if row["truth"]["adjudication"] != "DENIED":
            continue
        result = simulate(row, outputs[row["case_id"]], config)
        if result["decision"] != "APPROVED":
            continue
        feat = row["features"]
        details.append(
            {
                "case_id": row["case_id"],
                "fields": row["fields"],
                "state": row["state"],
                "model_output": outputs[row["case_id"]],
                "simulated": result,
                "evidence_features": {
                    key: feat.get(key)
                    for key in (
                        "observed_flags_seen",
                        "n_flags",
                        "has_dq",
                        "n_conflicts",
                        "n_untrusted_conflicts",
                        "src_intake",
                        "src_fee",
                        "src_registry",
                        "src_biometric",
                        "src_adjudicator",
                        "n_fields_from_native",
                        "n_fields_from_standard_ocr",
                        "n_fields_from_specialized_ocr",
                    )
                },
            }
        )
    return details


def score_rule_only(rows: list[dict]) -> dict[str, Any]:
    raw = 0.0
    false_approvals = 0
    correct = 0
    for row in rows:
        predicted = row["state"]["rule_decision"]
        truth = row["truth"]["adjudication"]
        points, false_approval = classification_points(truth, predicted)
        raw += points
        false_approvals += int(false_approval)
        correct += int(predicted == truth)
    return {
        "n": len(rows),
        "classification_raw": raw,
        "classification_score": 80.0 * raw / (8.0 * len(rows)),
        "false_approvals": false_approvals,
        "correct": correct,
        "accuracy": correct / len(rows),
    }


def threshold_grid() -> list[dict[str, float]]:
    values = itertools.product(
        (0.35, 0.5, 0.65, 0.8),
        (0.2, 0.35, 0.5),
        (0.5, 0.58, 0.7, 0.82),
        (0.42, 0.52, 0.65, 0.8),
        (0.45, 0.6, 0.72, 0.84),
    )
    configs = []
    for dq_deny, dq_ceiling, approve, deny, review in values:
        if dq_ceiling >= dq_deny:
            continue
        configs.append(
            {
                "dq_deny_threshold": dq_deny,
                "dq_approve_ceiling": dq_ceiling,
                "approve_threshold": approve,
                "deny_threshold": deny,
                "review_threshold": review,
            }
        )
    configs.append(dict(DEFAULT_RUNTIME_CONFIG))
    return configs


def metric_key(metrics: dict[str, Any], config: dict[str, float]) -> tuple:
    # False approvals are a hard first-order objective. Remaining ties prefer
    # higher challenge points, then accuracy, then the more conservative
    # approval/review configuration.
    return (
        -metrics["false_approvals"],
        metrics["classification_raw"],
        metrics["correct"],
        config["approve_threshold"],
        -config["review_threshold"],
    )


def select_config(
    rows: list[dict], outputs: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], dict[str, Any]]:
    best_config: dict[str, float] | None = None
    best_metrics: dict[str, Any] | None = None
    for config in threshold_grid():
        metrics = score_config(rows, outputs, config)
        if best_metrics is None or metric_key(metrics, config) > metric_key(
            best_metrics, best_config or config
        ):
            best_config, best_metrics = config, metrics
    assert best_config is not None and best_metrics is not None
    return best_config, best_metrics


def feature_importance(
    model: Pipeline, names: list[str], *, limit: int = 15
) -> list[dict[str, float]]:
    estimator = model.named_steps["gb"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.max(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        return []
    order = np.argsort(values)[::-1][:limit]
    return [
        {"feature": names[int(index)], "importance": float(values[int(index)])}
        for index in order
    ]


def calibration_features(row: dict, simulated: dict[str, Any]) -> list[float]:
    feat = row["features"]
    packet = ParsedPacket(case_id=row["case_id"])
    packet.risk_flags = [
        name.removeprefix("flag_")
        for name, value in feat.items()
        if name.startswith("flag_") and value
    ]
    packet.observed_flags_seen = bool(feat.get("observed_flags_seen", 0))
    packet.conflicts = {
        f"conflict_{index}" for index in range(int(feat.get("n_conflicts", 0)))
    }
    if row["state"]["has_untrusted_conflicts"]:
        packet.untrusted_conflicts = {"field"}
    packet.sources_seen = {
        source
        for source in ("intake", "fee", "registry", "biometric", "adjudicator")
        if feat.get(f"src_{source}", 0)
    }
    built = build_cal_features(
        decision=simulated["decision"],
        raw_confidence=simulated["raw_confidence"],
        packet=packet,
        missing=row["state"]["missing"],
        reasons=simulated["reasons"],
    )
    return [float(built.get(name, 0.0)) for name in LOGREG_FEATURES]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def old_calibrator_predictions(x: np.ndarray) -> np.ndarray | None:
    path = REPO / "models" / "confidence_calibrator.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    if blob.get("type") != "logreg" or blob.get("feature_names") != list(
        LOGREG_FEATURES
    ):
        return None
    coef = np.asarray(blob["coef"], dtype=float)
    intercept = float(blob["intercept"])
    return np.asarray([sigmoid(intercept + float(row @ coef)) for row in x])


def fit_calibrator(
    rows: list[dict],
    mapping: dict[str, int],
    outputs: dict[str, dict[str, Any]],
    config: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    simulated = [
        simulate(row, outputs[row["case_id"]], config) for row in rows
    ]
    x = np.asarray(
        [calibration_features(row, result) for row, result in zip(rows, simulated)]
    )
    y = np.asarray(
        [
            int(result["decision"] == row["truth"]["adjudication"])
            for row, result in zip(rows, simulated)
        ],
        dtype=int,
    )
    crossfit = np.zeros(len(rows), dtype=float)
    for held in sorted(set(mapping.values())):
        train_idx = np.asarray(
            [mapping[row["case_id"]] != held for row in rows], dtype=bool
        )
        valid_idx = ~train_idx
        model = LogisticRegression(
            C=0.3, max_iter=2000, random_state=SEED
        ).fit(x[train_idx], y[train_idx])
        crossfit[valid_idx] = model.predict_proba(x[valid_idx])[:, 1]
    raw = np.asarray([result["raw_confidence"] for result in simulated])
    old = old_calibrator_predictions(x)
    full = LogisticRegression(C=0.3, max_iter=2000, random_state=SEED).fit(x, y)
    brier = float(np.mean((crossfit - y) ** 2))
    report = {
        "n": len(rows),
        "correct_rate": float(np.mean(y)),
        "raw_brier": float(np.mean((raw - y) ** 2)),
        "old_dev800_calibrator_brier": (
            float(np.mean((old - y) ** 2)) if old is not None else None
        ),
        "fit_crossfit_brier": brier,
        "fit_crossfit_calibration_score": 20.0 * max(0.0, 1.0 - 2.0 * brier),
    }
    artifact = {
        "type": "logreg",
        "method": "logreg_l2",
        "C": 0.3,
        "coef": [float(value) for value in full.coef_[0]],
        "intercept": float(full.intercept_[0]),
        "feature_names": list(LOGREG_FEATURES),
        "train": "fit600_e011_oof",
        "oof_brier": brier,
        "fit_manifest_sha256": hashlib.sha256(
            (MANIFESTS / "fit600.txt").read_bytes()
        ).hexdigest(),
    }
    return artifact, report


def nested_select(
    rows: list[dict], mapping: dict[str, int], feature_names: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    all_folds = set(mapping.values())
    outer_report = []
    nested_decisions: dict[str, str] = {}
    for outer in sorted(all_folds):
        inner_folds = all_folds - {outer}
        train_rows = [
            row for row in rows if mapping[row["case_id"]] in inner_folds
        ]
        candidate_rows = []
        for spec in SPECS:
            outputs = crossfit_outputs(
                rows, mapping, inner_folds, spec, feature_names
            )
            config, metrics = select_config(train_rows, outputs)
            candidate_rows.append((spec, config, metrics))
        spec, config, inner_metrics = max(
            candidate_rows, key=lambda item: metric_key(item[2], item[1])
        )
        fitted = fit_pair(train_rows, spec, feature_names)
        held_rows = [row for row in rows if mapping[row["case_id"]] == outer]
        held_outputs = predict_pair(fitted, held_rows, feature_names)
        held_metrics = score_config(held_rows, held_outputs, config)
        for row in held_rows:
            nested_decisions[row["case_id"]] = simulate(
                row, held_outputs[row["case_id"]], config
            )["decision"]
        outer_report.append(
            {
                "outer_fold": outer,
                "selected_spec": spec.name,
                "selected_config": config,
                "inner_metrics": inner_metrics,
                "held_metrics": held_metrics,
            }
        )
        print(
            f"nested outer={outer} spec={spec.name} "
            f"inner_fa={inner_metrics['false_approvals']} "
            f"held_fa={held_metrics['false_approvals']} "
            f"held_class={held_metrics['classification_score']:.3f}",
            flush=True,
        )
    nested_outputs = {
        row["case_id"]: {"decision": nested_decisions[row["case_id"]]}
        for row in rows
    }
    raw = 0.0
    false_approvals = 0
    correct = 0
    for row in rows:
        decision = nested_outputs[row["case_id"]]["decision"]
        points, false_approval = classification_points(
            row["truth"]["adjudication"], decision
        )
        raw += points
        false_approvals += int(false_approval)
        correct += int(decision == row["truth"]["adjudication"])
    summary = {
        "n": len(rows),
        "classification_raw": raw,
        "classification_score": 80.0 * raw / (8.0 * len(rows)),
        "false_approvals": false_approvals,
        "correct": correct,
        "accuracy": correct / len(rows),
        "outer_folds": outer_report,
    }
    return summary, nested_decisions


def final_select(
    rows: list[dict], mapping: dict[str, int], feature_names: list[str],
    specs: tuple[ModelSpec, ...] = SPECS,
) -> tuple[ModelSpec, dict[str, float], dict[str, Any], dict[str, Any]]:
    candidates = []
    all_folds = set(mapping.values())
    for spec in specs:
        outputs = crossfit_outputs(rows, mapping, all_folds, spec, feature_names)
        config, metrics = select_config(rows, outputs)
        candidates.append((spec, config, metrics, outputs))
        print(
            f"final-oof spec={spec.name} fa={metrics['false_approvals']} "
            f"class={metrics['classification_score']:.3f}",
            flush=True,
        )
    return max(candidates, key=lambda item: metric_key(item[2], item[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--adjudication-out", type=Path)
    parser.add_argument("--dq-out", type=Path)
    parser.add_argument("--calibrator-out", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.cache)
    mapping = fold_map(rows)
    all_names = sorted(rows[0]["features"])
    feature_names = [
        name for name in all_names if name not in MODEL_EXCLUDED_FEATURES
    ]
    if MODEL_EXCLUDED_FEATURES & set(feature_names):
        raise SystemExit("excluded envelope/text-length feature entered model")
    feature_schema_sha256 = hashlib.sha256(
        ("\n".join(feature_names) + "\n").encode()
    ).hexdigest()

    reachable = [row for row in rows if model_reachable(row)]
    dq_rows = [row for row in rows if dq_reachable(row)]
    audit = {
        "n": len(rows),
        "n_features_cache": len(all_names),
        "n_features_model": len(feature_names),
        "excluded_features": sorted(MODEL_EXCLUDED_FEATURES),
        "feature_schema_sha256": feature_schema_sha256,
        "model_reachable": len(reachable),
        "model_reachable_labels": {
            label: sum(row["truth"]["adjudication"] == label for row in reachable)
            for label in ("APPROVED", "DENIED", "NEEDS_REVIEW")
        },
        "dq_reachable": len(dq_rows),
        "dq_reachable_positive": sum(row["truth"]["has_dq"] for row in dq_rows),
        "rule_only_metrics": score_rule_only(rows),
        "extraction_metrics": extraction_metrics(rows),
    }
    nested, nested_decisions = nested_select(rows, mapping, feature_names)
    spec, config, final_oof_metrics, final_oof_outputs = final_select(
        rows, mapping, feature_names
    )
    full_models = fit_pair(rows, spec, feature_names)
    calibrator, calibration_report = fit_calibrator(
        rows, mapping, final_oof_outputs, config
    )
    report = {
        "version": "e011-v1",
        "cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "fit_manifest_sha256": hashlib.sha256(
            (MANIFESTS / "fit600.txt").read_bytes()
        ).hexdigest(),
        "audit": audit,
        "nested_selection": nested,
        "nested_false_approval_case_ids": [
            row["case_id"]
            for row in rows
            if row["truth"]["adjudication"] == "DENIED"
            and nested_decisions[row["case_id"]] == "APPROVED"
        ],
        "final_selection": {
            "spec": spec.name,
            "spec_kind": spec.kind,
            "spec_params": spec.params,
            "runtime_config": config,
            "oof_metrics": final_oof_metrics,
            "false_approval_details": false_approval_details(
                rows, final_oof_outputs, config
            ),
            "adjudication_importance": feature_importance(
                full_models[0], feature_names
            ),
            "dq_importance": feature_importance(full_models[1], feature_names),
            "calibration": calibration_report,
            "oof_total_score": (
                audit["extraction_metrics"]["score"]
                + final_oof_metrics["classification_score"]
                + calibration_report["fit_crossfit_calibration_score"]
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.fit:
        return
    if nested["false_approvals"] or final_oof_metrics["false_approvals"]:
        raise SystemExit("refusing to fit release artifacts with OOF false approvals")
    if not args.adjudication_out or not args.dq_out or not args.calibrator_out:
        raise SystemExit("--fit requires all three artifact output paths")
    metadata = {
        "version": "e011-v1",
        "train": "fit600",
        "fit_manifest_sha256": report["fit_manifest_sha256"],
        "selection": "nested_6x5_fit_folds",
        "excluded_features": sorted(MODEL_EXCLUDED_FEATURES),
        "feature_schema_sha256": feature_schema_sha256,
    }
    args.adjudication_out.parent.mkdir(parents=True, exist_ok=True)
    args.dq_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibrator_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": full_models[0],
            "features": feature_names,
            "runtime_config": config,
            "metadata": metadata | {"task": "adjudication", "spec": spec.name},
        },
        args.adjudication_out,
    )
    joblib.dump(
        {
            "model": full_models[1],
            "features": feature_names,
            "metadata": metadata | {"task": "has_dq", "spec": "logreg_c03"},
        },
        args.dq_out,
    )
    args.calibrator_out.write_text(
        json.dumps(calibrator, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
