"""Cross-fitted three-class arbiter over the retained runtime's own state.

The rule engine and the inherited meta model are deliberately conservative:
whenever trusted evidence is incomplete they route a packet to ``NEEDS_REVIEW``.
That is the right default, but it is not free — the evaluator pays 2 of 8 points
for an unnecessary review.  This module estimates
``P(APPROVED | evidence state)`` from evidence-only features and converts a
review to an approval only when the expected evaluator utility (which prices a
false approval at -4) clearly favours it.

Design constraints:

- Features describe extracted policy content and the provenance/quality of the
  evidence that produced it.  PDF envelope inputs (page count, byte size), case
  IDs, filenames, and split membership are never used.
- ``replace`` mode also withholds every flag describing which rule branch fired,
  because those branches are themselves fitted to FIT600.  Hard policy denials
  and trusted adjudicator findings keep their precedence regardless.
- The artifact is fitted out of fold; the shipped model never saw the packet it
  scores at evaluation time.
"""
from __future__ import annotations

import datetime
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from .constants import (
    DISQUALIFYING_FLAGS,
    HOME_WORLDS,
    PURPOSES,
    REVIEW_ONLY_FLAGS,
    SPECIES_CODES,
    VISA_CLASSES,
)

_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "arbiter.joblib"

LABELS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
UTILITY = {
    "APPROVED": {"APPROVED": 8, "DENIED": -4, "NEEDS_REVIEW": 1},
    "DENIED": {"APPROVED": 0, "DENIED": 8, "NEEDS_REVIEW": 1},
    "NEEDS_REVIEW": {"APPROVED": 2, "DENIED": 2, "NEEDS_REVIEW": 8},
}
DEFAULT_MARGIN = 2.0

# Features that encode the runtime's own verdict.  A model given these learns to
# copy the decision and can never disagree with it usefully; a model denied them
# forms an independent opinion, which is what ``replace`` mode needs.
DECISION_FEATURES = frozenset(
    {
        "final_approved",
        "final_denied",
        "final_review",
        "rule_approved",
        "rule_denied",
        "rule_review",
        "hard_denied",
        "raw_confidence",
    }
)

# Every ``why_*`` flag reports which branch of the rule engine fired.  Those
# branches were selected over roughly a hundred experiments on FIT600, so the
# flags carry that fitting with them: measured on packets nobody has fitted they
# cost 2.94 points and double the false approvals, while on FIT600 they look
# mildly useful.  ``replace`` mode reads the evidence itself instead.
def is_rule_reason(name: str) -> bool:
    return name.startswith("why_")

RECEIPT = datetime.date(2026, 7, 7)
FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)
STANDARD_SOURCES = frozenset(
    {"native", "page_ocr", "embedded_ocr", "ppocr_ocr", "oriented_ocr"}
)
MARKER_KEYS = (
    "_observed_none_any",
    "_complete_no_b13_layout",
    "_review_flag_asserted",
    "_printed_fee_unknown_votes",
    "_adjudicator_finding",
    "_damage_implies_illegible",
    "_missing_biometric_roles_layout",
    "_manual_identity_conflict",
    "_redacted_identity_damage",
    "_occluded_biohazard_visual",
    "biometric_flags",
    "risk_flags_embargo_home",
)
REASON_KEYS = (
    "insufficient_evidence",
    "missing_fields",
    "review_flags",
    "conflicts",
    "unknown_fee",
    "waived_non_dip",
    "missing_sponsor",
    "suspicious_sponsor",
    "clean_packet",
    "injection_only",
    "unreadable_arrival",
    "fee_absent_imputed",
    "explicit_clean_risk_and_fee",
    "meta_approve_complete",
    "meta_approve_with_flags",
    "meta_review",
    "meta_deny",
    "meta_has_dq",
    "adjudicator_finding_approved",
    "adjudicator_note_review",
    "adjudicator_note_denied",
    "xw2_medical_no_biometric",
    "disqualifying_flags",
    "transit_visa",
    "stale_arrival",
    "unpaid_fee",
    "revoked_sponsor",
    "rescinded_transit",
    "unresolved_transit_evidence",
    "unresolved_stale_evidence",
    "review_monotonic_meta_deny",
)


def packet_state(
    packet,
    pred: dict,
    *,
    rule_decision: str,
    hard_decision: str | None,
    reasons: list[str],
    raw_confidence: float,
    missing: list[str],
    meta: tuple[str, float] | None,
    dq_prob: float,
    meta_features: dict | None = None,
) -> dict:
    """Snapshot the evidence state the arbiter reasons about.

    Taken at the very end of the runtime pass so the extracted values it sees
    are the ones actually submitted.
    """
    return {
        "case_id": pred.get("case_id"),
        "rule_decision": rule_decision,
        "hard": hard_decision,
        "final_decision": pred.get("adjudication"),
        "final_reasons": list(reasons),
        "raw_confidence": float(raw_confidence),
        "runtime_confidence": float(pred.get("confidence") or 0.0),
        "missing": list(missing),
        "meta_label": meta[0] if meta else None,
        "meta_conf": float(meta[1]) if meta else 0.0,
        "dq_prob": float(dq_prob),
        "adjudicator_finding": packet.adjudicator_finding,
        "risk_flags": [
            atom
            for atom in str(pred.get("risk_flags") or "none").split("|")
            if atom and atom != "none"
        ],
        "conflicts": sorted(packet.conflicts),
        "untrusted_conflicts": sorted(packet.untrusted_conflicts),
        "sources_seen": sorted(packet.sources_seen),
        "observed_flags_seen": bool(packet.observed_flags_seen),
        "positive_waiver_seen": bool(packet.positive_waiver_seen),
        "injection_heavy": bool(packet.injection_heavy),
        "trusted_text_chars": int(packet.trusted_text_chars),
        "packet_fields": {
            key: str(pred.get(key) or "") for key in FIELDS
        },
        "field_sources": {k: str(v) for k, v in packet.field_sources.items()},
        "meta_features": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in (meta_features or {}).items()
        },
    }


def _days_before_receipt(value: str) -> float:
    try:
        parsed = datetime.date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return -1.0
    return float((RECEIPT - parsed).days)


def build_features(state: dict) -> dict[str, float]:
    fields = state.get("packet_fields") or {}
    sources = state.get("field_sources") or {}
    flags = set(state.get("risk_flags") or [])
    reasons = set(state.get("final_reasons") or [])
    sources_seen = set(state.get("sources_seen") or [])
    untrusted = set(state.get("untrusted_conflicts") or [])
    visa = str(fields.get("visa_class") or "unknown")
    fee = str(fields.get("fee_status") or "unknown")
    sponsor = str(fields.get("sponsor_id") or "")

    feat: dict[str, float] = {
        "n_missing": float(len(state.get("missing") or [])),
        "n_conflicts": float(len(state.get("conflicts") or [])),
        "n_untrusted_conflicts": float(len(untrusted)),
        "n_flags": float(len(flags)),
        "has_dq_flag": float(bool(flags & DISQUALIFYING_FLAGS)),
        "has_review_flag": float(bool(flags & REVIEW_ONLY_FLAGS)),
        "observed_flags_seen": float(bool(state.get("observed_flags_seen"))),
        "positive_waiver_seen": float(bool(state.get("positive_waiver_seen"))),
        "injection_heavy": float(bool(state.get("injection_heavy"))),
        "trusted_chars": float(state.get("trusted_text_chars") or 0),
        "meta_conf": float(state.get("meta_conf") or 0.0),
        "dq_prob": float(state.get("dq_prob") or 0.0),
        "meta_approved": float(state.get("meta_label") == "APPROVED"),
        "meta_denied": float(state.get("meta_label") == "DENIED"),
        "meta_review": float(state.get("meta_label") == "NEEDS_REVIEW"),
        "rule_approved": float(state.get("rule_decision") == "APPROVED"),
        "rule_denied": float(state.get("rule_decision") == "DENIED"),
        "rule_review": float(state.get("rule_decision") == "NEEDS_REVIEW"),
        "final_approved": float(state.get("final_decision") == "APPROVED"),
        "final_denied": float(state.get("final_decision") == "DENIED"),
        "final_review": float(state.get("final_decision") == "NEEDS_REVIEW"),
        "hard_denied": float(state.get("hard") == "DENIED"),
        "raw_confidence": float(state.get("raw_confidence") or 0.0),
        "finding_approved": float(state.get("adjudicator_finding") == "APPROVED"),
        "finding_denied": float(state.get("adjudicator_finding") == "DENIED"),
        "finding_review": float(state.get("adjudicator_finding") == "NEEDS_REVIEW"),
        "days_before_receipt": _days_before_receipt(fields.get("arrival_date")),
        "sponsor_wellformed": float(
            sponsor.startswith("SPN-") and sponsor != "SPN-0000" and len(sponsor) == 8
        ),
        "species_known": float(str(fields.get("species_code") or "") in SPECIES_CODES),
        "world_known": float(str(fields.get("home_world") or "") in HOME_WORLDS),
        "purpose_known": float(str(fields.get("declared_purpose") or "") in PURPOSES),
        "visa_known": float(visa in VISA_CLASSES),
    }
    for source in ("intake", "fee", "registry", "biometric", "adjudicator"):
        feat[f"src_{source}"] = float(source in sources_seen)
    for value in sorted(VISA_CLASSES):
        feat[f"visa_{value}"] = float(visa == value)
    for value in ("paid", "waived", "unpaid", "unknown"):
        feat[f"fee_{value}"] = float(fee == value)
    for value in sorted(PURPOSES):
        feat[f"purpose_{value.replace(' ', '_')}"] = float(
            str(fields.get("declared_purpose") or "") == value
        )
    for atom in sorted(DISQUALIFYING_FLAGS | REVIEW_ONLY_FLAGS):
        feat[f"flag_{atom}"] = float(atom in flags)
    for key in FIELDS:
        source = str(sources.get(key) or "")
        feat[f"has_{key}"] = float(bool(source))
        feat[f"native_{key}"] = float(source.startswith("native"))
        feat[f"stdocr_{key}"] = float(source in STANDARD_SOURCES - {"native"})
        feat[f"special_{key}"] = float(bool(source) and source not in STANDARD_SOURCES)
        feat[f"untrusted_{key}"] = float(key in untrusted)
    for key in MARKER_KEYS:
        feat[f"mark_{key}"] = float(bool(sources.get(key)))
    feat["risk_source_present"] = float(bool(sources.get("risk_flags")))
    for reason in REASON_KEYS:
        feat[f"why_{reason}"] = float(reason in reasons)
    feat["trusted_chars_lt_120"] = float(feat["trusted_chars"] < 120)
    feat["trusted_chars_lt_400"] = float(feat["trusted_chars"] < 400)
    feat["trusted_chars_ge_1000"] = float(feat["trusted_chars"] >= 1000)
    return feat


def correctness_features(state: dict, proba: list[float], action: str) -> dict[str, float]:
    """Features calibrated against correctness of the exact emitted pathway."""
    evidence = build_features(state)
    current = str(state.get("final_decision") or "")
    chosen = float(proba[LABELS.index(action)])
    runtime = float(state.get("runtime_confidence") or 0.0)
    feat = {
        "p_approved": float(proba[0]), "p_denied": float(proba[1]),
        "p_review": float(proba[2]), "chosen_p": chosen,
        "chosen_logit": math.log(max(chosen, 1e-4) / max(1 - chosen, 1e-4)),
        "runtime_confidence": runtime, "changed": float(action != current),
        "hard_denied": float(state.get("hard") == "DENIED"),
        "finding_present": float(state.get("adjudicator_finding") in LABELS),
        "n_missing": evidence["n_missing"], "n_conflicts": evidence["n_conflicts"],
        "n_untrusted_conflicts": evidence["n_untrusted_conflicts"],
        "n_flags": evidence["n_flags"],
        "observed_flags_seen": evidence["observed_flags_seen"],
        "has_dq_flag": evidence["has_dq_flag"],
        "has_review_flag": evidence["has_review_flag"],
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


@lru_cache(maxsize=1)
def load_arbiter() -> dict[str, Any] | None:
    if not _ARTIFACT.is_file():
        return None
    try:
        import joblib

        return joblib.load(_ARTIFACT)
    except (OSError, ValueError, TypeError, ImportError):
        return None


def class_probabilities_batch(states: list[dict]) -> list[list[float]] | None:
    """Score a whole batch in one pass.

    Tree-ensemble inference has a large fixed cost per call, so the runtime
    scores every packet together at the end of the run instead of paying that
    cost once per PDF.
    """
    blob = load_arbiter()
    if blob is None or not states:
        return None
    import numpy as np

    feats = [build_features(state) for state in states]
    x = np.asarray(
        [[feat.get(name, 0.0) for name in blob["features"]] for feat in feats],
        dtype=float,
    )
    acc = np.zeros((len(states), 3))
    for model in blob["models"]:
        proba = model.predict_proba(x)
        for j, cls in enumerate(model.classes_):
            acc[:, int(cls)] += proba[:, j]
    return (acc / len(blob["models"])).tolist()


def class_probabilities(state: dict) -> list[float] | None:
    batch = class_probabilities_batch([state])
    return batch[0] if batch else None


def apply_arbiter_batch(preds: list[dict], states: list[dict]) -> None:
    """Arbitrate a whole run in one model pass.  Mutates ``preds`` in place."""
    batch = class_probabilities_batch(states)
    if batch is None:
        return
    for pred, state, proba in zip(preds, states, batch):
        _apply_one(pred, state, proba)


def apply_arbiter(pred: dict, state: dict) -> None:
    """Single-packet arbitration; see :func:`apply_arbiter_batch`."""
    proba = class_probabilities(state)
    if proba is not None:
        _apply_one(pred, state, proba)


def _calibrated_correctness(blob: dict, state: dict, proba: list[float],
                            chosen: str) -> float | None:
    calibrator = blob.get("correctness_calibrator")
    names = blob.get("correctness_features")
    if calibrator is None or not names:
        return None
    feat = correctness_features(state, proba, chosen)
    row = [[feat.get(name, 0.0) for name in names]]
    return float(calibrator.predict_proba(row)[0][1])


def runtime_majority_action(state: dict, current: str) -> str:
    """Return a two-of-three evidence-policy vote, preserving trusted locks."""
    finding = state.get("adjudicator_finding")
    if finding in LABELS:
        return str(finding)
    if state.get("hard") == "DENIED":
        return "DENIED"
    votes = [current, state.get("rule_decision"), state.get("meta_label")]
    for action in LABELS:
        if sum(vote == action for vote in votes) >= 2:
            return action
    return current


def _apply_one(pred: dict, state: dict, proba: list[float]) -> None:
    """Relax an over-conservative review when expected utility clearly favours it.

    Mutates ``pred`` in place.  Confidence always tracks the class probability of
    the decision that is actually submitted, which is exactly the quantity the
    Brier calibration term scores.
    """
    blob = load_arbiter()
    if blob is None:
        return
    current = str(pred.get("adjudication") or "")
    pathway = blob.get("decision_pathway")
    if pathway in {"runtime", "runtime_majority"}:
        # The low-variance policy retains the runtime verdict unless two of the
        # three already-computed evidence policies agree on a residual repair.
        chosen = runtime_majority_action(state, current) \
            if pathway == "runtime_majority" else current
        pred["adjudication"] = chosen
        exact = _calibrated_correctness(blob, state, proba, chosen)
        if exact is not None:
            pred["confidence"] = round(min(max(exact, 0.01), 0.99), 4)
        return
    expected_all = {
        action: sum(UTILITY[action][LABELS[k]] * proba[k] for k in range(3))
        for action in LABELS
    }
    if blob.get("mode") == "replace":
        # The model here never saw the runtime's verdict, so it is an
        # independent reading of the same evidence rather than a corrector.
        # FIELD_MANUAL precedence still wins: a trusted signed finding and a
        # hard policy denial are documentary facts, not predictions.
        finding = state.get("adjudicator_finding")
        if finding in LABELS:
            chosen = str(finding)
        elif state.get("hard") == "DENIED":
            chosen = "DENIED"
        else:
            chosen = max(LABELS, key=lambda action: expected_all[action])
        pred["adjudication"] = chosen
        # Only a decision that *overrides* the rule engine is capped, and it is
        # capped at the measured reliability of overriding rather than at the
        # class average: a disagreement is far less reliable than a typical
        # prediction of the same class.  Capping every prediction at its class
        # precision was measured and regressed fresh data.
        exact = _calibrated_correctness(blob, state, proba, chosen)
        if exact is not None:
            confidence = exact
        else:
            probability = float(proba[LABELS.index(chosen)])
            ceiling = float(blob.get("promoted_precision", 1.0))
            confidence = min(probability, ceiling) if chosen != current else probability
        pred["confidence"] = round(min(max(confidence, 0.01), 0.99), 4)
        return
    # FIELD_MANUAL precedence 1: a visible signed adjudicator review note is
    # trusted evidence for review, not a conservatism artifact.
    eligible = (
        current == "NEEDS_REVIEW"
        and state.get("adjudicator_finding") != "NEEDS_REVIEW"
    )
    if eligible:
        margin = float(blob.get("margin", DEFAULT_MARGIN))
        if expected_all["APPROVED"] - expected_all["NEEDS_REVIEW"] > margin:
            pred["adjudication"] = "APPROVED"
            # A promoted approval must not claim more confidence than this rule
            # has been measured to deserve.  The posterior for these packets
            # averages far above the out-of-fold precision of the promotion
            # itself, and the evaluator scores confidence against correctness,
            # so the class probability is capped by that measured reliability.
            ceiling = float(blob.get("promoted_precision", 1.0))
            confidence = min(float(proba[LABELS.index("APPROVED")]), ceiling)
            pred["confidence"] = round(min(max(confidence, 0.01), 0.99), 4)
            return
    if current not in LABELS:
        return
    blended = 0.5 * float(proba[LABELS.index(current)]) + 0.5 * float(
        state.get("runtime_confidence") or 0.0
    )
    pred["confidence"] = round(min(max(blended, 0.01), 0.99), 4)
