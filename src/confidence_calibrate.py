"""R20 confidence-only calibration (decisions/fields unchanged).

Maps raw P-ish confidence + stable internal evidence → P(final decision correct).
Artifact: models/confidence_calibrator.json (small, deterministic coefficients/bins).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from .constants import DISQUALIFYING_FLAGS, REVIEW_ONLY_FLAGS
from .parse_fields import ParsedPacket

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "models" / "confidence_calibrator.json"
_CAL: dict[str, Any] | None = None
_CAL_LOADED = False

REASON_ATOMS = (
    "disqualifying_flags",
    "transit_visa",
    "stale_arrival",
    "unpaid_fee",
    "revoked_sponsor",
    "adjudicator_note_denied",
    "meta_has_dq",
    "meta_deny",
    "missing_fields",
    "unknown_fee",
    "conflicts",
    "review_flags",
    "injection_only",
    "suspicious_sponsor",
    "missing_sponsor",
    "waived_non_dip",
    "unreadable_arrival",
    "insufficient_evidence",
    "xw2_medical_no_biometric",
    "adjudicator_note_review",
    "meta_review",
    "meta_approve_with_flags",
    "meta_approve_complete",
    "clean_packet",
    "adjudicator_approved_clean",
    "adjudicator_finding_approved",
)

HARD_REASONS = frozenset(
    {
        "disqualifying_flags",
        "transit_visa",
        "stale_arrival",
        "unpaid_fee",
        "revoked_sponsor",
        "adjudicator_note_denied",
        "meta_has_dq",
        "meta_deny",
    }
)
POLICY_REVIEW_REASONS = frozenset(
    {
        "review_flags",
        "adjudicator_note_review",
        "unreadable_arrival",
        "xw2_medical_no_biometric",
    }
)
META_APPROVE_REASONS = frozenset(
    {
        "meta_approve_with_flags",
        "meta_approve_complete",
    }
)
CLEAN_REASONS = frozenset(
    {"clean_packet", "adjudicator_approved_clean", "adjudicator_finding_approved"}
)

# Ordered feature names for logistic calibrator.
LOGREG_FEATURES = (
    "raw_confidence",
    "logit_raw",
    "dec_APPROVED",
    "dec_DENIED",
    "dec_NEEDS_REVIEW",
    "logit_x_APPROVED",
    "logit_x_DENIED",
    "logit_x_REVIEW",
    "raw_x_APPROVED",
    "raw_x_DENIED",
    "raw_x_REVIEW",
    "n_missing",
    "observed_flags_seen",
    "n_flags",
    "has_dq_flag",
    "has_review_flag",
    "n_conflicts",
    "has_untrusted_conflicts",
    "src_intake",
    "src_fee",
    "src_registry",
    "src_biometric",
    "src_adjudicator",
    "src_complete",
    "hard_policy",
    "policy_review",
    "meta_approve",
    "clean_reason",
    "meta_review",
    "evidence_gap",
    "n_reasons",
)


def _logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def reason_group(reasons: list[str] | set[str]) -> str:
    rs = set(reasons)
    if rs & HARD_REASONS:
        return "hard"
    if rs & POLICY_REVIEW_REASONS:
        return "policy_review"
    if rs & META_APPROVE_REASONS or rs & CLEAN_REASONS:
        return "approve_path"
    if "meta_review" in rs:
        return "meta_review"
    if rs & {
        "missing_fields",
        "unknown_fee",
        "conflicts",
        "insufficient_evidence",
        "injection_only",
        "suspicious_sponsor",
        "missing_sponsor",
        "waived_non_dip",
    }:
        return "evidence_gap"
    return "other"


def conf_bucket(decision: str, raw: float) -> str:
    if decision == "NEEDS_REVIEW":
        if raw < 0.35:
            return "r_lo"
        if raw < 0.55:
            return "r_mid"
        if raw < 0.75:
            return "r_hi"
        return "r_vhi"
    if decision == "DENIED":
        if raw < 0.6:
            return "d_lo"
        if raw < 0.8:
            return "d_mid"
        return "d_hi"
    # APPROVED
    if raw < 0.7:
        return "a_lo"
    if raw < 0.85:
        return "a_mid"
    return "a_hi"


def build_cal_features(
    *,
    decision: str,
    raw_confidence: float,
    packet: ParsedPacket,
    missing: list[str],
    reasons: list[str],
) -> dict[str, float]:
    flags = set(packet.risk_flags)
    rs = set(reasons)
    raw = float(raw_confidence)
    logit = _logit(raw)
    d_a = 1.0 if decision == "APPROVED" else 0.0
    d_d = 1.0 if decision == "DENIED" else 0.0
    d_r = 1.0 if decision == "NEEDS_REVIEW" else 0.0
    src_intake = 1.0 if "intake" in packet.sources_seen else 0.0
    src_fee = 1.0 if "fee" in packet.sources_seen else 0.0
    observed = 1.0 if packet.observed_flags_seen else 0.0
    feat: dict[str, float] = {
        "raw_confidence": raw,
        "logit_raw": logit,
        "dec_APPROVED": d_a,
        "dec_DENIED": d_d,
        "dec_NEEDS_REVIEW": d_r,
        "logit_x_APPROVED": logit * d_a,
        "logit_x_DENIED": logit * d_d,
        "logit_x_REVIEW": logit * d_r,
        "raw_x_APPROVED": raw * d_a,
        "raw_x_DENIED": raw * d_d,
        "raw_x_REVIEW": raw * d_r,
        "n_missing": float(len(missing)),
        "observed_flags_seen": observed,
        "n_flags": float(len(flags)),
        "has_dq_flag": float(bool(flags & DISQUALIFYING_FLAGS)),
        "has_review_flag": float(bool(flags & REVIEW_ONLY_FLAGS)),
        "n_conflicts": float(len(packet.conflicts)),
        "has_untrusted_conflicts": float(bool(packet.untrusted_conflicts)),
        "src_intake": src_intake,
        "src_fee": src_fee,
        "src_registry": 1.0 if "registry" in packet.sources_seen else 0.0,
        "src_biometric": 1.0 if "biometric" in packet.sources_seen else 0.0,
        "src_adjudicator": 1.0 if "adjudicator" in packet.sources_seen else 0.0,
        "src_complete": float(src_intake and src_fee and observed),
        "hard_policy": float(bool(rs & HARD_REASONS)),
        "policy_review": float(bool(rs & POLICY_REVIEW_REASONS)),
        "meta_approve": float(bool(rs & META_APPROVE_REASONS)),
        "clean_reason": float(bool(rs & CLEAN_REASONS)),
        "meta_review": float("meta_review" in rs),
        "evidence_gap": float(
            bool(
                rs
                & {
                    "missing_fields",
                    "unknown_fee",
                    "conflicts",
                    "insufficient_evidence",
                    "injection_only",
                    "suspicious_sponsor",
                    "missing_sponsor",
                    "waived_non_dip",
                }
            )
        ),
        "n_reasons": float(len(reasons)),
    }
    # Compact reason atoms (optional extras for bin keys / debugging).
    for atom in REASON_ATOMS:
        feat[f"reason_{atom}"] = float(atom in rs)
    feat["reason_group_hard"] = float(reason_group(rs) == "hard")
    feat["reason_group_policy_review"] = float(reason_group(rs) == "policy_review")
    feat["reason_group_approve_path"] = float(reason_group(rs) == "approve_path")
    feat["reason_group_meta_review"] = float(reason_group(rs) == "meta_review")
    feat["reason_group_evidence_gap"] = float(reason_group(rs) == "evidence_gap")
    feat["bin_key_decision"] = {"APPROVED": 0.0, "DENIED": 1.0, "NEEDS_REVIEW": 2.0}.get(
        decision, -1.0
    )
    return feat


def feature_vector(feat: dict[str, float], names: tuple[str, ...] | list[str]) -> list[float]:
    return [float(feat.get(n, 0.0)) for n in names]


def bin_key(decision: str, raw: float, reasons: list[str] | set[str]) -> str:
    return f"{decision}|{conf_bucket(decision, raw)}|{reason_group(reasons)}"


def load_calibrator(force: bool = False) -> dict[str, Any] | None:
    global _CAL, _CAL_LOADED
    if os.environ.get("MIB_DISABLE_CONF_CAL") == "1":
        return None
    if _CAL_LOADED and not force:
        return _CAL
    _CAL_LOADED = True
    if not _ARTIFACT.exists():
        _CAL = None
        return None
    with _ARTIFACT.open(encoding="utf-8") as f:
        _CAL = json.load(f)
    return _CAL


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_logreg(feat: dict[str, float], cal: dict[str, Any]) -> float:
    names = cal["feature_names"]
    coef = cal["coef"]
    intercept = float(cal["intercept"])
    x = feature_vector(feat, names)
    s = intercept + sum(c * v for c, v in zip(coef, x))
    return _sigmoid(s)


def apply_bins(feat: dict[str, float], decision: str, reasons: list[str], cal: dict[str, Any]) -> float:
    raw = float(feat["raw_confidence"])
    key = bin_key(decision, raw, reasons)
    bins = cal["bins"]
    if key in bins:
        return float(bins[key]["rate"])
    # Fallback: decision prior, then global.
    dec_key = f"__decision__|{decision}"
    if dec_key in bins:
        return float(bins[dec_key]["rate"])
    return float(cal.get("global_rate", raw))


def calibrate_confidence(
    *,
    decision: str,
    raw_confidence: float,
    packet: ParsedPacket,
    missing: list[str],
    reasons: list[str],
) -> float:
    """Return calibrated P(decision correct); identity if no artifact."""
    raw = min(max(float(raw_confidence), 0.0), 1.0)
    cal = load_calibrator()
    if cal is None:
        return raw
    feat = build_cal_features(
        decision=decision,
        raw_confidence=raw,
        packet=packet,
        missing=missing,
        reasons=reasons,
    )
    ctype = cal.get("type")
    if ctype == "logreg":
        out = apply_logreg(feat, cal)
    elif ctype == "shrink_bins":
        out = apply_bins(feat, decision, reasons, cal)
    else:
        out = raw
    return min(max(float(out), 0.0), 1.0)


def reset_calibrator_cache() -> None:
    global _CAL, _CAL_LOADED
    _CAL = None
    _CAL_LOADED = False
