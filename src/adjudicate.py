from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from .constants import (
    CRITICAL_FIELDS,
    DISQUALIFYING_FLAGS,
    PACKET_RECEIPT_DATE,
    REVIEW_ONLY_FLAGS,
    STALE_DAYS,
    SUSPICIOUS_SPONSORS,
)
from .confidence_calibrate import calibrate_confidence
from .features import feature_dict
from .meta_model import (
    adjudication_runtime_config,
    predict_adjudication,
    predict_has_dq,
)
from .parse_fields import ParsedPacket
from .fee_absent import fee_is_absent_imputed
from .sponsor_policy import is_revoked_sponsor


def _placeholder_fields(packet: ParsedPacket) -> dict[str, str]:
    f = packet.fields
    risk = "|".join(packet.risk_flags) if packet.risk_flags else "none"
    return {
        "case_id": packet.case_id,
        "applicant_name": f.get("applicant_name", "unknown"),
        "species_code": f.get("species_code", "unknown"),
        "home_world": f.get("home_world", "unknown"),
        "visa_class": f.get("visa_class", "unknown"),
        "sponsor_id": f.get("sponsor_id", "SPN-0000"),
        "arrival_date": f.get("arrival_date", "1900-01-01"),
        "declared_purpose": f.get("declared_purpose", "unknown"),
        "risk_flags": risk,
        "fee_status": f.get("fee_status", "unknown"),
    }


def _missing_critical(fields: dict[str, str]) -> list[str]:
    missing = []
    defaults = {
        "applicant_name": "unknown",
        "species_code": "unknown",
        "home_world": "unknown",
        "visa_class": "unknown",
        "sponsor_id": "SPN-0000",
        "arrival_date": "1900-01-01",
        "declared_purpose": "unknown",
        "fee_status": "unknown",
    }
    for key in CRITICAL_FIELDS:
        if fields.get(key, defaults[key]) == defaults[key]:
            missing.append(key)
    return missing


def _is_stale(arrival_date: str, visa_class: str) -> bool:
    if visa_class == "DIP-1":
        return False
    try:
        arrival = datetime.strptime(arrival_date, "%Y-%m-%d")
        receipt = datetime.strptime(PACKET_RECEIPT_DATE, "%Y-%m-%d")
    except ValueError:
        return False
    if arrival_date == "1900-01-01":
        return False
    return (receipt - arrival) > timedelta(days=STALE_DAYS)


def _rule_adjudicate(packet: ParsedPacket) -> tuple[dict, list[str], str | None]:
    """Return fields, reasons, hard_decision (DENIED locked or None)."""
    fields = _placeholder_fields(packet)
    missing = _missing_critical(fields)
    flags = set(packet.risk_flags)
    dq = flags & DISQUALIFYING_FLAGS
    review_flags = flags & REVIEW_ONLY_FLAGS
    visa = fields["visa_class"]
    fee = fields["fee_status"]
    sponsor = fields["sponsor_id"]

    reasons: list[str] = []
    hard: str | None = None
    soft: str | None = None

    # FIELD_MANUAL trusted-evidence #1: visible MIB adjudicator stamp / signed
    # manual note. After active-case validation in the parser, the finding locks
    # the decision — lower-precedence OCR fee/visa/flags/missing must not override.
    finding = packet.adjudicator_finding
    if finding == "DENIED":
        reasons.append("adjudicator_note_denied")
        fields["adjudication"] = "DENIED"
        return fields, reasons, "DENIED"
    if finding == "NEEDS_REVIEW":
        reasons.append("adjudicator_note_review")
        fields["adjudication"] = "NEEDS_REVIEW"
        return fields, reasons, None
    if finding == "APPROVED":
        reasons.append("adjudicator_finding_approved")
        fields["adjudication"] = "APPROVED"
        return fields, reasons, None

    stale = _is_stale(fields["arrival_date"], visa)
    visa_source = str(packet.field_sources.get("visa_class", ""))
    arrival_source = str(packet.field_sources.get("arrival_date", ""))
    stale_evidence_unresolved = (
        stale
        and bool(missing)
        and not visa_source.startswith("native")
        and not arrival_source.startswith("native")
    )
    unpaid = fee == "unpaid"
    # DIP-1 is sponsor-exempt, so an unreadable visa cannot establish the
    # non-DIP half of a revoked-sponsor denial.  Keep it reviewable until a
    # trusted visa class is present instead of treating ``unknown`` as proof.
    revoked = is_revoked_sponsor(sponsor) and visa not in {"DIP-1", "unknown"}
    if dq:
        hard = "DENIED"
        reasons.append("disqualifying_flags")
    else:
        # TRANSIT-7 is normally denied, but an explicit rescission is a
        # review-bearing adjudication event rather than an ordinary damaged
        # biometric.  Preserve that review state instead of silently
        # reinstating the denial that the visible record says was rescinded.
        rescinded_transit = review_flags == {"rescinded_denial"}
        transit_evidence_unresolved = (
            visa == "TRANSIT-7"
            and len(missing) >= 2
            and len(packet.untrusted_conflicts) >= 2
            and not visa_source.startswith("native")
        )
        if visa == "TRANSIT-7" and not (
            rescinded_transit or transit_evidence_unresolved
        ):
            hard = "DENIED"
            reasons.append("transit_visa")
        elif rescinded_transit:
            reasons.append("rescinded_transit")
        elif transit_evidence_unresolved:
            reasons.append("unresolved_transit_evidence")

        if hard is None:
            if stale and not stale_evidence_unresolved:
                hard = "DENIED"
                reasons.append("stale_arrival")
            elif stale_evidence_unresolved:
                reasons.append("unresolved_stale_evidence")
            elif unpaid:
                hard = "DENIED"
                reasons.append("unpaid_fee")
            elif revoked:
                hard = "DENIED"
                reasons.append("revoked_sponsor")

    if hard:
        return fields, reasons, hard

    needs_review = soft == "NEEDS_REVIEW"
    if packet.injection_heavy and packet.trusted_text_chars < 120:
        needs_review = True
        reasons.append("injection_only")
    if missing:
        needs_review = True
        reasons.append("missing_fields")
    if review_flags:
        needs_review = True
        reasons.append("review_flags")
    if fee == "unknown":
        needs_review = True
        reasons.append("unknown_fee")
    if packet.conflicts:
        needs_review = True
        reasons.append("conflicts")
    if "arrival_date_unreadable" in packet.conflicts:
        needs_review = True
        reasons.append("unreadable_arrival")
    # FIELD_MANUAL fee rules: ``waived`` is "acceptable only for DIP-1 *or a
    # visible hardship waiver*".  A non-DIP waiver with visible authorization
    # therefore satisfies the fee rule outright; only an unauthorized one is a
    # policy review.  R38 already requires positive waiver evidence on the
    # receipt itself, and the post-meta guard still blocks approval when that
    # evidence is absent.
    authorized_waiver = fee == "waived" and packet.positive_waiver_seen
    if fee == "waived" and visa != "DIP-1" and not authorized_waiver:
        needs_review = True
        reasons.append("waived_non_dip")
    if visa != "DIP-1" and sponsor == "SPN-0000":
        needs_review = True
        reasons.append("missing_sponsor")
    if sponsor in SUSPICIOUS_SPONSORS:
        needs_review = True
        reasons.append("suspicious_sponsor")

    complete = (
        not missing
        and not flags
        and not packet.conflicts
        and not packet.untrusted_conflicts
    )
    fee_ok = (
        fee == "paid"
        or (fee == "waived" and visa == "DIP-1")
        or authorized_waiver
    )
    clean_enough = (
        complete
        and fee_ok
        and "intake" in packet.sources_seen
        and "fee" in packet.sources_seen
        and packet.observed_flags_seen
        and sponsor not in SUSPICIOUS_SPONSORS
    )
    if needs_review:
        soft = "NEEDS_REVIEW"
    elif clean_enough:
        soft = "APPROVED"
        reasons.append("clean_packet")
    else:
        soft = "NEEDS_REVIEW"
        reasons.append("insufficient_evidence")

    fields["adjudication"] = soft
    return fields, reasons, None


def _has_trusted_biometric_evidence(packet: ParsedPacket) -> bool:
    """Trusted biometric / clean-biohazard evidence via source provenance only.

    True when a biometric stream was seen, Observed-flags evidence exists
    (including specialized biometric-header OCR), or biometric_flags were
    filled from that provenance. Page counts / case IDs are never used.
    """
    if "biometric" in packet.sources_seen:
        return True
    if packet.observed_flags_seen:
        return True
    if packet.field_sources.get("biometric_flags") == "biometric_header_ocr":
        return True
    return False


def _r30_xw2_medical_approval_guard(
    decision: str,
    fields: dict,
    packet: ParsedPacket,
    reasons: list[str],
) -> tuple[str, list[str]]:
    """Post-meta: XW-2 + medical consult without biometric evidence → REVIEW.

    Never hard-DENY. Trusted adjudicator findings (R25/R28/R29) bypass.
    """
    if decision != "APPROVED":
        return decision, reasons
    if packet.adjudicator_finding is not None:
        return decision, reasons
    if fields.get("visa_class") != "XW-2":
        return decision, reasons
    if fields.get("declared_purpose") != "medical consult":
        return decision, reasons
    if _has_trusted_biometric_evidence(packet):
        return decision, reasons
    reasons = list(reasons)
    reasons.append("xw2_medical_no_biometric")
    return "NEEDS_REVIEW", reasons


def _counterfactual_review_denial(
    packet: ParsedPacket, pdf_path: Path | None
) -> float | None:
    """Return an independent meta-denial confidence hidden by review atoms.

    Review-only risk evidence must not erase stronger, independent denial
    evidence.  Re-evaluate the exact same packet after removing only the
    review-only atoms; output fields and the real packet are never mutated.
    This is deliberately gated to a confident DENIED meta result rather than
    treating absence of the review atoms as new evidence.
    """
    flags = set(packet.risk_flags)
    if pdf_path is None or not (flags & REVIEW_ONLY_FLAGS) or (flags & DISQUALIFYING_FLAGS):
        return None
    counterfactual = deepcopy(packet)
    counterfactual.risk_flags = sorted(flags - REVIEW_ONLY_FLAGS)
    cf_fields, _cf_reasons, cf_hard = _rule_adjudicate(counterfactual)
    if cf_hard is not None:
        return None
    cf_rule = str(cf_fields["adjudication"])
    meta = predict_adjudication(feature_dict(counterfactual, pdf_path, cf_rule))
    if meta is None:
        return None
    label, confidence = meta
    if (
        label == "DENIED"
        and confidence >= adjudication_runtime_config()["deny_threshold"]
    ):
        return float(confidence)
    return None


def _finalize_confidence(
    fields: dict,
    decision: str,
    raw_confidence: float,
    packet: ParsedPacket,
    missing: list[str],
    reasons: list[str],
    collect: dict | None = None,
) -> dict:
    """Clamp + optional R20 calibration; never alter adjudication/fields."""
    raw = float(raw_confidence)
    if collect is not None:
        collect["raw_confidence"] = raw
        collect["decision"] = decision
        collect["reasons"] = list(reasons)
        collect["missing"] = list(missing)
    cal = calibrate_confidence(
        decision=decision,
        raw_confidence=raw,
        packet=packet,
        missing=missing,
        reasons=reasons,
    )
    fields["adjudication"] = decision
    fields["confidence"] = float(min(max(cal, 0.0), 1.0))
    return fields


def adjudicate(
    packet: ParsedPacket,
    pdf_path: Path | None = None,
    collect: dict | None = None,
) -> dict:
    fields, reasons, hard = _rule_adjudicate(packet)
    missing = _missing_critical(fields)

    if hard:
        raw = _confidence(hard, packet, missing, reasons)
        return _finalize_confidence(
            fields, hard, raw, packet, missing, reasons, collect=collect
        )

    rule_decision = fields["adjudication"]
    decision = rule_decision
    confidence = _confidence(decision, packet, missing, reasons)

    # Trusted adjudicator finding already locked APPROVED / NEEDS_REVIEW.
    # Meta fee/flag heuristics must not override FIELD_MANUAL precedence, and
    # must not mutate extracted fields (e.g. risk_flags) to fit a decision.
    if packet.adjudicator_finding in {"APPROVED", "NEEDS_REVIEW"}:
        return _finalize_confidence(
            fields, decision, confidence, packet, missing, reasons, collect=collect
        )

    if pdf_path is not None:
        feats = feature_dict(packet, pdf_path, rule_decision)
        model_config = adjudication_runtime_config()
        dq_prob = 0.0
        if not packet.observed_flags_seen and not packet.risk_flags:
            dq_prob = predict_has_dq(feats)
            if dq_prob >= model_config["dq_deny_threshold"]:
                packet.risk_flags = ["biohazard_red"]
                fields["risk_flags"] = "biohazard_red"
                decision = "DENIED"
                confidence = max(0.65, dq_prob)
                reasons.append("meta_has_dq")
                return _finalize_confidence(
                    fields,
                    decision,
                    confidence,
                    packet,
                    missing,
                    reasons,
                    collect=collect,
                )

        meta = predict_adjudication(feats)
        if meta is not None:
            meta_label, meta_conf = meta
            complete = (
                len(missing) == 0
                and not packet.risk_flags
                and not packet.conflicts
                and not packet.untrusted_conflicts
            )
            if meta_label == "APPROVED":
                if (
                    packet.observed_flags_seen
                    and complete
                    and meta_conf >= model_config["observed_approve_threshold"]
                ):
                    decision = "APPROVED"
                    confidence = max(meta_conf, 0.75)
                    reasons.append("meta_approve_with_flags")
                elif (
                    not packet.observed_flags_seen
                    and complete
                    and meta_conf >= model_config["approve_threshold"]
                    and "fee" in packet.sources_seen
                    and dq_prob < model_config["dq_approve_ceiling"]
                ):
                    # Only auto-approve without Observed-flags when DQ detector
                    # is confidently negative.
                    decision = "APPROVED"
                    confidence = meta_conf
                    reasons.append("meta_approve_complete")
                elif rule_decision == "APPROVED":
                    decision = "APPROVED"
                    confidence = max(confidence, meta_conf)
            elif meta_label == "DENIED":
                if meta_conf >= model_config["deny_threshold"]:
                    decision = "DENIED"
                    confidence = meta_conf
                    reasons.append("meta_deny")
            elif meta_label == "NEEDS_REVIEW":
                if (
                    rule_decision != "APPROVED"
                    or meta_conf >= model_config["review_threshold"]
                ):
                    decision = "NEEDS_REVIEW"
                    # Higher confidence when model agrees with policy review.
                    confidence = max(0.4, min(0.88, meta_conf))
                    reasons.append("meta_review")

    # R30: post-meta guard — never auto-approve XW-2 medical consult without
    # trusted biometric / clean-biohazard evidence (FIELD_MANUAL: MED-3 owns
    # medical consult + clean biohazard check). Never hard DENY.
    decision, reasons = _r30_xw2_medical_approval_guard(
        decision, fields, packet, reasons
    )
    if "xw2_medical_no_biometric" in reasons:
        confidence = _confidence(decision, packet, missing, reasons)

    # A visible ``waived`` value is not authorization by itself. The public
    # policy permits a non-DIP waiver only with visible hardship/waiver
    # evidence, so a statistical model may not promote that policy-review path
    # to APPROVED when R38 found no positive authorization.
    if (
        decision == "APPROVED"
        and fields.get("fee_status") == "waived"
        and fields.get("visa_class") != "DIP-1"
        and not packet.positive_waiver_seen
    ):
        decision = "NEEDS_REVIEW"
        if "waived_non_dip" not in reasons:
            reasons = list(reasons) + ["waived_non_dip"]
        confidence = _confidence(decision, packet, missing, reasons)

    # R42: never APPROVED on absent-fee imputation (unpaid absents exist; no
    # visible receipt). Trusted Finding lock already returned earlier.
    if decision == "APPROVED" and fee_is_absent_imputed(packet):
        decision = "NEEDS_REVIEW"
        reasons = list(reasons) + ["fee_absent_imputed"]
        confidence = _confidence(decision, packet, missing, reasons)

    # A source-local visible ``Observed: none`` is direct biometric risk
    # evidence, not merely absence of a parsed flag.  When the fee is also a
    # visible, non-imputed acceptable value, those two policy-bearing sources
    # outrank OCR uncertainty in descriptive fields.  Trusted adjudicator
    # findings returned above, every hard denial, absent-fee imputation, and
    # non-DIP waiver review remain untouched.
    explicit_clean_risk = bool(packet.field_sources.get("_observed_none_any"))
    visible_fee_ok = (
        fields.get("fee_status") == "paid"
        or (
            fields.get("fee_status") == "waived"
            and fields.get("visa_class") == "DIP-1"
        )
    )
    if (
        decision == "NEEDS_REVIEW"
        and explicit_clean_risk
        and not packet.risk_flags
        and visible_fee_ok
        and not fee_is_absent_imputed(packet)
        and (
            not packet.untrusted_conflicts
            or fields.get("visa_class") == "DIP-1"
        )
    ):
        decision = "APPROVED"
        reasons = list(reasons) + ["explicit_clean_risk_and_fee"]
        confidence = 0.72

    # Monotonic policy interaction: adding a review-only atom can legitimately
    # turn APPROVED into REVIEW, but it cannot demote independent, confident
    # DENIED evidence into REVIEW. Keep the extracted risk atom unchanged.
    if decision == "NEEDS_REVIEW" and packet.adjudicator_finding is None:
        counterfactual_denial = _counterfactual_review_denial(packet, pdf_path)
        if counterfactual_denial is not None:
            decision = "DENIED"
            reasons = list(reasons) + ["review_monotonic_meta_deny"]
            confidence = counterfactual_denial

    return _finalize_confidence(
        fields, decision, confidence, packet, missing, reasons, collect=collect
    )


def _confidence(
    decision: str,
    packet: ParsedPacket,
    missing: list[str],
    reasons: list[str],
) -> float:
    policy_review = bool(
        (
            {"review_flags", "adjudicator_note_review", "unreadable_arrival"}
        )
        & set(reasons)
    )

    if "injection_only" in reasons:
        return 0.15
    if decision == "NEEDS_REVIEW":
        if "xw2_medical_no_biometric" in reasons and not missing:
            return 0.72
        if policy_review and len(missing) <= 1:
            return 0.85
        if policy_review:
            return 0.7
        if "suspicious_sponsor" in reasons and not missing:
            return 0.55
        return 0.28
    if decision == "DENIED":
        if "disqualifying_flags" in reasons or "transit_visa" in reasons:
            return 0.9 if not missing else 0.55
        if "stale_arrival" in reasons or "unpaid_fee" in reasons:
            return 0.88 if not missing else 0.55
        if "revoked_sponsor" in reasons or "meta_deny" in reasons:
            return 0.8 if not missing else 0.5
        if "adjudicator_note_denied" in reasons:
            return 0.88
        return 0.55
    if "adjudicator_finding_approved" in reasons:
        return 0.88
    if packet.observed_flags_seen and not missing:
        return 0.84
    if not missing and "intake" in packet.sources_seen:
        return 0.76
    return 0.45
