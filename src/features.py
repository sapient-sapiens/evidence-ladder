from __future__ import annotations

import subprocess
from pathlib import Path

from .constants import (
    DISQUALIFYING_FLAGS,
    REVIEW_ONLY_FLAGS,
    REVOKED_SPONSORS,
    SUSPICIOUS_SPONSORS,
    VISA_CLASSES,
)
from .parse_fields import ParsedPacket


_FIELD_KEYS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)
_STANDARD_FIELD_SOURCES = frozenset(
    {"native", "page_ocr", "embedded_ocr", "ppocr_ocr", "oriented_ocr"}
)


def pdf_page_count(pdf_path: Path) -> int:
    try:
        out = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return -1
    for line in out.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return -1
    return -1


def feature_dict(packet: ParsedPacket, pdf_path: Path, rule_decision: str) -> dict:
    fields = packet.fields
    flags = set(packet.risk_flags)
    visa = fields.get("visa_class", "unknown")
    fee = fields.get("fee_status", "unknown")
    sponsor = fields.get("sponsor_id", "SPN-0000")
    pages = pdf_page_count(pdf_path)
    try:
        size = pdf_path.stat().st_size
    except OSError:
        size = 0

    feat: dict = {
        "pages": pages,
        "pdf_bytes": size,
        "trusted_chars": packet.trusted_text_chars,
        "injection_heavy": int(packet.injection_heavy),
        "observed_flags_seen": int(packet.observed_flags_seen),
        "n_flags": len(flags),
        "has_dq": int(bool(flags & DISQUALIFYING_FLAGS)),
        "has_review_flag": int(bool(flags & REVIEW_ONLY_FLAGS)),
        # Keep inherited model semantics frozen: it was trained with only the
        # public-manual sponsor set. FIT-derived sponsor policy is enforced by
        # the transparent rule path and must not shift this legacy feature.
        "revoked_sponsor": int(sponsor in REVOKED_SPONSORS),
        "suspicious_sponsor": int(sponsor in SUSPICIOUS_SPONSORS),
        "fee_paid": int(fee == "paid"),
        "fee_waived": int(fee == "waived"),
        "fee_unpaid": int(fee == "unpaid"),
        "fee_unknown": int(fee == "unknown"),
        "src_intake": int("intake" in packet.sources_seen),
        "src_fee": int("fee" in packet.sources_seen),
        "src_registry": int("registry" in packet.sources_seen),
        "src_biometric": int("biometric" in packet.sources_seen),
        "src_adjudicator": int("adjudicator" in packet.sources_seen),
        "n_conflicts": len(packet.conflicts),
        "finding_approved": int(packet.adjudicator_finding == "APPROVED"),
        "finding_denied": int(packet.adjudicator_finding == "DENIED"),
        "finding_review": int(packet.adjudicator_finding == "NEEDS_REVIEW"),
        "rule_approved": int(rule_decision == "APPROVED"),
        "rule_denied": int(rule_decision == "DENIED"),
        "rule_review": int(rule_decision == "NEEDS_REVIEW"),
        "missing_name": int("applicant_name" not in fields),
        "missing_species": int("species_code" not in fields),
        "missing_world": int("home_world" not in fields),
        "missing_visa": int("visa_class" not in fields),
        "missing_sponsor": int("sponsor_id" not in fields),
        "missing_date": int("arrival_date" not in fields),
        "missing_purpose": int("declared_purpose" not in fields),
        "missing_fee": int("fee_status" not in fields),
        "n_missing": sum(1 for k in _FIELD_KEYS if k not in fields),
    }
    for v in VISA_CLASSES:
        feat[f"visa_{v}"] = int(visa == v)
    for atom in sorted(DISQUALIFYING_FLAGS | REVIEW_ONLY_FLAGS):
        feat[f"flag_{atom}"] = int(atom in flags)

    # Evidence-path features for cleanly retrained models. The inherited
    # artifacts ignore these because they select their frozen feature lists.
    # Keep categories coarse so they describe provenance, not layout or a
    # particular case/template instance.
    source_values = [packet.field_sources.get(key, "") for key in _FIELD_KEYS]
    feat.update(
        {
            "n_field_sources": sum(bool(source) for source in source_values),
            "n_fields_from_native": sum(source == "native" for source in source_values),
            "n_fields_from_standard_ocr": sum(
                source in _STANDARD_FIELD_SOURCES - {"native"}
                for source in source_values
            ),
            "n_fields_from_specialized_ocr": sum(
                bool(source) and source not in _STANDARD_FIELD_SOURCES
                for source in source_values
            ),
            "n_untrusted_conflicts": len(packet.untrusted_conflicts),
            "n_explicit_risk_flags": len(packet.explicit_risk_flags),
            "n_r35_repairs": len(packet.r35_repair_evidence),
            "positive_waiver_seen": int(packet.positive_waiver_seen),
            "trusted_chars_ge_120": int(packet.trusted_text_chars >= 120),
            "trusted_chars_ge_400": int(packet.trusted_text_chars >= 400),
            "trusted_chars_ge_1000": int(packet.trusted_text_chars >= 1000),
        }
    )
    for key, source in zip(_FIELD_KEYS, source_values):
        feat[f"field_ocr_{key}"] = int(bool(source) and source != "native")
        feat[f"field_specialized_{key}"] = int(
            bool(source) and source not in _STANDARD_FIELD_SOURCES
        )
        feat[f"untrusted_conflict_{key}"] = int(key in packet.untrusted_conflicts)
        feat[f"r35_repair_{key}"] = int(key in packet.r35_repair_evidence)
    return feat
