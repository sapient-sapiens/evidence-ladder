"""R42: evidence-gated fee imputation when the packet has no fee evidence.

DISTINCTION:
- Absent: after injection strip, no 'fee' token in native/page/embedded streams
  and no FEE STATUS OBSCURED marker. Receipt page is missing from the packet.
- Obscured: explicit OBSCURED / redacted fee status — do NOT impute.
- Present-but-unread: 'fee' appears but status unknown — left to OCR/ROI paths.

Policy (FIT600 prior): among absent packets, paid is the plurality overall and
also within DIP-1. Every one of the six leave-one-fold-out FIT subsets retains
that DIP-1 majority, so impute paid for every visa class.
Never override a non-unknown fee. Provenance `r42_fee_absent:*` must NOT be
treated as a trusted fee source for auto-APPROVE (unpaid absents exist).
"""
from __future__ import annotations

import re

from .parse_fields import ParsedPacket, strip_injection
from .text_extract import TextSources

_OBSCURED_RE = re.compile(
    r"(?i)\[\s*FEE\s*STATUS\s*OBSCURED\s*\]|FEE\s*STATUS\s*OBSCURED|fee\s*status\s*obscured"
)
_FEE_TOKEN_RE = re.compile(r"(?i)\bfee\b")


def streams_have_fee_evidence(sources: TextSources) -> bool:
    """True when any trusted OCR/native stream mentions fee (or obscured)."""
    blob = "\n".join(
        t
        for t in (
            sources.native,
            sources.page_ocr,
            sources.embedded_ocr,
            getattr(sources, "threshold_ocr", "") or "",
            getattr(sources, "best_ocr", "") or "",
        )
        if t
    )
    cleaned, _ = strip_injection(blob)
    if _OBSCURED_RE.search(cleaned):
        return True
    return bool(_FEE_TOKEN_RE.search(cleaned))


def fee_status_obscured(sources: TextSources) -> bool:
    blob = "\n".join(
        t
        for t in (sources.native, sources.page_ocr, sources.embedded_ocr)
        if t
    )
    cleaned, _ = strip_injection(blob)
    return bool(_OBSCURED_RE.search(cleaned))


def apply_absent_fee_imputation(packet: ParsedPacket, sources: TextSources) -> bool:
    """Fill fee_status when absent; return True if mutated.

    Does not add `fee` to sources_seen — adjudication must not treat this as a
    visible fee receipt.
    """
    cur = (packet.fields.get("fee_status") or "").strip()
    if cur and cur != "unknown":
        return False
    src = packet.field_sources.get("fee_status", "")
    if src.startswith("native") or "manual" in src:
        return False
    if fee_status_obscured(sources):
        return False
    if streams_have_fee_evidence(sources):
        return False

    value = "paid"
    packet.fields["fee_status"] = value
    packet.field_sources["fee_status"] = f"r42_fee_absent:{value}"
    # Intentionally do NOT add "fee" to sources_seen.
    return True


def fee_is_absent_imputed(packet: ParsedPacket) -> bool:
    src = packet.field_sources.get("fee_status", "")
    return src.startswith("r42_fee_absent:")
