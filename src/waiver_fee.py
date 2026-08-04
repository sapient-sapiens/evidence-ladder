"""Round 38: explicit fee-receipt Waiver Code → fee_status fill.

FIELD_MANUAL: waived is acceptable for DIP-1 or a visible hardship waiver.
DEV audit: among fee receipts, `Waiver Code DIP-WAIVER` co-occurs with truth
`waived` for every observed case; `Waiver Code N/A` is mixed and must not
impute paid/waived. The positive code is the authorization and therefore
overrides a contradictory printed paid/unpaid status, but never a manual
correction. N/A implies nothing.

No demographic / label-frequency fee imputation. No bare word "waiver".
Requires OCR-tolerant `Waiver Code` label (or signed hardship-authorization
context) plus an active case id in the cleaned stream.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .noisy_channel_ocr import (
    ConfusionModel,
    levenshtein,
    load_confusion_model,
    log_emit_prob,
)

if TYPE_CHECKING:
    from .parse_fields import ParsedPacket
    from .text_extract import TextSources


def _strip_injection(text: str) -> tuple[str, Any]:
    # Lazy import avoids parse_fields ↔ waiver_fee cycle.
    from .parse_fields import strip_injection

    return strip_injection(text)

# Tiny public/observed code vocabulary (no case IDs / frequency tables).
WAIVER_CODE_VOCAB = ("DIP-WAIVER", "N/A", "NA", "NONE")
# Only DIP-WAIVER is positive fee evidence under DEV controls.
POSITIVE_WAIVER_CODES = frozenset({"DIP-WAIVER"})
NULL_WAIVER_CODES = frozenset({"N/A", "NA", "NONE", "NULL", "-", "—", "UNKNOWN"})

_FEE_HDR_RE = re.compile(r"(?i)\bMIB\s*Fee\s*Receipt\b|\bFee\s*Receipt\b")
# OCR-tolerant Waiver Code label — not bare "waiver".
_WAIVER_LABEL_RE = re.compile(
    r"(?i)\bw[ao0][il1]v[ae]?r\s*c[o0]d[e3]\b"
)
_HARDSHIP_AUTH_RE = re.compile(
    r"(?i)\b(?:hard\s*ship\s+waiv(?:er|ed)|waiv(?:er|ed)\s+authori[sz]ation|"
    r"authori[sz]ation\s+to\s+waive(?:\s+fee)?|"
    r"fee\s+waiv(?:er|ed)\s+authori[sz]ation)\b"
)
_SIGNED_CTX_RE = re.compile(
    r"(?i)\b(?:signed|signature|/s/|officer\s+sign|manual\s+correction)\b"
)
_CASE_ID_RE = re.compile(r"\bMIB-\d{6}\b")
_CODE_TOKEN_RE = re.compile(
    r"[\s:.\-_\'\"\`\|\]\[\,“”‘’´]*([A-Za-z0-9][A-Za-z0-9\-_/]{0,28})"
)

STREAM_ORDER = (
    "native",
    "page_ocr",
    "embedded_ocr",
    "threshold_ocr",
    "best_ocr",
)

_FEE_STATUS_VALUE_RE = re.compile(
    r"(?i)\bFee\s+Status\b\s*[:\n ]+\s*(paid|unpaid|waived|unknown)\b"
)
_AMOUNT_VALUE_RE = re.compile(
    r"(?i)\bAmount\b\s*[:\n ]+\s*\$?\s*([0-9]+)(?:\.([0-9]{2}))?"
)
_STANDARD_FEE_CENTS = 80900


def _norm_code_token(raw: str) -> str:
    t = (raw or "").strip().upper().replace("_", "-").replace(" ", "")
    t = t.replace("–", "-").replace("—", "-")
    return t


def _canonicalize_waiver_code(raw: str, model: ConfusionModel | None = None) -> str | None:
    """Map OCR token onto tiny vocab; None if null-family or unrecognized."""
    if not raw:
        return None
    token = _norm_code_token(raw)
    if not token:
        return None
    # Exact / near-exact null family first.
    if token in NULL_WAIVER_CODES or token.replace("/", "") in {"NA", "N-A"}:
        return "N/A"
    if token == "DIP-WAIVER":
        return "DIP-WAIVER"
    # Glyph-tolerant DIP-WAIVER shape before noisy channel.
    compact = (
        token.replace("0", "O")
        .replace("1", "I")
        .replace("|", "I")
        .replace(" ", "")
    )
    if re.fullmatch(r"D[I1L]P\-?W[A4][I1L]?V(?:ER|E|R)?", compact):
        return "DIP-WAIVER"

    # R35 noisy-channel correction against tiny vocab only.
    conf = model or load_confusion_model()
    ranked: list[tuple[float, str]] = []
    obs = token
    for cand in ("DIP-WAIVER", "N/A"):
        d = levenshtein(obs.casefold(), cand.casefold())
        maxlen = max(len(obs), len(cand), 1)
        if d / maxlen > 0.45 and d > 3:
            continue
        # Prefer channel likelihood; fall back to negative edit.
        try:
            score = float(log_emit_prob(obs, cand, conf)) - 0.15 * float(d)
        except Exception:  # noqa: BLE001
            score = -float(d)
        ranked.append((score, cand))
    if not ranked:
        return None
    ranked.sort(key=lambda x: (-x[0], x[1]))
    best_s, best = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else best_s - 99.0
    # Require unique margin + DIP shape for positive code.
    if best == "DIP-WAIVER":
        if second >= best_s - 0.5:
            return None
        if "W" not in token and "V" not in token:
            return None
        if levenshtein(compact[:3], "DIP") > 1:
            return None
        return "DIP-WAIVER"
    if best == "N/A":
        return "N/A"
    return None


def _stream_has_active_case(text: str, case_id: str) -> bool:
    if not case_id or not _CASE_ID_RE.fullmatch(case_id):
        return True  # synthetic tests without MIB ids
    if case_id not in (text or ""):
        return False
    return True


def _window_dominated_by_foreign_case(text: str, start: int, case_id: str) -> bool:
    """Reject Waiver Code windows that only name a different MIB case."""
    if not case_id or not _CASE_ID_RE.fullmatch(case_id):
        return False
    lo = max(0, start - 350)
    hi = min(len(text), start + 220)
    win = text[lo:hi]
    ids = _CASE_ID_RE.findall(win)
    if not ids:
        return False
    foreign = [i for i in ids if i != case_id]
    return bool(foreign) and case_id not in ids


def extract_waiver_code_evidence(
    text: str,
    *,
    case_id: str = "",
    model: ConfusionModel | None = None,
) -> dict[str, Any]:
    """Source-local Waiver Code / hardship-auth parse (one cleaned stream).

    Returns dict with keys: code, implies_waived, via, raw, label_span.
    """
    empty = {
        "code": None,
        "implies_waived": False,
        "via": None,
        "raw": None,
        "label_span": None,
    }
    if not text or not text.strip():
        return empty
    if case_id and not _stream_has_active_case(text, case_id):
        return empty
    # Prefer fee-receipt context when present; still allow labeled Waiver Code
    # on a fee page that OCR truncated the header for, but require either a
    # fee-receipt header somewhere in the stream OR an Amount/$0 cue near code.
    has_fee_hdr = bool(_FEE_HDR_RE.search(text))

    conf = model
    hits: list[dict[str, Any]] = []
    for m in _WAIVER_LABEL_RE.finditer(text):
        if _window_dominated_by_foreign_case(text, m.start(), case_id):
            continue
        # Layout-preserving native extraction can place a wide table cell
        # (50+ spaces) between the label and value.
        tail = text[m.end() : m.end() + 160]
        tm = _CODE_TOKEN_RE.match(tail)
        if not tm:
            continue
        raw = tm.group(1)
        # Refuse reading Fee Status chrome as a code.
        if raw.casefold() in {"status", "fee", "amount", "case", "id"}:
            continue
        code = _canonicalize_waiver_code(raw, model=conf)
        if code is None:
            continue
        # Local fee-receipt cue if global header missing.
        local = text[max(0, m.start() - 280) : m.end() + 40]
        local_fee = bool(
            _FEE_HDR_RE.search(local)
            or re.search(r"(?i)\bAmount\b|\bFee\s*Status\b|\$\s*0\.00", local)
        )
        if not (has_fee_hdr or local_fee):
            continue
        hits.append(
            {
                "code": code,
                "implies_waived": code in POSITIVE_WAIVER_CODES,
                "via": "waiver_code",
                "raw": raw,
                "label_span": (m.start(), m.end()),
            }
        )

    # Signed hardship authorization (no bare "waiver"). DEV support was 0;
    # keep ultra-conservative: require hardship-auth phrase + signed context
    # + fee-receipt cue; still only implies waived when those co-occur.
    if not hits:
        for m in _HARDSHIP_AUTH_RE.finditer(text):
            win = text[max(0, m.start() - 200) : m.end() + 200]
            if not _SIGNED_CTX_RE.search(win):
                continue
            if not (
                has_fee_hdr
                or _FEE_HDR_RE.search(win)
                or re.search(r"(?i)\bAmount\b|\bFee\s*Status\b", win)
            ):
                continue
            if _window_dominated_by_foreign_case(text, m.start(), case_id):
                continue
            hits.append(
                {
                    "code": "HARDSHIP-AUTH",
                    "implies_waived": True,
                    "via": "hardship_auth",
                    "raw": m.group(0),
                    "label_span": (m.start(), m.end()),
                }
            )

    if not hits:
        return empty
    # Ambiguity: any positive + null in same stream → reject (do not guess).
    codes = {h["code"] for h in hits}
    positives = [h for h in hits if h["implies_waived"]]
    nulls = [h for h in hits if h["code"] == "N/A"]
    if positives and nulls:
        return empty
    if positives:
        # Unique positive family.
        via_set = {h["via"] for h in positives}
        if len({h["code"] for h in positives}) != 1:
            return empty
        return positives[0]
    # Null-only: report N/A but do not imply waived.
    if "N/A" in codes:
        return {
            "code": "N/A",
            "implies_waived": False,
            "via": "waiver_code",
            "raw": next(h["raw"] for h in hits if h["code"] == "N/A"),
            "label_span": next(h["label_span"] for h in hits if h["code"] == "N/A"),
        }
    return empty


def fee_fill_from_waiver_evidence(evidence: dict[str, Any]) -> str | None:
    """Map evidence → fee fill value, or None (N/A / absent / no imply)."""
    if not evidence or not evidence.get("implies_waived"):
        return None
    return "waived"


def _fee_missing_or_unknown(fields: dict[str, str]) -> bool:
    fee = fields.get("fee_status")
    return fee is None or fee == "" or fee == "unknown"


def apply_waiver_fee_fill(
    packet: "ParsedPacket",
    sources: "TextSources",
    *,
    model: ConfusionModel | None = None,
) -> str | None:
    """Record positive waiver evidence and apply its authoritative status.

    Never overrides a manual correction and never invents from N/A or bare
    ``waiver``. Does not mutate sources_seen / conflicts / trusted_text_chars.
    """
    src = packet.field_sources.get("fee_status", "")
    can_apply = "manual" not in src

    conf = model or load_confusion_model()
    case_id = packet.case_id or ""
    for stream_name in STREAM_ORDER:
        raw = getattr(sources, stream_name, "") or ""
        if not str(raw).strip():
            continue
        cleaned, _ = _strip_injection(str(raw))
        if not cleaned.strip():
            continue
        evidence = extract_waiver_code_evidence(
            cleaned, case_id=case_id, model=conf
        )
        fill = fee_fill_from_waiver_evidence(evidence)
        if not fill:
            continue
        packet.positive_waiver_seen = True
        if not can_apply:
            return None
        if packet.fields.get("fee_status") == fill:
            return None
        packet.fields["fee_status"] = fill
        packet.field_sources["fee_status"] = (
            f"r38_waiver:{stream_name}:{evidence.get('via') or 'code'}"
        )
        return fill
    return None


def apply_fee_ledger_consistency(
    packet: "ParsedPacket", sources: "TextSources"
) -> str | None:
    """Resolve a contradictory receipt from status, amount, and waiver code.

    These are three assertions on one document, not population priors.  A
    standard $809 charge with no waiver is paid even when the status glyph is
    stale; a zero-dollar DIP-WAIVER is waived.  An explicit zero-dollar
    ``unknown`` receipt remains unknown and must not be statistically imputed.
    """
    # Signed/manual adjudicator notes are the highest-precedence visible
    # evidence in the field manual.  They can explicitly state that the fee is
    # unknown even when a lower-precedence receipt contains a stale value.
    case_id = packet.case_id or ""
    manual_unknown = re.compile(
        r"(?is)Finding\s*:?\s*NEEDS[_ ]REVIEW.{0,220}?"
        r"Reason\s*:?\s*Fee\s+status\s+unknown\b"
    )
    manual_unknown_reason = re.compile(
        r"(?is)Reason\s*:?\s*Fee\s+status\s+unknown\b"
    )
    for stream_name in STREAM_ORDER:
        raw = str(getattr(sources, stream_name, "") or "")
        if not raw.strip():
            continue
        cleaned, _ = _strip_injection(raw)
        if case_id and not _stream_has_active_case(cleaned, case_id):
            continue
        if manual_unknown.search(cleaned) or (
            getattr(packet, "adjudicator_finding", None) == "NEEDS_REVIEW"
            and manual_unknown_reason.search(cleaned)
        ):
            packet.fields["fee_status"] = "unknown"
            packet.field_sources["fee_status"] = (
                f"manual_note:{stream_name}:fee_unknown"
            )
            return "unknown"

    src = packet.field_sources.get("fee_status", "")
    if "manual" in src:
        return None
    for stream_name in STREAM_ORDER:
        raw = str(getattr(sources, stream_name, "") or "")
        if not raw.strip():
            continue
        cleaned, _ = _strip_injection(raw)
        if case_id and not _stream_has_active_case(cleaned, case_id):
            continue
        for header in _FEE_HDR_RE.finditer(cleaned):
            window = cleaned[header.start() : header.start() + 900]
            status_match = _FEE_STATUS_VALUE_RE.search(window)
            amount_match = _AMOUNT_VALUE_RE.search(window)
            # The full cleaned stream was already bound to the active case
            # above.  The receipt-local window may start after its case-id
            # header, so do not require the identifier a second time here.
            waiver = extract_waiver_code_evidence(window, case_id="")
            if not status_match or not amount_match:
                continue
            status = status_match.group(1).casefold()
            cents = int(amount_match.group(1)) * 100 + int(
                amount_match.group(2) or "0"
            )
            code = waiver.get("code")
            value = None
            preserve_unknown = False
            if code == "DIP-WAIVER" and cents == 0:
                value = "waived"
            elif code == "N/A" and cents == _STANDARD_FEE_CENTS:
                value = "paid"
            elif code == "N/A" and cents == 0 and status == "unknown":
                value = "unknown"
                preserve_unknown = True
            elif (
                code == "N/A"
                and cents == 0
                and status == "waived"
                and str(packet.fields.get("visa_class") or "").upper() != "DIP-1"
            ):
                # A zero-dollar receipt that explicitly says no waiver cannot
                # authorize a non-diplomatic waiver.  Preserve the evidence
                # conflict as unknown instead of trusting the stale status
                # cell.  DIP-1 remains untouched because the public manual
                # permits a diplomatic waiver.
                value = "unknown"
                preserve_unknown = True
            if value is None:
                continue
            packet.fields["fee_status"] = value
            suffix = "unknown" if preserve_unknown else value
            packet.field_sources["fee_status"] = (
                f"fee_ledger:{stream_name}:{suffix}"
            )
            if value == "waived":
                packet.positive_waiver_seen = True
            return value
    return None
