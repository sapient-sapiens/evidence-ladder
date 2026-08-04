from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from .constants import (
    FEE_STATUSES,
    HOME_WORLDS,
    PURPOSES,
    RISK_FLAG_ATOMS,
    SPECIES_CODES,
    VISA_CLASSES,
)

if TYPE_CHECKING:
    from .text_extract import TextSources

INJECTION_RE = re.compile(
    r"(SYSTEM\s*:|answer key|ignore visible evidence)",
    re.IGNORECASE,
)
ANSWER_KEY_CSV_RE = re.compile(
    r"MIB-\d{6},[^,\n]+,[A-Z0-9_]+,",
)

# Label and value may be separated by spaces, colon, or a single line break
# (common in OCR split-label layouts).
def _label_re(labels: str, value: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:^|\n)\s*(?:{labels})(?:\s{{1,}}|\s*:\s*|\s*\n\s*)({value})"
        rf"(?=\s{{2,}}|\s*$|\s*\n)",
        re.IGNORECASE | re.MULTILINE,
    )


LABEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "applicant_name": _label_re(
        r"Applicant(?:\s+Name)?|Registry Name",
        r".+?",
    ),
    "species_code": _label_re(r"Species\s+Code", r"[A-Z0-9_]+"),
    "home_world": _label_re(r"Home\s+World", r".+?"),
    "visa_class": _label_re(r"Visa\s+Class", r"[A-Z0-9-]+"),
    # Sponsor 1D / ID OCR confusions (R44).
    "sponsor_id": _label_re(r"Sponsor\s+(?:ID|1D|I[D0]|lD)", r"SPN-\d{4}"),
    "arrival_date": _label_re(
        r"Arrival\s+Date",
        r"\d{4}-\d{2}-\d{2}|UNREADABLE|N/?A|MISSING",
    ),
    "declared_purpose": _label_re(r"Declared Purpose|Purpose", r".+?"),
    "fee_status": _label_re(r"Fee\s+Status", r"[A-Za-z\[\] ]+?"),
}

# Nearest-label windows for constrained vocab recovery when regex misses.
# Allow OCR split labels: "Fee\nStatus" / "Visa\nClass".
_NEAR_LABELS: dict[str, re.Pattern[str]] = {
    "fee_status": re.compile(r"Fee(?:\s+Status|\s*\n\s*Status)", re.I),
    "visa_class": re.compile(r"Visa(?:\s+Class|\s*\n\s*Class)", re.I),
    # Exact Arrival label (single-source fills allowed).
    "arrival_date": re.compile(r"Arrival(?:\s+Date|\s*\n\s*Date)", re.I),
    # Fuzzy OCR Arrival labels (R44 expanded from residual DEV gaps).
    "arrival_date_fuzzy": re.compile(
        r"(?:Anwval|Arnval|Arnvai|Arval|Amval|Anivel|Arrivai|Arriva|Antval|"
        r"Anitval|Anwal|Antival|Anival|Amivat|Armvat|Ariival|Arrivel|"
        r"Arriival|Arrivl|Arrival)"
        r"(?:\s+Date|\s*\n\s*Date|\s+Cate|\s+Dabe)?",
        re.I,
    ),
    "sponsor_id": re.compile(
        r"Sponsor(?:\s+(?:ID|1D|I[D0]|lD)|\s*\n\s*(?:ID|1D))",
        re.I,
    ),
    "species_code": re.compile(r"Species(?:\s+Code|\s*\n\s*Code)", re.I),
    "declared_purpose": re.compile(
        r"(?:Dec(?:lared|tored|iered|iored|iared)|Declared)\s+Purpose|"
        r"(?:^|\n)\s*Purpose\b",
        re.I,
    ),
}

# Plausible challenge arrival window (DEV label span is 2025-06 .. 2026-07).
_PLAUSIBLE_DATE_MIN = "2025-01-01"
_PLAUSIBLE_DATE_MAX = "2026-12-31"
_FOOTER_OR_ID_LINE_RE = re.compile(
    r"(?i)\bPacket\s+MIB-\d{6}\b|\bpage\s+\d+\b|\bMIB-\d{6}\b\s*/|\bSYSTEM\s*:",
)

MANUAL_CORRECTION_RE = re.compile(
    r"Manual correction:\s*"
    r"(fee status|visa class|sponsor|applicant|species(?: code)?|"
    r"home world|arrival date|declared purpose|purpose)"
    r"\s+is\s+(.+?)\.",
    re.IGNORECASE,
)

SPONSOR_ANYWHERE_RE = re.compile(r"\bSPN-\d{4}\b")
DATE_ANYWHERE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
# Arrival label omitted: date line between Sponsor ID and Declared Purpose.
_SPONSOR_DATE_PURPOSE_RE = re.compile(
    r"(?is)Sponsor\s+(?:ID|1D)[^\n]{0,40}\n\s*(20\d{2}-\d{2}-\d{2})\s*[^\n]{0,8}\n"
    r"\s*(?:\|?\s*)?(?:Decla|Purpose)",
)
# OCR-tolerant Observed-flags header (R44): Dbserved/Obverved variants.
OBSERVED_FLAGS_RE = re.compile(
    r"(?:Observed|Dbserved|Obverved|Observcd|Ubserved)\s+"
    r"(?:flags|flogs|flegs|fiegs)\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
REGISTRY_STATUS_RE = re.compile(
    r"(?:^|\n)\s*Registry Status(?:\s{1,}|\s*:\s*|\s*\n\s*)(.+?)"
    r"(?=\s{2,}|\s*$|\s*\n)",
    re.IGNORECASE | re.MULTILINE,
)
# Round 28: OCR-tolerant Finding / Findg label; optional colon; NEEDS REVIEW spacing.
ADJUDICATOR_FINDING_RE = re.compile(
    r"\b(?:Findg|Finding)\s*:?\s*(APPROVED|DENIED|NEEDS(?:[_\s-]*REVIEW))\b",
    re.IGNORECASE,
)
# Exact / near-exact Manual Adjudicator Note / stamp headers.
_ADJUDICATOR_HEADER_RE = re.compile(
    r"(?:manual\s+adjudicator\s+note|adjudicator\s+note|adjudicator\s+stamp|"
    r"signed\s+manual\s+note|mib\s+adjudicator|"
    r"manuva\s*:?\s*acjudicator|"
    r"al\s+adjudicator\s+note)",
    re.IGNORECASE,
)
_MIB_EYES_ONLY_RE = re.compile(r"\bMIB\s+Eyes\s+Only\b", re.IGNORECASE)
# Official note body always pairs Finding with a Reason line (OCR-tolerant label).
_ADJUDICATOR_REASON_RE = re.compile(
    r"(?:Reason|Resson|Resor|Resscrc|Reesor|Renson|Monson|Reson|Reeson)\s*:?\s*"
    r"(?:"
    r"Clean\b|Approval\s+supported|Review[- ]?(?:only|cnly|orty|crly)|Revoked\s+sponsor|"
    r"Packet\s+contains|Unpaid\b|Stale\b|Disqualif|TRANSIT|Missing\b|"
    r"exception-qualified|surviving\s+visible|identity[_\s]?conflict|"
    r"illegible|rescinded|biohazard|warrant|embargo|damaged|contradictory|"
    r"Arrival\s+date\s+mi|missing\s+from\s+trusted|"
    # Common OCR garble of the same Reason templates
    r"Cle[a-z]{0,4}\b|Revicw|Packst|demaged|fleg|cnly|"
    r"Glear|ralifiad|except|Ceara|exceptionquai"
    r")",
    re.IGNORECASE,
)
# Reason body without a recoverable Reason: label (OCR dropped the label).
_REASON_BODY_LOOSE_RE = re.compile(
    r"(?:Arrival\s+date\s+miss|missing\s+from\s+trusted\s+visible|"
    r"Review[- ]?(?:only|cnly|orty)\s+risk|Packet\s+contains\s+dam|"
    r"Disqualifying\s+n(?:sk|isk)\s+flag|Clean\s+or\s+exception)",
    re.IGNORECASE,
)
_CROSS_OUT_RE = re.compile(
    r"(?:cross(?:ed)?[\s-]*out|strikethrough|void(?:ed)?|rescinded\s+by\s+later|"
    r"later\s+signed\s+(?:approval|note|correction)|superseded\s+by)",
    re.IGNORECASE,
)
_SPONSOR_PROSE_RE = re.compile(
    r"Sponsor\s+Attestation\s+Letter|attests\s+that\b",
    re.IGNORECASE,
)
_SAMPLE_DENIAL_LINE_RE = re.compile(
    r"(?:sample|sampee|sampie)\s+denial",
    re.IGNORECASE,
)
# Generic Adjudication: label is unsafe (archived adjacent / MIB-000000).
_GENERIC_ADJUDICATION_RE = re.compile(
    r"\bAdjudication\s*:\s*(APPROVED|DENIED|NEEDS_REVIEW)\b",
    re.IGNORECASE,
)
CASE_ID_RE = re.compile(r"\bMIB-\d{6}\b")

_FIELD_SOURCE_TAG = {
    "fee_status": "fee",
    "applicant_name": "intake",
    "species_code": "intake",
    "home_world": "intake",
    "visa_class": "intake",
    "sponsor_id": "intake",
    "arrival_date": "intake",
    "declared_purpose": "intake",
}

# Canonical recoverable fee values (never fuzzy-map onto unknown).
_FEE_FUZZY_TARGETS = ("paid", "waived", "unpaid")
_FEE_STOPWORDS = {
    "status",
    "fee",
    "amount",
    "code",
    "waiver",
    "receipt",
    "payment",
    "the",
    "and",
    "for",
    "n/a",
    "na",
    "null",
    "nil",
    "mib",
    "raceinpr",
    "recept",
    "receip",
}
# Explicit Fee Status label (comma/punct OCR noise allowed between Fee and Status).
_FEE_STATUS_LABEL_RE = re.compile(
    r"(?i)\bFee\s*[,\.]?\s*Status\b"
)
# Status OCR near-miss after Fee (e.g. Stati / Statu / Staius), still label-anchored.
_FEE_STATUS_FUZZY_LABEL_RE = re.compile(
    r"(?i)\bFee\s*[,\.]?\s*([A-Za-z]{4,8})\b"
)
# Bare "Fee" immediately before a short value token (Status dropped: "Fee oaid").
_FEE_BARE_VALUE_RE = re.compile(
    r"(?i)\bFee\b\s*[:\-]?\s*([A-Za-z][A-Za-z']{1,10})\b"
)


@dataclass
class ParsedPacket:
    case_id: str
    fields: dict[str, str] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    conflicts: set[str] = field(default_factory=set)
    injection_heavy: bool = False
    trusted_text_chars: int = 0
    adjudicator_finding: str | None = None
    sources_seen: set[str] = field(default_factory=set)
    observed_flags_seen: bool = False
    field_sources: dict[str, str] = field(default_factory=dict)
    untrusted_conflicts: set[str] = field(default_factory=set)
    # Flags drawn only from Observed-flags / Registry Status lines.
    explicit_risk_flags: list[str] = field(default_factory=list)
    # R39: compact R35 repair evidence flags (no raw tokens / case IDs).
    r35_repair_evidence: dict[str, dict] = field(default_factory=dict)
    # Visible, active-case waiver authorization. Kept separately from the fee
    # value because a parsed ``waived`` status is not itself authorization.
    positive_waiver_seen: bool = False


def strip_injection(text: str) -> tuple[str, bool]:
    """Remove prompt-injection / fake answer-key lines. Return cleaned text + whether packet looked injection-heavy."""
    lines = text.splitlines()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if INJECTION_RE.search(line) or ANSWER_KEY_CSV_RE.search(line):
            dropped += 1
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    total = max(len(lines), 1)
    injection_heavy = dropped / total >= 0.4 or (
        dropped > 0 and len(cleaned.strip()) < 80
    )
    return cleaned, injection_heavy


def _alpha_compact(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _looks_like_adjudicator_header_line(line: str) -> bool:
    """OCR-tolerant Manual Adjudicator Note / stamp header detector."""
    if _ADJUDICATOR_HEADER_RE.search(line or ""):
        return True
    compact = _alpha_compact(line)
    if len(compact) < 8:
        return False
    has_note = any(tok in compact for tok in ("note", "nore", "nate", "biote")) or compact.endswith(
        "not"
    )
    has_adj = any(
        tok in compact
        for tok in (
            "adjud",
            "adjua",
            "adiud",
            "aajua",
            "acjud",
            "dicator",
            "dinct",
            "djud",
            "ajudi",
            "dicet",
            "dioetar",
            "diastio",
        )
    )
    if has_note and has_adj:
        return True
    if has_adj and "manual" in compact:
        return True
    # Extreme garble seen on DEV scans: MariaarRajdinctnot / bfsavaldidiastioetar Nate
    if "dinctnot" in compact or "dicatornot" in compact:
        return True
    if "manual" in compact and has_note and len(compact) >= 14:
        return True
    # R28: Manuva / MANU… / Manual-OCR + Note fragments (conservative).
    if has_note and any(
        tok in compact for tok in ("manuva", "manua", "mrrna", "nalee", "bfsavald")
    ):
        return True
    if compact.startswith("manu") and len(compact) >= 12 and (
        "dic" in compact or has_note or "jud" in compact
    ):
        return True
    return False


def _normalize_finding_decision(raw: str) -> str:
    decision = re.sub(r"[\s-]+", "_", (raw or "").upper()).strip("_")
    if decision.startswith("NEEDS") and "REVIEW" in decision:
        return "NEEDS_REVIEW"
    if decision in {"APPROVED", "DENIED", "NEEDS_REVIEW"}:
        return decision
    return decision


def _active_case_finding_association(
    before: str, after: str, case_id: str, full_line: str = ""
) -> bool:
    """Strong active-case/page association when note chrome is OCR-garbled."""
    if _local_unsafe_finding_context(before, full_line, after):
        return False
    if _SPONSOR_PROSE_RE.search(before[-220:]):
        return False
    near = (before[-420:] if before else "") + (after[:140] if after else "")
    ids = set(CASE_ID_RE.findall(near))
    if case_id not in ids:
        return False
    if "MIB-000000" in ids:
        return False
    foreign = {i for i in ids if i != case_id}
    if foreign:
        return False
    # Prefer note-structure residue, else Packet/Case-ID page chrome for active case.
    if _ADJUDICATOR_REASON_RE.search(after[:200]) or _REASON_BODY_LOOSE_RE.search(
        after[:220]
    ):
        return True
    if re.search(rf"Packet\s+{re.escape(case_id)}\b", before[-450:], re.IGNORECASE):
        return True
    if re.search(
        rf"Case\s*ID\s*:?\s*{re.escape(case_id)}\b", before[-450:], re.IGNORECASE
    ):
        return True
    return False


def _header_case_ids(before: str) -> set[str]:
    """Case IDs attached to adjudicator header lines (not following page footers)."""
    ids: set[str] = set()
    lines = (before or "").splitlines()
    for line in lines[-10:]:
        if _looks_like_adjudicator_header_line(line) or _MIB_EYES_ONLY_RE.search(line):
            ids.update(CASE_ID_RE.findall(line))
        # Native layout: header and "MIB-###### | MIB Eyes Only" on one line.
        if _ADJUDICATOR_HEADER_RE.search(line) or "eyes only" in line.lower():
            ids.update(CASE_ID_RE.findall(line))
    return ids


def _local_unsafe_finding_context(before: str, full_line: str, after: str) -> bool:
    """True when Finding sits inside an untrusted local block (not prior pages)."""
    if _SAMPLE_DENIAL_LINE_RE.search(full_line):
        return True
    if _GENERIC_ADJUDICATION_RE.search(full_line):
        return True
    # If a Manual Adjudicator / Eyes-Only header appears after the last unsafe
    # marker, the Finding belongs to that note — not the archived adjacent page.
    header_spans = [
        m.end()
        for m in re.finditer(
            r"(?:manual\s+adjudicator|adjudicator\s+note|adjudicator\s+stamp|"
            r"manuva\s*:?\s*acjudicator|mib\s+eyes\s+only|"
            r"al\s+adjudicator\s+note)",
            before or "",
            re.IGNORECASE,
        )
    ]
    last_header = max(header_spans) if header_spans else -1
    unsafe_spans = [
        m.start()
        for m in re.finditer(
            r"(?:archived\s+adjacent\s+applicant|adjacent\s+applicant\s*-\s*not\s+active|"
            r"\bMIB-000000\b|barcode\s+payload|force\s+adjudication\s*=)",
            before or "",
            re.IGNORECASE,
        )
    ]
    last_unsafe = max(unsafe_spans) if unsafe_spans else -1
    if last_unsafe >= 0 and last_unsafe > last_header:
        return True
    # Unsafe marker on the finding line or immediately after.
    if re.search(
        r"(?:archived\s+adjacent\s+applicant|\bMIB-000000\b|"
        r"barcode\s+payload|force\s+adjudication\s*=)",
        full_line + (after[:80] if after else ""),
        re.IGNORECASE,
    ):
        return True
    return False


def _trusted_finding_context(
    cleaned: str, match: re.Match[str], case_id: str
) -> bool:
    """True when Finding sits in a trusted adjudicator-note/stamp context."""
    start = match.start()
    end = match.end()
    before = cleaned[max(0, start - 480) : start]
    after = cleaned[end : min(len(cleaned), end + 180)]
    window = before + cleaned[start:end] + after
    finding_line = cleaned[max(0, start) : min(len(cleaned), end + 40)]
    # Expand to full physical line for watermark checks.
    line_start = cleaned.rfind("\n", 0, start) + 1
    line_end = cleaned.find("\n", end)
    if line_end < 0:
        line_end = len(cleaned)
    full_line = cleaned[line_start:line_end]

    # Reject findings that still sit on injection/answer-key residue.
    if INJECTION_RE.search(window) or ANSWER_KEY_CSV_RE.search(window):
        return False
    if _local_unsafe_finding_context(before, full_line, after):
        return False

    header_ids = _header_case_ids(before)
    if header_ids and case_id not in header_ids:
        # Explicit other-case adjudicator header — not the active case.
        return False

    header_hit = bool(_ADJUDICATOR_HEADER_RE.search(before)) or any(
        _looks_like_adjudicator_header_line(line) for line in before.splitlines()[-12:]
    )
    eyes_hit = bool(_MIB_EYES_ONLY_RE.search(before) or _MIB_EYES_ONLY_RE.search(after))
    reason_hit = bool(
        _ADJUDICATOR_REASON_RE.search(after[:180])
        or _REASON_BODY_LOOSE_RE.search(after[:220])
    )

    if header_hit or eyes_hit:
        return True

    # Structural Manual Adjudicator Note body: Finding + official Reason.
    # Accept only when not an in-prose sponsor mention without note structure.
    if reason_hit:
        sponsor_ctx = bool(_SPONSOR_PROSE_RE.search(before[-200:]))
        # Sponsor page immediately above is OK if Reason template is present —
        # the note is usually the next page — unless Finding itself is inside
        # the attestation paragraph (no line break structure).
        if sponsor_ctx and "\n" not in cleaned[max(0, start - 40) : start]:
            return False
        if re.search(r"Find(?:ing|g)", finding_line, re.IGNORECASE):
            return True

    # R28: strong active-case/page association when note chrome is degraded.
    if _active_case_finding_association(before, after, case_id, full_line):
        return True
    return False


def extract_trusted_finding(cleaned: str, case_id: str) -> str | None:
    """Extract trusted visible adjudicator Finding for the active case.

    Requires OCR-tolerant ``Finding`` / ``Findg`` label (colon optional) plus
    adjudicator/manual-note header, MIB Eyes Only stamp chrome, Finding+Reason
    note-body structure, or strong active-case page association. Rejects
    generic ``Adjudication:``, archived/adjacent / MIB-000000, barcode force,
    sponsor prose, sample-denial watermark, and injection residue. When multiple
    trusted findings exist, the latest non-crossed-out note wins.
    """
    if not cleaned:
        return None
    trusted: list[tuple[int, str, bool]] = []
    for match in ADJUDICATOR_FINDING_RE.finditer(cleaned):
        if not _trusted_finding_context(cleaned, match, case_id):
            continue
        decision = _normalize_finding_decision(match.group(1))
        if decision not in {"APPROVED", "DENIED", "NEEDS_REVIEW"}:
            continue
        around = cleaned[max(0, match.start() - 120) : match.end() + 120]
        crossed = bool(_CROSS_OUT_RE.search(around))
        trusted.append((match.start(), decision, crossed))
    if not trusted:
        return None
    # Prefer latest non-crossed-out finding; else latest overall.
    live = [t for t in trusted if not t[2]]
    chosen = (live or trusted)[-1]
    return chosen[1]


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


_FEE_PUNCT = ".,;:|[](){}\"'`~_-/\\“”‘’´'"


def _normalize_fee_glyph(token: str) -> str:
    """Normalize common OCR glyph confusions for fee-status tokens."""
    t = token.casefold().strip().strip(_FEE_PUNCT)
    t = t.replace("0", "o").replace("1", "l").replace("|", "l").replace("$", "s")
    t = t.replace("rn", "m")  # warved already 1-edit; keep mild rn→m
    return t


def _fee_token_compatible(raw: str, cand: str) -> bool:
    """Reject false friends that are 1-edit from a fee value (e.g. Waid→paid)."""
    if cand == "unpaid":
        return raw.startswith("u")
    if cand == "waived":
        # Require waived shape: leading w plus 'v' or known OCR forms.
        # Rejects Home-World garble "waid" (would be dist-2 → waived).
        return raw.startswith("w") and (
            "v" in raw or raw in {"waved", "waied", "walved", "wvived"}
        )
    if cand == "paid":
        # Allow p* near-misses and Status-dropped OCR forms oaid/naid.
        # Reject Home-World garble "waid" and other *aid false friends.
        if raw.startswith("p"):
            return True
        if raw in {"oaid", "naid"}:
            return True
        if raw.endswith("aid") and raw[:1] in {"o", "n", "p"}:
            return True
        return False
    return False


def fuzzy_match_fee_token(token: str) -> str | None:
    """Map one OCR token onto paid/waived/unpaid. Never returns unknown.

    Conservative edit-distance: typically <=1; distance 2 only for longer
    targets with a unique margin of >=2 vs the runner-up. Rejects ties,
    stopwords, and unknown near-misses (anknown / sanknown / unknown).
    """
    raw = _normalize_fee_glyph(token)
    if not raw or raw in _FEE_STOPWORDS or "obscur" in raw:
        return None
    if raw in {"unknown", "unkown", "unknwn"}:
        return None

    # Prefer explicit unknown family over paid/waived/unpaid near-misses.
    unk_dist = _levenshtein(raw, "unknown")
    if unk_dist <= 2:
        best_fee = min(_levenshtein(raw, c) for c in _FEE_FUZZY_TARGETS)
        if unk_dist <= best_fee:
            return None

    if raw in _FEE_FUZZY_TARGETS:
        return raw

    ranked = sorted(
        (_levenshtein(raw, c), c)
        for c in _FEE_FUZZY_TARGETS
        if _fee_token_compatible(raw, c)
    )
    if not ranked:
        return None
    # Reject ties / ambiguity at the best distance.
    best_d, best = ranked[0]
    second_d = ranked[1][0] if len(ranked) > 1 else 99
    if second_d == best_d:
        return None
    if best_d <= 1:
        return best
    # Dist 2 only for longer tokens with a strong unique margin.
    if best_d == 2 and len(best) >= 5 and (second_d - best_d) >= 2:
        return best
    return None


def _fee_status_label_end(text: str, start: int = 0) -> list[int]:
    """Return end offsets of fuzzy-safe Fee Status labels (not Waiver Code / Home)."""
    ends: list[int] = []
    for match in _FEE_STATUS_LABEL_RE.finditer(text, start):
        ends.append(match.end())
    for match in _FEE_STATUS_FUZZY_LABEL_RE.finditer(text, start):
        word = match.group(1).casefold()
        if word == "status":
            continue  # already covered by exact label
        if _levenshtein(word, "status") <= 2:
            ends.append(match.end())
    return ends


def _immediate_fee_token(text: str, label_end: int) -> str | None:
    """Extract only the immediate value token after a Fee Status label."""
    tail = text[label_end : label_end + 48]
    if re.search(r"(?i)obscur", tail[:24]):
        return None
    # Same-line preference: stop at newline before scanning far.
    same_line = tail.split("\n", 1)[0]
    # Skip OCR chrome / curly quotes before the value token.
    skip = r"[\s:.\-_\'\"\`\|\]\[\,“”‘’´]+"
    match = re.match(
        skip + r"([A-Za-z][A-Za-z']{1,10})",
        same_line,
    )
    if not match:
        # Nearest local line: allow one short wrap.
        match = re.match(
            r"[\s:.\-_\'\"\`\|\]\[\,“”‘’´\n]{0,12}([A-Za-z][A-Za-z']{1,10})",
            tail,
        )
    if not match:
        return None
    return match.group(1)


def extract_fuzzy_fee_status(text: str) -> str | None:
    """Label-anchored fuzzy fee decode from one cleaned text stream.

    Requires a Fee Status label (exact, fuzzy Status, or bare Fee + immediate
    fee-like token). Never scans arbitrary document words; never uses Waiver
    Code / Home Waid; never maps unknown near-misses onto paid/waived/unpaid.
    """
    if not text:
        return None
    hits: list[str] = []

    for label_end in _fee_status_label_end(text):
        tok = _immediate_fee_token(text, label_end)
        if not tok:
            continue
        value = fuzzy_match_fee_token(tok)
        if value:
            hits.append(value)

    # Bare "Fee <token>" only when Status was dropped and token is fee-like.
    if not hits:
        for match in _FEE_BARE_VALUE_RE.finditer(text):
            tok = match.group(1)
            # Refuse non-status fee phrases.
            if tok.casefold() in _FEE_STOPWORDS or tok.casefold() in {
                "receipt",
                "amount",
                "code",
                "waiver",
                "status",
            }:
                continue
            # Must not be "Fee" inside "Waiver Code" / home-world lines.
            line_start = text.rfind("\n", 0, match.start()) + 1
            line = text[line_start : match.start()]
            if re.search(r"(?i)\b(waiver|home|world|visa|sponsor|registry)\b", line):
                continue
            value = fuzzy_match_fee_token(tok)
            if value:
                hits.append(value)

    if not hits:
        return None
    # Reject cross-hit ambiguity in the same stream.
    uniq = sorted(set(hits))
    if len(uniq) != 1:
        return None
    return uniq[0]


def _fee_is_missing_or_unknown(fields: dict[str, str]) -> bool:
    fee = fields.get("fee_status")
    return fee is None or fee == "unknown"


def _field_missing_or_unknown(fields: dict[str, str], key: str) -> bool:
    value = fields.get(key)
    return value is None or value == "" or value == "unknown"


def _clean_value(raw: str) -> str:
    value = " ".join(raw.split())
    # Strip trailing page chrome that sometimes glues to the value.
    value = re.split(r"\s{2,}(?:PASSPORT|REGISTRY|PACKET)\b", value, maxsplit=1)[0]
    # Single-space PASSPORT IMAGE glue common in OCR home-world lines.
    value = re.split(r"\s+PASSPORT\s+IMAGE\b", value, maxsplit=1, flags=re.I)[0]
    value = value.strip(" |\"'`“”~")
    return value.strip()


def _normalize_home_ocr(text: str) -> str:
    """Case/whitespace + unambiguous glyph fixes for home-world matching."""
    t = (text or "").replace("¢", "c").replace("©", "c")
    t = re.sub(r"\s+", " ", t)
    return t


def extract_allowlisted_home_world(raw: str) -> str | None:
    """If raw contains exactly one HOME_WORLDS value (OCR-normalized), return it.

    Used before retaining free-form OCR home-world text so suffix/chrome noise
    (`Luyten-b te`, `Wolf-1061¢ PASSPORT IMAGE`) collapses to the canonical
    allowlist entry. Ambiguous multi-hits return None.
    """
    if not raw:
        return None
    norm = _normalize_home_ocr(raw).casefold()
    hits = [
        world
        for world in HOME_WORLDS
        if _normalize_home_ocr(world).casefold() in norm
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def _is_exact_allowlisted_home(value: str | None) -> bool:
    if not value:
        return False
    return any(
        _normalize_home_ocr(value).casefold() == _normalize_home_ocr(w).casefold()
        for w in HOME_WORLDS
    )


def is_valid_calendar_date(value: str) -> bool:
    """True iff value is a real Gregorian calendar day (YYYY-MM-DD).

    Regex-shaped strings like 2025-07-36 / 2026-07-92 must fail here.
    """
    if not value or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _date_in_plausible_range(value: str) -> bool:
    """Accept only real calendar dates inside the challenge arrival window."""
    if not is_valid_calendar_date(value):
        return False
    return _PLAUSIBLE_DATE_MIN <= value <= _PLAUSIBLE_DATE_MAX


def _sanitize_arrival_date(packet: ParsedPacket) -> None:
    """Drop any shipped arrival_date that is not a real in-window calendar day."""
    value = packet.fields.get("arrival_date")
    if value is None:
        return
    if _date_in_plausible_range(value):
        return
    del packet.fields["arrival_date"]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and not is_valid_calendar_date(value):
        packet.conflicts.add("arrival_date_unreadable")


def _line_is_footer_or_id(line: str) -> bool:
    return bool(_FOOTER_OR_ID_LINE_RE.search(line))


def extract_plausible_iso_dates(text: str) -> list[str]:
    """Real calendar ISO dates in plausible range, excluding footer/injection."""
    if not text:
        return []
    hits: list[str] = []
    for match in DATE_ANYWHERE_RE.finditer(text):
        value = match.group(1)
        if not _date_in_plausible_range(value):
            continue
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start : line_end if line_end != -1 else len(text)]
        if _line_is_footer_or_id(line):
            continue
        if INJECTION_RE.search(line) or ANSWER_KEY_CSV_RE.search(line):
            continue
        hits.append(value)
    return hits


def extract_label_proximate_date(
    text: str,
    *,
    allow_fuzzy: bool = False,
) -> str | None:
    """Unique plausible ISO date sitting near an Arrival label.

    Exact `Arrival Date` windows are always considered. Fuzzy OCR labels
    (Anwval/Amval/…) are opt-in and intended only when paired with
    independent page+embedded agreement at fusion time.
    """
    labels = [_NEAR_LABELS["arrival_date"]]
    if allow_fuzzy:
        labels.append(_NEAR_LABELS["arrival_date_fuzzy"])
    hits: list[str] = []
    for label_re in labels:
        for window in _window_after_label(text, label_re, radius=48):
            hits.extend(extract_plausible_iso_dates(window))
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    return None


def extract_label_proximate_future_year_repair(text: str) -> str | None:
    """Repair the common 2026-06→2028-08 OCR glyph beside Arrival Date.

    The challenge receipt date makes year 2028 impossible.  Keep month/day
    untouched and require a unique real 2026 calendar date.
    """
    hits: list[str] = []
    for label_re in (
        _NEAR_LABELS["arrival_date"],
        _NEAR_LABELS["arrival_date_fuzzy"],
    ):
        for window in _window_after_label(text, label_re, radius=48):
            for raw in re.findall(r"\b2028-(\d{2})-(\d{2})\b", window):
                month = "06" if raw[0] == "08" else raw[0]
                value = f"2026-{month}-{raw[1]}"
                if _date_in_plausible_range(value):
                    hits.append(value)

    # Some scans damage most of the first word (``Astwal Date`` /
    # ``Asbeal Date``), making a finite spelling list brittle.  Classify the
    # label shape instead: it must be a two-word *Date label whose normalized
    # similarity to Arrival Date is high.  Independent-stream agreement is
    # still required by the caller before the repair is accepted.
    for line in text.splitlines():
        date_match = re.search(r"\b2028-(\d{2})-(\d{2})\b", line)
        if not date_match:
            continue
        prefix = line[: date_match.start()]
        for label_match in re.finditer(
            r"([A-Za-z]{3,12})\s+(Date|Cate|Dabe)\s*:?\s*$", prefix, re.I
        ):
            normalized = re.sub(r"[^a-z]", "", label_match.group(0).casefold())
            similarity = SequenceMatcher(None, normalized, "arrivaldate").ratio()
            if similarity < 0.62:
                continue
            month = "06" if date_match.group(1) == "08" else date_match.group(1)
            value = f"2026-{month}-{date_match.group(2)}"
            if _date_in_plausible_range(value):
                hits.append(value)
    unique = list(dict.fromkeys(hits))
    return unique[0] if len(unique) == 1 else None


def extract_label_proximate_visas(text: str) -> list[str]:
    """Allowlisted visas in Visa-Class label windows (punct-normalized)."""
    if not text:
        return []
    hits: list[str] = []
    for window in _window_after_label(text, _NEAR_LABELS["visa_class"], radius=40):
        hits.extend(extract_allowlisted_visas(window, punct_norm=True))
    return list(dict.fromkeys(hits))


def extract_label_proximate_species(text: str) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for window in _window_after_label(text, _NEAR_LABELS["species_code"], radius=48):
        hits.extend(extract_allowlisted_species(window))
    return list(dict.fromkeys(hits))


def extract_unique_plausible_date(text: str) -> str | None:
    """Exactly one plausible non-footer ISO date in the whole source text."""
    uniq = list(dict.fromkeys(extract_plausible_iso_dates(text)))
    if len(uniq) == 1:
        return uniq[0]
    return None


def _normalize_visa_ocr_text(text: str) -> str:
    """Map common OCR punct variants onto canonical visa tokens (XW.2 → XW-2)."""
    t = (text or "").replace("—", "-").replace("–", "-").replace("−", "-")
    # Trailing OCR glue (_ / .) must not block the digit end-boundary.
    t = re.sub(
        r"\b(XW|DIP|MED|TRANSIT)[.\s_](\d)(?![A-Za-z0-9])",
        r"\1-\2",
        t,
        flags=re.I,
    )
    return t


def extract_allowlisted_visas(text: str, *, punct_norm: bool = False) -> list[str]:
    """Return allowlisted visa tokens found in text.

    `punct_norm=False` preserves R9 exact-token behavior (avoids XW.2/DIP.1
    false friends from garbled Visa lines). `punct_norm=True` is for
    label-proximate windows only.
    """
    if not text:
        return []
    hay = _normalize_visa_ocr_text(text) if punct_norm else text
    hits: list[str] = []
    for visa in sorted(VISA_CLASSES):
        if punct_norm:
            if re.search(rf"\b{re.escape(visa)}(?![A-Za-z0-9])", hay):
                hits.append(visa)
        elif re.search(rf"\b{re.escape(visa)}\b", hay):
            hits.append(visa)
    return hits


def extract_allowlisted_species(text: str) -> list[str]:
    if not text:
        return []
    return [s for s in sorted(SPECIES_CODES) if re.search(rf"\b{re.escape(s)}\b", text)]


def extract_label_proximate_purpose(text: str) -> str | None:
    """Exact allowlisted purpose immediately after a Declared-Purpose-like label."""
    label_re = _NEAR_LABELS.get("declared_purpose")
    if not label_re or not text:
        return None
    hits: list[str] = []
    for window in _window_after_label(text, label_re, radius=48):
        lowered = window.casefold()
        local = [p for p in PURPOSES if p in lowered]
        if len(local) == 1:
            hits.append(local[0])
        elif len(local) > 1:
            # Ambiguous purpose window — reject this source path.
            return None
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    return None


def extract_purpose_candidates(text: str) -> list[str]:
    if not text:
        return []
    lowered = text.casefold()
    return [p for p in sorted(PURPOSES) if p in lowered]


def _collect_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    values: list[str] = []
    for match in pattern.finditer(text):
        value = _clean_value(match.group(1))
        if value:
            values.append(value)
    return values


def _pick_consistent(values: list[str], normalize=None) -> tuple[str | None, bool]:
    if not values:
        return None, False
    if normalize:
        keyed: dict[str, str] = {}
        for value in values:
            keyed.setdefault(normalize(value), value)
        uniq = list(keyed.values())
    else:
        uniq = list(dict.fromkeys(values))
    if len(uniq) == 1:
        return uniq[0], False
    return uniq[0], True


def _window_after_label(text: str, label_re: re.Pattern[str], radius: int = 80) -> list[str]:
    """Return short text windows immediately after each label hit."""
    windows: list[str] = []
    for match in label_re.finditer(text):
        chunk = text[match.end() : match.end() + radius]
        windows.append(chunk)
    return windows


def _nearest_allowlist(
    text: str,
    field: str,
    allowlist: set[str],
    *,
    normalize=None,
) -> list[str]:
    """Collect allowlisted tokens that sit near a field label."""
    label_re = _NEAR_LABELS.get(field)
    if not label_re:
        return []
    hits: list[str] = []
    for window in _window_after_label(text, label_re):
        lowered = window.casefold()
        for item in allowlist:
            key = normalize(item) if normalize else item
            needle = key.casefold() if isinstance(key, str) else str(key)
            if field == "fee_status":
                if re.search(rf"\b{re.escape(item)}\b", lowered):
                    hits.append(item)
            elif field == "arrival_date":
                for value in DATE_ANYWHERE_RE.findall(window):
                    if _date_in_plausible_range(value):
                        hits.append(value)
            elif field == "visa_class" or field == "species_code" or field == "sponsor_id":
                if re.search(rf"\b{re.escape(item)}\b", window if field != "fee_status" else lowered):
                    hits.append(item)
            else:
                if needle in lowered:
                    hits.append(item)
    return hits


_RISK_FLAG_MENTION_RE = re.compile(
    r"(?:disqualifying|review-only|raview-only)?\s*"
    r"(?:risk|rak|rik|nsk|fisk|isk|tisk)\s+"
    r"(?:flags?|fleg|fags?)\s*"  # flags / fleg OCR
    r"(?:present\s*)?:?\s*"
    r"([a-z_]+(?:\s+[a-z_]+)?)",
    re.IGNORECASE,
)

# OCR can destroy the word before ``flags:`` while preserving the delimiter
# and exact allowlisted payload. Anchor to one short line and still require an
# allowlisted atom; this is not a whole-document bare-token fallback.
_FLAG_PAYLOAD_LINE_RE = re.compile(
    r"^[^\n:]{0,32}\b(?:flags?|flogs?|fleg|fags?|fiegs?|flaas?)\s*:\s*([^\n]+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _normalize_flag_atom(raw: str) -> str | None:
    atom = raw.strip().lower().replace(" ", "_").replace("-", "_")
    atom = re.sub(r"_+", "_", atom).strip("_")
    if atom in RISK_FLAG_ATOMS:
        return atom
    return None


def _atoms_in_payload(payload: str) -> list[str]:
    """Extract allowlisted flag atoms from an Observed-flags payload."""
    found: list[str] = []
    low = payload.lower()
    for atom in RISK_FLAG_ATOMS:
        if re.search(rf"\b{re.escape(atom)}\b", low):
            found.append(atom)
            continue
        spaced = atom.replace("_", " ")
        if re.search(rf"\b{re.escape(spaced)}\b", low):
            found.append(atom)
    # Also split on commas/pipes for near-exact tokens.
    for part in re.split(r"[|,]", payload):
        atom = _normalize_flag_atom(part)
        if atom and atom not in found:
            found.append(atom)
            continue
        # Closed-vocabulary OCR repair is safe only inside an already explicit
        # flags payload.  Compare compact glyph strings and require a unique
        # winner with a clear margin; bare whole-document tokens never enter
        # this path.
        compact = re.sub(r"[^a-z]", "", part.casefold())
        if len(compact) < 8:
            continue
        ranked = sorted(
            (_levenshtein(compact, re.sub(r"[^a-z]", "", cand)), cand)
            for cand in RISK_FLAG_ATOMS
        )
        best_dist, best_atom = ranked[0]
        runner_dist = ranked[1][0]
        limit = max(1, round(len(best_atom.replace("_", "")) * 0.30))
        if best_dist <= limit and runner_dist - best_dist >= 2 and best_atom not in found:
            found.append(best_atom)
    return found


def _parse_explicit_risk_flags(text: str) -> tuple[list[str], bool]:
    """Flags from Observed-flags / Registry Status / explicit risk-flag mentions.

    `observed_flags_seen` is True only when an Observed-flags line exists.
    Registry Status may contribute embargo flags without flipping that bit.
    Does not scan bare flag tokens outside those contexts.
    """
    flags: set[str] = set()
    seen = False
    for match in OBSERVED_FLAGS_RE.finditer(text):
        seen = True
        payload = match.group(1).strip().lower()
        if payload in {"", "none", "n/a", "null"}:
            continue
        flags.update(_atoms_in_payload(payload))
    for match in REGISTRY_STATUS_RE.finditer(text):
        status = match.group(1).upper()
        if "EMBARGO" in status:
            flags.add("planetary_embargo")
    for match in _RISK_FLAG_MENTION_RE.finditer(text):
        flags.update(_atoms_in_payload(match.group(1)))
    for match in _FLAG_PAYLOAD_LINE_RE.finditer(text):
        flags.update(_atoms_in_payload(match.group(1)))
    # Strong structural residue: after Species Match inside a B-13 block, a
    # surviving ``...brics`` token is the tail of illegible_biometrics, not the
    # Biometric Scan Slip heading (which occurs before Species Match).
    for match in re.finditer(
        r"(?is)FORM\s+B-13.{0,500}?Sp\w{3,7}\s+Match[^\n]*\n(.{0,120}?)"
        r"(?:Packet\s+MIB|Synthetic\s+hiring)",
        text,
    ):
        residual = match.group(1)
        if re.search(r"(?i)\b[a-z]{0,8}bric[a-z]{0,5}\b", residual):
            flags.add("illegible_biometrics")
    # A visibly present but unreadable B-13 risk panel is itself the documented
    # review-only ``illegible_biometrics`` condition.
    if re.search(
        r"(?im)(?:observ\w{0,5}\s+)?fl(?:ag|eg|ieg)s?\s*:\s*\[?\s*"
        r"risk\s+panel\s+(?:illegible|unreadable|obscured)\b",
        text,
    ):
        flags.add("illegible_biometrics")
        seen = True
    # Official adjudicator-note prose used when the atom label is absent.
    if re.search(
        r"(?i)(?:prior\s+denial\s+stamp\s+rescinded|"
        r"denial\s+stamp\s+rescinded|rescinded\s+(?:prior\s+)?denial)",
        text,
    ):
        flags.add("rescinded_denial")
    # Extremely noisy adjudicator reasons can retain only approximate risk and
    # flag anchors. Within that explicit context, use a unique closed-vocab
    # fuzzy winner over the payload tail.
    for match in re.finditer(
        r"(?im)^\s*(?:Reason|Renson|Reosor|Resnen)\s*:\s*[^\n]{0,50}?"
        r"(?:risk|riak|rak|rk|isk|dak)\W{0,5}"
        r"(?:flags?|fap|flap|fag|sep)\W{0,5}([^\n.]{5,40})",
        text,
    ):
        payload = re.sub(r"[^a-z]", "", match.group(1).casefold())
        if len(payload) < 5:
            continue
        ratios = sorted(
            (
                SequenceMatcher(
                    None, payload, re.sub(r"[^a-z]", "", atom)
                ).ratio(),
                atom,
            )
            for atom in RISK_FLAG_ATOMS
        )
        best_ratio, best_atom = ratios[-1]
        runner_ratio = ratios[-2][0]
        if best_ratio >= 0.54 and best_ratio - runner_ratio >= 0.15:
            flags.add(best_atom)
    return sorted(flags), seen


def _apply_manual_corrections(cleaned: str, packet: ParsedPacket) -> None:
    """Native packet annotations override labeled form values."""
    for match in MANUAL_CORRECTION_RE.finditer(cleaned):
        kind = match.group(1).casefold()
        raw = _clean_value(match.group(2))
        if not raw:
            continue
        if kind == "fee status":
            value = raw.lower()
            if value in FEE_STATUSES:
                packet.fields["fee_status"] = value
                packet.conflicts.discard("fee_status")
                packet.sources_seen.add("fee")
        elif kind == "visa class":
            value = raw.upper()
            if value in VISA_CLASSES:
                packet.fields["visa_class"] = value
                packet.conflicts.discard("visa_class")
        elif kind == "sponsor":
            value = raw.upper()
            if SPONSOR_ANYWHERE_RE.fullmatch(value):
                packet.fields["sponsor_id"] = value
                packet.conflicts.discard("sponsor_id")
        elif kind == "applicant":
            if not INJECTION_RE.search(raw) and len(raw.split()) <= 6:
                packet.fields["applicant_name"] = raw
                packet.conflicts.discard("applicant_name")
        elif kind.startswith("species"):
            value = raw.upper()
            if value in SPECIES_CODES:
                packet.fields["species_code"] = value
                packet.conflicts.discard("species_code")
        elif kind == "home world":
            hit = next((w for w in HOME_WORLDS if w.casefold() == raw.casefold()), raw)
            if hit in HOME_WORLDS:
                packet.fields["home_world"] = hit
                packet.conflicts.discard("home_world")
        elif kind == "arrival date":
            if _date_in_plausible_range(raw):
                packet.fields["arrival_date"] = raw
                packet.conflicts.discard("arrival_date")
                packet.conflicts.discard("arrival_date_unreadable")
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                # ISO-shaped but impossible or out-of-window → never ship.
                packet.fields.pop("arrival_date", None)
                packet.conflicts.add("arrival_date_unreadable")
        elif kind in {"declared purpose", "purpose"}:
            lowered = raw.lower()
            hit = next((p for p in PURPOSES if p == lowered or p in lowered), None)
            if hit:
                packet.fields["declared_purpose"] = hit
                packet.conflicts.discard("declared_purpose")


def _parse_single(case_id: str, raw_text: str) -> ParsedPacket:
    cleaned, injection_heavy = strip_injection(raw_text)
    packet = ParsedPacket(
        case_id=case_id,
        injection_heavy=injection_heavy,
        trusted_text_chars=len(cleaned.strip()),
    )

    lower = cleaned.casefold()
    if "form i-8090" in lower:
        packet.sources_seen.add("intake")
    if "planetary registry extract" in lower:
        packet.sources_seen.add("registry")
    if "fee receipt" in lower:
        packet.sources_seen.add("fee")
    if "biometric" in lower:
        packet.sources_seen.add("biometric")
    if (
        "adjudicator note" in lower
        or "adjudicator stamp" in lower
        or "acjudicator note" in lower
        or "manuva" in lower
    ):
        packet.sources_seen.add("adjudicator")

    # FIELD_MANUAL #1: only trusted visible adjudicator-note/stamp findings.
    finding = extract_trusted_finding(cleaned, case_id)
    if finding:
        packet.adjudicator_finding = finding
        packet.sources_seen.add("adjudicator")

    # Species / visa / fee via allowlists when possible.
    species_hits = [
        v.upper()
        for v in _collect_matches(LABEL_PATTERNS["species_code"], cleaned)
        if v.upper() in SPECIES_CODES
    ]
    if not species_hits:
        species_hits = _nearest_allowlist(cleaned, "species_code", SPECIES_CODES)
    species, conflict = _pick_consistent(species_hits)
    if species:
        packet.fields["species_code"] = species
    if conflict:
        packet.conflicts.add("species_code")

    visa_hits = []
    for raw in _collect_matches(LABEL_PATTERNS["visa_class"], cleaned):
        cand = raw.upper()
        if cand in VISA_CLASSES:
            visa_hits.append(cand)
        else:
            normed = _normalize_visa_ocr_text(raw).upper()
            if normed in VISA_CLASSES:
                visa_hits.append(normed)
    if not visa_hits:
        # Label window may contain XW.2 / MED.3 OCR punct variants.
        visa_hits.extend(extract_label_proximate_visas(cleaned))
    if not visa_hits:
        visa_hits = _nearest_allowlist(cleaned, "visa_class", VISA_CLASSES)
    visa, conflict = _pick_consistent(visa_hits)
    if visa:
        packet.fields["visa_class"] = visa
    if conflict:
        packet.conflicts.add("visa_class")

    fee_hits = [
        v.lower()
        for v in _collect_matches(LABEL_PATTERNS["fee_status"], cleaned)
        if v.lower() in FEE_STATUSES
    ]
    if not fee_hits:
        fee_hits = _nearest_allowlist(cleaned, "fee_status", FEE_STATUSES)
    fee, conflict = _pick_consistent(fee_hits)
    if fee:
        packet.fields["fee_status"] = fee
    if conflict:
        packet.conflicts.add("fee_status")
    if "fee_status" not in packet.fields:
        # OCR / layout fallbacks
        m = re.search(
            r"Fee\s*Status\s*[:\s]+(paid|waived|unpaid|unknown)",
            cleaned,
            re.IGNORECASE,
        )
        if m:
            packet.fields["fee_status"] = m.group(1).lower()
            packet.sources_seen.add("fee")
        elif re.search(r"\bAmount\s*\$?\s*\d", cleaned, re.IGNORECASE) or (
            "fee receipt" in cleaned.casefold()
        ):
            if re.search(r"\bunpaid\b", cleaned, re.IGNORECASE):
                packet.fields["fee_status"] = "unpaid"
                packet.sources_seen.add("fee")
            elif re.search(r"\bwaived\b", cleaned, re.IGNORECASE):
                packet.fields["fee_status"] = "waived"
                packet.sources_seen.add("fee")
            elif re.search(r"\bpaid\b", cleaned, re.IGNORECASE):
                packet.fields["fee_status"] = "paid"
                packet.sources_seen.add("fee")

    name_hits = _collect_matches(LABEL_PATTERNS["applicant_name"], cleaned)
    name_hits = [n for n in name_hits if not INJECTION_RE.search(n)]
    # R43: drop field-label leaks ("Home World: …") that OCR associates with
    # a bare/next-line Applicant label. Keep [NAME CUT OUT] placeholders.
    from .name_resolve import _FIELD_LEAK_RE  # local import avoids cycles

    name_hits = [n for n in name_hits if not _FIELD_LEAK_RE.match((n or "").strip())]
    name, conflict = _pick_consistent(name_hits, normalize=lambda s: s.casefold())
    if name:
        packet.fields["applicant_name"] = name
    if conflict:
        packet.conflicts.add("applicant_name")

    world_hits = []
    for raw in _collect_matches(LABEL_PATTERNS["home_world"], cleaned):
        # Prefer exact allowlist hit; else collapse OCR noise containing one
        # canonical world (¢→c, PASSPORT IMAGE glue); else keep short free-form.
        match = next(
            (w for w in HOME_WORLDS if w.casefold() == raw.casefold()),
            None,
        )
        if match is None:
            match = extract_allowlisted_home_world(raw)
        if match is None:
            match = raw
        if match in HOME_WORLDS or len(match.split()) <= 4:
            world_hits.append(match)
    world, conflict = _pick_consistent(world_hits, normalize=lambda s: s.casefold())
    if world:
        # Final containment pass on free-form OCR residue.
        contained = extract_allowlisted_home_world(world)
        if contained and not _is_exact_allowlisted_home(world):
            world = contained
        packet.fields["home_world"] = world
    if conflict:
        packet.conflicts.add("home_world")

    sponsor_hits = _collect_matches(LABEL_PATTERNS["sponsor_id"], cleaned)
    if not sponsor_hits:
        for window in _window_after_label(cleaned, _NEAR_LABELS["sponsor_id"]):
            sponsor_hits.extend(SPONSOR_ANYWHERE_RE.findall(window))
    if not sponsor_hits:
        sponsor_hits = SPONSOR_ANYWHERE_RE.findall(cleaned)
    # Sponsor-only majority: repeated OCR of the true SPN beats a single flip.
    if sponsor_hits:
        from collections import Counter

        counts = Counter(s.upper() for s in sponsor_hits)
        best, n = counts.most_common(1)[0]
        if n > len(sponsor_hits) - n:
            sponsor, conflict = best, False
        else:
            sponsor, conflict = _pick_consistent([s.upper() for s in sponsor_hits])
    else:
        sponsor, conflict = None, False
    if sponsor:
        packet.fields["sponsor_id"] = sponsor.upper()
    if conflict:
        packet.conflicts.add("sponsor_id")

    date_hits = []
    unreadable_date = False
    for raw in _collect_matches(LABEL_PATTERNS["arrival_date"], cleaned):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            if _date_in_plausible_range(raw):
                date_hits.append(raw)
            elif not is_valid_calendar_date(raw):
                # Impossible calendar day (e.g. 2025-07-36) → unreadable.
                unreadable_date = True
            # else: real calendar but outside challenge window — reject silently.
        else:
            unreadable_date = True
    if not date_hits:
        for window in _window_after_label(cleaned, _NEAR_LABELS["arrival_date"]):
            date_hits.extend(extract_plausible_iso_dates(window))
    if not date_hits:
        # Fuzzy Arrival label + unique proximate plausible date (R44: enable
        # OCR-garbled Arrival stems on source-local parse).
        prox = extract_label_proximate_date(cleaned, allow_fuzzy=True)
        if prox:
            date_hits.append(prox)
    chosen_date, conflict = _pick_consistent(date_hits)
    if chosen_date and _date_in_plausible_range(chosen_date):
        packet.fields["arrival_date"] = chosen_date
    elif unreadable_date:
        packet.fields.pop("arrival_date", None)
        packet.conflicts.add("arrival_date_unreadable")
    if conflict:
        packet.conflicts.add("arrival_date")

    purpose_hits = []
    for raw in _collect_matches(LABEL_PATTERNS["declared_purpose"], cleaned):
        lowered = raw.lower()
        hit = next((p for p in PURPOSES if p == lowered or p in lowered), None)
        if hit:
            purpose_hits.append(hit)
    if not purpose_hits:
        prox_purpose = extract_label_proximate_purpose(cleaned)
        if prox_purpose:
            purpose_hits.append(prox_purpose)
    purpose, conflict = _pick_consistent(purpose_hits)
    if purpose:
        packet.fields["declared_purpose"] = purpose
    if conflict:
        packet.conflicts.add("declared_purpose")

    explicit_flags, explicit_seen = _parse_explicit_risk_flags(cleaned)
    packet.explicit_risk_flags = explicit_flags
    if explicit_seen:
        packet.observed_flags_seen = True

    flags: set[str] = set(explicit_flags)
    # Free-text fallback: underscore OR space forms as whole tokens.
    # Native stream keeps these on risk_flags; OCR contribution still requires
    # explicit_risk_flags (Observed / mention paths above).
    lowered = cleaned.lower()
    for atom in RISK_FLAG_ATOMS:
        spaced = atom.replace("_", " ")
        if re.search(rf"\b{re.escape(atom)}\b", lowered) or re.search(
            rf"\b{re.escape(spaced)}\b", lowered
        ):
            flags.add(atom)
            if re.search(r"(?:observed|dbserved|obverved)\s+flags?", lowered):
                packet.observed_flags_seen = True

    packet.risk_flags = sorted(flags)

    # Bare-token fallbacks when labeled parse missed (common after OCR).
    # Keep R9 whole-document unique exact tokens; punct-normalized visas are
    # additive via extract_allowlisted_visas.
    lowered = cleaned.lower()
    if "visa_class" not in packet.fields:
        hits = extract_allowlisted_visas(cleaned)
        if len(hits) == 1:
            packet.fields["visa_class"] = hits[0]
    if "species_code" not in packet.fields:
        hits = extract_allowlisted_species(cleaned)
        if len(hits) == 1:
            packet.fields["species_code"] = hits[0]
    if "home_world" not in packet.fields:
        # Prefer OCR-normalized containment over brittle casefold substring.
        hits = [
            w
            for w in HOME_WORLDS
            if _normalize_home_ocr(w).casefold() in _normalize_home_ocr(cleaned).casefold()
        ]
        if len(hits) == 1:
            packet.fields["home_world"] = hits[0]
    elif not _is_exact_allowlisted_home(packet.fields.get("home_world")):
        # Correct OCR-derived free-form residue; never runs after manual
        # correction below for exact allowlist values.
        contained = extract_allowlisted_home_world(packet.fields["home_world"])
        if contained:
            packet.fields["home_world"] = contained
    if "declared_purpose" not in packet.fields:
        prox_purpose = extract_label_proximate_purpose(cleaned)
        if prox_purpose:
            packet.fields["declared_purpose"] = prox_purpose
        else:
            hits = extract_purpose_candidates(cleaned)
            if len(hits) == 1:
                packet.fields["declared_purpose"] = hits[0]
    if "arrival_date" not in packet.fields:
        # Source-local: unique plausible date only with label proximity.
        prox = extract_label_proximate_date(cleaned, allow_fuzzy=True)
        if prox:
            packet.fields["arrival_date"] = prox
            packet.conflicts.discard("arrival_date_unreadable")
    if "fee_status" not in packet.fields:
        # Prefer explicit unpaid over paid substring.
        if re.search(r"\bunpaid\b", lowered):
            packet.fields["fee_status"] = "unpaid"
            packet.sources_seen.add("fee")
        elif re.search(r"\bwaived\b", lowered):
            packet.fields["fee_status"] = "waived"
            packet.sources_seen.add("fee")
        elif re.search(r"\bpaid\b", lowered) and (
            "fee" in lowered or "amount" in lowered or "waiver" in lowered
        ):
            packet.fields["fee_status"] = "paid"
            packet.sources_seen.add("fee")

    # Authoritative native annotations win over form OCR/labels.
    _apply_manual_corrections(cleaned, packet)

    # If Case ID in text conflicts with filename, treat as suspicious.
    ids = set(CASE_ID_RE.findall(cleaned))
    if ids and case_id not in ids and len(ids) >= 1:
        # Common when injection leaked another case id; already stripped usually.
        if case_id not in raw_text:
            packet.conflicts.add("case_id")

    _sanitize_arrival_date(packet)
    return packet


def _fuse_packets(case_id: str, sources: "TextSources") -> ParsedPacket:
    """Fuse independently parsed streams without cross-source conflicts.

    Native PDF text is authoritative. Page and embedded-image OCR may fill a
    missing field, but may not replace a native value or create model-visible
    conflicts against native. Meta-model features stay native-anchored unless an
    OCR field is actually accepted.
    """
    streams = (
        ("native", sources.native),
        ("page_ocr", sources.page_ocr),
        ("embedded_ocr", sources.embedded_ocr),
        # R45: neural union stream (lowest trust; fill-missing only).
        ("ppocr_ocr", getattr(sources, "ppocr_ocr", "") or ""),
        ("oriented_ocr", getattr(sources, "oriented_ocr", "") or ""),
    )
    parsed = [(name, _parse_single(case_id, text)) for name, text in streams if text]
    if not parsed:
        fused = ParsedPacket(case_id=case_id)
    else:
        first_source, fused = parsed[0]
        # Anchor meta features on the highest-trust stream (native when present).
        fused.field_sources = {key: first_source for key in fused.fields}
        # trusted_text_chars / sources_seen / conflicts / observed_flags_seen remain
        # as parsed from native; do not inflate with concatenated OCR length.

        for source_name, candidate in parsed[1:]:
            accepted_any = False
            for key, value in candidate.fields.items():
                if key in fused.fields:
                    if value != fused.fields[key]:
                        # Provenance-only signal; never feed n_conflicts.
                        fused.untrusted_conflicts.add(key)
                        # R44: prefer embedded sponsor only when page OCR itself
                        # also saw that SPN (page conflict / digit-flip pair).
                        if (
                            key == "sponsor_id"
                            and source_name == "embedded_ocr"
                            and fused.field_sources.get(key) == "page_ocr"
                            and "sponsor_id" not in candidate.conflicts
                            and value
                            in SPONSOR_ANYWHERE_RE.findall(
                                getattr(sources, "page_ocr", "") or ""
                            )
                        ):
                            fused.fields[key] = value
                            fused.field_sources[key] = source_name
                            accepted_any = True
                    continue
                fused.fields[key] = value
                fused.field_sources[key] = source_name
                accepted_any = True
                if key in candidate.conflicts:
                    fused.conflicts.add(key)
                tag = _FIELD_SOURCE_TAG.get(key)
                if tag:
                    fused.sources_seen.add(tag)
                if key == "fee_status":
                    fused.sources_seen.add("fee")

            if fused.adjudicator_finding is None and candidate.adjudicator_finding:
                fused.adjudicator_finding = candidate.adjudicator_finding

            # Risk flags: OCR may contribute only with explicit Observed-flags /
            # Registry Status / "risk flag:" context — never bare free tokens.
            # Propagate observed_flags_seen whenever OCR saw that section so the
            # has_dq meta detector is not spuriously triggered.
            if candidate.observed_flags_seen:
                fused.observed_flags_seen = True
            if candidate.explicit_risk_flags:
                fused.risk_flags = sorted(
                    set(fused.risk_flags) | set(candidate.explicit_risk_flags)
                )
                if candidate.sources_seen & {"registry", "biometric"}:
                    fused.sources_seen.update(
                        candidate.sources_seen & {"registry", "biometric"}
                    )

            if (
                "arrival_date" not in fused.fields
                and "arrival_date_unreadable" in candidate.conflicts
            ):
                fused.conflicts.add("arrival_date_unreadable")

            # Accept OCR document-source tags only when a field was filled from OCR.
            if accepted_any:
                for tag in ("registry", "biometric", "adjudicator"):
                    if tag in candidate.sources_seen:
                        fused.sources_seen.add(tag)

    # Specialized biometric-header OCR: flags only. Never mutates fields,
    # sources_seen, conflicts, or trusted_text_chars.
    bio_flags = tuple(getattr(sources, "biometric_flags", ()) or ())
    if bio_flags:
        fused.risk_flags = sorted(set(fused.risk_flags) | set(bio_flags))
        fused.observed_flags_seen = True
        fused.field_sources.setdefault("biometric_flags", "biometric_header_ocr")

    # Round-16 fee fuzzy: fill missing/unknown only from existing streams.
    # Never override a recognized paid/waived/unpaid value; never touch
    # trusted_text_chars / conflicts / sources_seen (meta-stable).
    if _fee_is_missing_or_unknown(fused.fields):
        for source_name in ("page_ocr", "embedded_ocr", "native"):
            raw = getattr(sources, source_name, "") or ""
            if not raw:
                continue
            cleaned, _ = strip_injection(raw)
            before = fused.fields.get("fee_status")
            _apply_fee_fuzzy_fill(
                fused, cleaned, provenance=f"fee_fuzzy:{source_name}"
            )
            if fused.fields.get("fee_status") != before and not _fee_is_missing_or_unknown(
                fused.fields
            ):
                break

    # Round-38: preserve explicit Waiver Code / signed hardship authorization
    # even when Fee Status was already parsed. The evidence may authorize a
    # visible non-DIP waiver; filling remains missing/unknown-only.
    from .waiver_fee import apply_waiver_fee_fill

    apply_waiver_fee_fill(fused, sources)

    # Round-17: source-local constrained fills / rare OCR overrides. Streams stay
    # separate (no global concatenation); trusted_text_chars / conflicts /
    # sources_seen unchanged except intentional field acceptance tags.
    _apply_r17_constrained_fills(fused, sources)

    # Round-21: provenance-isolated Sauvola OCR fills (missing/placeholder only).
    # Before name lexicon so R21 name candidates can be repaired by R18.
    _apply_r21_threshold_fills(fused, sources)

    # Round-22: tessdata_best fallback fills (missing only; after R21).
    _apply_r22_best_fills(fused, sources)

    # Round-18: provenance-aware applicant-name resolution (lexicon + chrome /
    # sponsor/registry). Name-only; no trusted_text / conflicts / sources_seen.
    from .name_resolve import apply_r18_name_resolution

    apply_r18_name_resolution(fused, sources)

    # Final source-local fee ledger arbitration. Run after OCR fills so a
    # stale status cell cannot outrank the internally consistent amount/code
    # pair on the same receipt.
    from .waiver_fee import apply_fee_ledger_consistency

    apply_fee_ledger_consistency(fused, sources)

    _sanitize_arrival_date(fused)
    return fused


def _cleaned_streams(sources: "TextSources") -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("native", "page_ocr", "embedded_ocr"):
        raw = getattr(sources, name, "") or ""
        if not raw:
            continue
        cleaned, _ = strip_injection(raw)
        if cleaned.strip():
            out[name] = cleaned
    return out


def _accept_field_fill(
    packet: ParsedPacket,
    key: str,
    value: str,
    *,
    provenance: str,
) -> None:
    """Accept a constrained fill without touching trusted_text_chars/conflicts."""
    if key == "arrival_date" and not _date_in_plausible_range(value):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) and not is_valid_calendar_date(
            value
        ):
            packet.fields.pop("arrival_date", None)
            packet.conflicts.add("arrival_date_unreadable")
        return
    packet.fields[key] = value
    packet.field_sources[key] = provenance
    tag = _FIELD_SOURCE_TAG.get(key)
    if tag:
        packet.sources_seen.add(tag)
    if key == "arrival_date":
        packet.conflicts.discard("arrival_date_unreadable")


def _apply_r17_constrained_fills(
    fused: ParsedPacket,
    sources: "TextSources",
) -> None:
    """High-precision source-local recoveries for R17 target fields."""
    streams = _cleaned_streams(sources)
    if not streams:
        return

    # --- arrival_date ---
    page_unique = (
        extract_unique_plausible_date(streams["page_ocr"])
        if "page_ocr" in streams
        else None
    )
    emb_unique = (
        extract_unique_plausible_date(streams["embedded_ocr"])
        if "embedded_ocr" in streams
        else None
    )
    page_emb_agree = (
        page_unique
        if page_unique and emb_unique and page_unique == emb_unique
        else None
    )

    if "arrival_date" not in fused.fields:
        if page_emb_agree:
            _accept_field_fill(
                fused,
                "arrival_date",
                page_emb_agree,
                provenance="r17_date:page+embedded",
            )
        else:
            # Single-source fill only with label proximity (already attempted
            # in per-stream parse; retry on cleaned streams for fusion gaps).
            for source_name in ("page_ocr", "embedded_ocr", "native"):
                text = streams.get(source_name)
                if not text:
                    continue
                prox = extract_label_proximate_date(text, allow_fuzzy=True)
                if prox:
                    _accept_field_fill(
                        fused,
                        "arrival_date",
                        prox,
                        provenance=f"r17_date:{source_name}",
                    )
                    break
        # Both standard OCR streams independently see the impossible 2028
        # year; normalize only that glyph and require agreement.
        if "arrival_date" not in fused.fields:
            page_repair = (
                extract_label_proximate_future_year_repair(streams["page_ocr"])
                if "page_ocr" in streams
                else None
            )
            embedded_repair = (
                extract_label_proximate_future_year_repair(
                    streams["embedded_ocr"]
                )
                if "embedded_ocr" in streams
                else None
            )
            if page_repair and page_repair == embedded_repair:
                _accept_field_fill(
                    fused,
                    "arrival_date",
                    page_repair,
                    provenance="r17_date_2028glyph:page+embedded",
                )
        # R44: Arrival label dropped by OCR — ISO date sandwiched between
        # Sponsor ID and Declared Purpose (DEV-measured 1/1, 0 false).
        if "arrival_date" not in fused.fields:
            for source_name in ("page_ocr", "embedded_ocr", "native"):
                text = streams.get(source_name)
                if not text:
                    continue
                m = _SPONSOR_DATE_PURPOSE_RE.search(text)
                if not m:
                    continue
                value = m.group(1)
                if _date_in_plausible_range(value):
                    _accept_field_fill(
                        fused,
                        "arrival_date",
                        value,
                        provenance=f"r44_date_struct:{source_name}",
                    )
                    break
    else:
        # Wrong nonempty override only with strong page+embedded agreement and
        # current value OCR-derived / non-native / non-manual.
        current = fused.fields["arrival_date"]
        src = fused.field_sources.get("arrival_date", "")
        ocr_derived = src.startswith("page_ocr") or src.startswith("embedded_ocr")
        if (
            page_emb_agree
            and page_emb_agree != current
            and ocr_derived
            and not src.startswith("native")
            and "manual" not in src
        ):
            _accept_field_fill(
                fused,
                "arrival_date",
                page_emb_agree,
                provenance="r17_date_override:page+embedded",
            )

    # --- visa_class / species_code (missing only; no native/manual override) ---
    label_extractors = {
        "visa_class": extract_label_proximate_visas,
        "species_code": extract_label_proximate_species,
    }
    whole_extractors = {
        "visa_class": extract_allowlisted_visas,
        "species_code": extract_allowlisted_species,
    }
    for key in ("visa_class", "species_code"):
        if not _field_missing_or_unknown(fused.fields, key):
            continue
        src = fused.field_sources.get(key, "")
        if src.startswith("native") or "manual" in src:
            continue
        # 1) Label-proximate unique token in one source.
        filled = False
        for source_name in ("page_ocr", "embedded_ocr", "native"):
            text = streams.get(source_name)
            if not text:
                continue
            hits = label_extractors[key](text)
            if len(hits) == 1:
                _accept_field_fill(
                    fused, key, hits[0], provenance=f"r17_{key}:{source_name}"
                )
                filled = True
                break
        if filled:
            continue
        # 2) Multiple / unlabeled: require cross-source agreement on one value.
        per_source = {
            name: whole_extractors[key](text)
            for name, text in streams.items()
            if text
        }
        evidence = [set(v) for v in per_source.values() if v]
        if len(evidence) >= 2:
            agree = set.intersection(*evidence)
            if len(agree) == 1:
                value = next(iter(agree))
                _accept_field_fill(
                    fused, key, value, provenance=f"r17_{key}:agree"
                )

    # --- declared_purpose: missing only; label-prox or full agreement ---
    if _field_missing_or_unknown(fused.fields, "declared_purpose"):
        src = fused.field_sources.get("declared_purpose", "")
        if not (src.startswith("native") or "manual" in src):
            filled = False
            for source_name in ("page_ocr", "embedded_ocr", "native"):
                text = streams.get(source_name)
                if not text:
                    continue
                prox = extract_label_proximate_purpose(text)
                if prox:
                    _accept_field_fill(
                        fused,
                        "declared_purpose",
                        prox,
                        provenance=f"r17_purpose:{source_name}",
                    )
                    filled = True
                    break
            if not filled:
                evidence = {
                    name: extract_purpose_candidates(text)
                    for name, text in streams.items()
                }
                bearing = [set(v) for v in evidence.values() if v]
                if bearing and all(len(s) == 1 for s in bearing):
                    values = {next(iter(s)) for s in bearing}
                    if len(values) == 1:
                        _accept_field_fill(
                            fused,
                            "declared_purpose",
                            next(iter(values)),
                            provenance="r17_purpose:agree",
                        )

    # --- home_world: correct OCR free-form fused value via containment ---
    current_home = fused.fields.get("home_world")
    home_src = fused.field_sources.get("home_world", "")
    if current_home and not _is_exact_allowlisted_home(current_home):
        if home_src.startswith("native") or "manual" in home_src:
            # Native/manual non-allowlist free-form: still allow containment
            # collapse only when the raw itself uniquely contains an allowlist
            # value (same stream-local rule); never replace an exact allowlist.
            contained = extract_allowlisted_home_world(current_home)
            if contained:
                fused.fields["home_world"] = contained
                # provenance unchanged — value cleaned in place
        else:
            contained = extract_allowlisted_home_world(current_home)
            if contained:
                fused.fields["home_world"] = contained
                fused.field_sources["home_world"] = f"r17_home:{home_src or 'ocr'}"


def _apply_fee_fuzzy_fill(
    packet: ParsedPacket,
    cleaned_text: str,
    *,
    provenance: str,
) -> None:
    """Fill missing/unknown fee from label-anchored fuzzy decode only."""
    if not _fee_is_missing_or_unknown(packet.fields):
        return
    fuzzy_fee = extract_fuzzy_fee_status(cleaned_text)
    if not fuzzy_fee:
        return
    packet.fields["fee_status"] = fuzzy_fee
    packet.field_sources["fee_status"] = provenance
    # Intentionally do not touch sources_seen / conflicts / trusted_text_chars.


def _apply_r21_threshold_fills(
    fused: ParsedPacket,
    sources: "TextSources",
) -> None:
    """Fill missing/placeholder fields from Sauvola threshold OCR only.

    Never overrides native/manual/current non-missing values. Never mutates
    trusted_text_chars. Risk flags require Observed-flags-anchored decode.
    """
    raw = getattr(sources, "threshold_ocr", "") or ""
    if not raw.strip():
        return
    from .threshold_ocr import extract_constrained_threshold_fields

    cand = extract_constrained_threshold_fields(raw)
    if not cand:
        return

    # Name/sponsor fills from Sauvola were FP-heavy on DEV; keep high-precision
    # closed fields only (visa/purpose/date/fee + Observed-flags).
    for key in (
        "fee_status",
        "visa_class",
        "arrival_date",
        "declared_purpose",
    ):
        if key not in cand:
            continue
        if key == "fee_status":
            if not _fee_is_missing_or_unknown(fused.fields):
                continue
        elif not _field_missing_or_unknown(fused.fields, key):
            continue
        src = fused.field_sources.get(key, "")
        if src.startswith("native") or "manual" in src:
            continue
        value = cand[key]
        if not isinstance(value, str) or not value:
            continue
        if key == "fee_status" and value == "unknown":
            continue
        _accept_field_fill(fused, key, value, provenance=f"r21_sauvola:{key}")

    flags = cand.get("risk_flags")
    if isinstance(flags, list) and flags:
        # Union only; Observed-flags already required inside extractor.
        before = set(fused.risk_flags)
        fused.risk_flags = sorted(before | set(flags))
        if set(fused.risk_flags) != before:
            fused.observed_flags_seen = True
            fused.field_sources.setdefault("risk_flags", "r21_sauvola:risk_flags")


def _apply_r22_best_fills(
    fused: ParsedPacket,
    sources: "TextSources",
) -> None:
    """Fill missing/placeholder fields from tessdata_best OCR only.

    After R21. Never overrides native/manual/current non-missing values.
    Never mutates trusted_text_chars. DEV grid: purpose + Observed-flags risk
    only (fee/name/sponsor FP-heavy; not applied).
    """
    raw = getattr(sources, "best_ocr", "") or ""
    if not raw.strip():
        return
    from .best_ocr import extract_constrained_best_fields

    cand = extract_constrained_best_fields(raw)
    if not cand:
        return

    for key in ("declared_purpose", "visa_class", "arrival_date"):
        if key not in cand:
            continue
        if not _field_missing_or_unknown(fused.fields, key):
            continue
        src = fused.field_sources.get(key, "")
        if src.startswith("native") or "manual" in src:
            continue
        # Do not override an R21 fill with best when already present.
        if src.startswith("r21_"):
            continue
        value = cand[key]
        if not isinstance(value, str) or not value:
            continue
        _accept_field_fill(fused, key, value, provenance=f"r22_best:{key}")

    flags = cand.get("risk_flags")
    if isinstance(flags, list) and flags:
        before = set(fused.risk_flags)
        fused.risk_flags = sorted(before | set(flags))
        if set(fused.risk_flags) != before:
            fused.observed_flags_seen = True
            fused.field_sources.setdefault("risk_flags", "r22_best:risk_flags")


def parse_packet(case_id: str, raw_text: str | "TextSources") -> ParsedPacket:
    """Parse one stream or fuse provenance-preserving extracted streams."""
    if isinstance(raw_text, str):
        packet = _parse_single(case_id, raw_text)
        cleaned, _ = strip_injection(raw_text)
        _apply_fee_fuzzy_fill(packet, cleaned, provenance="fee_fuzzy")
        from .name_resolve import apply_r18_name_resolution

        apply_r18_name_resolution(packet, raw_text)
        _sanitize_arrival_date(packet)
        return packet
    return _fuse_packets(case_id, raw_text)
