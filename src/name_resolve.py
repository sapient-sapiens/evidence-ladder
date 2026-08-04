"""Round-18 provenance-aware applicant-name resolution.

FIT-trained token lexicon + source-local candidates ranked by FIELD_MANUAL
precedence. Never ships case_id→name maps. Name-only: does not mutate
trusted_text_chars / conflicts / sources_seen.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .parse_fields import ParsedPacket
    from .text_extract import TextSources

_LEXICON_PATH = Path(__file__).resolve().parents[1] / "models" / "name_token_lexicon.json"

TIER_RANK = {
    "manual": 0,
    "intake": 1,
    "biometric": 2,
    "sponsor": 3,
    "registry": 4,
}

MANUAL_NAME_RE = re.compile(
    r"Manual correction:\s*applicant\s+is\s+(.+?)\.",
    re.IGNORECASE,
)
SPONSOR_ATTEST_RE = re.compile(
    r"Sponsor\s+(SPN-\d{4})\s+attests\s+that\s+(.+?)\s+is\s+expected\b",
    re.IGNORECASE,
)
# Label + value; value ends at multi-space, newline, or pipe table edge.
APP_LABEL_RE = re.compile(
    r"(?:^|\n)\s*(Applicant(?:\s+Name)?|Registry Name)"
    r"(?:\s{1,}|\s*:\s*|\s*\n\s*)([^\n|]{2,80})",
    re.IGNORECASE | re.MULTILINE,
)
CASE_ID_RE = re.compile(r"\b(MIB-\d{6})\b", re.IGNORECASE)

_CHROME_TRAIL_RE = re.compile(
    r"(?i)\s+(?:SCAN\s+IMAGE|PASSPORT\s+IMAGE|CASEWORK)\s*$"
)
# Trailing PORT only when it looks like form chrome (not a name token).
_PORT_TRAIL_RE = re.compile(r"(?i)\s+PORT\s*$")
_PLACEHOLDER_RE = re.compile(
    r"(?i)[\[\(\|\{]?\s*(?:NAME|WAME|TOI)\s*(?:CUT|GUT|TOI)?\s*(?:OUT|OLIT|Voy)?\s*[\]\)\|\}]?"
)
_FIELD_LEAK_RE = re.compile(
    r"(?i)^(Home\s+World|Species(?:\s+Code)?|Visa(?:\s+Class)?|Fee(?:\s+Status)?|"
    r"Sponsor(?:\s+ID)?|Arrival(?:\s+Date)?|Declared\s+Purpose|Purpose|"
    r"Registry\s+Status|Case\s+ID|Observed\s+flags)\b"
)
_DOC_BIO_RE = re.compile(r"(?i)\b(?:FORM\s*B-?13|biometric|Observed\s+flags)\b")
_DOC_REGISTRY_RE = re.compile(r"(?i)\b(?:Planetary\s+Registry|Registry\s+Extract|Registry\s+Name)\b")
_DOC_INTAKE_RE = re.compile(r"(?i)\b(?:FORM\s*I-?8090|Work\s+Authorization\s+Intake)\b")


@dataclass
class NameCandidate:
    raw: str
    name: str
    tier: str
    stream: str
    case_match: bool = False
    case_mismatch: bool = False
    chrome_stripped: bool = False
    edit_cost: int = 0
    lexicon_exact: bool = False
    sources: set[str] = field(default_factory=set)


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
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def load_name_lexicon(path: str | None = None) -> dict[str, object]:
    lex_path = Path(path) if path else _LEXICON_PATH
    if not lex_path.exists():
        return {
            "first_tokens": {},
            "last_tokens": {},
            "tokens": {},
            "_first_cf": {},
            "_last_cf": {},
            "_tokens_cf": {},
        }
    blob = json.loads(lex_path.read_text(encoding="utf-8"))
    first = dict(blob.get("first_tokens") or {})
    last = dict(blob.get("last_tokens") or {})
    tokens = dict(blob.get("tokens") or {})
    blob["_first_cf"] = {k.casefold(): k for k in first}
    blob["_last_cf"] = {k.casefold(): k for k in last}
    blob["_tokens_cf"] = {k.casefold(): k for k in tokens}
    blob["first_tokens"] = first
    blob["last_tokens"] = last
    blob["tokens"] = tokens
    return blob


def clear_lexicon_cache() -> None:
    load_name_lexicon.cache_clear()


def strip_name_chrome(raw: str) -> tuple[str, bool]:
    """Strip known OCR/page chrome; return (cleaned, did_strip)."""
    if not raw:
        return "", False
    text = " ".join(raw.split())
    trimmed = text.strip(" |\"'`“”~._=")
    stripped = trimmed != text
    text = trimmed
    for _ in range(3):
        nxt = _CHROME_TRAIL_RE.sub("", text)
        if nxt != text:
            text = nxt.strip(" |\"'`“”~._=")
            stripped = True
            continue
        nxt = _PORT_TRAIL_RE.sub("", text)
        if nxt != text:
            text = nxt.strip(" |\"'`“”~._=")
            stripped = True
            continue
        break
    # Trailing punct / OCR junk after a two-token name: "Name : — E", "Name i"
    m = re.match(
        r"^([A-Za-z][A-Za-z'`-]{1,24})\s+([A-Za-z][A-Za-z'`-]{1,24})"
        r"((?:[\s:;\-–—|./\\]+|\s+[A-Za-z0-9]{1,3})+)$",
        text,
    )
    if m and m.group(3).strip():
        # Reject if the trailing chunk still looks like a real name token.
        trail = re.sub(r"[^A-Za-z]", "", m.group(3))
        if len(trail) <= 3:
            text = f"{m.group(1)} {m.group(2)}"
            stripped = True
    return text.strip(), stripped


def is_name_placeholder(raw: str) -> bool:
    if not raw:
        return True
    t = raw.strip()
    if t.casefold() in {"unknown", "n/a", "na", "none", "null", "missing"}:
        return True
    if _PLACEHOLDER_RE.search(t):
        return True
    if _FIELD_LEAK_RE.match(t):
        return True
    return False


def _normalize_token_glyph(token: str) -> str:
    t = token.strip().strip("\"'`“”|_.,;:")
    # Leading digit/OCR I/l confusions common in names (lxotari / Inotari).
    if t[:1] in {"l", "1", "|", "!", "I"} and len(t) > 2 and t[1:].isalpha():
        # Prefer Title-case I… when rest looks like a name body.
        rest = t[1:]
        if rest[0].islower() or rest.isalpha():
            t = "I" + rest
    return t


def normalize_name_token(
    token: str,
    *,
    position: str,
    lexicon: dict[str, object],
    max_dist: int = 1,
) -> tuple[str | None, int, bool]:
    """Map one OCR token onto the lexicon.

    Returns (canonical, edit_cost, exact) or (None, -1, False) if ambiguous/unmatched.
    Dist-1 requires a unique best match (no tie). Dist-2 handled separately
    with a strong margin + multi-source gate.
    """
    raw = _normalize_token_glyph(token)
    if not raw or not re.fullmatch(r"[A-Za-z][A-Za-z'`-]{1,24}", raw):
        return None, -1, False
    # Drop internal punctuation for matching; keep alpha body.
    body = re.sub(r"[^A-Za-z]", "", raw)
    if len(body) < 3:
        return None, -1, False

    pos_map: dict[str, str] = lexicon.get(  # type: ignore[assignment]
        "_first_cf" if position == "first" else "_last_cf", {}
    )
    all_map: dict[str, str] = lexicon.get("_tokens_cf", {})  # type: ignore[assignment]
    # Prefer position-specific, fall back to unified token set.
    maps = [pos_map, all_map]

    cf = body.casefold()
    for m in maps:
        if cf in m:
            return m[cf], 0, True

    # Tesseract commonly collapses terminal ``rn`` to ``m`` (Qorzarn →
    # Qorzam). Expand only when that glyph repair lands exactly on a FIT token;
    # genuine exact tokens were already returned above.
    if cf.endswith("m"):
        expanded = f"{cf[:-1]}rn"
        for m in maps:
            if expanded in m:
                return m[expanded], 1, False

    for m in maps:
        ranked = sorted((_levenshtein(cf, cand_cf), canon) for cand_cf, canon in m.items())
        if not ranked:
            continue
        best_d, best = ranked[0]
        if best_d > max_dist or best_d < 1:
            continue
        peers = [c for d, c in ranked if d == best_d]
        if len(set(peers)) != 1:
            return None, -1, False
        return best, best_d, False
    return None, -1, False


def normalize_name_token_dist2(
    token: str,
    *,
    position: str,
    lexicon: dict[str, object],
) -> tuple[str | None, int, bool]:
    """Dist≤2 only for long tokens with unique margin ≥2."""
    raw = _normalize_token_glyph(token)
    body = re.sub(r"[^A-Za-z]", "", raw)
    if len(body) < 6:
        return None, -1, False
    pos_map: dict[str, str] = lexicon.get(  # type: ignore[assignment]
        "_first_cf" if position == "first" else "_last_cf", {}
    )
    all_map: dict[str, str] = lexicon.get("_tokens_cf", {})  # type: ignore[assignment]
    cf = body.casefold()
    for m in (pos_map, all_map):
        ranked = sorted((_levenshtein(cf, c), canon) for c, canon in m.items())
        if not ranked:
            continue
        best_d, best = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 99
        if best_d == 0:
            return best, 0, True
        if best_d <= 1:
            # uniqueness at best_d
            peers = [c for d, c in ranked if d == best_d]
            if len(set(peers)) == 1:
                return best, best_d, False
            return None, -1, False
        if best_d == 2 and (second - best_d) >= 2:
            return best, 2, False
        return None, -1, False
    return None, -1, False


def normalize_two_token_name(
    raw: str,
    lexicon: dict[str, object],
    *,
    allow_dist2: bool = False,
) -> tuple[str | None, int, bool]:
    """Return (Name, total_edit_cost, both_exact) or (None, …)."""
    cleaned, _ = strip_name_chrome(raw)
    if is_name_placeholder(cleaned):
        return None, -1, False
    if _FIELD_LEAK_RE.match(cleaned):
        return None, -1, False
    parts = cleaned.split()
    if len(parts) != 2:
        return None, -1, False
    a_raw, b_raw = parts
    a, da, ea = normalize_name_token(a_raw, position="first", lexicon=lexicon)
    b, db, eb = normalize_name_token(b_raw, position="last", lexicon=lexicon)
    if allow_dist2:
        if a is None:
            a, da, ea = normalize_name_token_dist2(
                a_raw, position="first", lexicon=lexicon
            )
        if b is None:
            b, db, eb = normalize_name_token_dist2(
                b_raw, position="last", lexicon=lexicon
            )
    if a is None or b is None or da < 0 or db < 0:
        return None, -1, False
    return f"{a} {b}", da + db, ea and eb


def _nearby_case_ids(text: str, start: int, end: int, radius: int = 220) -> set[str]:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return {m.group(1).upper() for m in CASE_ID_RE.finditer(text[lo:hi])}


def _infer_tier(label: str, context: str) -> str:
    lab = label.casefold()
    if "registry name" in lab:
        return "registry"
    if _DOC_BIO_RE.search(context):
        return "biometric"
    if _DOC_REGISTRY_RE.search(context):
        return "registry"
    if _DOC_INTAKE_RE.search(context):
        return "intake"
    # Bare Applicant defaults to intake (visible intake fields).
    return "intake"


def extract_name_candidates(
    text: str,
    *,
    stream: str,
    case_id: str,
    lexicon: dict[str, object],
) -> list[NameCandidate]:
    if not text:
        return []
    case_id_u = case_id.upper()
    out: list[NameCandidate] = []

    def _add(raw: str, tier: str, start: int, end: int) -> None:
        if not raw or is_name_placeholder(raw):
            return
        cleaned, chrome = strip_name_chrome(raw)
        if is_name_placeholder(cleaned):
            return
        # Try exact/dist1; keep candidate even if norm fails (for disagreement).
        norm, cost, exact = normalize_two_token_name(cleaned, lexicon)
        if norm is None:
            # Allow dist2 only later with multi-source support; store raw cleaned
            # two-token alpha for chrome-only paths.
            parts = cleaned.split()
            if len(parts) != 2:
                return
            if not all(re.fullmatch(r"[A-Za-z][A-Za-z'`-]{1,24}", p) for p in parts):
                return
            # Unnormalized — only useful if already clean-looking; skip garbage.
            return
        ids = _nearby_case_ids(text, start, end)
        case_match = case_id_u in ids if ids else False
        case_mismatch = bool(ids) and case_id_u not in ids
        out.append(
            NameCandidate(
                raw=raw,
                name=norm,
                tier=tier,
                stream=stream,
                case_match=case_match,
                case_mismatch=case_mismatch,
                chrome_stripped=chrome,
                edit_cost=cost,
                lexicon_exact=exact,
                sources={stream},
            )
        )

    for match in MANUAL_NAME_RE.finditer(text):
        _add(match.group(1).strip(), "manual", match.start(), match.end())

    for match in SPONSOR_ATTEST_RE.finditer(text):
        _add(match.group(2).strip(), "sponsor", match.start(), match.end())

    for match in APP_LABEL_RE.finditer(text):
        label = match.group(1)
        value = match.group(2).strip()
        # Truncate value at next labeled field fragment glued on same line.
        value = re.split(
            r"(?i)\s{2,}|\s+(?:Species|Home|Visa|Sponsor|Arrival|Fee|Declared|Purpose|Registry|Case)\b",
            value,
            maxsplit=1,
        )[0].strip()
        ctx_lo = max(0, match.start() - 180)
        context = text[ctx_lo : match.start()]
        tier = _infer_tier(label, context)
        _add(value, tier, match.start(), match.end())

    return out


def _candidate_sort_key(c: NameCandidate) -> tuple:
    return (
        TIER_RANK.get(c.tier, 99),
        0 if c.case_match else 1,
        1 if c.case_mismatch else 0,
        0 if c.lexicon_exact else 1,
        c.edit_cost,
        1 if c.chrome_stripped else 0,
        -len(c.sources),
        c.name,
    )


def _merge_candidates(cands: list[NameCandidate]) -> list[NameCandidate]:
    """Merge identical normalized names within the same tier."""
    buckets: dict[tuple[str, str], NameCandidate] = {}
    for c in cands:
        key = (c.tier, c.name.casefold())
        if key not in buckets:
            buckets[key] = NameCandidate(
                raw=c.raw,
                name=c.name,
                tier=c.tier,
                stream=c.stream,
                case_match=c.case_match,
                case_mismatch=c.case_mismatch,
                chrome_stripped=c.chrome_stripped,
                edit_cost=c.edit_cost,
                lexicon_exact=c.lexicon_exact,
                sources=set(c.sources),
            )
            continue
        cur = buckets[key]
        cur.sources |= c.sources
        cur.case_match = cur.case_match or c.case_match
        cur.case_mismatch = cur.case_mismatch and c.case_mismatch
        cur.edit_cost = min(cur.edit_cost, c.edit_cost)
        cur.lexicon_exact = cur.lexicon_exact or c.lexicon_exact
        cur.chrome_stripped = cur.chrome_stripped and c.chrome_stripped
        # Prefer native stream tag when merging identical names.
        stream_rank = {"native": 0, "page_ocr": 1, "embedded_ocr": 2}
        if stream_rank.get(c.stream, 9) < stream_rank.get(cur.stream, 9):
            cur.stream = c.stream
    return list(buckets.values())


def _is_clean_plausible(name: str | None, lexicon: dict[str, object]) -> bool:
    if not name or is_name_placeholder(name):
        return False
    cleaned, chrome = strip_name_chrome(name)
    if chrome:
        return False
    norm, cost, exact = normalize_two_token_name(cleaned, lexicon)
    return bool(norm and cost == 0 and exact and norm.casefold() == cleaned.casefold())


def _soft_normalize_current(
    name: str | None, lexicon: dict[str, object]
) -> tuple[str | None, int, bool, bool]:
    """Return (norm, cost, exact, had_chrome) for the current fused name."""
    if not name or is_name_placeholder(name):
        return None, -1, False, False
    cleaned, chrome = strip_name_chrome(name)
    if _FIELD_LEAK_RE.match(cleaned):
        return None, -1, False, chrome
    norm, cost, exact = normalize_two_token_name(cleaned, lexicon)
    if norm is None:
        norm, cost, exact = normalize_two_token_name(
            cleaned, lexicon, allow_dist2=True
        )
    return norm, cost, exact, chrome


def _is_garbled_current(name: str | None, lexicon: dict[str, object]) -> bool:
    if not name or is_name_placeholder(name):
        return True
    cleaned, chrome = strip_name_chrome(name)
    if _FIELD_LEAK_RE.match(cleaned):
        return True
    norm, cost, exact, _ = _soft_normalize_current(name, lexicon)
    if norm is None:
        return True
    # Chrome on an otherwise exact name is correctable garble.
    if chrome and cost == 0:
        return True
    # Far from lexicon → garbled.
    return cost >= 2 and not exact


def _shared_token_count(a: str, b: str) -> int:
    ta = set(a.casefold().split())
    tb = set(b.casefold().split())
    return len(ta & tb)


def _current_lexicon_tokens(name: str | None, lexicon: dict[str, object]) -> set[str]:
    if not name or is_name_placeholder(name):
        return set()
    cleaned, _ = strip_name_chrome(name)
    toks: set[str] = set()
    for i, part in enumerate(cleaned.split()[:2]):
        pos = "first" if i == 0 else "last"
        canon, cost, _ = normalize_name_token(part, position=pos, lexicon=lexicon)
        if canon and cost == 0:
            toks.add(canon.casefold())
        else:
            # Also accept dist2-unique long tokens as anchors (Qormor→Qormora).
            canon2, cost2, _ = normalize_name_token_dist2(
                part, position=pos, lexicon=lexicon
            )
            if canon2 and cost2 <= 2:
                toks.add(canon2.casefold())
    return toks


def _looks_like_two_token_name(name: str | None) -> bool:
    if not name or is_name_placeholder(name):
        return False
    cleaned, _ = strip_name_chrome(name)
    if _FIELD_LEAK_RE.match(cleaned):
        return False
    parts = cleaned.split()
    if len(parts) != 2:
        return False
    for part in parts:
        body = re.sub(r"[^A-Za-z]", "", part)
        if len(body) < 3:
            return False
    return True


def select_name_candidate(
    candidates: list[NameCandidate],
    *,
    case_id: str,
    current: str | None,
    lexicon: dict[str, object],
) -> NameCandidate | None:
    if not candidates:
        return None
    merged = _merge_candidates(candidates)
    # Drop hard case mismatches unless manual.
    filtered = [
        c for c in merged if c.tier == "manual" or not c.case_mismatch
    ] or merged

    support: dict[str, set[str]] = {}
    for c in candidates:
        support.setdefault(c.name.casefold(), set()).update(c.sources)

    cur_norm, cur_cost, cur_exact, cur_chrome = _soft_normalize_current(
        current, lexicon
    )
    cur_tokens = _current_lexicon_tokens(current, lexicon)
    placeholder = not current or is_name_placeholder(current or "")
    current_clean = _is_clean_plausible(current, lexicon)

    def repair_bonus(c: NameCandidate) -> tuple:
        """Prefer repairs of the current string / shared tokens over strangers."""
        share = _shared_token_count(c.name, cur_norm) if cur_norm else 0
        if not share and cur_tokens:
            share = len(cur_tokens & set(c.name.casefold().split()))
        multi = len(support.get(c.name.casefold(), set()))
        return (
            TIER_RANK.get(c.tier, 99),
            0 if share else 1,
            0 if c.case_match else 1,
            0 if c.lexicon_exact else 1,
            c.edit_cost,
            0 if multi >= 2 else 1,
            1 if c.chrome_stripped else 0,
            -multi,
            c.name,
        )

    filtered.sort(key=repair_bonus)
    if not filtered:
        return None

    # Genuine disagreement among top-tier clean exact candidates.
    best_tier = min(TIER_RANK.get(c.tier, 99) for c in filtered)
    top_clean = [
        c
        for c in filtered
        if TIER_RANK.get(c.tier, 99) == best_tier
        and c.edit_cost == 0
        and c.lexicon_exact
    ]
    top_names = {c.name.casefold() for c in top_clean}
    if len(top_names) > 1:
        matched = [c for c in top_clean if c.case_match]
        matched_names = {c.name.casefold() for c in matched}
        if len(matched_names) == 1:
            filtered = sorted(matched, key=repair_bonus) + [
                c for c in filtered if c not in matched
            ]
        elif current_clean or (cur_norm and cur_cost == 0 and cur_exact and not cur_chrome):
            return None
        elif cur_tokens:
            # Keep only candidates sharing a current lexicon token.
            share = [
                c
                for c in filtered
                if len(cur_tokens & set(c.name.casefold().split())) > 0
            ]
            if not share:
                return None
            filtered = sorted(share, key=repair_bonus)
        else:
            # Placeholder with conflicting clean intakes — need multi-source.
            multi = [
                c
                for c in top_clean
                if len(support.get(c.name.casefold(), set())) >= 2
            ]
            multi_names = {c.name.casefold() for c in multi}
            if len(multi_names) != 1:
                return None
            filtered = sorted(multi, key=repair_bonus)

    best = filtered[0]

    if current_clean:
        if cur_norm and cur_norm.casefold() == best.name.casefold():
            return best if current != best.name else None
        # Manual may override a clean value; nothing else may.
        if best.tier == "manual":
            return best
        return None

    # Soft-clean current (near-lexicon, possibly chrome): only canonicalize to
    # the soft-normalized form itself (or manual). Do not jump to a different
    # shared-token name (Andane Miraix ↛ Aridane Miraix).
    if cur_norm and cur_cost <= 2 and not placeholder and _looks_like_two_token_name(current):
        if best.tier == "manual":
            return best
        same = [c for c in filtered if c.name.casefold() == cur_norm.casefold()]
        if same:
            pick = sorted(same, key=repair_bonus)[0]
            if current != pick.name or cur_chrome:
                return pick
            return None
        # Chrome-only self cleanup (no lexicon hallucination without evidence).
        if cur_chrome and cur_cost == 0 and cur_exact and current != cur_norm:
            return NameCandidate(
                raw=current or cur_norm,
                name=cur_norm,
                tier="intake",
                stream="current",
                edit_cost=0,
                lexicon_exact=True,
                chrome_stripped=True,
                sources={"current"},
            )
        return None

    # Current has at least one exact lexicon token but full name won't normalize:
    # only accept candidates sharing that token (or manual).
    if cur_tokens and not placeholder and _looks_like_two_token_name(current):
        share_cands = [
            c
            for c in filtered
            if len(cur_tokens & set(c.name.casefold().split())) > 0
        ]
        if share_cands:
            best = sorted(share_cands, key=repair_bonus)[0]
            return best
        if best.tier == "manual":
            return best
        return None

    # Missing / placeholder / field-leak / severe garble.
    if placeholder or _is_garbled_current(current, lexicon):
        looks_name = _looks_like_two_token_name(current)
        cutout = bool(current and _PLACEHOLDER_RE.search(current))

        def acceptable_fill(c: NameCandidate) -> bool:
            multi = len(support.get(c.name.casefold(), set()))
            if c.tier == "manual":
                return True
            # [NAME CUT OUT] is often packet-wide; intake labels may be other
            # applicants. Only trust sponsor/biometric/registry/manual.
            if cutout and c.tier == "intake":
                return False
            # Require shared-token repair when current already anchors to the
            # lexicon. Fully garbled OCR (birequell Qcrd) may be replaced by a
            # corroborated non-intake exact name, but never by a random intake.
            if looks_name and (cur_tokens or cur_norm):
                share = 0
                if cur_norm:
                    share = _shared_token_count(c.name, cur_norm)
                if not share and cur_tokens:
                    share = len(cur_tokens & set(c.name.casefold().split()))
                if not share and cur_norm and c.name.casefold() == cur_norm.casefold():
                    share = 2
                if share < 1:
                    return False
            elif looks_name and not cur_tokens and not cur_norm:
                if c.tier == "intake":
                    return False
            if c.edit_cost >= 2 and multi < 2:
                if not (cur_norm and c.name.casefold() == cur_norm.casefold()):
                    return False
            if c.tier in {"sponsor", "registry"} and multi < 2:
                return False
            if c.tier == "intake" and not c.case_match and multi < 2:
                return False
            if placeholder and not cutout and c.tier == "intake" and multi < 2:
                return False
            return True

        for c in filtered:
            if acceptable_fill(c):
                return c
        # Last resort: soft-norm of current itself when chrome/garbled.
        if cur_norm and (cur_chrome or cur_cost > 0):
            return NameCandidate(
                raw=current or cur_norm,
                name=cur_norm,
                tier="intake",
                stream="current",
                case_match=False,
                edit_cost=max(cur_cost, 0),
                lexicon_exact=bool(cur_exact),
                chrome_stripped=bool(cur_chrome),
                sources={"current"},
            )
        return None

    # Fallback chrome equality
    if cur_norm and cur_chrome and best.name.casefold() == cur_norm.casefold():
        return best
    return None


def collect_packet_name_candidates(
    sources: "TextSources | str",
    case_id: str,
    lexicon: dict[str, object] | None = None,
) -> list[NameCandidate]:
    from .parse_fields import strip_injection

    lex = lexicon or load_name_lexicon()
    streams: list[tuple[str, str]] = []
    if isinstance(sources, str):
        cleaned, _ = strip_injection(sources)
        streams = [("native", cleaned)]
    else:
        for name in ("native", "page_ocr", "embedded_ocr"):
            raw = getattr(sources, name, "") or ""
            if not raw:
                continue
            cleaned, _ = strip_injection(raw)
            if cleaned.strip():
                streams.append((name, cleaned))
    cands: list[NameCandidate] = []
    for stream, text in streams:
        cands.extend(
            extract_name_candidates(
                text, stream=stream, case_id=case_id, lexicon=lex
            )
        )
    # Second pass: dist2 normalization for raw two-token labels that failed dist1,
    # only when the same raw chrome-stripped form appears or we'll gate on support.
    # (extract already uses dist1; add explicit dist2 candidates from label hits)
    extra: list[NameCandidate] = []
    for stream, text in streams:
        for match in APP_LABEL_RE.finditer(text):
            value = match.group(2).strip()
            value = re.split(
                r"(?i)\s{2,}|\s+(?:Species|Home|Visa|Sponsor|Arrival|Fee|Declared|Purpose|Registry|Case)\b",
                value,
                maxsplit=1,
            )[0].strip()
            cleaned, chrome = strip_name_chrome(value)
            if is_name_placeholder(cleaned):
                continue
            norm1, _, _ = normalize_two_token_name(cleaned, lex)
            if norm1 is not None:
                continue
            norm2, cost2, exact2 = normalize_two_token_name(
                cleaned, lex, allow_dist2=True
            )
            if norm2 is None or cost2 < 2:
                continue
            label = match.group(1)
            ctx = text[max(0, match.start() - 180) : match.start()]
            tier = _infer_tier(label, ctx)
            ids = _nearby_case_ids(text, match.start(), match.end())
            extra.append(
                NameCandidate(
                    raw=value,
                    name=norm2,
                    tier=tier,
                    stream=stream,
                    case_match=case_id.upper() in ids if ids else False,
                    case_mismatch=bool(ids) and case_id.upper() not in ids,
                    chrome_stripped=chrome,
                    edit_cost=cost2,
                    lexicon_exact=exact2,
                    sources={stream},
                )
            )
        for match in SPONSOR_ATTEST_RE.finditer(text):
            value = match.group(2).strip()
            cleaned, chrome = strip_name_chrome(value)
            if is_name_placeholder(cleaned):
                continue
            norm1, _, _ = normalize_two_token_name(cleaned, lex)
            if norm1 is not None:
                continue
            norm2, cost2, exact2 = normalize_two_token_name(
                cleaned, lex, allow_dist2=True
            )
            if norm2 is None:
                continue
            ids = _nearby_case_ids(text, match.start(), match.end())
            extra.append(
                NameCandidate(
                    raw=value,
                    name=norm2,
                    tier="sponsor",
                    stream=stream,
                    case_match=case_id.upper() in ids if ids else False,
                    case_mismatch=bool(ids) and case_id.upper() not in ids,
                    chrome_stripped=chrome,
                    edit_cost=cost2,
                    lexicon_exact=exact2,
                    sources={stream},
                )
            )
    return cands + extra


def resolve_applicant_name(
    case_id: str,
    sources: "TextSources | str",
    current: str | None,
    lexicon: dict[str, object] | None = None,
) -> tuple[str | None, str | None]:
    """Return (new_name, provenance) or (None, None) if no change."""
    lex = lexicon or load_name_lexicon()
    if not lex.get("tokens"):
        return None, None

    # Cheap chrome/lexicon cleanup of the current value first.
    if current and not is_name_placeholder(current):
        cleaned, chrome = strip_name_chrome(current)
        norm, cost, exact = normalize_two_token_name(cleaned, lex)
        if norm and chrome and cost == 0:
            # Prefer cleaned current if no better candidate later.
            current_cleanup = norm
        elif norm and cost == 0 and exact and norm != current and not chrome:
            current_cleanup = norm
        else:
            current_cleanup = None
    else:
        current_cleanup = None

    cands = collect_packet_name_candidates(sources, case_id, lex)
    chosen = select_name_candidate(
        cands, case_id=case_id, current=current, lexicon=lex
    )

    if chosen is None:
        if current_cleanup and current_cleanup != current:
            return current_cleanup, "r18_name:chrome_strip"
        return None, None

    if current and chosen.name == current:
        return None, None

    # If chosen equals cleanup, use richer provenance from chosen.
    return chosen.name, f"r18_name:{chosen.tier}:{chosen.stream}"


def apply_r18_name_resolution(
    packet: "ParsedPacket",
    sources: "TextSources | str",
) -> None:
    """Name-only fill/correction. Leaves meta/conflict/trusted text untouched."""
    current = packet.fields.get("applicant_name")
    # Do not override an already-applied native manual correction value unless
    # the resolver itself found a manual candidate (handled inside select).
    new_name, prov = resolve_applicant_name(packet.case_id, sources, current)
    if not new_name or not prov:
        return
    packet.fields["applicant_name"] = new_name
    packet.field_sources["applicant_name"] = prov
