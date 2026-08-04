"""Round 35: cross-fitted noisy-channel closed-vocab OCR repair (+ R39 reconcile).

Literature:
- Kolak & Resnik, HLT 2002 — maximize P(obs|true) with lexicon prior; empirical
  error model beats nearest edit distance.
- Confidence-Aware Document OCR Error Detection, arXiv 2409.04117 — separate
  detection from correction; confidence/context matter; real-word errors need caution.
- Domain lexicon pipelines: candidate generation + OCR confusion likelihood + context.

Production apply uses a generic smoothed character-error prior; see
`load_confusion_model`.  No case IDs / token pairs.  Preserves trusted
native/manual provenance.

Round 39: store compact repair evidence flags and reconcile untrusted_conflicts
after gated repairs (policy B) without changing field values.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

from .constants import (
    FEE_STATUSES,
    HOME_WORLDS,
    PURPOSES,
    RISK_FLAG_ATOMS,
    SPECIES_CODES,
    VISA_CLASSES,
)
from .parse_fields import strip_injection

if TYPE_CHECKING:
    from .parse_fields import ParsedPacket
    from .text_extract import TextSources

LITERATURE = [
    "Kolak & Resnik, HLT 2002 — OCR Error Correction Using a Noisy Channel Model",
    "Confidence-Aware Document OCR Error Detection, arXiv 2409.04117",
    "Domain lexicon: candidate generation + OCR confusion + context scoring",
]

CLOSED_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "fee_status",
    "risk_flags",
)

STREAM_NAMES = (
    "native",
    "page_ocr",
    "embedded_ocr",
    "threshold_ocr",
    "best_ocr",
    "oriented_ocr",
)

_FEE_OBS_RE = re.compile(r"(?i)\[?\s*FEE STATUS OBSCURED\s*\]?")
_OBSCURED_RE = re.compile(
    r"(?i)obscur|unreadable|washed|whiteout|cut\s*out|blank|\[.*missing"
)
_PLACEHOLDER_RE = re.compile(
    r"(?i)^\s*(?:unknown|n/?a|none|null|[-_.]+|"
    r"\[?\s*(?:cut\s*out|whiteout|obscured|unreadable|missing)\s*\]?)\s*$"
)

# Label patterns (OCR-tolerant). home_world / Observed-flags included for R35.
_LABEL_RES: dict[str, re.Pattern[str]] = {
    "species_code": re.compile(
        r"(?i)(?:Species(?:\s+Code)?|Spaces(?:\s+Code)?|Spacies(?:\s+Code)?)"
    ),
    "home_world": re.compile(r"(?i)Home(?:\s+World|\s*\n\s*World)"),
    "visa_class": re.compile(
        r"(?i)Visa(?:\s+Class|\s*\n\s*Class|\s+Close|\s+Ciass)"
    ),
    "declared_purpose": re.compile(
        r"(?i)(?:Dec(?:lared|tored|iered|iored|iared|isred)|Declared)\s+Purpose|"
        r"(?:^|\n)\s*Purpose\b"
    ),
    "fee_status": re.compile(
        r"(?i)Fee(?:\s+Status|\s*\n\s*Status|\s+Staius|\s+Statu[s5])"
    ),
    "risk_flags": re.compile(
        r"(?i)Observed\s+flags|Observed\s+fiags|Cbserved\s+flags"
    ),
}

# Value token grab after label (field-specific).
_VALUE_RES: dict[str, re.Pattern[str]] = {
    "species_code": re.compile(r"^[A-Za-z_]{3,24}"),
    "home_world": re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{1,40}"),
    "visa_class": re.compile(
        r"(?i)^(?:XW|DIP|MED|TRANSIT)[\-.\s_]?[0-9OIl]{1,2}"
    ),
    "declared_purpose": re.compile(r"^[A-Za-z][A-Za-z \-]{2,40}"),
    "fee_status": re.compile(r"(?i)^[A-Za-z]{3,12}"),
    "risk_flags": re.compile(r"^[A-Za-z_|,\s]{3,80}"),
}

_STOP_AFTER = re.compile(
    r"(?i)\b(?:Species|Home|Visa|Fee|Sponsor|Arrival|Declared|Purpose|"
    r"Registry|Observed|Case|Packet|FORM|MIB-|SCAN|PASSPORT)\b"
)

# Dirichlet / Laplace smoothing for char channel.
_ALPHA_SUB = 0.25
_ALPHA_INS = 0.25
_ALPHA_DEL = 0.25
_P_MATCH_PRIOR = 0.90
_P_INS_PRIOR = 0.04
_P_DEL_PRIOR = 0.04
_P_SUB_PRIOR = 0.02

# Gates
_MIN_MARGIN_LEV = 1
_MIN_MARGIN_LOGP = 1.5
_MAX_REL_EDIT = 0.45
_MIN_ABS_CONF = 0.35  # normalized source-agreement / confidence proxy

# R39 provenance reconciliation — clear untrusted_conflicts after a gated repair.
# Never clears applicant_name / sponsor_id / arrival_date (outside CLOSED_FIELDS).
RECONCILE_POLICIES = ("A", "B", "C")
# A: wrong-nonempty only, >=2 independent source streams (existing strongest gate)
# B: A + missing fills with >=2 independent + high posterior/margin
# C: no clearing (pre-R39 baseline)
# Production: B kept after R39 DEV ablation (A no-op on residuals; C baseline).
_DEFAULT_RECONCILE_POLICY = "B"
_HIGH_MARGIN = 3.0
_HIGH_POSTERIOR = 0.95  # sigmoid(3.0) ≈ 0.95
_PROTECTED_FROM_RECONCILE = frozenset(
    {"applicant_name", "sponsor_id", "arrival_date"}
)


@dataclass(frozen=True)
class ObservedToken:
    field: str
    raw: str
    stream: str
    has_label: bool
    mean_conf_proxy: float  # 0..1 from stream trust / length heuristics


@dataclass
class ConfusionModel:
    """Smoothed character confusion — no tokens, no case IDs."""

    version: str = "r35-v1"
    n_pairs: int = 0
    n_aligned: int = 0
    n_match: int = 0
    n_sub: int = 0
    n_ins: int = 0
    n_del: int = 0
    # true_char -> obs_char -> count (substitutions only)
    sub_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # obs inserted char counts / true deleted char counts
    ins_counts: dict[str, int] = field(default_factory=dict)
    del_counts: dict[str, int] = field(default_factory=dict)

    def add_alignment(self, ops: list[tuple[str, str | None, str | None]]) -> None:
        """ops: list of ('M'|'S'|'I'|'D', true_char|None, obs_char|None)."""
        self.n_pairs += 1
        for kind, t, o in ops:
            self.n_aligned += 1
            if kind == "M":
                self.n_match += 1
            elif kind == "S" and t is not None and o is not None:
                self.n_sub += 1
                self.sub_counts.setdefault(t, {})
                self.sub_counts[t][o] = self.sub_counts[t].get(o, 0) + 1
            elif kind == "I" and o is not None:
                self.n_ins += 1
                self.ins_counts[o] = self.ins_counts.get(o, 0) + 1
            elif kind == "D" and t is not None:
                self.n_del += 1
                self.del_counts[t] = self.del_counts.get(t, 0) + 1

    def merge(self, other: "ConfusionModel") -> "ConfusionModel":
        out = ConfusionModel(version=self.version)
        out.n_pairs = self.n_pairs + other.n_pairs
        out.n_aligned = self.n_aligned + other.n_aligned
        out.n_match = self.n_match + other.n_match
        out.n_sub = self.n_sub + other.n_sub
        out.n_ins = self.n_ins + other.n_ins
        out.n_del = self.n_del + other.n_del
        out.sub_counts = _merge_nested(self.sub_counts, other.sub_counts)
        out.ins_counts = dict(Counter(self.ins_counts) + Counter(other.ins_counts))
        out.del_counts = dict(Counter(self.del_counts) + Counter(other.del_counts))
        return out

    def to_shippable(self) -> dict[str, Any]:
        """Aggregate rates only — never includes training pairs or case IDs."""
        return {
            "version": self.version,
            "n_pairs": int(self.n_pairs),
            "n_aligned": int(self.n_aligned),
            "rates": {
                "p_match": _safe_div(self.n_match, self.n_aligned, _P_MATCH_PRIOR),
                "p_sub": _safe_div(self.n_sub, self.n_aligned, _P_SUB_PRIOR),
                "p_ins": _safe_div(self.n_ins, self.n_aligned, _P_INS_PRIOR),
                "p_del": _safe_div(self.n_del, self.n_aligned, _P_DEL_PRIOR),
            },
            "sub_counts": {
                t: dict(sorted(obs.items()))
                for t, obs in sorted(self.sub_counts.items())
            },
            "ins_counts": dict(sorted(self.ins_counts.items())),
            "del_counts": dict(sorted(self.del_counts.items())),
            "smoothing": {
                "alpha_sub": _ALPHA_SUB,
                "alpha_ins": _ALPHA_INS,
                "alpha_del": _ALPHA_DEL,
            },
            "literature": LITERATURE,
            "anti_leakage": {
                "contains_case_ids": False,
                "contains_token_pairs": False,
                "contains_truth_values": False,
                "shipped": "smoothed_char_edit_rates_only",
            },
        }

    @classmethod
    def from_shippable(cls, blob: dict[str, Any]) -> "ConfusionModel":
        m = cls(version=str(blob.get("version") or "r35-v1"))
        m.n_pairs = int(blob.get("n_pairs") or 0)
        m.n_aligned = int(blob.get("n_aligned") or 0)
        rates = blob.get("rates") or {}
        # Reconstruct counts from rates when only rates present.
        if blob.get("sub_counts") is not None:
            m.sub_counts = {
                str(t): {str(o): int(c) for o, c in obs.items()}
                for t, obs in (blob.get("sub_counts") or {}).items()
            }
            m.ins_counts = {str(k): int(v) for k, v in (blob.get("ins_counts") or {}).items()}
            m.del_counts = {str(k): int(v) for k, v in (blob.get("del_counts") or {}).items()}
            m.n_sub = sum(sum(d.values()) for d in m.sub_counts.values())
            m.n_ins = sum(m.ins_counts.values())
            m.n_del = sum(m.del_counts.values())
            m.n_match = max(0, m.n_aligned - m.n_sub - m.n_ins - m.n_del)
        else:
            # Priors-only fallback.
            m.n_aligned = max(1, m.n_aligned)
            m.n_match = int(float(rates.get("p_match", _P_MATCH_PRIOR)) * m.n_aligned)
            m.n_sub = int(float(rates.get("p_sub", _P_SUB_PRIOR)) * m.n_aligned)
            m.n_ins = int(float(rates.get("p_ins", _P_INS_PRIOR)) * m.n_aligned)
            m.n_del = int(float(rates.get("p_del", _P_DEL_PRIOR)) * m.n_aligned)
        return m


def _merge_nested(
    a: dict[str, dict[str, int]], b: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {k: dict(v) for k, v in a.items()}
    for t, obs in b.items():
        out.setdefault(t, {})
        for o, c in obs.items():
            out[t][o] = out[t].get(o, 0) + c
    return out


def _safe_div(num: int, den: int, default: float) -> float:
    if den <= 0:
        return float(default)
    return float(num) / float(den)


def classify_template_group(native: str, page_ocr: str, embedded_ocr: str) -> str:
    n, p, e = len(native or ""), len(page_ocr or ""), len(embedded_ocr or "")
    if e > 200 and e >= p:
        return "embedded_image"
    if n > 400 and p < 80:
        return "native_rich"
    if p > 80:
        return "native_scan"
    return "sparse"


def classify_corruption_from_text(text: str) -> str:
    t = text or ""
    if _OBSCURED_RE.search(t) or _FEE_OBS_RE.search(t):
        return "obscured_or_blank"
    weird = sum(1 for c in t if c in "|~`^¦»«°")
    if weird >= 8:
        return "glyph_noise"
    lower = t.casefold()
    if "anwval" in lower or "arnval" in lower or "findg" in lower:
        return "label_ocr_degraded"
    return "nominal"


def diagnostic_group_split(group_keys: list[str]) -> tuple[set[str], set[str]]:
    diag: set[str] = set()
    held: set[str] = set()
    for g in sorted(set(group_keys)):
        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
        if h % 2 == 0:
            diag.add(g)
        else:
            held.add(g)
    return diag, held


def lexicon_for(field: str) -> list[str]:
    if field == "species_code":
        return sorted(SPECIES_CODES)
    if field == "home_world":
        return sorted(HOME_WORLDS)
    if field == "visa_class":
        return sorted(VISA_CLASSES)
    if field == "declared_purpose":
        return sorted(PURPOSES)
    if field == "fee_status":
        # Never map onto unknown via correction.
        return sorted(s for s in FEE_STATUSES if s != "unknown")
    if field == "risk_flags":
        return sorted(RISK_FLAG_ATOMS)
    return []


def is_allowlisted(field: str, value: Any) -> bool:
    if value in (None, "", [], "unknown", "none"):
        return False
    if field == "risk_flags":
        atoms = value if isinstance(value, (list, tuple, set)) else [value]
        return bool(atoms) and all(str(a) in RISK_FLAG_ATOMS for a in atoms)
    return str(value) in set(lexicon_for(field))


def is_missing_or_placeholder(field: str, value: Any) -> bool:
    if value in (None, "", [], "unknown", "none"):
        return True
    if field == "risk_flags":
        return not value
    s = str(value).strip()
    if not s or _PLACEHOLDER_RE.match(s):
        return True
    return False


def looks_fee_obscured(text: str) -> bool:
    return bool(_FEE_OBS_RE.search(text or "") or _OBSCURED_RE.search(text or ""))


def levenshtein(a: str, b: str) -> int:
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


def align_ops(true: str, obs: str) -> list[tuple[str, str | None, str | None]]:
    """Wagner–Fischer alignment ops mapping true→obs (channel direction)."""
    T, O = (true or "").casefold(), (obs or "").casefold()
    n, m = len(T), len(O)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if T[i - 1] == O[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    i, j = n, m
    ops_rev: list[tuple[str, str | None, str | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and T[i - 1] == O[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops_rev.append(("M", T[i - 1], O[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops_rev.append(("S", T[i - 1], O[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops_rev.append(("D", T[i - 1], None))
            i -= 1
        else:
            ops_rev.append(("I", None, O[j - 1]))
            j -= 1
    ops_rev.reverse()
    return ops_rev


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _stream_conf_proxy(stream: str, raw: str) -> float:
    """Cheap confidence proxy without TSV (stream trust × token cleanliness)."""
    base = {
        "native": 0.95,
        "page_ocr": 0.70,
        "embedded_ocr": 0.65,
        "threshold_ocr": 0.55,
        "best_ocr": 0.60,
    }.get(stream, 0.5)
    if not raw:
        return 0.0
    alnum = sum(c.isalnum() or c in "-_ " for c in raw) / max(1, len(raw))
    weird = sum(c in "|~`^¦»«°" for c in raw) / max(1, len(raw))
    return max(0.05, min(1.0, base * (0.5 + 0.5 * alnum) * (1.0 - 0.7 * weird)))


def _trim_value(raw: str, field: str) -> str:
    s = _norm_spaces(raw)
    s = s.lstrip(":|=·• ")
    # Cut at next field chrome.
    mstop = _STOP_AFTER.search(s)
    if mstop and mstop.start() > 2:
        s = s[: mstop.start()]
    s = s.strip(" .:;|")
    if field == "risk_flags":
        s = s.split("\n", 1)[0].strip()
        if s.casefold() in {"none", "n/a", "null", ""}:
            return ""
    if field == "home_world":
        s = re.sub(r"(?i)\s+(?:PASSPORT\s+IMAGE|SCAN\s+IMAGE|CASEWORK)\s*$", "", s)
    if field == "fee_status":
        s = s.split()[0] if s.split() else s
    if field == "visa_class":
        s = re.sub(r"(?i)^(XW|DIP|MED|TRANSIT)[\-.\s_]+([0-9OIl])\b", r"\1-\2", s)
        s = s.upper().replace("O", "0").replace("I", "1").replace("L", "1")
        # Keep canonical shape if digits present.
        m = re.match(r"(?i)^(XW|DIP|MED|TRANSIT)-?(\d)$", s)
        if m:
            s = f"{m.group(1).upper()}-{m.group(2)}"
    if field == "declared_purpose":
        # Keep up to two words for multi-word purposes.
        parts = s.split()
        if len(parts) >= 2:
            s = " ".join(parts[:2])
        elif parts:
            s = parts[0]
    if field == "species_code":
        s = re.sub(r"[^A-Za-z_]", "", s).upper()
    if field == "risk_flags":
        s = s.lower().replace(" ", "_")
    return s.strip()


def extract_observed_tokens(text: str, field: str, stream: str) -> list[ObservedToken]:
    """Source-local label-proximate observed tokens (may be garbled / non-lexicon)."""
    if not text or field not in _LABEL_RES:
        return []
    cleaned, _ = strip_injection(text)
    if not cleaned.strip():
        return []
    if field == "fee_status" and looks_fee_obscured(cleaned):
        return []
    label_re = _LABEL_RES[field]
    val_re = _VALUE_RES[field]
    out: list[ObservedToken] = []
    for m in label_re.finditer(cleaned):
        window = cleaned[m.end() : m.end() + 64]
        window = window.lstrip(" \t:|-·•=")
        # Skip empty / newline-only
        if not window.strip():
            continue
        vm = val_re.search(window.strip())
        if not vm:
            continue
        raw = _trim_value(vm.group(0), field)
        if not raw or len(raw) < 2:
            continue
        if _PLACEHOLDER_RE.match(raw):
            continue
        if field == "risk_flags":
            # Split atoms; keep garbled single tokens too.
            atoms = [p.strip() for p in re.split(r"[|,]", raw) if p.strip()]
            if not atoms:
                continue
            for atom in atoms:
                atom = atom.strip().lower().replace(" ", "_")
                if len(atom) < 4:
                    continue
                out.append(
                    ObservedToken(
                        field=field,
                        raw=atom,
                        stream=stream,
                        has_label=True,
                        mean_conf_proxy=_stream_conf_proxy(stream, atom),
                    )
                )
            continue
        out.append(
            ObservedToken(
                field=field,
                raw=raw,
                stream=stream,
                has_label=True,
                mean_conf_proxy=_stream_conf_proxy(stream, raw),
            )
        )
    # Dedup by casefold raw+stream
    seen: set[tuple[str, str]] = set()
    uniq: list[ObservedToken] = []
    for tok in out:
        key = (tok.raw.casefold(), tok.stream)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tok)
    return uniq


def collect_stream_tokens(
    sources: Any, field: str
) -> list[ObservedToken]:
    """Gather observed tokens across all available OCR/native streams."""
    toks: list[ObservedToken] = []
    for name in STREAM_NAMES:
        text = getattr(sources, name, None)
        if text is None and isinstance(sources, dict):
            text = sources.get(name)
        if not text:
            continue
        toks.extend(extract_observed_tokens(str(text), field, name))
    return toks


def build_confusion_from_pairs(
    pairs: Iterable[tuple[str, str]],
) -> ConfusionModel:
    """pairs: (obs, true) — learns P(obs|true) channel. No IDs stored."""
    model = ConfusionModel()
    for obs, true in pairs:
        if not obs or not true:
            continue
        # Skip exact allowlist identity-only pairs lightly (still count matches).
        ops = align_ops(str(true), str(obs))
        if not ops:
            continue
        # Cap absurd pairs (unrelated strings).
        d = levenshtein(str(obs).casefold(), str(true).casefold())
        maxlen = max(len(obs), len(true), 1)
        if d / maxlen > 0.6 and d > 4:
            continue
        model.add_alignment(ops)
    return model


def crossfit_models(
    pairs_by_group: dict[str, list[tuple[str, str]]],
) -> dict[str, ConfusionModel]:
    """For each group, fit confusion on all OTHER groups only."""
    groups = sorted(pairs_by_group.keys())
    # Precompute per-group models then leave-one-out merge.
    per: dict[str, ConfusionModel] = {
        g: build_confusion_from_pairs(pairs_by_group[g]) for g in groups
    }
    out: dict[str, ConfusionModel] = {}
    for held in groups:
        acc = ConfusionModel()
        for g in groups:
            if g == held:
                continue
            acc = acc.merge(per[g])
        # If empty (single group), fall back to weak prior-only model.
        if acc.n_pairs == 0:
            acc = ConfusionModel(n_pairs=0, n_aligned=100, n_match=90, n_sub=5, n_ins=3, n_del=2)
        out[held] = acc
    return out


def log_emit_prob(obs: str, cand: str, model: ConfusionModel) -> float:
    """log P(obs|cand) under smoothed char channel (Kolak-style)."""
    ops = align_ops(cand, obs)
    if not ops:
        return 0.0
    n = max(1, model.n_aligned)
    p_match = _safe_div(model.n_match, n, _P_MATCH_PRIOR)
    p_ins = _safe_div(model.n_ins, n, _P_INS_PRIOR)
    p_del = _safe_div(model.n_del, n, _P_DEL_PRIOR)
    p_sub = _safe_div(model.n_sub, n, _P_SUB_PRIOR)
    # Clamp
    p_match = min(0.99, max(0.5, p_match))
    p_ins = min(0.2, max(1e-4, p_ins))
    p_del = min(0.2, max(1e-4, p_del))
    p_sub = min(0.3, max(1e-4, p_sub))

    # Vocab of chars seen in subs + alphabet extras
    alphabet = set("abcdefghijklmnopqrstuvwxyz0123456789-_ ")
    for t, obs_map in model.sub_counts.items():
        alphabet.add(t)
        alphabet.update(obs_map.keys())
    alphabet.update(model.ins_counts.keys())
    alphabet.update(model.del_counts.keys())
    v = max(16, len(alphabet))

    logp = 0.0
    for kind, t, o in ops:
        if kind == "M":
            logp += math.log(p_match)
        elif kind == "I":
            # P(insert obs char)
            c = model.ins_counts.get(o or "", 0)
            p_char = (c + _ALPHA_INS) / (model.n_ins + _ALPHA_INS * v)
            logp += math.log(p_ins * p_char)
        elif kind == "D":
            c = model.del_counts.get(t or "", 0)
            p_char = (c + _ALPHA_DEL) / (model.n_del + _ALPHA_DEL * v)
            logp += math.log(p_del * p_char)
        else:  # S
            row = model.sub_counts.get(t or "", {})
            row_tot = sum(row.values())
            c = row.get(o or "", 0)
            p_char = (c + _ALPHA_SUB) / (row_tot + _ALPHA_SUB * v)
            logp += math.log(p_sub * p_char)
    return logp


def field_prior_log(field: str, cand: str) -> float:
    """Conservative uniform prior over public lexicon (no truth frequencies)."""
    lex = lexicon_for(field)
    if not lex:
        return 0.0
    return -math.log(len(lex))


def score_candidates(
    obs: str,
    field: str,
    model: ConfusionModel,
    *,
    method: str,
    agreement: float = 0.0,
    conf_proxy: float = 0.0,
) -> list[tuple[float, str, dict[str, float]]]:
    """Return ranked (score, candidate, parts) for lexicon members."""
    lex = lexicon_for(field)
    if not lex or not obs:
        return []
    ranked: list[tuple[float, str, dict[str, float]]] = []
    obs_n = obs.casefold().replace(" ", "")
    for cand in lex:
        cand_n = cand.casefold().replace(" ", "")
        d = levenshtein(obs_n, cand_n)
        maxlen = max(len(obs_n), len(cand_n), 1)
        # Prefix / containment near-misses must stay eligible (AQUARIAN→AQUARIAN_MANTIS).
        contained = (
            obs_n in cand_n
            or cand_n in obs_n
            or cand_n.startswith(obs_n)
            or obs_n.startswith(cand_n)
        )
        if d / maxlen > _MAX_REL_EDIT and d > 3 and not contained:
            continue
        if method == "levenshtein":
            # Prefer containment / prefix strongly (AQUARIAN ⊂ AQUARIAN_MANTIS).
            prefix = cand_n.startswith(obs_n) or obs_n.startswith(cand_n)
            if prefix:
                # Length residual only; beat any non-prefix near-miss.
                score = 10.0 - abs(len(cand_n) - len(obs_n)) / maxlen
            elif contained:
                score = 5.0 - float(d) / maxlen
            else:
                score = -float(d)
            parts = {"neg_edit": -float(d), "prior": 0.0, "agree": 0.0, "conf": 0.0}
        else:
            emit = log_emit_prob(obs, cand, model)
            prior = field_prior_log(field, cand)
            agree_b = 0.75 * math.log(max(1e-3, agreement)) if agreement > 0 else 0.0
            conf_b = 0.35 * math.log(max(1e-3, conf_proxy)) if conf_proxy > 0 else 0.0
            if method == "weighted_confusion":
                score = emit + prior
                parts = {
                    "emit": emit,
                    "prior": prior,
                    "agree": 0.0,
                    "conf": 0.0,
                }
            else:  # conf_aware
                score = emit + prior + agree_b + conf_b
                parts = {
                    "emit": emit,
                    "prior": prior,
                    "agree": agree_b,
                    "conf": conf_b,
                }
            # Mild length-normalized edit penalty to break ties.
            score -= 0.05 * d
            prefix = cand_n.startswith(obs_n) or obs_n.startswith(cand_n)
            if prefix:
                score += 4.0
            elif contained:
                score += 1.5
            parts["neg_edit"] = -0.05 * d
        ranked.append((score, cand, parts))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked


def _sigmoid(x: float) -> float:
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class CorrectionDecision:
    value: Any | None
    method: str
    obs: str
    score: float
    margin: float
    n_agree_streams: int
    rejected_reason: str = ""
    parts: dict[str, float] = field(default_factory=dict)
    # R39 non-sensitive evidence metadata (no raw tokens / case IDs shipped).
    mode: str = ""
    n_independent: int = 0
    posterior: float = 0.0
    token_agree: float = 0.0

    def evidence_flags(self) -> dict[str, Any]:
        """Compact repair evidence for in-memory/provenance use."""
        return {
            "mode": self.mode or "",
            "n_ind": int(self.n_independent or self.n_agree_streams),
            "margin": round(float(self.margin), 4),
            "posterior": round(float(self.posterior), 4),
            "token_agree": round(float(self.token_agree), 4),
            "method": self.method or "",
        }


def _unique_with_margin(
    ranked: list[tuple[float, str, dict[str, float]]],
    *,
    method: str,
) -> tuple[str | None, float, float, dict[str, float]]:
    if not ranked:
        return None, -1e9, 0.0, {}
    best_s, best_v, best_p = ranked[0]
    if len(ranked) == 1:
        return best_v, best_s, 999.0, best_p
    second = ranked[1][0]
    margin = best_s - second
    if method == "levenshtein":
        # scores are -edit; margin in edit units
        if margin < _MIN_MARGIN_LEV - 1e-9:
            return None, best_s, margin, best_p
    else:
        if margin < _MIN_MARGIN_LOGP:
            return None, best_s, margin, best_p
    return best_v, best_s, margin, best_p


def propose_correction(
    field: str,
    tokens: list[ObservedToken],
    model: ConfusionModel,
    *,
    method: str,
    current: Any,
    current_source: str = "",
    mode: str = "fill_missing",  # fill_missing | wrong_nonempty
) -> CorrectionDecision:
    """Propose a closed-vocab correction under method + gates."""
    if not tokens:
        return CorrectionDecision(None, method, "", -1e9, 0.0, 0, "no_observed_tokens")
    labeled = [t for t in tokens if t.has_label]
    if not labeled:
        return CorrectionDecision(None, method, "", -1e9, 0.0, 0, "no_explicit_label")

    # Trusted native/manual never overridden.
    src = (current_source or "").lower()
    if src.startswith("native") or "manual" in src:
        if not is_missing_or_placeholder(field, current):
            return CorrectionDecision(
                None, method, "", -1e9, 0.0, 0, "trusted_native_or_manual"
            )

    missing = is_missing_or_placeholder(field, current)
    current_allow = is_allowlisted(field, current)

    if mode == "fill_missing":
        # Missing/placeholder OR non-allowlisted garbled current.
        if not missing and current_allow:
            return CorrectionDecision(
                None, method, "", -1e9, 0.0, 0, "already_valid_real_word"
            )
    elif mode == "wrong_nonempty":
        # Real-word errors only: current is a different allowlisted value.
        if missing or not current_allow:
            return CorrectionDecision(
                None, method, "", -1e9, 0.0, 0, "not_real_word_error"
            )
    else:
        return CorrectionDecision(None, method, "", -1e9, 0.0, 0, "bad_mode")

    # Aggregate by normalized obs raw; count agreeing streams.
    by_raw: dict[str, list[ObservedToken]] = defaultdict(list)
    for t in labeled:
        by_raw[t.raw.casefold()].append(t)
    n_labeled_streams = len({t.stream for t in labeled})

    best_dec: CorrectionDecision | None = None
    for raw_key, group in by_raw.items():
        obs = group[0].raw
        streams = {t.stream for t in group}
        n_agree = len(streams)
        agreement = n_agree / max(1, len(STREAM_NAMES))
        token_agree = n_agree / max(1, n_labeled_streams)
        conf_proxy = max(t.mean_conf_proxy for t in group)
        # For wrong-nonempty real-word corrections, require cross-source agreement.
        if mode == "wrong_nonempty" and n_agree < 2:
            continue
        # Conf-aware abstention: require minimum evidence.
        if method == "conf_aware" and conf_proxy < _MIN_ABS_CONF and n_agree < 2:
            continue
        # Risk atoms: require stronger string proximity (avoid loose garble→atom).
        if field == "risk_flags":
            obs_c = obs.casefold().replace("_", "")
            # Defer until after ranking; filter ranked list below.

        ranked = score_candidates(
            obs,
            field,
            model,
            method=method,
            agreement=agreement,
            conf_proxy=conf_proxy,
        )
        if field == "risk_flags":
            obs_c = obs.casefold().replace("_", "")
            ranked = [
                r
                for r in ranked
                if levenshtein(obs_c, r[1].casefold().replace("_", ""))
                <= max(4, len(r[1]) // 3)
            ]
        val, score, margin, parts = _unique_with_margin(ranked, method=method)
        if val is None:
            continue
        # Do not "correct" to the same current allowlisted value.
        if not missing and str(current) == val:
            continue
        # Real-word: must differ from current and have agreement.
        if mode == "wrong_nonempty" and str(current) == val:
            continue
        # Exact allowlisted token in stream — fine for fill.
        dec = CorrectionDecision(
            value=val if field != "risk_flags" else [val],
            method=method,
            obs=obs,
            score=score,
            margin=margin,
            n_agree_streams=n_agree,
            rejected_reason="",
            parts=parts,
            mode=mode,
            n_independent=n_agree,
            posterior=_sigmoid(margin),
            token_agree=token_agree,
        )
        if best_dec is None or (
            (dec.n_agree_streams, dec.margin, dec.score)
            > (best_dec.n_agree_streams, best_dec.margin, best_dec.score)
        ):
            best_dec = dec

    if best_dec is None:
        reason = (
            "insufficient_cross_source"
            if mode == "wrong_nonempty"
            else "no_unique_margin_candidate"
        )
        if method == "conf_aware":
            reason = "conf_aware_abstain_or_" + reason
        return CorrectionDecision(
            None, method, "", -1e9, 0.0, 0, reason, mode=mode
        )
    return best_dec


def values_match(field: str, pred: Any, truth: Any) -> bool:
    if pred in (None, "", []):
        return False
    if field == "risk_flags":
        p = set(pred if isinstance(pred, (list, tuple, set)) else [pred])
        t = set(truth if isinstance(truth, (list, tuple, set)) else [truth])
        # Incremental atom recovery: count TP if predicted atom ∈ truth
        # and was missing from current — handled by caller; here exact set.
        return p == t
    return str(pred) == str(truth)


def risk_atom_match(pred: Any, truth: Any) -> bool:
    """True if predicted atom list is non-empty subset of truth (fill-safe)."""
    if not pred:
        return False
    p = set(pred if isinstance(pred, (list, tuple, set)) else [pred])
    t = set(truth if isinstance(truth, (list, tuple, set)) else [truth])
    return bool(p) and p.issubset(t)


def incremental_outcome(
    *,
    proposed: Any,
    truth: Any,
    current: Any,
    field: str,
    fill_ok: bool,
    control: bool,
) -> str:
    if control:
        if proposed in (None, "", []):
            return "control_rejected"
        if field == "risk_flags":
            ok = risk_atom_match(proposed, truth) or values_match(field, proposed, truth)
        else:
            ok = values_match(field, proposed, truth)
        return "control_correct_blocked" if ok else "control_hallucination"
    if not fill_ok or proposed in (None, "", []):
        return "no_fill"
    if field == "risk_flags":
        # Fill missing atoms: TP if new atoms ⊆ truth and improve recall.
        cur = set(current or [])
        prop = set(proposed if isinstance(proposed, (list, tuple, set)) else [proposed])
        t = set(truth or [])
        if not prop:
            return "no_fill"
        if prop.issubset(t) and prop - cur:
            return "inc_tp"
        if prop - t:
            return "inc_fp"
        return "no_fill"
    if values_match(field, proposed, truth):
        return "inc_tp"
    return "inc_fp"


@lru_cache(maxsize=1)
def load_confusion_model() -> ConfusionModel:
    """Generic character-error prior.

    The cross-fitted rate table this used to load carried no training manifest,
    so it was dropped when every learned artifact was refitted on DEV800.  The
    prior below is the fallback that shipped in its place; it is stated here
    rather than reached by a missing-file branch.
    """
    return ConfusionModel(
        n_pairs=0, n_aligned=100, n_match=90, n_sub=5, n_ins=3, n_del=2
    )


def _packet_current(packet: "ParsedPacket", field: str) -> Any:
    if field == "risk_flags":
        return list(packet.risk_flags or [])
    return (packet.fields.get(field) or "").strip()


def _packet_source(packet: "ParsedPacket", field: str) -> str:
    if field == "risk_flags":
        return packet.field_sources.get("biometric_flags") or packet.field_sources.get(
            "risk_flags", ""
        )
    return packet.field_sources.get(field, "")


def should_clear_untrusted(
    evidence: dict[str, Any],
    policy: str,
    *,
    high_margin: float = _HIGH_MARGIN,
    high_posterior: float = _HIGH_POSTERIOR,
) -> bool:
    """Whether R39 should discard a repaired field from untrusted_conflicts.

    Never clears protected demographics (name/sponsor/date) or unrepaired fields
    (caller must only pass repaired CLOSED_FIELDS evidence). Never clears on the
    strength of trusted native/manual (those repairs are rejected upstream).
    """
    pol = (policy or "C").upper()
    if pol == "C":
        return False
    mode = str(evidence.get("mode") or "")
    n_ind = int(evidence.get("n_ind") or 0)
    margin = float(evidence.get("margin") or 0.0)
    posterior = float(evidence.get("posterior") or 0.0)
    if mode == "wrong_nonempty" and n_ind >= 2:
        return pol in ("A", "B")
    if (
        pol == "B"
        and mode == "fill_missing"
        and n_ind >= 2
        and margin >= high_margin
        and posterior >= high_posterior
    ):
        return True
    return False


def reconcile_untrusted_after_repairs(
    packet: "ParsedPacket",
    *,
    policy: str = _DEFAULT_RECONCILE_POLICY,
    high_margin: float = _HIGH_MARGIN,
    high_posterior: float = _HIGH_POSTERIOR,
) -> list[str]:
    """Clear untrusted_conflicts for repaired closed fields under policy.

    Does not mutate field values. Skips protected name/sponsor/date and any
    field lacking r35_repair_evidence. Trusted ``conflicts`` from native/manual
    disagreement are left untouched here (R35 never overrides those sources).
    """
    cleared: list[str] = []
    evidence_map = getattr(packet, "r35_repair_evidence", None) or {}
    for field_name, evidence in list(evidence_map.items()):
        if field_name in _PROTECTED_FROM_RECONCILE:
            continue
        if field_name not in CLOSED_FIELDS:
            continue
        if field_name not in packet.untrusted_conflicts:
            evidence["cleared_untrusted"] = False
            continue
        # Refuse to clear if current provenance is trusted native/manual.
        src = _packet_source(packet, field_name).lower()
        if src.startswith("native") or "manual" in src:
            evidence["cleared_untrusted"] = False
            continue
        if should_clear_untrusted(
            evidence,
            policy,
            high_margin=high_margin,
            high_posterior=high_posterior,
        ):
            packet.untrusted_conflicts.discard(field_name)
            evidence["cleared_untrusted"] = True
            cleared.append(field_name)
        else:
            evidence["cleared_untrusted"] = False
    return cleared


def apply_noisy_channel_repairs(
    packet: "ParsedPacket",
    sources: "TextSources",
    *,
    model: ConfusionModel | None = None,
    method: str = "weighted_confusion",
    reconcile_policy: str = _DEFAULT_RECONCILE_POLICY,
    high_margin: float = _HIGH_MARGIN,
    high_posterior: float = _HIGH_POSTERIOR,
) -> list[str]:
    """Apply gated closed-vocab repairs. Returns list of changed fields.

    Fill missing/placeholder/garble first; wrong-nonempty real-word only with
    cross-source agreement (conf_aware gate). Never overrides native/manual.

    R39: stores compact repair evidence flags and reconciles untrusted_conflicts
    under ``reconcile_policy`` (A/B/C). Field values are unchanged by reconcile.
    """
    model = model or load_confusion_model()
    changed: list[str] = []
    if not hasattr(packet, "r35_repair_evidence") or packet.r35_repair_evidence is None:
        packet.r35_repair_evidence = {}
    for field_name in CLOSED_FIELDS:
        tokens = collect_stream_tokens(sources, field_name)
        if not tokens:
            continue
        current = _packet_current(packet, field_name)
        src = _packet_source(packet, field_name)

        if "manual" in str(src).lower() or (
            field_name == "fee_status"
            and str(src).startswith("fee_ledger:")
        ):
            # An explicit unknown from a manual note/receipt ledger is evidence,
            # not a missing slot.  Protect it before either proposal mode.
            continue

        # A document-role semantic (for example a visa explicitly stated in a
        # sponsor letter) is stronger than the generic token consensus below.
        # Letting this stage overwrite it collapses the very source distinction
        # that the structural OCR pass recovered.
        if str(src).startswith("ppocr_semantic:"):
            continue

        dec = propose_correction(
            field_name,
            tokens,
            model,
            method=method,
            current=current,
            current_source=src,
            mode="fill_missing",
        )
        if dec.value is None:
            # Real-word override path (independent evidence required).
            dec = propose_correction(
                field_name,
                tokens,
                model,
                method="conf_aware",
                current=current,
                current_source=src,
                mode="wrong_nonempty",
            )
        if dec.value is None:
            continue

        if field_name == "risk_flags":
            atoms = dec.value if isinstance(dec.value, list) else [dec.value]
            merged = sorted(set(packet.risk_flags or []) | set(atoms))
            if merged != sorted(packet.risk_flags or []):
                packet.risk_flags = merged
                packet.explicit_risk_flags = sorted(
                    set(packet.explicit_risk_flags or []) | set(atoms)
                )
                packet.field_sources["risk_flags"] = f"r35_nc:{field_name}"
                if "biometric" not in packet.sources_seen:
                    packet.sources_seen.add("intake")
                packet.r35_repair_evidence[field_name] = dec.evidence_flags()
                packet.conflicts.discard(field_name)
                changed.append(field_name)
            continue

        new_val = str(dec.value)
        old = (packet.fields.get(field_name) or "").strip()
        if new_val == old:
            continue
        packet.fields[field_name] = new_val
        packet.field_sources[field_name] = f"r35_nc:{field_name}"
        packet.conflicts.discard(field_name)
        packet.r35_repair_evidence[field_name] = dec.evidence_flags()
        # Garble replacement may clear unknown-ish state.
        if field_name in ("species_code", "home_world", "visa_class", "declared_purpose"):
            packet.sources_seen.add("intake")
        elif field_name == "fee_status":
            packet.sources_seen.add("fee")
        changed.append(field_name)

    reconcile_untrusted_after_repairs(
        packet,
        policy=reconcile_policy,
        high_margin=high_margin,
        high_posterior=high_posterior,
    )
    return changed


def packet_needs_noisy_channel(packet: "ParsedPacket") -> bool:
    """Cheap gate: any closed field missing/placeholder or non-allowlisted."""
    for field_name in CLOSED_FIELDS:
        cur = _packet_current(packet, field_name)
        src = _packet_source(packet, field_name)
        if str(src).startswith("native") or "manual" in str(src).lower():
            continue
        if is_missing_or_placeholder(field_name, cur):
            return True
        if field_name != "risk_flags" and cur and not is_allowlisted(field_name, cur):
            return True
    return False
