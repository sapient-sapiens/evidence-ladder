"""Self-supervised page-quality probe via known-plaintext case-id CER.

The printed ``MIB-######`` string is compared to OCR mentions of the same
pattern.  Only the resulting continuous CER is used — as a stream reliability
weight / escalation gate.  Case-id tokens never enter adjudication features or
label shortcuts (AGENTS identity-free contract; Exp2 exception).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.noisy_channel_ocr import levenshtein

_CASE_ID_RE = re.compile(r"\bMIB-\d{6}\b", re.IGNORECASE)


def char_error_rate(expected: str, observed: str) -> float:
    exp = (expected or "").strip().upper()
    obs = (observed or "").strip().upper()
    if not exp:
        return 1.0
    if not obs:
        return 1.0
    return levenshtein(exp, obs) / max(len(exp), 1)


def stream_case_id_cer(text: str, case_id: str) -> float | None:
    """Mean CER of OCR case-id mentions vs the expected id, or None if absent."""
    if not text or not case_id:
        return None
    mentions = _CASE_ID_RE.findall(text)
    if not mentions:
        return None
    return sum(char_error_rate(case_id, m) for m in mentions) / len(mentions)


@dataclass(frozen=True)
class PageQuality:
    """Per-stream and aggregate case-id CER (lower is better)."""

    stream_cer: dict[str, float]
    mean_cer: float
    n_streams: int

    def cer_for(self, stream: str) -> float:
        if stream in self.stream_cer:
            return self.stream_cer[stream]
        return self.mean_cer if self.n_streams else 1.0


@lru_cache(maxsize=256)
def page_quality_for_texts(
    case_id: str,
    native: str,
    page_ocr: str,
    embedded_ocr: str,
    threshold_ocr: str = "",
    best_ocr: str = "",
    oriented_ocr: str = "",
) -> PageQuality:
    streams = {
        "native": native,
        "page_ocr": page_ocr,
        "embedded_ocr": embedded_ocr,
        "threshold_ocr": threshold_ocr,
        "best_ocr": best_ocr,
        "oriented_ocr": oriented_ocr,
    }
    cers: dict[str, float] = {}
    for name, text in streams.items():
        cer = stream_case_id_cer(text or "", case_id)
        if cer is not None:
            cers[name] = cer
    if not cers:
        return PageQuality(stream_cer={}, mean_cer=1.0, n_streams=0)
    mean = sum(cers.values()) / len(cers)
    return PageQuality(stream_cer=cers, mean_cer=mean, n_streams=len(cers))


def page_quality_from_sources(case_id: str, sources) -> PageQuality:
    return page_quality_for_texts(
        case_id,
        getattr(sources, "native", "") or "",
        getattr(sources, "page_ocr", "") or "",
        getattr(sources, "embedded_ocr", "") or "",
        getattr(sources, "threshold_ocr", "") or "",
        getattr(sources, "best_ocr", "") or "",
        getattr(sources, "oriented_ocr", "") or "",
    )
