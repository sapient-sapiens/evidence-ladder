"""R46: FORM B-13 Observed-flags band legibility (not token recovery).

Two high-precision outcomes from the biometric header crop + Tesseract TSV:

- ``readable_none``: Observed-flags payload is clearly ``none`` at high word
  confidence → set ``observed_flags_seen`` so clean packets can approve.
- ``illegible``: B-13 header is present, ink exists in the flags band, but
  payload is neither ``none`` nor an allowlisted flag and TSV confidence is
  poor → emit ``illegible_biometrics``.

Fit/thresholds frozen from stratified probe100 (DEV). No case-ID memory.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .biometric_header_ocr import (
    _header_variants,
    _tesseract,
    decode_biometric_flags,
    has_observed_flags_label,
    looks_like_biometric_header,
)
from .constants import RISK_FLAG_ATOMS
from .ocr_cache import ocr_cache_dir, ocr_code_version

_PAGE_DPI = 288
_NONE_RE = re.compile(
    r"(?i)(?:(?:observ\w{0,6}|[a-z]{4,10})\s+)?[ft]l?ags?"
    r"\s*[:\-]?\s*none\b|\b[a-z]{2,20}ag[a-z]{0,3}\s+none\b"
)
# Precision-first thresholds (probe100: readable_none 13/35 AR, 0/30 miss_illeg).
_NONE_MIN_CONF = 70.0
_ILLEG_MAX_CONF = 55.0
_ILLEG_MIN_INK = 0.05
_ILLEG_MIN_LOW_FRAC = 0.35


@dataclass(frozen=True)
class LegibilityResult:
    status: str  # none | readable_none | illegible | flags
    flags: tuple[str, ...] = ()
    mean_conf: float = -1.0
    ink_frac: float = 0.0
    snip: str = ""


def _tsv_stats(image: Path) -> tuple[float, float, int]:
    """Return (mean_conf, low_frac, n_words) from Tesseract TSV."""
    try:
        out = subprocess.run(
            ["tesseract", str(image), "stdout", "-l", "eng", "--psm", "6", "tsv"],
            check=True,
            capture_output=True,
            text=True,
            timeout=25,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return -1.0, 1.0, 0
    confs: list[float] = []
    for line in (out.stdout or "").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            conf = float(parts[10])
            txt = parts[11].strip()
        except ValueError:
            continue
        if conf >= 0 and txt:
            confs.append(conf)
    if not confs:
        return -1.0, 1.0, 0
    mean = sum(confs) / len(confs)
    low = sum(1 for c in confs if c < 50) / len(confs)
    return mean, low, len(confs)


def _ink_frac(image: Path) -> float:
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(image).convert("L"), dtype="float32")
        if arr.size == 0:
            return 0.0
        return float((arr < float(arr.mean())).mean())
    except Exception:  # noqa: BLE001
        return 0.0


def _layout_b13_score(page: Path) -> float:
    """Cheap photo-right / form-left score for B-13-like pages (no OCR)."""
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(page).convert("L"), dtype="float32")
        if arr.size == 0:
            return 0.0
        h, w = arr.shape
        left = arr[:, : int(w * 0.55)]
        right = arr[:, int(w * 0.55) :]
        lstd = float(left.std()) + 1e-6
        rstd = float(right.std())
        ul = arr[: int(h * 0.35), : int(w * 0.6)]
        ul_ink = float((ul < float(ul.mean())).mean())
        # Prefer high right/left contrast with some header ink.
        return (rstd / lstd) * (0.5 + ul_ink)
    except Exception:  # noqa: BLE001
        return 0.0


def inspect_b13_legibility(pdf_path: Path, dpi: int = _PAGE_DPI) -> LegibilityResult:
    """Rasterize PDF and classify Observed-flags band legibility."""
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return LegibilityResult("none")
    with tempfile.TemporaryDirectory(prefix="mib-r46-") as tmp:
        tmp_path = Path(tmp)
        prefix = tmp_path / "page"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-r",
                    str(dpi),
                    "-gray",
                    "-png",
                    "-f",
                    "1",
                    "-l",
                    "6",
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return LegibilityResult("none")
        pages = sorted(tmp_path.glob("page-*.png"))
        # Text-gated only (layout-only illegible over-fired on probe100).
        ranked = sorted(pages, key=_layout_b13_score, reverse=True)
        pages = list(dict.fromkeys(ranked + pages))
        for page in pages:
            try:
                variants = _header_variants(page)
            except Exception:  # noqa: BLE001
                continue
            if not variants:
                continue
            text = _tesseract(variants[0], "6")
            text2 = _tesseract(variants[0], "11")
            if not any(
                looks_like_biometric_header(candidate)
                or has_observed_flags_label(candidate)
                for candidate in (text, text2)
            ):
                continue
            text = f"{text}\n{text2}"
            if len(variants) > 1:
                text = f"{text}\n{_tesseract(variants[1], '6')}"
            if len(variants) > 3:
                text = f"{text}\n{_tesseract(variants[3], '11')}"

            flags = tuple(decode_biometric_flags(text))
            mean_conf, low_frac, _n = _tsv_stats(variants[0])
            ink = _ink_frac(variants[0])
            snip = " ".join(text.split())[:120]
            readable_none = bool(_NONE_RE.search(text)) and not flags

            if flags:
                return LegibilityResult(
                    "flags", flags=flags, mean_conf=mean_conf, ink_frac=ink, snip=snip
                )
            if readable_none:
                return LegibilityResult(
                    "readable_none",
                    mean_conf=mean_conf,
                    ink_frac=ink,
                    snip=snip,
                )
            # Illegible only with text-gated B-13 header (precision-first).
            if ink >= _ILLEG_MIN_INK and (
                mean_conf < 0
                or (
                    mean_conf <= _ILLEG_MAX_CONF
                    and low_frac >= _ILLEG_MIN_LOW_FRAC
                )
            ):
                return LegibilityResult(
                    "illegible",
                    mean_conf=mean_conf,
                    ink_frac=ink,
                    snip=snip,
                )
            return LegibilityResult(
                "none", mean_conf=mean_conf, ink_frac=ink, snip=snip
            )
    return LegibilityResult("none")


def legibility_cached(pdf_path: Path) -> LegibilityResult:
    """Disk-cached legibility inspection."""
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"v10-r46leg-readable-none:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{pdf_path.stem}_{key[:16]}_r46leg.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return LegibilityResult(
                status=payload.get("status", "none"),
                flags=tuple(payload.get("flags") or ()),
                mean_conf=float(payload.get("mean_conf", -1)),
                ink_frac=float(payload.get("ink_frac", 0)),
                snip=payload.get("snip", ""),
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    result = inspect_b13_legibility(pdf_path)
    cache_path.write_text(
        json.dumps(
            {
                "status": result.status,
                "flags": list(result.flags),
                "mean_conf": result.mean_conf,
                "ink_frac": result.ink_frac,
                "snip": result.snip,
            }
        ),
        encoding="utf-8",
    )
    return result


def packet_needs_legibility(packet) -> bool:
    """Gate: run when flags unseen / empty, or review-only uncertainty."""
    if getattr(packet, "risk_flags", None):
        # Already have atoms — only skip unless we still lack observed bit.
        if getattr(packet, "observed_flags_seen", False):
            return False
    # Always useful when Observed section never confirmed.
    if not getattr(packet, "observed_flags_seen", False):
        return True
    return False


def apply_legibility_result(packet, result: LegibilityResult) -> str | None:
    """Mutate packet from legibility result. Returns provenance tag or None.

    Illegible emission is disabled in production apply: review-only flag can
    demote correct DENIED → NEEDS_REVIEW. Keep readable_none (obs_seen unlock)
    and decoded allowlisted flags only. Illegible status remains available for
    offline benchmarks.
    """
    if result.status == "readable_none":
        packet.risk_flags = []
        packet.observed_flags_seen = True
        packet.field_sources["risk_flags"] = "r46_legible_none"
        return "r46_legible_none"
    if result.status == "flags" and result.flags:
        before = set(packet.risk_flags)
        allowed = [f for f in result.flags if f in RISK_FLAG_ATOMS]
        if not allowed:
            return None
        packet.risk_flags = sorted(before | set(allowed))
        packet.observed_flags_seen = True
        packet.field_sources.setdefault("risk_flags", "r46_flags")
        packet.sources_seen.add("biometric")
        return "r46_flags"
    return None
