"""Round 22: provenance-isolated tessdata_best OCR fallback.

Uses official Apache-2.0 float LSTM `eng.traineddata` from
tesseract-ocr/tessdata_best via an isolated `--tessdata-dir` + OEM 1.
Never modifies system tessdata. Never concatenates into trusted text;
callers may only fill missing/placeholder fields after current/R21 streams.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .biometric_header_ocr import decode_biometric_flags, has_observed_flags_label
from .ocr_cache import ocr_cache_dir, ocr_code_version
from .parse_fields import (
    extract_allowlisted_visas,
    extract_fuzzy_fee_status,
    extract_label_proximate_date,
    extract_label_proximate_purpose,
    extract_label_proximate_visas,
    strip_injection,
)

_DPI = 220
_PSMS = ("4", "6", "11")
_OEM = "1"
_TESSDATA_DIR = Path(__file__).resolve().parents[1] / "tessdata_best"
_EXPECTED_SHA256 = (
    "8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba"
)

_VISA_LABEL_RE = re.compile(r"(?i)\bvisa\b")
_PURPOSE_LABEL_RE = re.compile(r"(?i)\bpurpose|\bdeclared")


def tessdata_best_available() -> bool:
    eng = _TESSDATA_DIR / "eng.traineddata"
    if not eng.is_file():
        return False
    # Cheap size gate; full sha verified in tests / once at cache build.
    return eng.stat().st_size == 15_400_601


def verify_tessdata_best_sha256() -> str:
    eng = _TESSDATA_DIR / "eng.traineddata"
    digest = hashlib.sha256(eng.read_bytes()).hexdigest()
    if digest != _EXPECTED_SHA256:
        raise ValueError(f"tessdata_best sha256 mismatch: {digest}")
    return digest


def _structure_hits(text: str) -> int:
    from .text_extract import _structure_hits as hits

    return hits(text)


def _tesseract_best(image: Path, psm: str) -> str:
    """OEM1 + isolated tessdata_best dir; default thresholding."""
    if not tessdata_best_available():
        return ""
    cmd = [
        "tesseract",
        str(image),
        "stdout",
        "-l",
        "eng",
        "--oem",
        _OEM,
        "--psm",
        psm,
        "--dpi",
        str(_DPI),
        "--tessdata-dir",
        str(_TESSDATA_DIR),
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=90
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return ""


def _best_for_image(image: Path) -> str:
    best = ""
    for psm in _PSMS:
        cand = _tesseract_best(image, psm)
        if _structure_hits(cand) > _structure_hits(best):
            best = cand
        elif not best:
            best = cand
    return best


def packet_needs_best_fallback(packet, sources) -> bool:
    """Post-R21 gate: unresolved purpose or Observed-flags risk only.

    Cohort grid: best_default added high-precision purpose + risk TPs vs
    std_sauvola; fee/name were FP-heavy and are never filled from this stream.
    """
    if not tessdata_best_available():
        return False
    page = getattr(sources, "page_ocr", "") or ""
    emb = getattr(sources, "embedded_ocr", "") or ""
    # Prefer evidence from existing OCR; also accept threshold stream labels.
    thr = getattr(sources, "threshold_ocr", "") or ""
    if not (page.strip() or emb.strip() or thr.strip()):
        return False
    cleaned, _ = strip_injection(f"{page}\n{emb}\n{thr}")
    fields = getattr(packet, "fields", {}) or {}

    purpose = fields.get("declared_purpose")
    if purpose is None or purpose == "" or purpose == "unknown":
        if _PURPOSE_LABEL_RE.search(cleaned) and not extract_label_proximate_purpose(
            cleaned
        ):
            return True

    if not getattr(packet, "risk_flags", None):
        if has_observed_flags_label(cleaned) and not decode_biometric_flags(cleaned):
            if not re.search(
                r"(?i)observ\w{0,4}\s+flags?\s*[:\-]?\s*(none|n/?a|null|clear)\b",
                cleaned,
            ):
                return True
    return False


def best_ocr_cached(pdf_path: Path) -> str:
    """Cached tessdata_best OCR text for one PDF (isolated stream)."""
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"v1-best:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{pdf_path.stem}_{key[:16]}_best.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload.get("best_ocr", "")
    text = ocr_best_text_for_pdf(pdf_path)
    cache_path.write_text(json.dumps({"best_ocr": text}), encoding="utf-8")
    return text


def ocr_best_text_for_pdf(pdf_path: Path, dpi: int = _DPI) -> str:
    """Rasterize pages + large embeds; return tessdata_best OCR (isolated)."""
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return ""
    if not tessdata_best_available():
        return ""
    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mib-r22-") as tmp:
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
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return ""
        images = list(sorted(tmp_path.glob("page-*.png")))
        if shutil.which("pdfimages"):
            emb = tmp_path / "emb"
            try:
                subprocess.run(
                    ["pdfimages", "-png", str(pdf_path), str(emb)],
                    check=True,
                    capture_output=True,
                )
                for image in sorted(tmp_path.glob("emb-*.png")):
                    if image.stat().st_size >= 20_000:
                        images.append(image)
            except (subprocess.CalledProcessError, OSError):
                pass
        for image in images:
            text = _best_for_image(image)
            if text.strip():
                parts.append(text)
    return "\n".join(parts)


def extract_constrained_best_fields(text: str) -> dict[str, object]:
    """Closed-vocab / label-anchored fields from tessdata_best stream.

    High-precision allowlist from DEV grid: purpose + Observed-flags risk.
    Visa/date/fee parsed for diagnostics but fee/name/sponsor are not applied
    by the fill path (FP-heavy on cohort).
    """
    cleaned, _ = strip_injection(text or "")
    out: dict[str, object] = {}
    purpose = extract_label_proximate_purpose(cleaned)
    if purpose:
        out["declared_purpose"] = purpose
    # Risk flags: Observed-flags anchor required.
    flags = decode_biometric_flags(cleaned)
    if flags:
        out["risk_flags"] = flags
    # Optional high-precision closed fields (fill path may ignore some).
    visas = extract_label_proximate_visas(cleaned)
    if not visas:
        for window in re.finditer(
            r"(?i)Visa\s+Cla[a-z]{0,4}\s*[:\-]?\s*([A-Z0-9\.\-]{3,12})",
            cleaned,
        ):
            visas.extend(
                extract_allowlisted_visas(window.group(1), punct_norm=True)
            )
        visas = list(dict.fromkeys(visas))
    if len(visas) == 1:
        out["visa_class"] = visas[0]
    elif not visas and _VISA_LABEL_RE.search(cleaned):
        whole = extract_allowlisted_visas(cleaned, punct_norm=True)
        if len(whole) == 1:
            out["visa_class"] = whole[0]
    date = extract_label_proximate_date(cleaned)
    if date:
        out["arrival_date"] = date
    fee = extract_fuzzy_fee_status(cleaned)
    if fee and fee != "unknown":
        out["fee_status"] = fee
    return out
