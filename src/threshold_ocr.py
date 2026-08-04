"""Round 21: provenance-isolated Tesseract Sauvola thresholding fallback.

Uses Tesseract internal thresholding_method=2 (Sauvola) on the same raster
DPI/PSM path as page OCR. Never concatenates into trusted text; callers may
only fill missing/placeholder fields via constrained parsers.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .biometric_header_ocr import decode_biometric_flags, has_observed_flags_label
from .ocr_cache import ocr_cache_dir, ocr_code_version
from .parse_fields import (
    SPONSOR_ANYWHERE_RE,
    extract_allowlisted_visas,
    extract_fuzzy_fee_status,
    extract_label_proximate_date,
    extract_label_proximate_purpose,
    extract_label_proximate_visas,
    strip_injection,
)

# Cohort-selected Sauvola params (DEV grid; high-precision incremental visas).
_SAUVOLA_K = "0.25"
_SAUVOLA_W = "0.25"
_DPI = 220
_PSMS = ("4", "6", "11")

_VISA_LABEL_RE = re.compile(r"(?i)\bvisa\b")
_FEE_LABEL_RE = re.compile(r"(?i)\bfee\s*[,\.]?\s*stat|\bfee\b\s*[:\-]")
_PURPOSE_LABEL_RE = re.compile(r"(?i)\bpurpose|\bdeclared")
_ARRIVAL_LABEL_RE = re.compile(r"(?i)\barriv|\banwval|\barnval")
_SPONSOR_LABEL_RE = re.compile(r"(?i)\bsponsor")


def _structure_hits(text: str) -> int:
    from .text_extract import _structure_hits as hits

    return hits(text)


def _tesseract_sauvola(image: Path, psm: str) -> str:
    cmd = [
        "tesseract",
        str(image),
        "stdout",
        "-l",
        "eng",
        "--psm",
        psm,
        "--dpi",
        str(_DPI),
        "-c",
        "thresholding_method=2",
        "-c",
        f"thresholding_kfactor={_SAUVOLA_K}",
        "-c",
        f"thresholding_window_size={_SAUVOLA_W}",
    ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=45
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return ""


def _best_sauvola_for_image(image: Path) -> str:
    """Sauvola OCR over the same PSM set as page OCR; keep best structure hits."""
    best = ""
    for psm in _PSMS:
        cand = _tesseract_sauvola(image, psm)
        if _structure_hits(cand) > _structure_hits(best):
            best = cand
        elif not best:
            best = cand
    return best


def should_run_threshold_ocr(page_ocr: str, embedded_ocr: str) -> bool:
    """Cheap text-only gate: OCR present with an unresolved labeled field."""
    combined = f"{page_ocr or ''}\n{embedded_ocr or ''}"
    if len(combined.strip()) < 20:
        return False
    cleaned, _ = strip_injection(combined)

    if _VISA_LABEL_RE.search(cleaned):
        visas = extract_label_proximate_visas(cleaned) or extract_allowlisted_visas(
            cleaned, punct_norm=True
        )
        if not visas:
            return True
    if _FEE_LABEL_RE.search(cleaned):
        if not extract_fuzzy_fee_status(cleaned):
            # Obscured administrative blanks are not recoverable — skip those.
            if not re.search(r"(?i)\[?\s*FEE STATUS OBSCURED\s*\]?", cleaned):
                return True
    if _PURPOSE_LABEL_RE.search(cleaned):
        if not extract_label_proximate_purpose(cleaned):
            return True
    if _ARRIVAL_LABEL_RE.search(cleaned):
        if not extract_label_proximate_date(cleaned):
            return True
    if _SPONSOR_LABEL_RE.search(cleaned):
        # Label-proximate SPN only.
        hit = False
        for m in SPONSOR_ANYWHERE_RE.finditer(cleaned):
            ctx = cleaned[max(0, m.start() - 40) : m.end() + 5]
            if re.search(r"(?i)sponsor", ctx):
                hit = True
                break
        if not hit:
            return True
    # Risk: Observed-flags-ish label without decoded allowlisted flags.
    if has_observed_flags_label(cleaned) and not decode_biometric_flags(cleaned):
        return True
    return False


def packet_needs_threshold_fallback(packet, sources) -> bool:
    """Post-fusion gate for high-precision Sauvola recoveries only.

    Cohort grid showed incremental TPs mainly for truncated visas, garbled
    purpose labels, and Observed-flags pages — not fee-unknown alone (high FP).
    """
    page = getattr(sources, "page_ocr", "") or ""
    emb = getattr(sources, "embedded_ocr", "") or ""
    if not (page.strip() or emb.strip()):
        return False
    cleaned, _ = strip_injection(f"{page}\n{emb}")
    fields = getattr(packet, "fields", {}) or {}

    visa = fields.get("visa_class")
    if visa is None or visa == "" or visa == "unknown":
        if _VISA_LABEL_RE.search(cleaned):
            visas = extract_label_proximate_visas(cleaned) or extract_allowlisted_visas(
                cleaned, punct_norm=True
            )
            if not visas:
                return True

    purpose = fields.get("declared_purpose")
    if purpose is None or purpose == "" or purpose == "unknown":
        if _PURPOSE_LABEL_RE.search(cleaned) and not extract_label_proximate_purpose(
            cleaned
        ):
            return True

    if not getattr(packet, "risk_flags", None):
        # Only when Observed-flags looks nonempty but failed closed-set decode.
        if has_observed_flags_label(cleaned) and not decode_biometric_flags(cleaned):
            if not re.search(
                r"(?i)observ\w{0,4}\s+flags?\s*[:\-]?\s*(none|n/?a|null|clear)\b",
                cleaned,
            ):
                return True

    return False


def threshold_ocr_cached(pdf_path: Path) -> str:
    """Cached Sauvola OCR text for one PDF (isolated stream)."""
    import hashlib
    import json

    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"v5-sauvola:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{pdf_path.stem}_{key[:16]}_sauvola.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload.get("threshold_ocr", "")
    text = ocr_threshold_text_for_pdf(pdf_path)
    cache_path.write_text(
        json.dumps({"threshold_ocr": text}), encoding="utf-8"
    )
    return text


def ocr_threshold_text_for_pdf(pdf_path: Path, dpi: int = _DPI) -> str:
    """Rasterize pages + large embeds; return Sauvola OCR text (isolated)."""
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return ""
    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mib-r21-") as tmp:
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
            text = _best_sauvola_for_image(image)
            if text.strip():
                parts.append(text)
    return "\n".join(parts)


def extract_constrained_threshold_fields(text: str) -> dict[str, object]:
    """Closed-vocab / label-anchored fields from a threshold OCR stream."""
    cleaned, _ = strip_injection(text or "")
    out: dict[str, object] = {}
    fee = extract_fuzzy_fee_status(cleaned)
    if fee and fee != "unknown":
        out["fee_status"] = fee
    date = extract_label_proximate_date(cleaned)
    if date:
        out["arrival_date"] = date
    # Visa: OCR-tolerant label window, else unique allowlisted token when a
    # Visa label is evidenced (handles "Visa Clase: TRANSIT-7").
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
    elif not visas and re.search(r"(?i)\bvisa\b", cleaned):
        whole = extract_allowlisted_visas(cleaned, punct_norm=True)
        if len(whole) == 1:
            out["visa_class"] = whole[0]
    purpose = extract_label_proximate_purpose(cleaned)
    if purpose:
        out["declared_purpose"] = purpose
    # Risk flags: Observed-flags anchor required.
    flags = decode_biometric_flags(cleaned)
    if flags:
        out["risk_flags"] = flags
    return out
