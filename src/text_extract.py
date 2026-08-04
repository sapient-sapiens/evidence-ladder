from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .ocr_cache import ocr_cache_dir, ocr_code_version

STRUCTURE_LABEL_RE = re.compile(
    r"(Species Code|Visa Class|Fee Status|Sponsor ID|Arrival Date|"
    r"Declared Purpose|Purpose|Home World|Applicant|Registry Name|Observed flags)",
    re.IGNORECASE,
)

_CACHE_VERSION = "v5-budget"

# Number of distinct structural labels a fully legible intake page exposes.
_FULL_STRUCTURE_HITS = 7


@dataclass(frozen=True)
class TextSources:
    """Text streams ordered from highest to lowest trust."""

    native: str
    page_ocr: str = ""
    embedded_ocr: str = ""
    # Provenance-isolated flags from specialized biometric-header OCR only.
    biometric_flags: tuple[str, ...] = ()
    # Provenance-isolated Sauvola-threshold OCR (R21); never trusted-text concat.
    threshold_ocr: str = ""
    # Provenance-isolated tessdata_best OCR (R22); never trusted-text concat.
    best_ocr: str = ""
    # Provenance-isolated full-page PP-OCRv5 / RapidOCR (R45); fusion fill-only.
    ppocr_ocr: str = ""
    # Orientation-normalized degraded-form OCR; lowest trust and fill-only.
    oriented_ocr: str = ""

    def combined(self) -> str:
        return "\n\n".join(
            text for text in (self.native, self.page_ocr, self.embedded_ocr) if text
        )


def extract_text(pdf_path: Path, *, allow_ocr: bool = True) -> TextSources:
    """Native PDF text, with OCR fallback when structured labels are incomplete."""
    native = _extract_native(pdf_path)
    if not allow_ocr:
        return TextSources(native=native)
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return TextSources(native=native)

    # Rasterizing a packet whose text layer is already complete cannot recover a
    # document role the packet does not contain.  A missing ``Observed flags``
    # line usually means there is no biometric slip at all, so it used to send a
    # quarter of all packets through a full OCR pass that had nothing to find.
    needs_ocr = _structure_hits(native) < 5 or not _has_fee_and_visa(native)
    if not needs_ocr:
        return TextSources(native=native)

    page_ocr, embedded_ocr, biometric_flags = _ocr_pdf_cached(pdf_path, native=native)
    return TextSources(
        native=native,
        page_ocr=page_ocr,
        embedded_ocr=embedded_ocr,
        biometric_flags=biometric_flags,
    )


def _has_fee_and_visa(text: str) -> bool:
    lower = (text or "").casefold()
    return ("fee status" in lower) and ("visa class" in lower)


def _structure_hits(text: str) -> int:
    return len(STRUCTURE_LABEL_RE.findall(text or ""))


def _extract_native(pdf_path: Path) -> str:
    if shutil.which("pdftotext"):
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout or ""
        except (subprocess.CalledProcessError, OSError):
            pass
    return _extract_with_pypdf(pdf_path)


def _extract_with_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pdftotext not found and pypdf is not installed; "
            "pip install -r baselines/mvp/requirements.txt"
        ) from exc

    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _cache_path_for(pdf_path: Path, version: str) -> Path:
    key = hashlib.sha1(
        f"{version}:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    return ocr_cache_dir() / f"{pdf_path.stem}_{key[:16]}.json"


def _ocr_pdf_cached(
    pdf_path: Path, *, native: str = ""
) -> tuple[str, str, tuple[str, ...]]:
    import json

    from .biometric_header_ocr import (
        ocr_biometric_flags_for_pdf,
        should_run_biometric_header,
    )

    cache_path = _cache_path_for(pdf_path, _CACHE_VERSION)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        flags = tuple(payload.get("biometric_flags") or ())
        return (
            payload.get("page_ocr", ""),
            payload.get("embedded_ocr", ""),
            flags,
        )

    page_ocr, embedded_ocr = _ocr_pdf(pdf_path, native=native)

    biometric_flags: tuple[str, ...] = ()
    if should_run_biometric_header(native, page_ocr, embedded_ocr):
        biometric_flags = tuple(ocr_biometric_flags_for_pdf(pdf_path))

    cache_path.write_text(
        json.dumps(
            {
                "page_ocr": page_ocr,
                "embedded_ocr": embedded_ocr,
                "biometric_flags": list(biometric_flags),
            }
        ),
        encoding="utf-8",
    )
    return page_ocr, embedded_ocr, biometric_flags


def _tesseract(image: Path, psm: str) -> str:
    try:
        result = subprocess.run(
            ["tesseract", str(image), "stdout", "-l", "eng", "--psm", psm],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, OSError):
        return ""


def _best_ocr_text(image: Path) -> str:
    """Read one image with the two complementary Tesseract layout modes.

    ``--psm 6`` was measured never to win outright against the column-aware and
    sparse modes on these forms, so it is not worth a third of the page budget.
    A page whose column mode already exposes the full label set is accepted
    immediately; only weak reads pay for the sparse pass.
    """
    best = _tesseract(image, "4")
    if _structure_hits(best) >= _FULL_STRUCTURE_HITS:
        return best
    candidate = _tesseract(image, "11")
    if _structure_hits(candidate) > _structure_hits(best) or not best:
        best = candidate or best
    return best


def _ocr_pdf(pdf_path: Path, dpi: int = 220, *, native: str = "") -> tuple[str, str]:
    """Rasterize pages, then OCR embedded images only for unresolved packets.

    Re-reading the original embedded rasters is the single most expensive step
    in the pass and it supplies a field value for well under one packet in
    twenty.  Once the rendered pages plus the native layer already expose the
    full structural label set, that spend buys nothing, so it is gated on the
    packet still looking incomplete.
    """
    page_parts: list[str] = []
    embedded_parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="mib-ocr-") as tmp:
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
            pass

        for image in sorted(tmp_path.glob("page-*.png")):
            page_text = _best_ocr_text(image)
            if page_text.strip():
                page_parts.append(page_text)

        # Embedded images (scanned letters / slips) often hold Fee/Visa fields.
        rendered = "\n".join(page_parts)
        resolved = _structure_hits(f"{native}\n{rendered}") >= _FULL_STRUCTURE_HITS and (
            _has_fee_and_visa(f"{native}\n{rendered}")
        )
        if not resolved and shutil.which("pdfimages"):
            img_prefix = tmp_path / "emb"
            try:
                subprocess.run(
                    ["pdfimages", "-png", str(pdf_path), str(img_prefix)],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, OSError):
                pass
            for image in sorted(tmp_path.glob("emb-*.png")):
                # Skip tiny icons; keep letter-sized / portrait-sized embeds.
                if image.stat().st_size < 20_000:
                    continue
                emb_text = _best_ocr_text(image)
                if _structure_hits(emb_text) > 0 or re.search(
                    r"\b(paid|waived|unpaid|SPN-\d{4}|APPROVED|DENIED)\b",
                    emb_text,
                    re.I,
                ):
                    embedded_parts.append(emb_text)
    return "\n".join(page_parts), "\n".join(embedded_parts)
