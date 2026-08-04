"""Specialized OCR for degraded FORM B-13 biometric header risk flags.

Only decodes closed-set risk flags from an upper-left header crop. Does not
emit other fields or free text for global concatenation.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .constants import RISK_FLAG_ATOMS

# Derived from DEV degraded B-13 slips: label + Observed-flags sit in the
# upper-left header band (not the portrait body). Slightly padded vs the
# tightest DEV crops so mild deskew/margins still land inside the window.
_HEADER_W = 0.60
_HEADER_H = 0.32

# Effective ~300–400 DPI from a 220 DPI page render via 1.5–2× upscale.
_PAGE_DPI = 220
_UPSCALE = 2

_OBS_LABEL_RE = re.compile(
    r"\b[a-z0-9]bser\w{0,5}\s+f(?:l|i|1)ag\w{0,2}\b",
    re.IGNORECASE,
)
_BIO_CLUE_RE = re.compile(
    r"(form\s*b[\-\s]*13|b\-?13|biometric|scan\s*slip)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


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
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (
                ca != cb
            )
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def has_observed_flags_label(text: str) -> bool:
    """True when OCR shows an Observed-flags label (modest fuzziness)."""
    return bool(_OBS_LABEL_RE.search(text or ""))


def looks_like_biometric_header(text: str) -> bool:
    t = text or ""
    return bool(_BIO_CLUE_RE.search(t) or has_observed_flags_label(t))


def fuzzy_match_flag(token: str) -> str | None:
    """Map one OCR token onto the closed risk-flag allowlist."""
    atom = token.strip().lower().strip(".,;:|[](){}\"'").replace(" ", "_")
    if not atom or atom in {"none", "n/a", "null", "nil", "clear"}:
        return None
    if atom in RISK_FLAG_ATOMS:
        return atom
    best: str | None = None
    best_d = 99
    for cand in RISK_FLAG_ATOMS:
        dist = _levenshtein(atom, cand)
        # Conservative: 1 edit for shorter atoms, 2 for longer ones.
        thresh = 1 if len(cand) <= 14 else 2
        if dist <= thresh and dist < best_d:
            best, best_d = cand, dist
    if best is not None:
        return best
    # Rotated/faint glyphs commonly confuse il/bl/m/n clusters.  Once an
    # Observed-flags label has gated this decoder, accept only a unique,
    # high-similarity closed-vocabulary winner.
    from difflib import SequenceMatcher

    compact = re.sub(r"[^a-z]", "", atom)
    ranked = sorted(
        (
            SequenceMatcher(None, compact, re.sub(r"[^a-z]", "", cand)).ratio(),
            cand,
        )
        for cand in RISK_FLAG_ATOMS
    )
    if ranked[-1][0] >= 0.64 and ranked[-1][0] - ranked[-2][0] >= 0.10:
        return ranked[-1][1]
    return best


def decode_biometric_flags(text: str) -> list[str]:
    """Decode risk flags only when Observed-flags evidence is present."""
    if not has_observed_flags_label(text):
        return []
    flags: set[str] = set()
    observed_none = False
    # Prefer the payload after the (possibly fuzzy) label.
    for match in _OBS_LABEL_RE.finditer(text):
        raw_tail = text[match.end() : match.end() + 120]
        tail = raw_tail
        # Drop pipe/box-drawing chrome common on degraded scans.
        tail = re.split(r"[\n|]", tail, maxsplit=1)[0]
        if re.search(r"(?i)\b(?:none|nil|clear)\b", tail):
            observed_none = True
        for tok in _TOKEN_RE.findall(tail):
            hit = fuzzy_match_flag(tok)
            if hit:
                flags.add(hit)
        if not flags:
            wrapped = re.split(
                r"(?i)\b(?:CASEWORK|Packet\s+MIB|Applicant|Species\s+Match)\b",
                raw_tail,
                maxsplit=1,
            )[0]
            joined = re.sub(r"[^A-Za-z]", "", "\n".join(wrapped.splitlines()[:2]))
            if 5 <= len(joined) <= 32:
                hit = fuzzy_match_flag(joined)
                if hit:
                    flags.add(hit)
    return sorted(flags)


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


def _otsu_threshold(gray):
    """Return a bilevel image via Otsu (Pillow-only)."""
    from PIL import Image

    hist = gray.histogram()
    total = sum(hist)
    sum_total = sum(i * c for i, c in enumerate(hist))
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    threshold = 127
    for t, count in enumerate(hist):
        w_b += count
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * count
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            threshold = t
    return gray.point(lambda x: 255 if x > threshold else 0, mode="1").convert("L")


def _adaptive_threshold(gray, block: int = 31, c: int = 10):
    """Simple local-mean adaptive threshold (Pillow-only)."""
    from PIL import ImageFilter

    # Odd radius for box blur ≈ block size.
    radius = max(1, block // 2)
    local = gray.filter(ImageFilter.BoxBlur(radius))
    from PIL import ImageChops

    # Image.point can't access two images; use chroma math via offset.
    # gray - local + (255-c) then threshold mid.
    diff = ImageChops.subtract(gray, local, offset=255 - c)
    return diff.point(lambda x: 255 if x > 128 else 0, mode="L")


def _header_variants(page_image: Path) -> list[Path]:
    """Crop upper-left header and emit bounded preprocess variants."""
    from PIL import Image, ImageOps, ImageFilter

    im = Image.open(page_image).convert("L")
    w, h = im.size
    crop = im.crop((0, 0, max(1, int(w * _HEADER_W)), max(1, int(h * _HEADER_H))))
    crop = ImageOps.autocontrast(crop)
    up = crop.resize(
        (max(1, crop.width * _UPSCALE), max(1, crop.height * _UPSCALE)),
        Image.Resampling.LANCZOS,
    )
    tmp_dir = page_image.parent
    paths: list[Path] = []
    native = tmp_dir / f"{page_image.stem}_bio_native.png"
    crop.save(native)
    base = tmp_dir / f"{page_image.stem}_bio_up.png"
    up.save(base)
    paths.append(base)

    otsu = _otsu_threshold(up)
    op = tmp_dir / f"{page_image.stem}_bio_otsu.png"
    otsu.save(op)
    paths.append(op)

    adaptive = _adaptive_threshold(up)
    # Mild morphology: open small speckles then lightly dilate strokes.
    adaptive = adaptive.filter(ImageFilter.MinFilter(3))
    adaptive = adaptive.filter(ImageFilter.MaxFilter(3))
    ap = tmp_dir / f"{page_image.stem}_bio_adapt.png"
    adaptive.save(ap)
    paths.append(ap)
    paths.append(native)
    return paths


def ocr_biometric_flags_from_pages(page_images: list[Path]) -> list[str]:
    """Run specialized header OCR on candidate pages; return allowlisted flags."""
    if not page_images or not shutil.which("tesseract"):
        return []
    found: set[str] = set()
    for page in page_images:
        try:
            variants = _header_variants(page)
        except Exception:
            continue
        # Cheap probe: first variant + PSM 6.
        probe = _tesseract(variants[0], "6")
        if not looks_like_biometric_header(probe):
            # One more PSM before giving up on this page.
            probe11 = _tesseract(variants[0], "11")
            if not looks_like_biometric_header(probe11):
                continue
            probe = probe11
        texts = [probe]
        for img in variants:
            for psm in ("6", "11", "12"):
                if img == variants[0] and psm in {"6", "11"} and texts:
                    # Already probed.
                    if psm == "6" or (psm == "11" and len(texts) > 1):
                        continue
                texts.append(_tesseract(img, psm))
        for text in texts:
            for flag in decode_biometric_flags(text):
                found.add(flag)
        if found:
            # Bound runtime: stop after first page that yields flags.
            break
    return sorted(found)


def ocr_biometric_flags_for_pdf(pdf_path: Path, dpi: int = _PAGE_DPI) -> list[str]:
    """Rasterize PDF pages and decode biometric-header flags if present."""
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return []
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return []
    with tempfile.TemporaryDirectory(prefix="mib-bio-") as tmp:
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
            return []
        pages = sorted(tmp_path.glob("page-*.png"))
        return ocr_biometric_flags_from_pages(pages)


_STRICT_OBS_RE = re.compile(
    r"Observed flags\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def should_run_biometric_header(
    native: str,
    page_ocr: str,
    embedded_ocr: str,
) -> bool:
    """Gate expensive header OCR: only when base streams lack usable flags.

    Mirrors the main OCR fusion's strict colon parse so trailing punctuation
    or missing colons still trigger this specialist path.
    """
    if re.search(r"observ\w{0,4}\s+flags?", native or "", re.I):
        return False

    for text in (native, page_ocr, embedded_ocr):
        for match in _STRICT_OBS_RE.finditer(text or ""):
            # Same atom cleaning as parse_fields._parse_explicit_risk_flags.
            payload = match.group(1).strip().lower()
            head = payload.split("|")[0].strip()
            if head in {"", "none", "n/a", "null", "-"}:
                return False
            for part in re.split(r"[|,]", head):
                atom = part.strip().replace(" ", "_")
                if atom in RISK_FLAG_ATOMS:
                    return False
    return True
