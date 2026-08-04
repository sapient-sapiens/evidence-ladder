"""Round 29: gated adjudicator-note Finding OCR fallback.

When existing streams show a Manual/Adjudicator Note header but R28
``extract_trusted_finding`` returns nothing, locate the note page, crop a
vertical band around the header→Finding/Reason region, and OCR with vendored
tessdata_best (OEM1) across default + Sauvola configs.

Emits only a trusted Finding value (APPROVED / DENIED / NEEDS_REVIEW). Never
concatenates crop text into global OCR, never mutates other fields, never
infers a Finding from Reason text alone.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from .best_ocr import _TESSDATA_DIR, tessdata_best_available
from .ocr_cache import ocr_cache_dir, ocr_code_version
from .parse_fields import (
    ADJUDICATOR_FINDING_RE,
    INJECTION_RE,
    ANSWER_KEY_CSV_RE,
    _looks_like_adjudicator_header_line,
    _local_unsafe_finding_context,
    _normalize_finding_decision,
    extract_trusted_finding,
    strip_injection,
)

_DPI_PRIMARY = 300
_DPI_RETRY = 400
_UPSCALE = 1.5
_PSMS = ("4", "6", "7", "11")
_BANDS = (
    (0.05, 0.70),
    (0.0, 0.55),
    (0.0, 1.0),  # full page when localization is weak
)
_SAUVOLA = (False, True)
_MAX_PAGES_FULL = 2
_MAX_TESS_CALLS = 64
_CACHE_TAG = "v1-note-finding"

_FIND_LABEL_RE = re.compile(
    r"(?i)\b(?:Findg|Finding|Findinq|Fmding|Findlng|Findin)\b"
)
_HEADER_PROBE_RE = re.compile(
    r"(?i)(?:manual\s+adjud|adjudicat(?:or)?\s+note|adjudicator\s+stamp|"
    r"manuva|acjudic|adiudic|adiudir|adudcat|adjuc|adjud)"
)
_PAGE_FOOTER_RE = re.compile(
    r"Packet\s+(MIB-\d{6})\s*/\s*page\s+(\d+)",
    re.IGNORECASE,
)


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


def stream_has_note_header(text: str) -> bool:
    """True when OCR/native shows a visible/fuzzy adjudicator-note header."""
    cleaned, _ = strip_injection(text or "")
    if not cleaned.strip():
        return False
    for line in cleaned.splitlines():
        if _looks_like_adjudicator_header_line(line):
            return True
        if _HEADER_PROBE_RE.search(line) and re.search(
            r"(?i)\b(?:note|nore|nate|stamp)\b", line
        ):
            return True
    return False


def packet_needs_note_finding_ocr(packet, sources) -> bool:
    """Gate: note header present in existing streams and R28 finding missing.

    R28 findings are only those already fused onto the packet from
    native/page/embedded parses. Threshold/best streams are not fused for
    adjudicator findings, so an extractable Finding there must not suppress
    this fallback.
    """
    if not tessdata_best_available():
        return False
    if getattr(packet, "adjudicator_finding", None):
        return False
    native = getattr(sources, "native", "") or ""
    page = getattr(sources, "page_ocr", "") or ""
    emb = getattr(sources, "embedded_ocr", "") or ""
    thr = getattr(sources, "threshold_ocr", "") or ""
    best = getattr(sources, "best_ocr", "") or ""
    combined = "\n".join((native, page, emb, thr, best))
    if not stream_has_note_header(combined):
        return False
    # Confirm fused streams still lack a trusted finding (matches parse fusion).
    for blob in (native, page, emb):
        cleaned, _ = strip_injection(blob)
        if extract_trusted_finding(cleaned, packet.case_id):
            return False
    return True


def _tesseract_best(
    image: Path,
    psm: str,
    *,
    sauvola: bool = False,
    dpi: int = _DPI_PRIMARY,
) -> str:
    if not tessdata_best_available():
        return ""
    cmd = [
        "tesseract",
        str(image),
        "stdout",
        "-l",
        "eng",
        "--oem",
        "1",
        "--psm",
        str(psm),
        "--dpi",
        str(dpi),
        "--tessdata-dir",
        str(_TESSDATA_DIR),
    ]
    if sauvola:
        cmd += [
            "-c",
            "thresholding_method=2",
            "-c",
            "thresholding_kfactor=0.25",
            "-c",
            "thresholding_window_size=0.25",
        ]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=45
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return ""


def _unsafe_crop_context(cleaned: str) -> bool:
    """Reject archived / barcode / injection / sample-denial crops."""
    if INJECTION_RE.search(cleaned) or ANSWER_KEY_CSV_RE.search(cleaned):
        return True
    # Reuse R28 local unsafe check around each Finding label.
    for m in _FIND_LABEL_RE.finditer(cleaned):
        before = cleaned[max(0, m.start() - 480) : m.start()]
        after = cleaned[m.end() : min(len(cleaned), m.end() + 180)]
        line_start = cleaned.rfind("\n", 0, m.start()) + 1
        line_end = cleaned.find("\n", m.end())
        if line_end < 0:
            line_end = len(cleaned)
        full_line = cleaned[line_start:line_end]
        if _local_unsafe_finding_context(before, full_line, after):
            return True
    if re.search(
        r"(?i)(?:archived\s+adjacent|adjacent\s+applicant\s*-\s*not\s+active|"
        r"\bMIB-000000\b|barcode\s+payload|force\s+adjudication\s*=|"
        r"sample\s+denial)",
        cleaned,
    ):
        return True
    return False


def decode_note_finding(text: str, case_id: str) -> tuple[str | None, bool, bool]:
    """Decode Finding from note-crop OCR.

    Returns (value, has_finding_label, exact_allowlisted).
    Fuzzy values require an explicit Finding-label anchor; Reason-only text
    never yields a value.
    """
    raw = text or ""
    # Reject before strip so SYSTEM/answer-key residue cannot leave a bare Finding.
    if INJECTION_RE.search(raw) or ANSWER_KEY_CSV_RE.search(raw):
        return None, bool(_FIND_LABEL_RE.search(raw)), False
    cleaned, injection_heavy = strip_injection(raw)
    if injection_heavy:
        return None, bool(_FIND_LABEL_RE.search(cleaned)), False
    if not cleaned.strip():
        return None, False, False
    if _unsafe_crop_context(cleaned):
        return None, bool(_FIND_LABEL_RE.search(cleaned)), False

    has_label = bool(_FIND_LABEL_RE.search(cleaned))
    # Prefer full R28 trusted extractor (header / reason / active-case gates).
    trusted = extract_trusted_finding(cleaned, case_id)
    if trusted:
        return trusted, True, True

    if not has_label:
        return None, False, False

    vals: list[str] = []
    exact = False
    for m in ADJUDICATOR_FINDING_RE.finditer(cleaned):
        d = _normalize_finding_decision(m.group(1))
        if d in {"APPROVED", "DENIED", "NEEDS_REVIEW"}:
            vals.append(d)
            exact = True

    for m in _FIND_LABEL_RE.finditer(cleaned):
        before = cleaned[max(0, m.start() - 200) : m.start()]
        after = cleaned[m.end() : min(len(cleaned), m.end() + 180)]
        line_start = cleaned.rfind("\n", 0, m.start()) + 1
        line_end = cleaned.find("\n", m.end())
        if line_end < 0:
            line_end = min(len(cleaned), m.end() + 48)
        full_line = cleaned[line_start:line_end]
        if _local_unsafe_finding_context(before, full_line, after):
            continue

        tail = cleaned[m.end() : m.end() + 28]
        # A scan line can erase the leading NEEDS_REV characters and leave
        # ``Finding: 1 IEW``. Drop non-letter scan debris before decoding the
        # surviving suffix; acceptance below still requires an official review
        # Reason body and the normal unsafe-context gates.
        tail = re.sub(r"^[^A-Za-z]+", "", tail)
        tok_m = re.match(r"([A-Za-z]{2,14})", tail)
        if not tok_m:
            continue
        tok = tok_m.group(1).upper()
        if tok in {"APPROVED", "DENIED"}:
            vals.append(tok)
            exact = True
            continue
        if len(tok) >= 5 and _levenshtein(tok, "DENIED") <= 1:
            vals.append("DENIED")
            continue
        if len(tok) >= 7 and _levenshtein(tok, "APPROVED") <= 1:
            vals.append("APPROVED")
            continue
        if tok.startswith("NEED"):
            rest = re.sub(r"^[\s_\-:]+", "", tail[tok_m.end() :])
            compact = re.sub(r"[^A-Z]", "", (tok + rest[:14]).upper())
            if "REVIEW" in compact or compact.startswith("NEEDSRE"):
                vals.append("NEEDS_REVIEW")
                continue
            rev_m = re.match(r"([A-Za-z]{3,10})", rest)
            if rev_m:
                rev = rev_m.group(1).upper()
                if rev == "REVIEW" or _levenshtein(rev, "REVIEW") <= 2:
                    vals.append("NEEDS_REVIEW")
                    continue
            # Same-line REVIEW residue after NEEDS…
            if re.search(r"(?i)rev(?:iew|iey|iew|vey)", full_line):
                vals.append("NEEDS_REVIEW")
                continue
        if tok in {"IEW", "VIEW", "EVIEW", "REVIEW"} and re.search(
            r"(?i)Reason\s*:?.{0,45}(?:damaged\s+or\s+contradictory|"
            r"review[- ]?only\s+risk|missing\s+from\s+trusted\s+visible)",
            after,
        ):
            vals.append("NEEDS_REVIEW")
            continue

    if not vals:
        return None, has_label, False
    return Counter(vals).most_common(1)[0][0], has_label, exact


def _hint_note_pages(sources) -> list[int]:
    """1-based page hints from existing OCR footers near note headers."""
    blobs = [
        getattr(sources, "page_ocr", "") or "",
        getattr(sources, "embedded_ocr", "") or "",
        getattr(sources, "native", "") or "",
    ]
    hints: list[int] = []
    for blob in blobs:
        cleaned, _ = strip_injection(blob)
        # Split on packet footers; associate header lines with nearest footer page.
        parts = re.split(r"(?=Packet\s+MIB-\d+\s*/\s*page\s+\d+)", cleaned)
        for part in parts:
            if not stream_has_note_header(part) and not _FIND_LABEL_RE.search(part):
                continue
            m = _PAGE_FOOTER_RE.search(part)
            if m:
                try:
                    hints.append(int(m.group(2)))
                except ValueError:
                    pass
    # Unique preserve order
    out: list[int] = []
    for p in hints:
        if p not in out:
            out.append(p)
    return out


def _prepare_crop(page_image: Path, y0: float, y1: float, tmp: Path) -> tuple[Path, int]:
    from PIL import Image, ImageOps

    im = Image.open(page_image).convert("L")
    w, h = im.size
    top = max(0, int(h * y0))
    bot = min(h, max(top + 1, int(h * y1)))
    crop = im.crop((int(w * 0.03), top, int(w * 0.97), bot))
    crop = ImageOps.autocontrast(crop, cutoff=1)
    up_w = max(1, int(crop.width * _UPSCALE))
    up_h = max(1, int(crop.height * _UPSCALE))
    up = crop.resize((up_w, up_h), Image.Resampling.LANCZOS)
    out = tmp / f"{page_image.stem}_{y0:.2f}_{y1:.2f}.png"
    up.save(out)
    eff_dpi = int(_DPI_PRIMARY * _UPSCALE)
    return out, eff_dpi


def _select_finding(
    configs: list[dict],
) -> tuple[str | None, str]:
    """Require Finding-label evidence; >=2 independent configs or one HQ exact."""
    by_val: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        if c.get("value") and c.get("has_label"):
            by_val[c["value"]].append(c)

    best: str | None = None
    best_meta = ""
    best_n = -1
    for val, rows in by_val.items():
        keys = {
            (r["page"], r["band"], r["sauvola"], r["psm"], r.get("dpi"))
            for r in rows
        }
        hq = [r for r in rows if r.get("exact")]
        if len(keys) >= 2 and len(keys) > best_n:
            best, best_meta, best_n = val, f"consensus:{len(keys)}", len(keys)
        elif hq and best is None:
            best, best_meta, best_n = val, "hq_exact:1", 1
    # Ambiguity: competing values with comparable consensus → reject.
    if best and len(by_val) > 1:
        rival_ns = []
        for val, rows in by_val.items():
            if val == best:
                continue
            rival_ns.append(
                len(
                    {
                        (r["page"], r["band"], r["sauvola"], r["psm"], r.get("dpi"))
                        for r in rows
                    }
                )
            )
        if rival_ns and max(rival_ns) >= best_n:
            return None, "ambiguous"
    return best, best_meta


def ocr_note_finding_for_pdf(
    pdf_path: Path,
    case_id: str,
    sources=None,
) -> str | None:
    """Run bounded note-crop OCR; return trusted Finding or None."""
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return None
    if not tessdata_best_available():
        return None
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return None

    hints = _hint_note_pages(sources) if sources is not None else []
    tess_calls = 0
    configs: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="mib-r29-") as tmp:
        tmp_path = Path(tmp)
        prefix = tmp_path / "page"
        try:
            subprocess.run(
                [
                    "pdftoppm",
                    "-r",
                    str(_DPI_PRIMARY),
                    "-gray",
                    "-png",
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return None
        pages = sorted(tmp_path.glob("page-*.png"))
        if not pages:
            return None

        # Probe order: footer hints first, then remaining pages. Always scan
        # all pages — footer association is often off-by-one on scanned packs.
        order: list[Path] = []
        for h in hints:
            if 1 <= h <= len(pages):
                p = pages[h - 1]
                if p not in order:
                    order.append(p)
        for p in pages:
            if p not in order:
                order.append(p)

        def _page_looks_like_note(text: str) -> bool:
            if stream_has_note_header(text):
                return True
            if _FIND_LABEL_RE.search(text or ""):
                return True
            # Bare adjud* token alone is too loose (sponsor prose FPs).
            return bool(
                re.search(
                    r"(?i)(?:manual\s+adjud|adjudicat(?:or)?\s+note|"
                    r"adjudicator\s+stamp|manuva\s*:?\s*acjudic)",
                    text or "",
                )
            )

        hit_pages: list[Path] = []
        for p in order:
            if tess_calls >= _MAX_TESS_CALLS:
                break
            # Full-page cheap probe (notes often sit mid/lower; upper crop misses).
            probe = _tesseract_best(p, "6", sauvola=False, dpi=_DPI_PRIMARY)
            tess_calls += 1
            if _page_looks_like_note(probe):
                hit_pages.append(p)
            if len(hit_pages) >= _MAX_PAGES_FULL:
                break

        if not hit_pages:
            # Localization failed: crop full pages (bounded).
            hit_pages = pages[:_MAX_PAGES_FULL]

        for p in hit_pages[:_MAX_PAGES_FULL]:
            for y0, y1 in _BANDS:
                if tess_calls >= _MAX_TESS_CALLS:
                    break
                crop_path, eff = _prepare_crop(p, y0, y1, tmp_path)
                for sau in _SAUVOLA:
                    for psm in _PSMS:
                        if tess_calls >= _MAX_TESS_CALLS:
                            break
                        text = _tesseract_best(
                            crop_path, psm, sauvola=sau, dpi=eff
                        )
                        tess_calls += 1
                        value, has_label, exact = decode_note_finding(
                            text, case_id
                        )
                        if has_label or value:
                            configs.append(
                                {
                                    "page": p.name,
                                    "band": (y0, y1),
                                    "sauvola": sau,
                                    "psm": psm,
                                    "dpi": _DPI_PRIMARY,
                                    "value": value,
                                    "has_label": has_label,
                                    "exact": exact,
                                }
                            )
            chosen, meta = _select_finding(configs)
            if chosen and meta.startswith("consensus"):
                return chosen

        chosen, meta = _select_finding(configs)
        if chosen:
            return chosen

        # Optional 400 DPI retry on first hit page when label seen but no value.
        label_only = any(c.get("has_label") and not c.get("value") for c in configs)
        if label_only and hit_pages and tess_calls < _MAX_TESS_CALLS:
            p = hit_pages[0]
            retry_prefix = tmp_path / "hi"
            try:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-f",
                        str(pages.index(p) + 1),
                        "-l",
                        str(pages.index(p) + 1),
                        "-r",
                        str(_DPI_RETRY),
                        "-gray",
                        "-png",
                        str(pdf_path),
                        str(retry_prefix),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, OSError):
                return None
            hi_pages = sorted(tmp_path.glob("hi-*.png"))
            if not hi_pages:
                return None
            hp = hi_pages[0]
            for y0, y1 in _BANDS[:2]:
                from PIL import Image, ImageOps

                im = Image.open(hp).convert("L")
                w, h = im.size
                crop = im.crop(
                    (
                        int(w * 0.03),
                        int(h * y0),
                        int(w * 0.97),
                        max(int(h * y0) + 1, int(h * y1)),
                    )
                )
                crop = ImageOps.autocontrast(crop, cutoff=1)
                cpath = tmp_path / f"hi_{y0:.2f}_{y1:.2f}.png"
                crop.save(cpath)
                for sau in _SAUVOLA:
                    for psm in ("4", "6", "11"):
                        if tess_calls >= _MAX_TESS_CALLS:
                            break
                        text = _tesseract_best(
                            cpath, psm, sauvola=sau, dpi=_DPI_RETRY
                        )
                        tess_calls += 1
                        value, has_label, exact = decode_note_finding(
                            text, case_id
                        )
                        if has_label or value:
                            configs.append(
                                {
                                    "page": hp.name,
                                    "band": (y0, y1),
                                    "sauvola": sau,
                                    "psm": psm,
                                    "dpi": _DPI_RETRY,
                                    "value": value,
                                    "has_label": has_label,
                                    "exact": exact,
                                }
                            )
            chosen, _meta = _select_finding(configs)
            return chosen
    return None


def note_finding_ocr_cached(
    pdf_path: Path,
    case_id: str,
    sources=None,
) -> str | None:
    """Cached note-Finding OCR for one gated PDF."""
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"{_CACHE_TAG}:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}:{case_id}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{pdf_path.stem}_{key[:16]}_notefind.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        val = payload.get("finding")
        return val if val in {"APPROVED", "DENIED", "NEEDS_REVIEW"} else None
    finding = ocr_note_finding_for_pdf(pdf_path, case_id, sources=sources)
    cache_path.write_text(
        json.dumps({"finding": finding}, sort_keys=True),
        encoding="utf-8",
    )
    return finding
