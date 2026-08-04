"""Writable OCR cache directory (Docker --read-only safe)."""
from __future__ import annotations

import hashlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path

# Every module that writes an OCR stream to disk, plus the helpers whose
# parameters change what those streams contain.  The digest of these files is
# part of every OCR cache key, so a tuned DPI, gate or threshold invalidates the
# entries it would otherwise silently reuse.  Hand-maintained version literals
# do not catch that: they only move when someone remembers to move them, and a
# warm cache serving output from another code state is indistinguishable from a
# real score change.
_OCR_SOURCES = (
    "best_ocr.py",
    "biometric_header_ocr.py",
    "legibility_b13.py",
    "note_finding_ocr.py",
    "oriented_form_ocr.py",
    "rapid_ocr.py",
    "text_extract.py",
    "threshold_ocr.py",
    "visual_damage.py",
    "visual_risk_marks.py",
)


@lru_cache(maxsize=1)
def ocr_cache_dir() -> Path:
    """Prefer MIB_OCR_CACHE, else local outputs/.ocr_cache, else /tmp."""
    env = os.environ.get("MIB_OCR_CACHE", "").strip()
    if env:
        path = Path(env)
        path.mkdir(parents=True, exist_ok=True)
        return path

    local = Path(__file__).resolve().parents[1] / "outputs" / ".ocr_cache"
    try:
        local.mkdir(parents=True, exist_ok=True)
        probe = local / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return local
    except OSError:
        path = Path(tempfile.gettempdir()) / "mib_ocr_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def ocr_code_version() -> str:
    """Digest of the OCR-producing source, for inclusion in every cache key."""
    here = Path(__file__).resolve().parent
    digest = hashlib.sha1()
    for name in _OCR_SOURCES:
        path = here / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:12]
