"""Orientation-normalized OCR for severely degraded multi-field forms.

The regular page OCR assumes upright pages.  Some packets contain whole forms
rotated by 90/180/270 degrees or displaced into scan bands.  This fallback is
gated to packets with at least three unresolved critical fields, probes all four
orthogonal orientations at modest resolution, then uses tessdata_best on only
the structurally strongest page/orientation.  Output remains a lowest-trust,
fill-missing text stream.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .best_ocr import _TESSDATA_DIR, tessdata_best_available
from .constants import (
    CRITICAL_FIELDS,
    FEE_STATUSES,
    HOME_WORLDS,
    PURPOSES,
    SPECIES_CODES,
    VISA_CLASSES,
)
from .ocr_cache import ocr_cache_dir, ocr_code_version
from .parse_fields import strip_injection

_DPI = 180
_ROTATIONS = (0, 90, 180, 270)
_CACHE_VERSION = "v4-oriented-degraded-label-consensus"
_MAX_PAGES = 6

_STRUCTURE_CLUES = (
    re.compile(r"(?i)\bapplicant\b"),
    re.compile(r"(?i)\b(?:species|spacies|spaces)\b"),
    re.compile(r"(?i)\b(?:home|mowe)\s+(?:world|weld)\b"),
    re.compile(r"(?i)\bvisa\b"),
    re.compile(r"(?i)\bsponsor\b"),
    re.compile(r"(?i)\b(?:arrival|anwval|arnval)\b"),
    re.compile(r"(?i)\b(?:declared|purpose)\b"),
    re.compile(r"(?i)\bfee\b"),
    re.compile(r"(?i)\b(?:observed|observcd|dbserved)\b"),
)
_INTAKE_FORM_RE = re.compile(r"(?i)\bform\s+i[-\s]*8090\b|\bintake\b")

_VOCAB_FIELDS = {
    "species_code": (SPECIES_CODES, "Species Code"),
    "home_world": (HOME_WORLDS, "Home World"),
    "visa_class": (VISA_CLASSES, "Visa Class"),
    "declared_purpose": (PURPOSES, "Declared Purpose"),
    "fee_status": ({v for v in FEE_STATUSES if v != "unknown"}, "Fee Status"),
}


def packet_needs_oriented_form_ocr(packet) -> bool:
    if not (shutil.which("pdftoppm") and shutil.which("tesseract")):
        return False
    if not tessdata_best_available():
        return False
    fields = getattr(packet, "fields", {}) or {}
    unresolved = sum(
        str(fields.get(field) or "").strip().casefold()
        in {"", "unknown", "n/a", "none", "null", "1900-01-01", "spn-0000"}
        for field in CRITICAL_FIELDS
    )
    risk = set(getattr(packet, "risk_flags", ()) or ())
    unresolved_biometric_relation = (
        "illegible_biometrics" in risk
        and not bool(getattr(packet, "observed_flags_seen", False))
    )
    missing_sponsor_and_date = (
        str(fields.get("sponsor_id") or "").strip().casefold()
        in {"", "unknown", "n/a", "none", "null", "spn-0000"}
        and str(fields.get("arrival_date") or "").strip().casefold()
        in {"", "unknown", "n/a", "none", "null", "1900-01-01"}
    )
    return unresolved >= 3 or unresolved_biometric_relation or missing_sponsor_and_date


def structural_score(text: str) -> int:
    cleaned, injection_heavy = strip_injection(text or "")
    if injection_heavy or not cleaned.strip():
        return 0
    score = sum(2 for clue in _STRUCTURE_CLUES if clue.search(cleaned))
    score += 8 * int(bool(_INTAKE_FORM_RE.search(cleaned)))
    # Reward plausible field values without learning any packet-specific token.
    score += 2 * len(set(re.findall(r"\bSPN-\d{4}\b", cleaned, re.I)))
    score += 2 * len(set(re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", cleaned)))
    score += 2 * len(
        set(re.findall(r"\b(?:XW-[12]|DIP-1|MED-3|TRANSIT-7)\b", cleaned, re.I))
    )
    return score


def _norm_vocab(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


def canonical_vocab_lines(text: str) -> list[str]:
    """Decode unique closed-vocabulary values even when their label is lost.

    Full-page parsers need a recognizable label. Corrupted intake bands often
    preserve a value such as ``KALIU MICRO`` while mangling ``Species Code``.
    Search short, line-local token windows and emit a synthetic canonical label
    only for a strict unique match. Conflicting values tie and are rejected.
    """
    observed: list[str] = []
    for line in (text or "").splitlines():
        tokens = re.findall(r"[A-Za-z0-9_-]+", line)
        for width in (1, 2, 3):
            for index in range(len(tokens) - width + 1):
                value = _norm_vocab(" ".join(tokens[index : index + width]))
                if len(value) >= 3:
                    observed.append(value)

    out: list[str] = []
    emitted: set[str] = set()
    for field, (vocab, label) in _VOCAB_FIELDS.items():
        scored: list[tuple[int, str]] = []
        for value in vocab:
            target = _norm_vocab(value)
            distances = [
                _edit_distance(target, candidate)
                for candidate in observed
                if abs(len(candidate) - len(target))
                <= max(1, int(0.25 * len(target)))
            ]
            if distances:
                scored.append((min(distances), value))
        scored.sort()
        if not scored:
            continue
        best_distance, best_value = scored[0]
        second_distance = scored[1][0] if len(scored) > 1 else 99
        target_len = len(_norm_vocab(best_value))
        max_distance = 1 if field in {"visa_class", "fee_status"} else max(
            2, int(0.25 * target_len)
        )
        if best_distance <= max_distance and second_distance >= best_distance + 2:
            out.append(f"{label}: {best_value}")
            emitted.add(field)

    # Long-vocabulary recovery when both label and value are smeared. Require
    # repeated line agreement, except for a uniquely decoded home value behind
    # two still-near Home World label words.
    for field, vocab, label in (
        ("species_code", SPECIES_CODES, "Species Code"),
        ("home_world", HOME_WORLDS, "Home World"),
    ):
        if field in emitted:
            continue
        votes: dict[str, int] = {}
        strong_home: dict[str, int] = {}
        for line in (text or "").splitlines():
            if re.search(
                r"(?i)whiteout|registry\s+lost|cut\s+out|illeg|torn|"
                r"synthetic|packet\s+mib",
                line,
            ):
                continue
            tokens = re.findall(r"[A-Za-z0-9_-]+", line)
            if len(tokens) < 3:
                continue
            first, second = _norm_vocab(tokens[0]), _norm_vocab(tokens[1])
            if field == "species_code":
                first_distance = _edit_distance(first, "species")
                second_distance = min(
                    _edit_distance(second, "code"),
                    _edit_distance(second, "match"),
                )
                label_like = (
                    not first.startswith("spon")
                    and second not in {"id", "1d"}
                    and first_distance <= 5
                    and second_distance <= 3
                )
            else:
                first_distance = _edit_distance(first, "home")
                second_distance = _edit_distance(second, "world")
                label_like = first_distance <= 2 and second_distance <= 4
            if not label_like:
                continue

            observed_values: list[str] = []
            for width in (1, 2, 3):
                for index in range(2, len(tokens) - width + 1):
                    value = _norm_vocab(" ".join(tokens[index : index + width]))
                    if value:
                        observed_values.append(value)
            ranked: list[tuple[int, str]] = []
            for value in vocab:
                target = _norm_vocab(value)
                ranked.append(
                    (min(_edit_distance(obs, target) for obs in observed_values), value)
                )
            ranked.sort()
            best_distance, best_value = ranked[0]
            runner_distance = ranked[1][0]
            ratio = best_distance / max(1, len(_norm_vocab(best_value)))
            if ratio > 0.65 or runner_distance - best_distance < 1:
                continue
            votes[best_value] = votes.get(best_value, 0) + 1
            if (
                field == "home_world"
                and first_distance + second_distance <= 5
                and ratio <= 0.65
            ):
                strong_home[best_value] = strong_home.get(best_value, 0) + 1

        if not votes:
            continue
        ranked_votes = sorted(
            ((count, value) for value, count in votes.items()), reverse=True
        )
        best_count, best_value = ranked_votes[0]
        runner_count = ranked_votes[1][0] if len(ranked_votes) > 1 else 0
        repeated = best_count >= 2 and best_count >= runner_count + 2
        strong_single = (
            field == "home_world"
            and strong_home.get(best_value, 0) >= 1
            and runner_count == 0
        )
        if repeated or strong_single:
            out.append(f"{label}: {best_value}")
    return out


def _render_pages(pdf_path: Path, tmp: Path) -> list[Path]:
    prefix = tmp / "page"
    try:
        subprocess.run(
            [
                "pdftoppm", "-f", "1", "-l", str(_MAX_PAGES),
                "-r", str(_DPI), "-gray", "-png", str(pdf_path), str(prefix),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return []
    return sorted(tmp.glob("page-*.png"))


def _variant(image: Image.Image, angle: int) -> Image.Image:
    rotated = image.rotate(angle, expand=True, fillcolor=255)
    rotated = ImageOps.autocontrast(rotated, cutoff=(0.2, 0.8))
    rotated = ImageEnhance.Contrast(rotated).enhance(1.25)
    return rotated.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))


def _tesseract(image: Path, *, psm: str, best: bool) -> str:
    cmd = [
        "tesseract", str(image), "stdout", "-l", "eng", "--psm", psm,
        "--dpi", str(_DPI),
    ]
    if best:
        cmd += ["--oem", "1", "--tessdata-dir", str(_TESSDATA_DIR)]
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=60
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return ""


def oriented_form_ocr(pdf_path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="core-oriented-") as tmp_name:
        tmp = Path(tmp_name)
        pages = _render_pages(pdf_path, tmp)
        ranked: list[tuple[int, Path, str]] = []
        for page_index, page in enumerate(pages):
            try:
                base = Image.open(page).convert("L")
            except OSError:
                continue
            best_score = -1
            best_path: Path | None = None
            best_probe = ""
            for angle in _ROTATIONS:
                path = tmp / f"p{page_index}-r{angle}.png"
                _variant(base, angle).save(path)
                probe = _tesseract(path, psm="11", best=False)
                score = structural_score(probe)
                if score > best_score:
                    best_score, best_path, best_probe = score, path, probe
            if best_path is not None:
                ranked.append((best_score, best_path, best_probe))

        if not ranked or max(score for score, _path, _probe in ranked) < 4:
            return ""
        _score, best_path, probe = max(ranked, key=lambda item: item[0])
        texts = [probe]
        for psm in ("6", "11"):
            text = _tesseract(best_path, psm=psm, best=True)
            if text.strip():
                texts.append(text)
        # Horizontal corruption tends to spare the compact field block at the
        # top of a page while confusing full-page segmentation. Re-read that
        # band on every page in its best orthogonal orientation. This also
        # reaches intake pages whose damaged header scored below a clean B-13.
        for _page_score, page_path, _page_probe in ranked:
            try:
                image = Image.open(page_path).convert("L")
            except OSError:
                continue
            band = image.crop((0, 0, image.width, max(1, int(image.height * 0.44))))
            band = ImageOps.autocontrast(band, cutoff=(0.1, 0.6))
            band_path = tmp / f"band-{len(texts)}.png"
            band.save(band_path)
            band_text = _tesseract(band_path, psm="11", best=True)
            if band_text.strip():
                texts.append(band_text)
        texts.extend(canonical_vocab_lines("\n".join(texts)))
        return "\n\n".join(texts)


def oriented_form_ocr_cached(pdf_path: Path) -> str:
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"{_CACHE_VERSION}:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    path = cache_dir / f"{pdf_path.stem}_{key[:16]}_oriented.json"
    if path.exists():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("text") or "")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    text = oriented_form_ocr(pdf_path)
    path.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text
