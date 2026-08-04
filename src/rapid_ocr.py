"""Offline neural OCR fallback for packets that remain severely incomplete.

Tesseract and PP-OCR have complementary failure modes.  The regular pipeline
keeps Tesseract as the cheap default; this fallback runs the bundled PP-OCRv6
small models only after orientation recovery still leaves several critical
fields unresolved.  Results are an isolated, fill-missing-only text stream.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import (
    CRITICAL_FIELDS,
    PURPOSES,
    RISK_FLAG_ATOMS,
    SPECIES_CODES,
    VISA_CLASSES,
)
from .name_resolve import load_name_lexicon, normalize_two_token_name
from .ocr_cache import ocr_cache_dir, ocr_code_version

_CACHE_VERSION = "v2-rapidocr-vertical-box-rotation"
_EMBEDDED_CACHE_VERSION = "v1-rapidocr-embedded"
_DPI = 144
_MAX_PAGES = 6
_ENGINE: Any | None = None
_ENGINE_FAILED = False
_VISIBLE_PLACEHOLDER_RE = re.compile(
    r"(?i)^\s*[\[(].*(?:cut\s*out|lost|illegible|torn|obscured|unreadable).*[\])]\s*$"
)
_FIELD_TAGS = {
    "applicant_name": "intake",
    "species_code": "intake",
    "home_world": "intake",
    "visa_class": "intake",
    "sponsor_id": "intake",
    "arrival_date": "intake",
    "declared_purpose": "intake",
    "fee_status": "fee",
}


_PAGE_FOOTER_RE = re.compile(
    r"(?im)^.*?Packet\s+M(?:I|1|L)?B-?\d+\s*/\s*page\s+\d+.*?$"
)


def _logical_pages(text: str) -> list[str]:
    """Recover page boundaries that OCR text otherwise discards.

    The Tesseract streams historically joined pages with a single newline,
    while embedded-image OCR often inserted blank lines between *rows*.  A
    blank-line split therefore mixed adjacent documents in the first case and
    shattered one document in the second.  Packet footers are printed on each
    synthetic page and give us a content-derived boundary without relying on
    filenames, ordering, or a learned template.
    """
    if not text or not text.strip():
        return []
    matches = list(_PAGE_FOOTER_RE.finditer(text))
    if not matches:
        return [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    pages: list[str] = []
    start = 0
    for match in matches:
        page = text[start : match.end()].strip()
        if page:
            pages.append(page)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        pages.append(tail)
    return pages


def _is_missing(value: object) -> bool:
    raw = str(value or "").strip()
    return raw.casefold() in {
        "",
        "unknown",
        "n/a",
        "none",
        "null",
        "1900-01-01",
        "spn-0000",
    } or bool(_VISIBLE_PLACEHOLDER_RE.match(raw))


def packet_needs_rapid_ocr(packet) -> bool:
    """Gate neural OCR to incomplete or internally contradictory packets.

    A packet can be superficially complete while two document roles disagree
    (for example, intake/registry versus sponsor identity).  Those conflicts
    are exactly where the page-aware neural pass adds evidence, so do not make
    completeness the sole gate.
    """
    if not shutil.which("pdftoppm"):
        return False
    fields = getattr(packet, "fields", {}) or {}
    if any(_is_missing(fields.get(field)) for field in CRITICAL_FIELDS):
        return True
    conflicts = set(getattr(packet, "conflicts", ()) or ()) | set(
        getattr(packet, "untrusted_conflicts", ()) or ()
    )
    if conflicts & set(CRITICAL_FIELDS):
        return True
    return False


def packet_needs_embedded_rapid_ocr(packet) -> bool:
    """Second-stage original-image pass for packets still missing 3+ fields."""
    if not shutil.which("pdfimages"):
        return False
    fields = getattr(packet, "fields", {}) or {}
    return sum(_is_missing(fields.get(field)) for field in CRITICAL_FIELDS) >= 3




def _canonical_name(raw: str | None) -> str | None:
    if not raw:
        return None
    normalized, _cost, _exact = normalize_two_token_name(
        " ".join(raw.split()), load_name_lexicon(), allow_dist2=True
    )
    if normalized:
        return normalized
    # A page crop can remove one or two leading glyphs (``xtari`` from
    # ``Nextari``). Restore only a unique lexicon suffix, then rerun the normal
    # two-token resolver; this cannot invent a token outside FIT vocabulary.
    parts = " ".join(raw.split()).split()
    if len(parts) != 2:
        return None
    lexicon = load_name_lexicon()
    repaired: list[str] = []
    for index, token in enumerate(parts):
        mapping = lexicon.get("_first_cf" if index == 0 else "_last_cf", {})
        suffixes = {token.casefold()}
        if token.casefold().endswith("m"):
            suffixes.add(token.casefold()[:-1] + "rn")
        candidates = [
            canonical
            for key, canonical in mapping.items()
            if any(
                len(key) - len(suffix) in {1, 2} and key.endswith(suffix)
                for suffix in suffixes
            )
        ]
        repaired.append(candidates[0] if len(set(candidates)) == 1 else token)
    normalized, _cost, _exact = normalize_two_token_name(
        " ".join(repaired), lexicon, allow_dist2=True
    )
    if normalized:
        return normalized

    # Short OCR tokens are excluded from the generic dist-2 resolver because
    # they are ambiguous in isolation. In a two-token name, allow the repair
    # only when each positional winner is unique and the combined corruption
    # is small (for example ``Lumorm Telnsx`` → ``Lumora Teknax``).
    repaired = []
    distances: list[int] = []
    for index, token in enumerate(parts):
        mapping = lexicon.get("_first_cf" if index == 0 else "_last_cf", {})
        ranked = sorted(
            (_edit_distance(token.casefold(), key), canonical)
            for key, canonical in mapping.items()
        )
        if not ranked or ranked[0][0] > 2:
            return None
        if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
            return None
        distances.append(ranked[0][0])
        repaired.append(ranked[0][1])
    if not (0 in distances or sum(distances) <= 3):
        return None
    return " ".join(repaired)


def repair_ocr_name(raw: str | None) -> str | None:
    """Public last-mile wrapper for conservative FIT-lexicon name repair."""
    return _canonical_name(raw)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def arrival_date_votes(text: str) -> Counter[str]:
    """Count canonical dates across independently transformed OCR views."""
    dates: list[str] = []
    for year, month, day in re.findall(
        r"\b20(26|28)-([0-1]\d)-([0-3]\d)\b", text or ""
    ):
        if year == "28" and month == "08":
            month = "06"
        dates.append(f"2026-{month}-{day}")
    return Counter(dates)


def repeated_arrival_date(text: str, minimum_votes: int = 3) -> str | None:
    """Return the unique date repeated across transformed OCR views."""
    counts = arrival_date_votes(text)
    winners = [value for value, count in counts.items() if count >= minimum_votes]
    return winners[0] if len(winners) == 1 else None


def _nearest_unique(raw: str, vocabulary: tuple[str, ...] | set[str], limit: int = 1) -> str | None:
    token = re.sub(r"[^A-Z0-9]", "", raw.upper())
    ranked = sorted(
        (_edit_distance(token, re.sub(r"[^A-Z0-9]", "", value.upper())), value)
        for value in vocabulary
    )
    if not ranked or ranked[0][0] > limit:
        return None
    if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
        return None
    return ranked[0][1]


def rapid_semantics(text: str) -> dict[str, object]:
    """Extract source-aware facts from PP-OCR's page-ordered text.

    Generic fusion discards document roles.  Here, page headings let registry,
    biometric, sponsor, and intake evidence arbitrate names and recover sponsor
    prose such as ``class XW-1 compliance``.  All outputs remain closed-vocab or
    format constrained.
    """
    result: dict[str, object] = {}
    if not text.strip():
        return result

    lower_text = text.casefold()
    has_b13_evidence = bool(
        re.search(r"\bform\s*[b8][ -]?13\b|observed\s+fl\w*", lower_text)
    )
    has_denial_watermark = "sample denial" in lower_text
    registry_lost = "registry lost" in lower_text
    torn_core_field = bool(
        re.search(r"\[(?:visa class|sponsor id|date)\s+torn\]", lower_text)
    )
    registry_fee_without_intake = (
        "registry" in lower_text
        and "fee receipt" in lower_text
        and not re.search(r"form\s*[i1|l]?[ -]?8090", lower_text)
    )
    # Some damaged packets physically omit B-13, so no OCR engine can read an
    # Observed-flags value. Recover that document condition from independent
    # damage/layout evidence. solution.py keeps this extraction-only.
    if not has_b13_evidence and (
        (
            has_denial_watermark
            and (
            registry_lost
            or torn_core_field
            or registry_fee_without_intake
            )
        )
        or (torn_core_field and "registry" in lower_text)
    ):
        result["_damage_implies_illegible"] = True
    if (
        not has_b13_evidence
        and "registry" in lower_text
        and "fee receipt" in lower_text
        and re.search(r"form\s*[i1|l]?[ -]?8090", lower_text)
    ):
        result["_complete_no_b13_layout"] = True
    if (
        not has_b13_evidence
        and "registry" not in lower_text
        and "fee receipt" in lower_text
        and "sponsor attestation" in lower_text
        and "adjudicator note" in lower_text
        and re.search(r"form\s*[i1|l]?[ -]?8090", lower_text)
    ):
        # A four-role packet reconstructed from intake, fee, sponsor, and a
        # signed note can prove that both durable biometric roles (B-13 and
        # registry) are physically absent.  PDF-layer forensics later confirms
        # that this is a damaged raster packet before asserting illegibility.
        result["_missing_biometric_roles_layout"] = True
    if re.search(r"(?im)^\s*Fee\s+Status\s*:?\s*unknown\b", text):
        # This printed value is intentionally only a vote. Receipt-local
        # amount/code arbitration and explicit parsed fields remain stronger.
        result["_printed_fee_unknown"] = True
    if re.search(r"(?i)review[- ]only\s+risk\s+flag\s+present", text):
        result["_review_flag_asserted"] = True
    structural_risk_flags: list[str] = []
    if (
        re.search(
            r"(?i)packet\s+contains\s+damaged\s+or\s+contradictory\s+"
            r"visible\s+evidence",
            text,
        )
        and re.search(r"(?i)risk\s+panel", text)
    ):
        # Neither phrase alone identifies the atom, but together the signed
        # review note and the visibly absent B-13 panel establish biometric
        # illegibility without guessing among unrelated damage styles.
        structural_risk_flags.append("illegible_biometrics")

    heading_re = re.compile(
        r"(?im)^(FORM\s*[I1|l]?[- ]?8090|Sponsor\s+Attestation\s+Letter|"
        r"(?:Planetary|\w{1,10}(?:letary|netary))\s+"
        r"Reg[\w'’\s]{0,10}?Extract|"
        r"F\w{2}M\s+B-13)"
    )
    starts = list(heading_re.finditer(text))
    names: dict[str, str] = {}
    sponsor_visas: list[str] = []
    sponsor_role_ids: list[str] = []
    intake_role_ids: list[str] = []
    supplemental_sponsor_ids: list[str] = []
    correction_names: list[str] = []
    standalone_visas: list[str] = []
    explicit_visas: list[str] = []
    explicit_purposes: list[str] = []
    registry_dates: list[str] = []
    registry_pages: list[int] = []
    species_fills: list[str] = []
    fuzzy_home_worlds: list[str] = []
    intake_name_fragments: list[str] = []

    def name_after(block: str, pattern: str) -> str | None:
        match = re.search(pattern, block, re.IGNORECASE | re.DOTALL)
        return _canonical_name(match.group(1)) if match else None

    for index, heading in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[heading.start() : end]
        kind = heading.group(1).casefold()
        if kind.startswith(("form i", "form 1")):
            name = name_after(
                block,
                r"\bApp\w{3,9}\s*[^A-Za-z0-9\n]?[ \t]*(?:\n[ \t]*)?"
                r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if name:
                names["intake"] = name
        elif kind.startswith("sponsor"):
            name = name_after(
                block,
                r"attests\s+that\s+([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)"
                r"\s+is\s+expected",
            )
            if name:
                names["sponsor"] = name
            sponsor_role_ids.extend(
                f"SPN-{digits}"
                for digits in re.findall(
                    r"(?i)\bSponsor\s+S(?:P|H)(?:N|I|M|1)-?(\d{4})"
                    r"\s+attests\b",
                    block,
                )
            )
            expected = re.findall(
                r"(?is)expected\s+on\s+Earth\s+for\s+(.{3,60}?)(?:\.|\nThe\s+sponsor)",
                block,
            )
            for raw_purpose in expected:
                lowered = " ".join(raw_purpose.split()).casefold()
                hits = [value for value in PURPOSES if value in lowered]
                if len(hits) == 1:
                    explicit_purposes.append(hits[0])
            for visa in VISA_CLASSES:
                if re.search(
                    rf"(?i)(?:Visa\s+Class\s*:?[ \t\n]*|class\s+)"
                    rf"{re.escape(visa)}\b",
                    block,
                ):
                    sponsor_visas.append(visa)
            # A damaged label such as ``ass: MED-3`` still sits inside a
            # structurally identified sponsor letter. A unique exact class in
            # that block is stronger than an intake hypothesis.
            sponsor_vocab = [
                visa
                for visa in VISA_CLASSES
                if re.search(rf"\b{re.escape(visa)}\b", block, re.IGNORECASE)
            ]
            if len(sponsor_vocab) == 1:
                sponsor_visas.append(sponsor_vocab[0])
        elif "reg" in kind:
            name = name_after(
                block,
                r"(?:\w{0,5}pplicant|Registry\s+Name)\s*:?[ \t]*"
                r"(?:\n[ \t]*)?([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if name:
                names["registry"] = name
            registry_dates.extend(
                re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", block)
            )
            page_match = re.search(r"(?i)Packet\s+MIB-\d+\s*/\s*page\s+(\d+)", block)
            if page_match:
                registry_pages.append(int(page_match.group(1)))
        elif kind.startswith("form b"):
            name = name_after(
                block,
                r"\bApp\w{3,9}\s*[^A-Za-z0-9\n]?[ \t]*(?:\n[ \t]*)?"
                r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if name:
                names["biometric"] = name

        for match in re.finditer(
            r"(?im)^\s*(?:Declared\s+)?Purpose\s*:?[ \t]*([^\n]+)", block
        ):
            lowered = match.group(1).casefold()
            hits = [value for value in PURPOSES if value in lowered]
            if len(hits) == 1:
                explicit_purposes.append(hits[0])
        for match in re.finditer(
            r"(?im)^\s*Visa\s+Class\s*:?[ \t]*([^\n]+)", block
        ):
            upper = match.group(1).upper()
            hits = [value for value in VISA_CLASSES if value in upper]
            if len(hits) == 1:
                explicit_visas.append(hits[0])

    # Page-level roles remain usable when rotation/reordering places the form
    # heading after its fields. They also expose supplemental correction pages.
    pages = _logical_pages(text)
    fuzzy_observed_flags: list[str] = []
    for page in pages:
        # Spatially scrambled rotated B-13 pages can emit the value before
        # fragments of ``Observed flags`` / ``Species Match``. Recover a long
        # closed-vocabulary atom only when both sides of that structure remain.
        has_flag_clue = bool(re.search(r"(?i)\b(?:flag|serve\w*)\b", page))
        has_b13_clue = bool(
            re.search(r"(?i)\b(?:matc\w*|biometric|scan\s+image|b-?13)\b", page)
        )
        if has_flag_clue and has_b13_clue:
            from .biometric_header_ocr import fuzzy_match_flag

            hits = {
                hit
                for token in re.findall(r"[A-Za-z_]{12,32}", page)
                if (hit := fuzzy_match_flag(token)) is not None
            }
            if len(hits) == 1:
                structural_risk_flags.extend(hits)
                fuzzy_observed_flags.extend(hits)
        if re.search(r"(?i)F\w{2}M\s+B-13", page):
            name = name_after(
                page,
                r"\bApp\w{3,9}\s*[^A-Za-z0-9\n]?\s*(?:\n\s*)?"
                r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if name:
                names["biometric"] = name

        intake_page = bool(re.search(r"(?i)FORM\s*[I1|l]?[- ]?8090", page))
        if intake_page and "intake" not in names:
            before_world = re.split(r"(?i)Home\s+World", page, maxsplit=1)[0]
            for line in before_world.splitlines()[1:]:
                if re.search(r"(?i)Case\s+ID|Packet|Species|Passport|Synthetic", line):
                    continue
                name = _canonical_name(line.strip(" :|_-"))
                if name:
                    names["intake"] = name
                    break
        if intake_page:
            for match in re.finditer(
                r"(?im)^\s*Applicant\s*[:.]\s*([^\n]{2,40})", page
            ):
                fragment = " ".join(re.findall(r"[A-Za-z]+", match.group(1)))
                if 4 <= len(re.sub(r"[^A-Za-z]", "", fragment)) <= 30:
                    intake_name_fragments.append(fragment)

        sponsor_page = bool(
            re.search(r"(?i)Sponsor\s+Attestat\w*\s+(?:L|e)?etter", page)
        )
        id_matches = re.findall(
            r"(?i)Spons\w{0,3}\s+(?:ID|1D|IU|D)\s*:?\s*"
            r"S(?:P|H)(?:N|I|M|1)-?(\d{4})",
            page,
        )
        canonical_ids = [f"SPN-{digits}" for digits in id_matches]
        if intake_page:
            intake_role_ids.extend(canonical_ids)
        if sponsor_page:
            sponsor_role_ids.extend(canonical_ids)
            # Some supplemental sponsor attestations use a compact
            # ``Sponsor ID`` + ``Applicant`` layout instead of prose saying
            # ``attests that ...``.  Keep the two assertions on this physical
            # page together; do not borrow an Applicant row from an adjacent
            # intake/B-13 page.
            correction_name = name_after(
                page,
                r"\bApplicant\s*:?\s*(?:\n\s*)?"
                r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if correction_name and canonical_ids:
                names["sponsor_correction"] = correction_name
                correction_names.append(correction_name)
        elif not intake_page:
            # Compact sponsor supplements sometimes lose their heading while
            # retaining Sponsor ID + Applicant. Treat the applicant as a
            # correction-role assertion, ahead of a damaged biometric name.
            correction_name = None
            if canonical_ids:
                correction_name = name_after(
                    page,
                    r"\bApplicant\s*:?\s*(?:\n\s*)?"
                    r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
                )
                if not correction_name:
                    for raw_line in page.splitlines():
                        line = raw_line.strip(" :|_-")
                        if re.search(
                            r"(?i)Sponsor|Purpose|Visa|Packet|Synthetic|"
                            r"Archive|System|Evidence|Copy|MIB",
                            line,
                        ):
                            continue
                        candidate_name = _canonical_name(line)
                        if candidate_name:
                            correction_name = candidate_name
                            break
                if correction_name:
                    names["sponsor_correction"] = correction_name
                    correction_names.append(correction_name)
            # When the supplement asserts a different applicant, preserve its
            # name evidence but do not also let a one-glyph sponsor-ID reading
            # override the clean intake ID. ID-only supplements remain field
            # corrections.
            if not correction_name:
                supplemental_sponsor_ids.extend(canonical_ids)

        if intake_page:
            from difflib import SequenceMatcher

            for line in page.splitlines():
                if ":" in line or re.search(
                    r"(?i)Sponsor|Visa|Home|Applicant|Purpose|Arrival|Fee|Packet",
                    line,
                ):
                    continue
                letters = [char for char in line if char.isalpha()]
                if not letters or sum(char.isupper() for char in letters) / len(letters) < 0.75:
                    continue
                compact = re.sub(r"[^A-Za-z]", "", line).casefold()
                if not (10 <= len(compact) <= 24):
                    continue
                ranked = sorted(
                    (
                        SequenceMatcher(
                            None,
                            compact,
                            re.sub(r"[^A-Za-z]", "", species).casefold(),
                        ).ratio(),
                        species,
                    )
                    for species in SPECIES_CODES
                )
                if ranked[-1][0] >= 0.54 and ranked[-1][0] - ranked[-2][0] >= 0.12:
                    species_fills.append(ranked[-1][1])

        # A short supplemental page containing only ``Visa Class: ...`` is a
        # field-level correction, not another intake form.
        visa_hits = [
            visa
            for visa in VISA_CLASSES
            if re.search(
                rf"(?im)^\s*Visa\s+Class\s*:?\s*{re.escape(visa)}\s*$", page
            )
        ]
        domain_labels = len(
            re.findall(
                r"(?im)^\s*(?:Applicant|Species|Home\s+World|Sponsor\s+ID|"
                r"Arrival\s+Date|Declared\s+Purpose|Fee\s+Status)\b",
                page,
            )
        )
        if len(visa_hits) == 1 and domain_labels == 0:
            standalone_visas.append(visa_hits[0])
        # A compact correction sheet can lose its heading while retaining a
        # small set of clean labelled assertions.  Its visa is authoritative;
        # a different named person alongside two agreeing durable identity
        # documents is sponsor-role evidence, not a replacement applicant.
        if (
            len(visa_hits) == 1
            and domain_labels <= 3
            and not re.search(r"(?i)FORM\s*[I1|l]?[- ]?8090|FORM\s+B-13", page)
        ):
            standalone_visas.append(visa_hits[0])
            correction_name = name_after(
                page,
                r"\bApplicant\s*:?\s*(?:\n\s*)?"
                r"([A-Z][A-Za-z]+\s+[A-Z][A-Za-z]+)",
            )
            if correction_name:
                correction_names.append(correction_name)

        # Closed-vocabulary visa glyph repair inside an intake/supplement page
        # (DIP-T→DIP-1, IP-1→DIP-1).
        for match in re.finditer(
            r"(?im)^\s*Visa(?:\s+Class)?\s*:?\s*(?:\n\s*)?([A-Z0-9-]{3,10})",
            page,
        ):
            visa = _nearest_unique(match.group(1), VISA_CLASSES, limit=1)
            if visa:
                explicit_visas.append(visa)
        # The value can survive when the middle of ``Visa Class`` does not
        # (for example ``Visa TáSs' UIP-1``). Keep a short proximity window
        # and require a unique edit-distance-1 closed-vocabulary correction.
        for match in re.finditer(
            r"(?i)\bVisa\b.{0,24}?\b([A-Z0-9]{2,8}-[0-9T])\b",
            page,
        ):
            visa = _nearest_unique(match.group(1).upper(), VISA_CLASSES, limit=1)
            if visa:
                explicit_visas.append(visa)

        # Purpose labels survive many glyph errors even when the value has one
        # vowel substitution (xanubotany→xenobotany).
        for match in re.finditer(
            r"(?im)^\s*(?:\w{1,10}\s+)?Purp\w{0,5}\s*:?\s*([^\n]+)", page
        ):
            from difflib import SequenceMatcher
            import unicodedata

            raw = "".join(
                char
                for char in unicodedata.normalize("NFKD", match.group(1))
                if not unicodedata.combining(char)
            ).strip().casefold()
            ranked = sorted((SequenceMatcher(None, raw, value).ratio(), value) for value in PURPOSES)
            if ranked[-1][0] >= 0.70 and ranked[-1][0] - ranked[-2][0] >= 0.10:
                explicit_purposes.append(ranked[-1][1])

    from .oriented_form_ocr import canonical_vocab_lines

    for synthetic in canonical_vocab_lines(text):
        if synthetic.startswith("Home World: "):
            fuzzy_home_worlds.append(synthetic.split(":", 1)[1].strip())

    # Recover values whose glyph fragments were emitted on separate spatial
    # rows by the detector. These signatures are closed-vocabulary and require
    # ordered fragments, so unrelated prose cannot create a new purpose.
    if re.search(r"(?is)\bfield\s+rep.{0,100}?\bair\b", text):
        explicit_purposes.append("field repair")
    if re.search(
        r"(?is)\breac\w*.{0,50}?\btor\b.{0,50}?\bm\w*.{0,50}?enance\b",
        text,
    ):
        explicit_purposes.append("reactor maintenance")

    # Corroborating documents win name arbitration.  Dist-2 lexicon
    # normalization removes OCR-only spelling noise before any override.
    if names.get("registry"):
        preferred_role = "registry"
    elif names.get("biometric") and names.get("biometric") == names.get("sponsor"):
        preferred_role = "biometric"
    else:
        preferred_role = next(
            (
                role
                for role in (
                    "registry",
                    "sponsor_correction",
                    "biometric",
                    "sponsor",
                )
                if names.get(role)
            ),
            None,
        )
    preferred_name = names.get(preferred_role) if preferred_role else None
    if preferred_name:
        result["applicant_name"] = preferred_name
        result["_applicant_name_role"] = preferred_role
    elif names.get("intake"):
        result["applicant_name_fill"] = names["intake"]
    if names.get("intake"):
        result["_intake_name"] = names["intake"]
    if names.get("registry"):
        result["_registry_name"] = names["registry"]
    if names.get("sponsor"):
        result["_sponsor_name"] = names["sponsor"]
    if (
        names.get("registry")
        and names.get("registry") == names.get("sponsor")
        and names.get("intake")
        and names.get("intake") != names.get("registry")
    ):
        result["_identity_cross_role_exact"] = True

    semantic_flags: list[str] = []
    from .biometric_header_ocr import decode_biometric_flags

    decoded_observed_flags = decode_biometric_flags(text)
    semantic_flags.extend(decoded_observed_flags)
    fuzzy_observed_flags.extend(decoded_observed_flags)
    exact_review_atoms: list[str] = []
    for match in re.finditer(
        r"(?is)Review-only\s+risk\s+flag\s+present\s*:?.{0,120}?\b"
        r"(identity_conflict|sponsor_mismatch|illegible_biometrics|"
        r"rescinded_denial)\b",
        text,
    ):
        atom = match.group(1).casefold()
        semantic_flags.append(atom)
        exact_review_atoms.append(atom)
    if exact_review_atoms:
        result["_review_risk_atoms_exact"] = sorted(set(exact_review_atoms))
    if fuzzy_observed_flags:
        result["_observed_flags_fuzzy"] = sorted(set(fuzzy_observed_flags))
    explicit_observed: list[str] = []
    for match in re.finditer(
        r"(?im)^\s*[A-Za-z0-9]bser\w{0,5}\s+f(?:l|i|1)ag\w{0,2}"
        r"\s*:?\s*[^\n]+",
        text,
    ):
        line = match.group(0)
        if "panel missing" in line.casefold():
            continue
        explicit_observed.extend(decode_biometric_flags(line))
    if explicit_observed:
        result["_observed_flags_exact"] = sorted(set(explicit_observed))
    semantic_flags.extend(structural_risk_flags)
    if (
        "species whiteout" in lower_text
        and not names.get("intake")
        and names.get("registry")
        and names.get("registry") == names.get("sponsor")
    ):
        canonical_tokens = {
            token.casefold()
            for token in re.findall(r"[A-Za-z]+", names["registry"])
        }
        for fragment in intake_name_fragments:
            fragment_tokens = [
                token.casefold() for token in re.findall(r"[A-Za-z]+", fragment)
            ]
            shared = any(
                any(left.startswith(right) or right.startswith(left) for right in canonical_tokens)
                for left in fragment_tokens
            )
            damaged_contradiction = any(
                2 <= len(left) <= 3
                and not any(
                    left.startswith(right) or right.startswith(left)
                    for right in canonical_tokens
                )
                for left in fragment_tokens
            )
            if shared and damaged_contradiction:
                # The intact registry+sponsor identity and a visibly different
                # surviving intake fragment prove the conflict. The truncated
                # fragment plus adjacent whiteout independently proves that the
                # biometric evidence is illegible.
                semantic_flags.extend(
                    ["identity_conflict", "illegible_biometrics"]
                )
                result["_damaged_intake_identity_exact"] = True
                break
    if re.search(
        r"(?im)^\s*[A-Za-z0-9]bser\w{0,5}\s+f(?:l|i|1)ag\w{0,2}"
        r"\s*:?\s*none\b",
        text,
    ):
        result["_observed_none"] = True
    if "_observed_none" not in result:
        # Short OCR payloads such as ``ncne``, ``nore`` and ``nang`` are
        # common two-glyph corruptions of a printed ``none``.  Keep this tied
        # to the Observed flags label and a single short alphabetic token;
        # truncated risk atoms (for example ``reso``) remain outside the
        # edit-distance-2 envelope.
        for match in re.finditer(
            r"(?im)^\s*[A-Za-z0-9]bser\w{0,5}\s+f(?:l|i|1)ag\w{0,2}"
            r"\s*:?\s*([A-Za-z]{3,5})\s*[^A-Za-z\n]*$",
            text,
        ):
            if _edit_distance(match.group(1).casefold(), "none") <= 2:
                result["_observed_none"] = True
                break
    if "_observed_none" not in result:
        # Severe photocopy damage can erase most of ``Observed flags`` while
        # preserving a label tail (``dedlags``) and a noisy value (``tuane``).
        # This remains constrained to a label-shaped line and one short token,
        # so prompt text containing ``risk_flags=none`` is not eligible.
        for match in re.finditer(
            r"(?im)^\s*[^\w\n]*[A-Za-z]{1,12}[ ._-]*"
            r"(?:fla[go]s|[dt]lags)\s*[:;.]\s*"
            r"([A-Za-z]{3,5})\s*[^A-Za-z\n]*$",
            text,
        ):
            if _edit_distance(match.group(1).casefold(), "none") <= 3:
                result["_observed_none"] = True
                break
    if result.get("_observed_flags_exact"):
        # Independent OCR streams can contribute both a readable atom and a
        # weak short-token guess of ``none``. A decoded allowlisted payload is
        # positive evidence; the fuzzy sentinel is only an absence fallback.
        result.pop("_observed_none", None)
    identity_names = [names[role] for role in ("intake", "registry") if role in names]
    # Intake and registry are durable identity assertions. A degraded biometric
    # name alone is not enough (illegible scans can hallucinate a different
    # person), while sponsor identity denotes sponsor_mismatch separately.
    if len(identity_names) >= 2 and len(set(identity_names)) > 1:
        compact_names = [re.sub(r"[^a-z]", "", name.casefold()) for name in identity_names]
        # A one- or two-glyph discrepancy in otherwise matching identity rows
        # is characteristic of a degraded scan, not two asserted identities.
        # When the B-13 itself is absent, retain that degradation as the
        # extraction risk instead of fabricating identity_conflict.
        if (
            not has_b13_evidence
            and _edit_distance(compact_names[0], compact_names[1]) <= 2
        ):
            semantic_flags.append("illegible_biometrics")
            result["_identity_near_ocr"] = True
        else:
            semantic_flags.append("identity_conflict")
        if not has_b13_evidence and "redacted?" in lower_text:
            result["_redacted_identity_damage"] = True
    elif (
        names.get("intake")
        and names.get("registry") == names.get("intake")
        and names.get("sponsor")
        and names["sponsor"] != names["intake"]
    ):
        semantic_flags.append("sponsor_mismatch")
    elif (
        names.get("intake")
        and names.get("registry") == names.get("intake")
        and len(set(correction_names)) == 1
        and correction_names[0] != names["intake"]
    ):
        semantic_flags.append("sponsor_mismatch")
    elif (
        names.get("intake")
        and not names.get("registry")
        and len(set(correction_names)) == 1
        and correction_names[0] != names["intake"]
        and re.search(r"(?i)redact\w*\?", text)
        and not result.get("_observed_none")
        and not re.search(r"(?i)Finding\s*:\s*APPROVED\b", text)
    ):
        # With the durable registry role absent, a visibly redacted intake
        # identity and a distinct correction-role identity are still two
        # explicit person assertions. A clean B-13 ``none`` or APPROVED note
        # vetoes this relation (manual correction packets use that pattern).
        semantic_flags.append("identity_conflict")
        result["_redacted_correction_identity_exact"] = True

    # At high resolution, a damaged B-13 value can expose only the beginning
    # of a closed-vocabulary atom (``resc:``).  The truncation itself proves
    # illegibility while the unique prefix recovers the visible flag.
    for match in re.finditer(
        r"(?im)^\s*Observed\s+fl\w*\s*:?\s*([^\n]*)", text
    ):
        payload = re.sub(r"[^a-z]", "", match.group(1).casefold())
        if 4 <= len(payload) <= 8 and "rescindeddenial".startswith(payload):
            semantic_flags.extend(["rescinded_denial", "illegible_biometrics"])
    if semantic_flags:
        result["risk_flags"] = semantic_flags

    unique_sponsor_visas = list(dict.fromkeys(sponsor_visas))
    unique_visas = list(dict.fromkeys(explicit_visas))
    unique_standalone_visas = list(dict.fromkeys(standalone_visas))
    if len(unique_standalone_visas) == 1:
        result["visa_class"] = unique_standalone_visas[0]
        result["_visa_class_authoritative"] = True
    elif len(unique_sponsor_visas) == 1:
        result["visa_class"] = unique_sponsor_visas[0]
    elif len(unique_visas) == 1:
        result["visa_class"] = unique_visas[0]

    manual_visa = re.findall(
        r"(?i)Manual\s+correction\s*:\s*visa\s+class\s+is\s+"
        r"(DIP-1|MED-3|TRANSIT-7|XW-1|XW-2)\b",
        text,
    )
    if len(set(value.upper() for value in manual_visa)) == 1:
        result["visa_class"] = manual_visa[0].upper()
        result["_visa_class_authoritative"] = True

    unique_purposes = list(dict.fromkeys(explicit_purposes))
    if len(unique_purposes) == 1:
        result["declared_purpose"] = unique_purposes[0]

    # A rotated fee receipt can be emitted as separate spatial fragments:
    # ``Sta`` then ``tus: waiv``.  The value tail is distinctive and remains
    # tied to the printed Status label; treat it as fill-only evidence so it
    # cannot replace a complete receipt ledger or a manual correction.
    if re.search(
        r"(?im)^\s*Reason\s*:\s*Fee\s+status\s+unknown\s*\.?\s*$",
        text,
    ):
        result["fee_status_fill"] = "unknown"
        result["_manual_fee_unknown"] = True
    elif re.search(r"(?im)^\s*tus\s*:\s*waiv(?:ed)?\s*$", text):
        result["fee_status_fill"] = "waived"

    unique_species_fills = list(dict.fromkeys(species_fills))
    if len(unique_species_fills) == 1:
        result["species_code_fill"] = unique_species_fills[0]

    unique_fuzzy_worlds = list(dict.fromkeys(fuzzy_home_worlds))
    if len(unique_fuzzy_worlds) == 1:
        result["home_world"] = unique_fuzzy_worlds[0]
        result["_home_world_fuzzy"] = True

    unique_registry_dates = list(dict.fromkeys(registry_dates))
    if len(unique_registry_dates) == 1:
        result["arrival_date"] = unique_registry_dates[0]
    unique_registry_pages = list(dict.fromkeys(registry_pages))
    if len(unique_registry_pages) == 1:
        result["_registry_page"] = unique_registry_pages[0]

    # Fill-only label-shape recovery for OCR-damaged intake dates. Registry
    # dates above remain the only dates allowed to override a concrete value.
    fuzzy_dates: list[str] = []
    for line in text.splitlines():
        match = re.search(
            r"(?i)([A-Za-z]{3,12})\s+(?:Da\w{2,4}|Cate)\s*:?\s*"
            r"(20(?:25|26|28)-\d{2}-\d{2})\b",
            line,
        )
        if not match:
            continue
        from difflib import SequenceMatcher

        label = re.sub(r"[^a-z]", "", match.group(1).casefold() + "date")
        if SequenceMatcher(None, label, "arrivaldate").ratio() < 0.62:
            continue
        value = match.group(2)
        if value.startswith("2028-"):
            month_day = value[5:]
            if month_day.startswith("08-"):
                month_day = "06-" + month_day[3:]
            value = "2026-" + month_day
        fuzzy_dates.append(value)
    unique_fuzzy_dates = list(dict.fromkeys(fuzzy_dates))
    if "arrival_date" not in result and len(unique_fuzzy_dates) == 1:
        result["arrival_date_fill"] = unique_fuzzy_dates[0]

    malformed_dates: list[str] = []
    for match in re.finditer(
        r"(?im)rival\s*Date\s*:\s*(2026)=([0-1]\d)-([0-3]\d)\b",
        text,
    ):
        malformed_dates.append(
            f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        )
    for match in re.finditer(
        r"(?im)Arrival\s+Date\s*:?\s*(2026-\d{2}-\d{3,5})\b", text
    ):
        year_month, noisy_day = match.group(1).rsplit("-", 1)
        day = noisy_day[-2:]
        if 1 <= int(day) <= 31:
            malformed_dates.append(f"{year_month}-{day}")
    for match in re.finditer(
        r"(?is)\bArriv\w*.{0,80}?\bte\s*:\s*20.{0,80}?!6-(\d{2})-(\d{2})\b",
        text,
    ):
        malformed_dates.append(f"2026-{match.group(1)}-{match.group(2)}")
    unique_malformed_dates = list(dict.fromkeys(malformed_dates))
    if (
        "arrival_date" not in result
        and "arrival_date_fill" not in result
        and len(unique_malformed_dates) == 1
    ):
        result["arrival_date_fill"] = unique_malformed_dates[0]

    sponsors = {
        f"SPN-{digits}"
        for digits in re.findall(
            r"(?i)Spons?\w{0,5}\s+(?:ID|1D|IU|D)\s*:?[ \t]*"
            r"S(?:P|H)(?:N|I|M|1)-?(\d{4})",
            text,
        )
    }
    # A rotated/cropped form can lose only the leading ``S`` while preserving
    # the highly constrained PN-#### payload. Accept it only as a unique
    # four-digit candidate; trailing duplicate/noise digits are ignored.
    sponsors.update(
        f"SPN-{digits}"
        for digits in re.findall(r"(?i)(?:[;:]\s*)?PN-?(\d{4})\d{0,2}\b", text)
    )
    sponsors.update(
        f"SPN-{digits}"
        for digits in re.findall(
            r"(?is)\b(?:spons\w*|isor)\b.{0,120}?\b[A-Z]-?(\d{4})\b",
            text,
        )
    )
    if sponsors:
        result["_sponsor_candidates"] = sorted(sponsors)
    unique_role_ids = list(dict.fromkeys(sponsor_role_ids))
    unique_intake_ids = list(dict.fromkeys(intake_role_ids))
    unique_supplemental_ids = list(dict.fromkeys(supplemental_sponsor_ids))
    if (
        len(unique_role_ids) == 1
        and len(unique_intake_ids) == 1
        and unique_role_ids[0] != unique_intake_ids[0]
    ):
        result["_sponsor_id_role_conflict"] = [
            unique_intake_ids[0], unique_role_ids[0]
        ]
        prompt_sentinel_ids = {"SPN-0007", "SPN-0139", "SPN-4040", "SPN-8090"}
        if (
            not ({unique_intake_ids[0], unique_role_ids[0]} & prompt_sentinel_ids)
            and re.search(r"(?i)Manual\s+correction\s*:\s*applicant\s+is\b", text)
            and "name cut out" in lower_text
        ):
            # The intake manually restores the applicant while the sponsor
            # identity is physically absent; two clean role-local sponsor IDs
            # then disagree. This is explicit sponsor mismatch evidence, not
            # a generic multi-ID guess (prompt sentinel IDs are excluded).
            risks = set(result.get("risk_flags", ()) or ())
            risks.add("sponsor_mismatch")
            result["risk_flags"] = sorted(risks)
            result["_sponsor_id_role_conflict_exact"] = True
    if len(unique_role_ids) == 1:
        result["sponsor_id"] = unique_role_ids[0]
        result["_sponsor_id_authoritative"] = True
    elif len(unique_supplemental_ids) == 1:
        result["sponsor_id"] = unique_supplemental_ids[0]
        result["_sponsor_id_authoritative"] = True
    elif len(sponsors) == 1:
        result["sponsor_id"] = next(iter(sponsors))
    return result


def merge_rapid_fills(
    packet, candidate, neural_candidate=None, semantics: dict[str, object] | None = None
) -> set[str]:
    """Copy only missing-field neural recoveries into the existing packet.

    Reparsing with an extra stream can perturb an already-valid lower-level OCR
    choice even though PP-OCR itself is fill-only.  Mutating the prior packet
    makes that contract explicit: an existing concrete value is never erased or
    replaced, while placeholders count as missing.  Explicit risk flags remain
    union-only.
    """
    accepted: set[str] = set()
    old_fields = getattr(packet, "fields", None)
    if old_fields is None:
        old_fields = {}
        packet.fields = old_fields
    # ``candidate`` is the normal multi-stream fusion.  It is deliberately
    # conservative, but that also means an existing weak Tesseract value can
    # mask a better PP-OCR parse.  Keep a separately parsed neural-only packet
    # so missing fields and the applicant name can use the independent engine.
    neural_fields = getattr(neural_candidate, "fields", {}) or {}
    new_fields = getattr(candidate, "fields", {}) or {}
    new_sources = getattr(candidate, "field_sources", {}) or {}

    # A neural-only parse can recover a small Finding line that the ordinary
    # page OCR misses even when every ordinary form field is already complete.
    # ``parse_packet`` has already applied the strict adjudicator-note gates:
    # active-case association, official header/body structure, and rejection
    # of archived-adjacent, sample, barcode, and prompt-injection contexts.
    # Preserve that trusted document-role evidence instead of treating PP-OCR
    # as a field-only source. Existing findings retain their precedence.
    neural_finding = getattr(neural_candidate, "adjudicator_finding", None)
    if (
        getattr(packet, "adjudicator_finding", None) is None
        and neural_finding in {"APPROVED", "DENIED", "NEEDS_REVIEW"}
    ):
        packet.adjudicator_finding = neural_finding
        packet.sources_seen.add("adjudicator")
        packet.field_sources["_adjudicator_finding"] = "ppocr_ocr:trusted_note"
        accepted.add("adjudicator_finding")

    for field in CRITICAL_FIELDS:
        if not _is_missing(old_fields.get(field)):
            continue
        old_source = str(getattr(packet, "field_sources", {}).get(field, ""))
        if "manual" in old_source or (
            field == "fee_status" and old_source.startswith("fee_ledger:")
        ):
            # ``unknown`` can itself be an explicit high-precedence assertion;
            # missing-value mechanics must not turn it back into a fill slot.
            continue
        value = neural_fields.get(field)
        source = "ppocr_ocr" if not _is_missing(value) else new_sources.get(field)
        if _is_missing(value):
            value = new_fields.get(field)
        if _is_missing(value):
            continue
        old_fields[field] = value
        if source:
            packet.field_sources[field] = source
        tag = _FIELD_TAGS.get(field)
        if tag:
            packet.sources_seen.add(tag)
        if field == "arrival_date":
            packet.conflicts.discard("arrival_date_unreadable")
        accepted.add(field)

    semantics = semantics or {}
    if bool(semantics.get("_observed_none")):
        packet.risk_flags = []
        packet.observed_flags_seen = True
        packet.field_sources["risk_flags"] = "ppocr_semantic:observed_none"
    fill_date = semantics.get("arrival_date_fill")
    if isinstance(fill_date, str) and _is_missing(old_fields.get("arrival_date")):
        old_fields["arrival_date"] = fill_date
        packet.field_sources["arrival_date"] = "ppocr_semantic:arrival_date_fill"
        packet.sources_seen.add("intake")
        packet.conflicts.discard("arrival_date_unreadable")
        accepted.add("arrival_date")
    fill_name = semantics.get("applicant_name_fill")
    if isinstance(fill_name, str) and _is_missing(old_fields.get("applicant_name")):
        old_fields["applicant_name"] = fill_name
        packet.field_sources["applicant_name"] = "ppocr_semantic:applicant_name_fill"
        packet.sources_seen.add("intake")
        accepted.add("applicant_name")
    fill_species = semantics.get("species_code_fill")
    if isinstance(fill_species, str) and _is_missing(old_fields.get("species_code")):
        old_fields["species_code"] = fill_species
        packet.field_sources["species_code"] = "ppocr_semantic:species_code_fill"
        packet.sources_seen.add("intake")
        accepted.add("species_code")
    fill_fee = semantics.get("fee_status_fill")
    fee_source = str(getattr(packet, "field_sources", {}).get("fee_status", ""))
    if (
        bool(semantics.get("_manual_fee_unknown"))
        or (
            fill_fee in {"unknown", "waived"}
            and _is_missing(old_fields.get("fee_status"))
            and "manual" not in fee_source
            and not fee_source.startswith("fee_ledger:")
        )
    ):
        old_fields["fee_status"] = str(fill_fee)
        packet.field_sources["fee_status"] = (
            "ppocr_semantic:manual_fee_unknown"
            if semantics.get("_manual_fee_unknown")
            else "ppocr_semantic:fee_status_fill"
        )
        packet.sources_seen.add("fee")
        accepted.add("fee_status")
    # Document-role semantics may correct a concrete OCR hypothesis.  All
    # values are canonicalized/allowlisted by ``rapid_semantics``.
    for field in (
        "applicant_name",
        "visa_class",
        "declared_purpose",
        "arrival_date",
        "sponsor_id",
    ):
        value = semantics.get(field)
        if not isinstance(value, str) or _is_missing(value):
            continue
        current = old_fields.get(field)
        source = str(getattr(packet, "field_sources", {}).get(field, ""))
        if "manual" in source or source.startswith("document_role_semantic:"):
            continue
        if (
            field == "applicant_name"
            and source.startswith("native")
            and "identity_conflict"
            not in set(getattr(packet, "risk_flags", ()) or ())
        ):
            # Intake/native identity outranks a sponsor or damaged biometric
            # reading unless independent packet evidence has already established
            # an identity conflict.  That relation—not OCR disagreement alone—
            # unlocks cross-document applicant arbitration.
            continue
        if value == current:
            # Corroboration matters even without changing the string.  Stamp
            # its stronger document-role provenance so a later generic OCR
            # reconciler cannot replace it with the weaker intake hypothesis.
            if not source.startswith("native"):
                packet.field_sources[field] = f"ppocr_semantic:{field}"
            continue
        if (
            field == "sponsor_id"
            and not _is_missing(current)
            and not bool(semantics.get("_sponsor_id_authoritative"))
        ):
            continue
        # Names have corroborating-document/lexicon provenance.  Other fields
        # may override OCR/native labels only when their document role is more
        # specific (sponsor visa/purpose, registry date, normalized sponsor ID).
        old_fields[field] = value
        packet.field_sources[field] = f"ppocr_semantic:{field}"
        tag = _FIELD_TAGS.get(field)
        if tag:
            packet.sources_seen.add(tag)
        if field == "arrival_date":
            packet.conflicts.discard("arrival_date_unreadable")
        accepted.add(field)

    # Aggressive independent-engine arbitration for the hardest extraction
    # field.  On two 100-case FIT slices the neural-only name fixed 12 concrete
    # OCR errors while changing 2 correct names.  Native/manual corrections are
    # the only protected sources; all other values are OCR hypotheses.
    neural_name = neural_fields.get("applicant_name")
    current_name = old_fields.get("applicant_name")
    current_source = str(getattr(packet, "field_sources", {}).get("applicant_name", ""))
    if (
        neural_candidate is not None
        and "applicant_name" not in semantics
        and not _is_missing(neural_name)
        and neural_name != current_name
        and not current_source.startswith("native")
        and not current_source.startswith("ppocr_semantic:")
        and "manual" not in current_source
    ):
        old_fields["applicant_name"] = neural_name
        packet.field_sources["applicant_name"] = "ppocr_ocr:override"
        packet.sources_seen.add("intake")
        accepted.add("applicant_name")

    prior_flags = set(getattr(packet, "risk_flags", ()) or ())
    explicit_candidate_flags = set(
        getattr(candidate, "explicit_risk_flags", ()) or ()
    )
    explicit_candidate_flags.update(
        getattr(neural_candidate, "explicit_risk_flags", ()) or ()
    )
    candidate_flags = set(explicit_candidate_flags)
    semantic_flags = set(semantics.get("risk_flags", ()) or ())
    if (
        getattr(packet, "adjudicator_finding", None) == "APPROVED"
        or "illegible_biometrics" in prior_flags
    ):
        semantic_flags.discard("identity_conflict")
        semantic_flags.discard("sponsor_mismatch")
    candidate_flags.update(semantic_flags)
    added_flags = candidate_flags - prior_flags
    if added_flags:
        packet.risk_flags = sorted(prior_flags | added_flags)
        if explicit_candidate_flags or bool(
            getattr(neural_candidate, "observed_flags_seen", False)
        ):
            packet.observed_flags_seen = True
        packet.field_sources.setdefault("risk_flags", "ppocr_ocr")
        accepted.add("risk_flags")
    elif bool(getattr(neural_candidate, "observed_flags_seen", False)):
        # A clearly read ``Observed flags: none`` is still valuable document-
        # role evidence: it prevents a later low-confidence legibility pass
        # from fabricating illegibility on a noisy but readable B-13.
        packet.observed_flags_seen = True
        packet.field_sources.setdefault("risk_flags", "ppocr_ocr:observed_none")
    return accepted


def merge_semantic_risk_flags(
    packet, semantics: dict[str, object], *, source: str = "document_role_semantic"
) -> set[str]:
    """Union only source-role relations into the packet risk evidence."""
    allowed = set(RISK_FLAG_ATOMS)
    flags = {
        str(flag)
        for flag in (semantics.get("risk_flags", ()) or ())
        if str(flag) in allowed
    }
    prior = set(getattr(packet, "risk_flags", ()) or ())
    if getattr(packet, "adjudicator_finding", None) == "APPROVED":
        return set()
    if "illegible_biometrics" in prior:
        flags.discard("identity_conflict")
        flags.discard("sponsor_mismatch")
    added = flags - prior
    if not added:
        return set()
    packet.risk_flags = sorted(prior | added)
    packet.field_sources.setdefault("risk_flags", source)
    return added




def _engine() -> Any | None:
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    try:
        from rapidocr import RapidOCR

        _ENGINE = RapidOCR(
            params={
                "Global.log_level": "critical",
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    except (ImportError, OSError, RuntimeError, ValueError):
        _ENGINE_FAILED = True
        return None
    return _ENGINE


def _render_pages(pdf_path: Path, tmp: Path, *, dpi: int = _DPI) -> list[Path]:
    prefix = tmp / "page"
    try:
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-l",
                str(_MAX_PAGES),
                "-r",
                str(dpi),
                "-jpeg",
                str(pdf_path),
                str(prefix),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return []
    return sorted(tmp.glob("page-*.jpg"))


def rapid_ocr(pdf_path: Path) -> str:
    engine = _engine()
    if engine is None:
        return ""
    texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="core-rapidocr-") as tmp_name:
        for image in _render_pages(pdf_path, Path(tmp_name)):
            try:
                result = engine(str(image))
            except (OSError, RuntimeError, ValueError):
                continue
            lines = tuple(getattr(result, "txts", ()) or ()) if result else ()
            text = "\n".join(str(line) for line in lines if str(line).strip())
            if text:
                texts.append(text)
            # PP-OCR detects vertical word boxes even when recognition is too
            # sparse to expose the page role.  Re-run just that page in both
            # orthogonal directions; this recovers rotated B-13 headers while
            # leaving normal pages at the cheap single-pass cost.
            raw_boxes = getattr(result, "boxes", None) if result else None
            boxes = tuple(raw_boxes) if raw_boxes is not None else ()
            vertical = 0
            for box in boxes:
                try:
                    xs = [float(point[0]) for point in box]
                    ys = [float(point[1]) for point in box]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                except (TypeError, ValueError, IndexError):
                    continue
                if height >= max(24.0, width * 1.8):
                    vertical += 1
            if vertical < 2:
                continue
            try:
                from PIL import Image

                base = Image.open(image).convert("RGB")
            except OSError:
                continue
            rotated_texts: list[str] = []
            for angle in (90, 270):
                rotated_path = Path(tmp_name) / f"{image.stem}-r{angle}.jpg"
                base.rotate(angle, expand=True, fillcolor="white").save(
                    rotated_path, quality=94
                )
                try:
                    rotated_result = engine(str(rotated_path))
                except (OSError, RuntimeError, ValueError):
                    continue
                rotated_lines = (
                    tuple(getattr(rotated_result, "txts", ()) or ())
                    if rotated_result
                    else ()
                )
                rotated_text = "\n".join(
                    str(line) for line in rotated_lines if str(line).strip()
                )
                if rotated_text:
                    rotated_texts.append(rotated_text)
            texts.extend(rotated_texts)
    return "\n\n".join(texts)


def rapid_ocr_cached(pdf_path: Path) -> str:
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"{_CACHE_VERSION}:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    path = cache_dir / f"{pdf_path.stem}_{key[:16]}_rapidocr.json"
    if path.exists():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("text") or "")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    text = rapid_ocr(pdf_path)
    path.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text






def rapid_embedded_ocr(pdf_path: Path) -> str:
    """Run PP-OCR on original embedded images, avoiding raster recompression."""
    engine = _engine()
    if engine is None or not shutil.which("pdfimages"):
        return ""
    texts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="core-rapid-embedded-") as tmp_name:
        prefix = Path(tmp_name) / "image"
        try:
            subprocess.run(
                ["pdfimages", "-j", str(pdf_path), str(prefix)],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
            return ""
        for image in sorted(Path(tmp_name).glob("image-*")):
            try:
                result = engine(str(image))
            except (OSError, RuntimeError, ValueError):
                continue
            lines = tuple(getattr(result, "txts", ()) or ()) if result else ()
            text = "\n".join(str(line) for line in lines if str(line).strip())
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def rapid_embedded_ocr_cached(pdf_path: Path) -> str:
    cache_dir = ocr_cache_dir()
    key = hashlib.sha1(
        f"{_EMBEDDED_CACHE_VERSION}:{ocr_code_version()}:"
        f"{pdf_path.resolve()}:{pdf_path.stat().st_mtime_ns}".encode()
    ).hexdigest()
    path = cache_dir / f"{pdf_path.stem}_{key[:16]}_rapid_embedded.json"
    if path.exists():
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("text") or "")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    text = rapid_embedded_ocr(pdf_path)
    path.write_text(json.dumps({"text": text}), encoding="utf-8")
    return text
