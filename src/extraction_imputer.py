"""Missing-only closed-vocabulary recovery from packet evidence features.

OCR remains authoritative whenever it emits a concrete value.  This module is
the last extraction stage and predicts only a sentinel field from broad packet
features plus the other already-extracted fields.  It runs after adjudication,
so it cannot alter policy decisions or confidence.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from .constants import FEE_STATUSES, HOME_WORLDS, PURPOSES, SPECIES_CODES, VISA_CLASSES

_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "extraction_imputer.joblib"
TARGET_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "fee_status",
)
_ALLOWLISTS = {
    "species_code": frozenset(SPECIES_CODES),
    "home_world": frozenset(HOME_WORLDS),
    "visa_class": frozenset(VISA_CLASSES),
    "declared_purpose": frozenset(PURPOSES),
    "fee_status": frozenset(FEE_STATUSES),
}
_MISSING = frozenset({"", "unknown", "none", "n/a", "null"})
_PROVENANCE_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "fee_status",
)
_STANDARD_SOURCES = frozenset(
    {"native", "page_ocr", "embedded_ocr", "ppocr_ocr", "oriented_ocr"}
)


def _is_missing(value: object) -> bool:
    return str(value or "").strip().casefold() in _MISSING


@lru_cache(maxsize=1)
def load_extraction_imputer() -> dict[str, Any] | None:
    if not _ARTIFACT.is_file():
        return None
    try:
        artifact = joblib.load(_ARTIFACT)
    except (OSError, ValueError, TypeError, ImportError):
        return None
    if not isinstance(artifact, dict):
        return None
    anti = artifact.get("anti_leakage") or {}
    if any(
        anti.get(key)
        for key in ("contains_case_ids", "contains_filenames", "contains_document_hashes")
    ):
        return None
    if not isinstance(artifact.get("models"), dict):
        return None
    return artifact


def _model_features(base_features: dict[str, object], prediction: dict) -> dict:
    features = dict(base_features)
    for field in TARGET_FIELDS:
        features[f"field_{field}"] = str(prediction.get(field, "unknown"))
    return features


def stabilize_provenance_features(
    base_features: dict[str, object], field_sources: dict[str, str]
) -> dict[str, object]:
    """Map post-training evidence refinements onto their original source class.

    The imputer was fit before high-resolution OCR, ledger arbitration, and
    cross-resolution consensus were added.  Treating those mechanisms as new
    ``specialized`` categories causes representation drift: the same document
    content can flip an unrelated missing field merely because provenance got
    more precise.  Collapse only those newer wrappers to the source class the
    fitted model saw; adjudication keeps the full provenance unchanged.
    """

    def canonical(source: str) -> str:
        if source.startswith(("fee_ledger:", "manual_note:")):
            parts = source.split(":", 2)
            return parts[1] if len(parts) >= 2 else source
        if source.startswith("native:"):
            return "native"
        if source.startswith("ocr_consensus:"):
            return "ppocr_ocr"
        return source

    features = dict(base_features)
    sources = [canonical(str(field_sources.get(field, ""))) for field in _PROVENANCE_FIELDS]
    features["n_field_sources"] = sum(bool(source) for source in sources)
    features["n_fields_from_native"] = sum(source == "native" for source in sources)
    features["n_fields_from_standard_ocr"] = sum(
        source in _STANDARD_SOURCES - {"native"} for source in sources
    )
    features["n_fields_from_specialized_ocr"] = sum(
        bool(source) and source not in _STANDARD_SOURCES for source in sources
    )
    for field, source in zip(_PROVENANCE_FIELDS, sources):
        features[f"field_ocr_{field}"] = int(bool(source) and source != "native")
        features[f"field_specialized_{field}"] = int(
            bool(source) and source not in _STANDARD_SOURCES
        )
    return features


def apply_missing_field_imputations(
    prediction: dict,
    base_features: dict[str, object],
    *,
    artifact: dict[str, Any] | None = None,
) -> set[str]:
    """Fill sentinel outputs; never replace an extracted concrete value."""
    artifact = artifact if artifact is not None else load_extraction_imputer()
    if not artifact:
        return set()
    changed: set[str] = set()
    models = artifact.get("models") or {}
    for field in TARGET_FIELDS:
        if not _is_missing(prediction.get(field)):
            continue
        bundle = models.get(field) or {}
        vectorizer = bundle.get("vectorizer")
        estimator = bundle.get("estimator")
        if vectorizer is None or estimator is None:
            continue
        features = _model_features(base_features, prediction)
        try:
            value = str(estimator.predict(vectorizer.transform([features]))[0])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

        # A small random forest has complementary behavior for the rare
        # archive-audit purpose.  It may only assert that one value; all other
        # purpose predictions stay with the primary ExtraTrees model.
        if field == "declared_purpose":
            specialist = artifact.get("purpose_archive_specialist") or {}
            special_vec = specialist.get("vectorizer")
            special_est = specialist.get("estimator")
            if special_vec is not None and special_est is not None:
                try:
                    special = str(
                        special_est.predict(special_vec.transform([features]))[0]
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    special = ""
                if special == "archive audit":
                    value = special

        if value not in _ALLOWLISTS[field] or _is_missing(value):
            continue
        prediction[field] = value
        changed.add(field)
    return changed
