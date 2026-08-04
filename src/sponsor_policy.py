"""Load the compact FIT-derived sponsor policy artifact."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .constants import REVOKED_SPONSORS

_ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "sponsor_policy.json"


@lru_cache(maxsize=1)
def learned_revoked_sponsors() -> frozenset[str]:
    if not _ARTIFACT.is_file():
        return frozenset()
    try:
        blob = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    anti = blob.get("anti_leakage") or {}
    if any(
        anti.get(key)
        for key in ("contains_case_ids", "contains_filenames", "contains_document_hashes")
    ):
        return frozenset()
    sponsors = blob.get("learned_revoked_sponsors") or {}
    required = int(
        (blob.get("criteria") or {}).get("required_leave_one_fold_selections", 6)
    )
    crossfit = blob.get("crossfit_selection_counts") or {}
    return frozenset(
        sponsor
        for sponsor, evidence in sponsors.items()
        if isinstance(sponsor, str) and sponsor.startswith("SPN-")
        and int(crossfit.get(sponsor, 0)) >= int(
            evidence.get("required_leave_one_fold_selections", required)
            if isinstance(evidence, dict) else required
        )
    )


def is_revoked_sponsor(sponsor: str) -> bool:
    return sponsor in REVOKED_SPONSORS or sponsor in learned_revoked_sponsors()
