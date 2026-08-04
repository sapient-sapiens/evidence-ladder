"""Per-path empirical EV arbiter with Dirichlet shrinkage.

Coarse rule-path outcome distributions (shrinkage m=10), EV argmax against the
evaluator utility table, behind a positive-evidence gate for APPROVED.  No
high-dimensional within-path resolver — that design overfit in public traps.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.arbiter import LABELS, UTILITY
from src.constants import DISQUALIFYING_FLAGS, REVIEW_ONLY_FLAGS

_ARTIFACT = Path(__file__).resolve().parents[1] / "dev" / "runs" / "path_ev.json"
DEFAULT_M = 10.0


def has_positive_evidence(state: dict) -> bool:
    """Require concrete policy evidence before an EV approval is allowed."""
    fields = state.get("packet_fields") or {}
    missing = state.get("missing") or []
    fee = str(fields.get("fee_status") or "unknown")
    flags = set(state.get("risk_flags") or [])
    if missing:
        return False
    if fee not in {"paid", "waived", "unpaid"}:
        return False
    if not state.get("observed_flags_seen") and not state.get("sources_seen"):
        return False
    # A clean packet still needs intake-like provenance or observed flags.
    sources = set(state.get("sources_seen") or [])
    if not (sources & {"intake", "fee", "registry", "biometric", "adjudicator"}):
        return False
    if flags & DISQUALIFYING_FLAGS:
        return False
    if state.get("hard") == "DENIED":
        return False
    return True


def path_key(state: dict) -> str:
    """Identity-free coarse path from rule/meta/evidence buckets."""
    fields = state.get("packet_fields") or {}
    fee = str(fields.get("fee_status") or "unknown")
    if fee not in {"paid", "waived", "unpaid", "unknown"}:
        fee = "other"
    flags = set(state.get("risk_flags") or [])
    flag_bucket = (
        "dq"
        if flags & DISQUALIFYING_FLAGS
        else ("rev" if flags & REVIEW_ONLY_FLAGS else "clean")
    )
    miss = "miss" if state.get("missing") else "full"
    conf = "conf" if state.get("conflicts") else "noconf"
    untrust = "ut" if state.get("untrusted_conflicts") else "nout"
    dq = "dqhi" if float(state.get("dq_prob") or 0.0) >= 0.5 else "dqlo"
    pos = "pos" if has_positive_evidence(state) else "weak"
    finding = state.get("adjudicator_finding") or "none"
    if finding not in LABELS:
        finding = "none"
    meta = state.get("meta_label") or "none"
    if meta not in LABELS:
        meta = "none"
    rule = state.get("rule_decision") or "none"
    if rule not in LABELS:
        rule = "none"
    hard = "hard" if state.get("hard") == "DENIED" else "nohard"
    return "|".join(
        [rule, hard, f"find_{finding}", f"meta_{meta}", dq, miss, conf, untrust,
         flag_bucket, f"fee_{fee}", pos]
    )


def _dirichlet_proba(counts: Counter[str], m: float) -> dict[str, float]:
    n = float(sum(counts.values()))
    prior = m / 3.0
    return {lab: (counts.get(lab, 0) + prior) / (n + m) for lab in LABELS}


def fit_path_tables(
    states: list[dict],
    labels: list[str],
    *,
    m: float = DEFAULT_M,
) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for state, label in zip(states, labels):
        if label not in LABELS:
            continue
        buckets[path_key(state)][label] += 1
    global_counts: Counter[str] = Counter()
    for counter in buckets.values():
        global_counts.update(counter)
    paths = {
        key: {
            "counts": {lab: int(counter.get(lab, 0)) for lab in LABELS},
            "n": int(sum(counter.values())),
            "proba": _dirichlet_proba(counter, m),
        }
        for key, counter in buckets.items()
    }
    return {
        "kind": "path_ev",
        "m": float(m),
        "paths": paths,
        "global_proba": _dirichlet_proba(global_counts, m),
        "n_train": len(states),
        "n_paths": len(paths),
    }


def path_proba(blob: dict[str, Any], state: dict) -> list[float]:
    key = path_key(state)
    entry = blob.get("paths", {}).get(key)
    proba = (entry or {}).get("proba") or blob.get("global_proba") or {}
    return [float(proba.get(lab, 1.0 / 3.0)) for lab in LABELS]


def decide_path_ev(blob: dict[str, Any], state: dict) -> tuple[str, list[float]]:
    """Trusted finding / hard denial precedence, then EV argmax with gate."""
    finding = state.get("adjudicator_finding")
    if finding in LABELS:
        proba = path_proba(blob, state)
        return str(finding), proba
    if state.get("hard") == "DENIED":
        proba = path_proba(blob, state)
        return "DENIED", proba
    proba = path_proba(blob, state)
    expected = {
        action: sum(UTILITY[action][LABELS[k]] * proba[k] for k in range(3))
        for action in LABELS
    }
    chosen = max(LABELS, key=lambda action: expected[action])
    if chosen == "APPROVED" and not has_positive_evidence(state):
        chosen = "NEEDS_REVIEW"
    return chosen, proba


@lru_cache(maxsize=1)
def load_path_ev() -> dict[str, Any] | None:
    if not _ARTIFACT.is_file():
        return None
    try:
        return json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def save_path_ev(blob: dict[str, Any], path: Path | None = None) -> Path:
    out = path or _ARTIFACT
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky nested structure is already compact; write sorted for stability.
    out.write_text(json.dumps(blob, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    load_path_ev.cache_clear()
    return out
