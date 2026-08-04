#!/usr/bin/env python3
"""Fit a self-supervised OCR confusion model from native↔OCR field tokens.

Digital native text is treated as the channel true side; page/embedded/threshold
OCR tokens for the same closed-vocab field are observations.  No labels.
Writes a shippable rates blob with the training manifest hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from src.noisy_channel_ocr import (  # noqa: E402
    CLOSED_FIELDS,
    build_confusion_from_pairs,
    extract_observed_tokens,
    levenshtein,
)
from src.text_extract import extract_text  # noqa: E402

OCR_STREAMS = ("page_ocr", "embedded_ocr", "threshold_ocr", "best_ocr", "oriented_ocr")


def manifest_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _closest_true(obs: str, truths: list[str]) -> str | None:
    best, best_d = None, 10**9
    for true in truths:
        d = levenshtein(obs.casefold(), true.casefold())
        maxlen = max(len(obs), len(true), 1)
        if d / maxlen > 0.6 and d > 4:
            continue
        if d < best_d:
            best, best_d = true, d
    return best


def pairs_for_pdf(pdf: Path) -> list[tuple[str, str]]:
    sources = extract_text(pdf, allow_ocr=True)
    native = sources.native or ""
    if len(native.strip()) < 40:
        return []
    pairs: list[tuple[str, str]] = []
    for field in CLOSED_FIELDS:
        true_toks = extract_observed_tokens(native, field, "native")
        if not true_toks:
            continue
        truths = [t.raw for t in true_toks]
        for stream in OCR_STREAMS:
            text = getattr(sources, stream, None) or ""
            if not text.strip():
                continue
            for tok in extract_observed_tokens(text, field, stream):
                true_raw = _closest_true(tok.raw, truths)
                if true_raw is not None:
                    pairs.append((tok.raw, true_raw))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "models" / "confusion_model.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    ids = args.manifest.read_text().split()
    if args.limit:
        ids = ids[: args.limit]
    all_pairs: list[tuple[str, str]] = []
    used = 0
    for i, cid in enumerate(ids, 1):
        pdf = CHALLENGE / "data" / "train" / f"{cid}.pdf"
        if not pdf.is_file():
            continue
        got = pairs_for_pdf(pdf)
        if got:
            used += 1
            all_pairs.extend(got)
        if i % 50 == 0:
            print(f"{i}/{len(ids)} packets, {len(all_pairs)} pairs", flush=True)

    model = build_confusion_from_pairs(all_pairs)
    blob = model.to_shippable()
    blob["training_manifest"] = args.manifest.name
    blob["training_manifest_sha"] = manifest_sha(args.manifest)
    blob["n_packets_with_pairs"] = used
    blob["version"] = "selfsup-native-ocr-v1"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(blob, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_pairs": model.n_pairs,
                "n_aligned": model.n_aligned,
                "n_match": model.n_match,
                "n_sub": model.n_sub,
                "n_ins": model.n_ins,
                "n_del": model.n_del,
                "packets": used,
                "manifest_sha": blob["training_manifest_sha"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    # Workers inherit cache.
    os.environ.setdefault("MIB_OCR_CACHE", "/tmp/mib-cold-dev")
    main()
