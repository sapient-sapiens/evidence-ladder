#!/usr/bin/env python3
"""Spike: constrained CTC scoring of closed-vocab candidates on RapidOCR preds.

Attribution note: technique originates in an MIT-licensed public MIB solution
(PR #68 lineage).  This spike only measures reachability, lift, and runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))

from src.constants import (  # noqa: E402
    FEE_STATUSES,
    HOME_WORLDS,
    PURPOSES,
    SPECIES_CODES,
    VISA_CLASSES,
)
from src.rapid_ocr import _engine, _render_pages, packet_needs_rapid_ocr  # noqa: E402
from src.text_extract import extract_text  # noqa: E402
from src.parse_fields import parse_packet  # noqa: E402

LEXICONS = {
    "species_code": sorted(SPECIES_CODES),
    "home_world": sorted(HOME_WORLDS),
    "visa_class": sorted(VISA_CLASSES),
    "declared_purpose": sorted(PURPOSES),
    "fee_status": sorted(FEE_STATUSES),
}


def ctc_logprob(log_probs: np.ndarray, blank: int, label_ids: list[int]) -> float:
    """Log P(label|frames) under CTC with blank collapses (forward algorithm)."""
    t_steps, _ = log_probs.shape
    if not label_ids:
        return float(log_probs[:, blank].sum())
    # Extended label with blanks: B a B b B ...
    ext = [blank]
    for idx in label_ids:
        ext.extend([idx, blank])
    u = len(ext)
    neg_inf = -1e30
    dp_prev = np.full(u, neg_inf, dtype=np.float64)
    dp_prev[0] = log_probs[0, ext[0]]
    if u > 1:
        dp_prev[1] = log_probs[0, ext[1]]
    for t in range(1, t_steps):
        dp = np.full(u, neg_inf, dtype=np.float64)
        for s in range(u):
            stay = dp_prev[s]
            step = dp_prev[s - 1] if s - 1 >= 0 else neg_inf
            skip = neg_inf
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                skip = dp_prev[s - 2]
            dp[s] = np.logaddexp(np.logaddexp(stay, step), skip) + log_probs[t, ext[s]]
        dp_prev = dp
    return float(np.logaddexp(dp_prev[-1], dp_prev[-2] if u > 1 else neg_inf))


def char_to_id(character: list[str]) -> dict[str, int]:
    return {c: i for i, c in enumerate(character)}


def encode(text: str, mapping: dict[str, int]) -> list[int] | None:
    ids = []
    for ch in text:
        if ch not in mapping:
            # try casefold variants
            alt = ch.upper() if ch.islower() else ch.lower()
            if alt not in mapping:
                return None
            ids.append(mapping[alt])
        else:
            ids.append(mapping[ch])
    return ids


def score_candidates(preds: np.ndarray, character: list[str], cands: list[str]) -> list[tuple[str, float]]:
    # preds: (T, C) raw logits or probs — normalize with log-softmax
    if preds.ndim != 2:
        raise ValueError(preds.shape)
    # numerical stability
    x = preds - preds.max(axis=1, keepdims=True)
    log_probs = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
    mapping = char_to_id(character)
    blank = 0
    scored = []
    for cand in cands:
        ids = encode(cand, mapping)
        if ids is None:
            continue
        scored.append((cand, ctc_logprob(log_probs, blank, ids)))
    scored.sort(key=lambda x: -x[1])
    return scored


def recognize_with_preds(engine, image_path: Path):
    """Run det+rec and return list of (text, score, preds_TxC) for each crop."""
    # Monkey-access recognizer internals.
    rec = getattr(engine, "text_rec", None) or getattr(engine, "rec", None)
    if rec is None:
        # RapidOCR stores pipeline pieces differently across versions.
        for name in dir(engine):
            obj = getattr(engine, name)
            if obj is not None and hasattr(obj, "session") and hasattr(obj, "postprocess_op"):
                rec = obj
                break
    if rec is None:
        raise RuntimeError("could not locate text recognizer on RapidOCR engine")

    result = engine(str(image_path))
    if result is None:
        return []
    # Re-run crops through session to capture preds. Use detection boxes if present.
    raw_boxes = getattr(result, "boxes", None)
    raw_txts = getattr(result, "txts", None)
    boxes = tuple(raw_boxes) if raw_boxes is not None else ()
    txts = tuple(raw_txts) if raw_txts is not None else ()
    if not boxes:
        return [(str(t), 0.0, None) for t in txts]

    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return []
    character = list(rec.postprocess_op.character)
    out = []
    for box, text in zip(boxes, txts):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0, x1 = int(max(0, min(xs))), int(min(img.shape[1], max(xs)))
        y0, y1 = int(max(0, min(ys))), int(min(img.shape[0], max(ys)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        crop = img[y0:y1, x0:x1]
        # Match recognizer preprocessing for a single image.
        h, w = crop.shape[:2]
        wh_ratio = w / float(h)
        norm = rec.resize_norm_img(crop, max(wh_ratio, rec.rec_image_shape[2] / rec.rec_image_shape[1]))
        batch = norm[np.newaxis, :].astype(np.float32)
        preds = rec.session(batch)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        preds = np.asarray(preds)
        if preds.ndim == 3:
            preds = preds[0]
        out.append((str(text), float(getattr(result, "scores", [0])[0] if False else 0.0), preds, character))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "dev" / "manifests" / "iter200.txt")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", type=Path, default=REPO / "dev" / "runs" / "exp5-ctc-spike.json")
    args = parser.parse_args()

    engine = _engine()
    if engine is None:
        raise SystemExit("RapidOCR engine unavailable")

    ids = args.manifest.read_text().split()[: args.limit]
    rows = []
    t0 = time.perf_counter()
    triggered = 0
    for cid in ids:
        pdf = CHALLENGE / "data" / "train" / f"{cid}.pdf"
        text = extract_text(pdf)
        packet = parse_packet(cid, text)
        if not packet_needs_rapid_ocr(packet):
            continue
        triggered += 1
        with tempfile.TemporaryDirectory(prefix="ctc-spike-") as tmp:
            pages = _render_pages(pdf, Path(tmp), dpi=180)
            if not pages:
                continue
            try:
                lines = recognize_with_preds(engine, pages[0])
            except Exception as exc:  # noqa: BLE001
                rows.append({"case_id": cid, "error": str(exc)})
                continue
        # For each lexicon, see if any line's CTC argmax-among-cands differs from decoded text.
        flips = []
        for field, cands in LEXICONS.items():
            for item in lines:
                if len(item) < 4 or item[2] is None:
                    continue
                text_i, _score, preds, character = item
                scored = score_candidates(preds, character, cands)
                if not scored:
                    continue
                best, best_lp = scored[0]
                decoded_fold = text_i.casefold().replace(" ", "")
                best_fold = best.casefold().replace(" ", "")
                if best_fold in decoded_fold or decoded_fold in best_fold:
                    continue
                # Only count when decoded doesn't already match a lexicon value
                if any(c.casefold() == text_i.casefold() for c in cands):
                    continue
                flips.append(
                    {
                        "field": field,
                        "decoded": text_i,
                        "ctc_best": best,
                        "logprob": best_lp,
                        "top3": scored[:3],
                    }
                )
        rows.append({"case_id": cid, "n_lines": len(lines), "flips": flips[:8]})
        if triggered >= args.limit:
            break

    elapsed = time.perf_counter() - t0
    report = {
        "n_scanned_hard": triggered,
        "elapsed_s": elapsed,
        "seconds_per_hard": elapsed / max(triggered, 1),
        "n_with_flips": sum(1 for r in rows if r.get("flips")),
        "rows": rows,
        "attribution": "Constrained CTC candidate scoring; MIT-licensed public solution lineage (PR #68).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("n_scanned_hard", "elapsed_s", "seconds_per_hard", "n_with_flips")}, sort_keys=True))


if __name__ == "__main__":
    main()
