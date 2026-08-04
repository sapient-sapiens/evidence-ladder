# Experiment ledger

Parent baseline: `82d0ea2`. Host scores sliced from `dev/runs/baseline-dev800/predictions.jsonl`.

## Score anchors

| split | total | ext | class | calib | FA | role |
|---|---:|---:|---:|---:|---:|---|
| iter200 | 131.607 | 46.183 | 67.800 | 17.623 | 0 | in-sample diagnosis |
| tune100 (host) | 126.078 | 45.856 | 63.500 | 16.722 | 0 | true approximate score |
| probe100 (host) | 130.728 | 45.078 | 68.300 | 17.350 | 1 | secondary confirm |
| DEV800 (host) | 130.580 | 45.892 | 67.313 | 17.376 | 2 | in-sample ceiling |
| HOLD200 (README) | 124.047 | 45.528 | 61.800 | 16.719 | 2 | milestone only |
| tune100 (container, README) | 126.576 | 45.700 | 64.100 | 16.775 | 0 | shipping authority |

Decision rule: host TUNE Δ ≈ public move; iter200 class alone is not a keep signal for decision-side changes.

## Milestone summary (2026-08-04)

All five experiments **rejected**. Production `src/` unchanged vs `82d0ea2`. Host TUNE still **126.078**. Tests **125 passed**. No Docker TUNE / HOLD200 milestone run — nothing retained to confirm. Experiment tooling left under `dev/` for anti-rediscovery.

| exp | decision | one-line why |
|---|---|---|
| 1 path-EV m=10 | KILL | matched iter200 class +0.4 but honest TUNE class −5.3 / FA+2 / calib collapse |
| 3 selfsup confusion | KILL | near-identity channel; Δ0 preds |
| 2 page CER | KILL | free probe variance ~0 (9/200, max CER 0.033); glyph re-OCR unaffordable |
| 4 redact-render | KILL | crafted decoys never reach page/threshold OCR |
| 5 CTC constrained | KILL | preds reachable but ungated spike hallucinates; 3.6 s/hard |

### ship-confirm
- **tag:** ship-confirm-none-retained
- **parent:** 82d0ea2
- **hypothesis:** n/a — no mechanism kept
- **protocol:** `git diff` on `src/`/`models`/`solution.py` empty; pytest 125 passed; re-sliced baseline iter200/tune100 unchanged
- **iter200:** 131.607 (unchanged)
- **host TUNE:** 126.078 (unchanged)
- **container TUNE / HOLD200:** skipped — nothing retained to ship
- **decision:** keep baseline as-is
- **notes:** Docker and HOLD200 gates apply only after a retained candidate.

---

## Entries

### anchors-lock
- **tag:** anchors-lock
- **parent:** 82d0ea2
- **hypothesis:** Lock measured host baselines before experiments.
- **cohort:** n/a
- **protocol:** evaluate_split on baseline-dev800
- **iter200:** 131.607 / 46.183 / 67.800 / 17.623 / FA0
- **host TUNE:** 126.078 / 45.856 / 63.500 / 16.722 / FA0
- **runtime Δ:** 0
- **decision:** keep (anchors only)
- **notes:** Container TUNE shipping gate 126.576 (~+0.50 vs host).

### exp5-ctc-constrained-spike
- **tag:** exp5-ctc-spike
- **parent:** 82d0ea2
- **hypothesis:** Scoring closed-vocab candidates against RapidOCR frame posteriors via CTC forward recovers truths argmax never emits, at ~0.3 s on ~20% of cases.
- **cohort:** `packet_needs_rapid_ocr` hard remainder
- **protocol:** spike only — reach `engine.text_rec.session` preds on page-1 det crops; CTC-forward all closed lexicons; no production wiring
- **iter200 spike (25 packets → 14 hard):** n_with_flips=14 but flips are hallucinations (page titles / case headers scored as species_code); no label-proximate crop gate
- **host TUNE:** not run — spike failed quality bar before full measure
- **runtime Δ:** **3.64 s/hard packet** on host (includes extract+render+re-rec); far above ~0.3 s claim and ~0.4 s/PDF headroom
- **decision:** REVERT / KILL
- **notes:** Posteriors are reachable (PP-OCRv6 ONNX). Without label-region crops + acceptance margins the scorer invents lexicon hits. Runtime alone kills shipping. Attribution would be required if ever revisited. Spike at `dev/exp5_ctc_spike.py`, report `dev/runs/exp5-ctc-spike.json`.

### exp4-redact-then-render-audit
- **tag:** exp4-decoy-audit
- **parent:** 82d0ea2
- **hypothesis:** Hidden spans (Tr=3, near-white, sub-6pt, off-page) are resurrected by Sauvola/threshold into OCR streams; pre-raster redaction would prevent injections.
- **cohort:** packets with appearance-stream hidden decoys
- **protocol:** craft four one-page PDFs with unique decoy `ZZHIDDENDECOY999`; run pdftotext + page OCR + threshold OCR
- **iter200 / host TUNE:** n/a (correctness audit)
- **results:** decoy in native for Tr3/white/tiny (stripped by text-layer path); **decoy_anywhere_ocr=false for all four modes** including threshold
- **runtime Δ:** 0 (not implemented)
- **decision:** REVERT / KILL (audit pass: no resurrection under current chain)
- **notes:** Score Δ0 with no OCR resurrection → do not pay for redact-then-render. Artifacts in `dev/runs/exp4-decoy-audit/`.

### exp2-pagecer-probe
- **tag:** exp2-pagecer (diagnostic; full split aborted)
- **parent:** 82d0ea2
- **hypothesis:** Case-id CER from existing OCR streams weights conflicts / targets escalation.
- **cohort:** packets with garbled/foreign MIB mentions; OCR–OCR field conflicts
- **protocol:** free probe over cached streams (no new `_OCR_SOURCES`); measured variance on iter200 then attempted gated Sauvola + fusion preference
- **iter200:** probe present on 40/40 microset but CER=0 everywhere; full iter200: only 9/200 streams with CER>0 (max 0.033, mostly adjacent/foreign ids)
- **host TUNE:** not scored — insufficient probe variance to expect extraction lift; first gate bug (absent→CER=1.0) caused runaway cold Sauvola and was reverted
- **runtime Δ:** would add cost if forced; no demonstrated save
- **decision:** REVERT / KILL
- **notes:** Exception (self-supervised quality-only use of printed case id) adopted for the attempt. True per-page glyph re-OCR of the header would be needed for a real quality signal and is not affordable without prior evidence. Code kept at `dev/page_quality.py` only; production wiring removed.

### exp3-selfsup-confusion
- **tag:** exp3-confusion-iter200 / exp3-confusion-tune100
- **parent:** 82d0ea2
- **hypothesis:** Self-supervised native↔OCR confusion beats the generic prior in load_confusion_model.
- **cohort:** closed-vocab noisy-channel repairs
- **protocol:** fit on non_iter200 digital pages (1148 pairs / 287 packets); warm re-score iter200+tune100
- **iter200:** 131.607 / 46.183 / 67.800 / 17.623 / FA0 (Δ0 vs baseline)
- **host TUNE:** 126.078 / 45.856 / 63.500 / 16.722 / FA0 (Δ0 vs baseline)
- **runtime Δ:** ~0
- **decision:** REVERT / KILL
- **notes:** Fitted channel was near-identity (11561/11788 match). Zero prediction diffs on iter200. Loader reverted; fit script kept at `dev/fit_selfsup_confusion.py` for anti-rediscovery.

### exp1-path-ev-m10
- **tag:** exp1-path-ev-m10
- **parent:** 82d0ea2
- **hypothesis:** Coarse per-path EV with Dirichlet shrinkage m=10 transfers better than a learned high-dim arbiter.
- **cohort:** over-conservative reviews; FA-sensitive approvals
- **protocol:** matched pair — fit HGB replace + path-EV on non_iter200 (600), score iter200; honest TUNE fit on non_iter200\tune (500)
- **iter200 matched (HGB600 → path_EV600):** class 68.15→68.55 (+0.40), FA 4→0, total 128.65→125.19 (calib 14.31→10.45)
- **host TUNE honest (HGB500 → path_EV500):** class 66.0→60.7 (−5.3), FA 1→2, total 124.70→116.31; shipped TUNE ref 126.078 / class 63.5 / FA0
- **runtime Δ:** ~0 (decision-only, offline patch)
- **decision:** REVERT / KILL
- **notes:** Matched iter200 class edged up and FA fell, but honest TUNE classification and total collapsed and FA rose. Calibration from raw path posteriors is unusable. Code left only under `dev/path_ev.py` + eval scripts; not shipped. Do not rediscover without a calibrator and a TUNE-held fit.

---
