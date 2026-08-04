# Evidence Ladder

A cost-gated OCR escalation over a field-manual rule engine, which reads as far
as the document allows and abstains when it doesn't.

Submission for the [8090 MIB document challenge](https://github.com/8090-inc/mib-doc-challenge).
Reads a directory of PDF packets, extracts nine fields per case, adjudicates
each against the field manual, and emits a calibrated confidence. Fully offline,
CPU-only, no LLMs or network at any point in the runtime.

## Run it

```bash
docker build -t evidence-ladder .
docker run --rm --network none --read-only \
  --cpus 4 --memory 8g --pids-limit 512 \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src="$PWD/pdfs",dst=/input,readonly \
  --mount type=bind,src="$PWD/out",dst=/output \
  evidence-ladder /input /output/predictions.jsonl
```

The entrypoint takes `<input_pdf_dir> <output_predictions_path>`. It writes
temporary files only under `/tmp` and is read-only-root compatible. Outside
Docker, `python3 solution.py <input_dir> <output_path>` does the same thing,
though host and container output are not identical — see the note under Score.

Requires `tesseract-ocr` and `poppler-utils` on the host; the Dockerfile
installs both.

## Score

| Split | Total | Extraction | Classification | Calibration | FA |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEV800 (in sample — every artifact is fitted on these 800) | 130.58 | 45.89 | 67.31 | 17.38 | 2 |
| **HOLD200 (untouched by every learned component)** | **124.05** | **45.53** | **61.80** | **16.72** | **2** |
| TUNE100, in-container | 126.58 | 45.70 | 64.10 | 16.78 | 0 |

HOLD200 is the only honest number here. Every learned artifact is fitted on
DEV800, so the DEV800 row is flattered by construction and exists only to bound
the in-sample ceiling.

**Host and container do not agree.** The same code over the same packets differs
on 17 of 100 TUNE cases between this host and the scoring image — different
Tesseract, Leptonica and poppler builds — worth 0.50 total points there, in the
container's favour. Only the container's numbers are authoritative, so the
submitted `predictions.jsonl` must be produced by the image, not locally. The
DEV800 and HOLD200 rows above are host runs and carry that uncertainty.

## Provenance

Every shipped artifact is fitted on all 800 DEV packets, selected by inner
eight-fold out-of-fold evaluation, and carries its training manifest hash
(`dev800.txt`, sha `f9e1c84c…`). No artifact fitted on unknown or unrecorded
data ships.

| Artifact | Role | Fitted on |
| --- | --- | --- |
| `adjudication.joblib` | evidence → action probabilities | 800, inner 8-fold OOF |
| `has_dq.joblib` | data-quality gate | 800, inner 8-fold OOF |
| `arbiter.joblib` | final decision, `replace` mode | 800 |
| ↳ correctness calibrator | confidence | 700, cross-fitted OOF |
| `sponsor_policy.json` | learned revoked-sponsor policy | 800, 8-fold crossfit |
| `confidence_calibrator.json` | evidence-path confidence | 800, OOF |
| `extraction_imputer.joblib` | closed-vocab field imputation | 800 |
| `name_token_lexicon.json` | name-token vocabulary | 800 |

Confidence is calibrated on out-of-fold predictions, never in-sample ones — that
is why the correctness calibrator inside the arbiter is fitted on 700 rather
than 800. Calibrating it on the in-sample production state would inflate the
calibration term without improving a single decision.

## Architecture

Extraction runs an escalating chain of OCR streams, each gated on the packet
still having unresolved fields, so the expensive passes touch only the hard
remainder: native text layer → page OCR → Sauvola threshold fallback →
`tessdata_best` → PP-OCRv6 neural pass → embedded-raster pass → oriented pass
for rotated forms. A noisy-channel repair stage then corrects closed-vocabulary
fields against a generic character-error prior, with real-word overrides
requiring cross-source agreement.

Adjudication applies the field manual as rules, then an arbiter re-decides from
the evidence. The arbiter runs in `replace` mode: it is denied the eight
features that encode the runtime's own verdict, so it forms an independent
opinion rather than learning to copy one. Given the runtime's verdict it simply
reproduces it and can never disagree usefully; denied it, the same features
reach 92.2% standalone accuracy against the rule engine's 85.0%. Trusted
adjudicator findings and hard policy denials keep field-manual precedence over
the model.

## What decides the score

Residual loss is overwhelmingly *conservatism*, not misclassification. Before
the arbiter, 73 of 600 true approvals sat in `NEEDS_REVIEW`, worth 2 points of 8
each and 74% of all classification loss. The final stage prices the retained
decision against the evaluator's utility table and relaxes a review to an
approval only when expected utility — which charges a false approval at -4 —
clearly favours it.

The residual is measured, and it is not addressable. Of 800 DEV packets, **74
have all nine fields extracted correctly and are still routed to
`NEEDS_REVIEW`** against a true APPROVED or DENIED — all 74 with `risk_flags`
reading `none` and fee settled, 71 of them truly APPROVED. Scaled to 200 cases
that is 5.6 classification points, and it is the whole gap between this system
and the leaders.

It is not reachable, because the cohort is not identifiable at inference time.
The runtime-observable version of it — reviewed, risk read `none`, fee read
settled — is 212 packets splitting 96 APPROVED / 78 NEEDS_REVIEW / 38 DENIED.
Promoting all of them costs **3.45 points and adds 38 catastrophic false
approvals**; every confidence threshold is worse, because the true reviews
destroyed (8 raw each) outnumber the approvals rescued. The only feature that
separates the 74 from the other 138 is whether the extraction happens to be
correct, which is exactly what cannot be known without the labels.

The arbiter is not the cause. Disabling it entirely leaves TUNE100
classification unchanged at 63.5 — it changes no decisions there and contributes
only calibration, worth 0.54.

## The ceiling

Fit the best possible decision function directly to the *trusted* field values —
out of fold, all labelled packets, scored with the evaluator's utility table —
and it reaches 95.0% accuracy for 76.40 classification. Five percent of
adjudications are simply not a function of the nine fields. So the architecture
tops out at:

```
   50.00  extraction, perfect
+  76.40  classification, perfect extraction, best decision function
+  18.10  calibration at 95% accuracy
=  144.50
```

Each point of extraction recovered is worth **4.01** total points, because
classification moves 2.47 and calibration 0.54 alongside it. Reaching 140
requires extraction of **48.88/50 — 97.8% weighted field accuracy.**

That is the part the pipeline cannot buy. Across all seven OCR streams, the
number of currently-wrong field values for which *any* engine produced the truth
is **4**. Three later probes agree from directions that share none of that
method: measured against the case id printed on every page as known plaintext,
only 9 pages in 200 carry any OCR error at all and the worst CER is 0.033;
1,148 aligned (native, OCR) pairs yield a near-identity confusion channel, so
there are no systematic character errors left to model; and crafted
render-mode-3, white, sub-6pt and off-cropbox decoys never reach any OCR stream.
The reading channel is clean. What remains missing is evidence the packet does
not contain.

One field dominates. Substituting trusted values into the real feature set, one
group at a time:

| feature set | classification | accuracy |
| --- | ---: | ---: |
| as shipped | 67.75 | 83.4% |
| + trusted visa, fee, sponsor, date | 69.45 | 85.3% |
| **+ trusted `risk_flags`** | **74.03** | **92.0%** |
| + both | 76.10 | 94.9% |

`risk_flags` alone is 6.28 of the 8.35-point gap, against 1.70 for the other
four policy fields combined. It is also the least recoverable: 664 of its 752
lost raw points are evidence that is physically absent from the packet.

**Every extraction number here understates the private one by about 1.7
points.** `EVALUATION.md` removes genuinely unrecoverable fields from a case's
extraction maximum, and 45% of measured loss sits exactly there — 83 of 600
packets carry no biometric slip, 60 more no fee receipt at all. Inspection
confirms rather than infers this: MIB-000018 contains intake, note, registry and
sponsor pages, no fee receipt, and its registry prints `[SPECIES WHITEOUT]`.

## Runtime budget

**5.46 seconds/PDF** measured in-container over TUNE100, four vCPUs,
`--network none`, read-only root, 8 GiB, against a contract of six. CPU sits at
386% of 400% mean and memory peaks at 2.07 GiB of 8 GiB, so time is the binding
resource and the cores are saturated. Across the 5,000-PDF validation set that
is 27,300 s against a 30,000 s hard kill — **9% margin**. On this host (M2 Pro,
10 workers) the same work runs at 2.72 s/PDF, but host output is not the
container's, so that figure is for iteration only.

The escalation chain that produced the
pre-arbiter extraction score measured 9.6 s/PDF; three cuts brought it back
under budget for 0.28 total points:

- A 288-DPI PP-OCR escalation ran on nearly half of all packets for 0.10 points
  and a fifth of the budget. Removed.
- The neural pass and its high-resolution escalation fired whenever no biometric
  panel had been observed — true of 46% of packets and usually not something
  another OCR pass can fix. Gated on unresolved fields and conflicts instead.
- Every page and raster was read three times with three Tesseract layout modes.
  `--psm 6` never won outright, so the sparse pass is now conditional and the
  embedded pass is gated on the packet still looking incomplete.

## Development boundary

DEV800 is partitioned into eight disjoint 100-packet folds (`fit_fold_0..5`,
`tune100`, `probe100`). HOLD200 is scored in aggregate only, through
`dev/score_holdout.py`, and never inspected per case.

The dev tooling expects this repository to sit inside a checkout of the
challenge repo, so that `../data/train/` and `../data/train_labels.csv` resolve.
Generate the DEV800 truth slice first — the organizer's labels are not vendored
here:

```bash
python3 dev/make_truth.py
```

```bash
python3 dev/run_split.py --split tune100 --tag baseline --truth dev/truth/dev800_labels.csv
python3 dev/run_split.py --split dev800  --tag full    --truth dev/truth/dev800_labels.csv
python3 dev/run_holdout.py --candidate <name>
```

`run_split.py` creates and cleans its temporary PDF directory, writes
predictions and timing metadata under ignored `dev/runs/<tag>/`, validates
completeness, and prints the aggregate score. Add `--docker --build` for the
exact four-vCPU container contract.

Timing does not need a container run: a fifteen-packet cold single-process
sample scales to the four-vCPU contract within a few percent, and any change
that only selects which cached engine output is used can be re-scored warm.

Rebuild the arbiter after any change to the extraction or decision path, since
its training data is the runtime's own state:

```bash
python3 dev/dump_state.py --split dev800 --out dev/runs/state.jsonl
python3 dev/train_arbiter.py --state dev/runs/state.jsonl --oof dev/runs/oof.jsonl \
  --truth dev/truth/dev800_labels.csv --fold-manifest dev/manifests/fit_fold_0.txt ...
python3 dev/train_arbiter.py --state dev/runs/state.jsonl --out models/arbiter.joblib ...
```

The fully nested eight-fold evaluation, which refits every learned component per
fold in an isolated repo copy, is `dev/run_nested_eval.py`. It is the only bench
that is not blind to upstream fitting; use it for any change to a learned
component.

See `AGENTS.md` for the iteration contract.

## License

MIT — see `LICENSE`. Bundled and invoked third-party components, their licenses,
and the provenance of the vendored Tesseract model are listed in
`THIRD_PARTY_NOTICES.md`.

## Experiment record

`dev/EXPERIMENT_LEDGER.md` records what was measured and rejected, including six
falsifiable negatives that bound what this architecture can reach. Ideas sourced
from public submissions are attributed there; none of their code is used.
