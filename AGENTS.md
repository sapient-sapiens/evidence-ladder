# Agent contract

Read the challenge README, PRD, FIELD_MANUAL, EVALUATION, and Docker contract in
the parent repository before changing this code.

## Non-negotiable rules

- Use only DEV800 for fitting, diagnosis, examples, and iteration decisions.
- Access HOLD200 only through `dev/score_holdout.py` and only at milestones.
- Never use case IDs, filenames, hashes, byte sizes, ordering, or split membership
  as prediction features or hardcoded conditions.
- Keep catastrophic false approvals at zero for retained candidates.
- One falsifiable hypothesis per change. Measure it, then keep or revert it.
- Commit only retained mechanisms. Leave no dormant switches or model variants
  in production; a removed mechanism's code path goes with it.
- Preserve offline, four-vCPU, read-only-root Docker behavior.

## Iteration loop

1. Start from a clean Git commit and record its hash.
2. Diagnose on one rotating FIT600 fold or a 20–50 case DEV microset.
3. State the expected affected cohort, score component, safety behavior, and
   runtime cost before editing.
4. Implement one conceptual change.
5. Check the microset plus matched unaffected controls.
6. Confirm on at least two FIT folds not used for diagnosis.
7. Use TUNE100 only after those checks pass.
8. Use PROBE100 only for milestone candidates.
9. Refit on DEV800 only after code and thresholds are frozen.
10. Use HOLD200 only for an occasional aggregate generalization check.

Fit learned artifacts with out-of-fold evaluation. Fit confidence calibration on
out-of-fold predictions, never in-sample predictions. Do not repeatedly select
against the same 100 examples.

Every classifier is evidence-only. The envelope inputs `pages`, `pdf_bytes` and
`trusted_chars` are explicitly excluded from the adjudication and DQ feature
sets, and no artifact without a recorded training manifest ships. Do not
reintroduce envelope features.
