#!/usr/bin/env python3
"""Run the eight-fold, fully refitted DEV800 evaluation in an isolated repo copy.

Run this script only from a disposable copy of ``solution-core``.  It overwrites
that copy's model artifacts while fitting each inner/outer partition.  It never
opens or resolves a HOLD manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "dev" / "manifests"
FOLD_PATHS = [MANIFESTS / f"fit_fold_{i}.txt" for i in range(6)] + [
    MANIFESTS / "tune100.txt", MANIFESTS / "probe100.txt",
]
LEARNED = (
    "adjudication.joblib", "has_dq.joblib", "confidence_calibrator.json",
    "extraction_imputer.joblib", "name_token_lexicon.json", "sponsor_policy.json",
    "arbiter.joblib",
)


def component_fingerprint(truth: Path, fixed_evidence_spec: str | None) -> str:
    """Bind reusable fits to the exact parser/trainer code and DEV truth."""
    paths = [REPO / "solution.py", truth]
    paths.extend(sorted((REPO / "src").glob("*.py")))
    paths.extend(
        REPO / "dev" / name for name in (
            "train_name_lexicon.py", "train_sponsor_policy.py",
            "train_evidence_models.py", "train_extraction_imputer.py",
            "fit_evidence_models.py", "fit_outer_evidence.py",
        )
    )
    digest = hashlib.sha256()
    digest.update(f"fixed_evidence_spec={fixed_evidence_spec or 'selected'}\n".encode())
    for path in paths:
        digest.update(str(path.relative_to(REPO) if path.is_relative_to(REPO) else path.name).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def call(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.check_call(command, cwd=REPO, env=env)


def write_manifest(path: Path, fold_indexes: list[int]) -> list[str]:
    ids = [case_id for index in fold_indexes for case_id in FOLD_PATHS[index].read_text().split()]
    if len(ids) != 100 * len(fold_indexes) or len(set(ids)) != len(ids):
        raise RuntimeError("nested fold union is not a unique 100-case partition")
    path.write_text("\n".join(ids) + "\n")
    return ids


def fold_ids(fold_index: int) -> set[str]:
    ids = FOLD_PATHS[fold_index].read_text().split()
    if len(ids) != 100 or len(set(ids)) != 100:
        raise RuntimeError(f"fold {fold_index} is not a unique 100-case partition")
    return set(ids)


def validate_state(path: Path, expected: set[str], label: str) -> None:
    """Reject truncated, duplicated, or cross-fold cached state."""
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    ids = [str(row.get("case_id") or "") for row in rows]
    if len(ids) != len(expected):
        raise RuntimeError(
            f"{label} has {len(ids)} rows, expected {len(expected)}: {path}"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{label} contains duplicate case IDs: {path}")
    actual = set(ids)
    if actual != expected:
        raise RuntimeError(
            f"{label} case membership differs from its manifest: {path}"
        )


def fit_components(
    train_folds: list[int], stage: Path, truth: Path, workers: int, ocr_cache: Path,
    fixed_evidence_spec: str | None,
) -> Path:
    stage.mkdir(parents=True, exist_ok=True)
    manifest = stage / "train.txt"
    ids = write_manifest(manifest, train_folds)
    fold_args = [arg for index in train_folds for arg in ("--fold-manifest", str(FOLD_PATHS[index]))]
    models = REPO / "models"
    snapshot = stage / "artifacts"
    complete = stage / "complete.json"
    fingerprint = component_fingerprint(truth, fixed_evidence_spec)
    for name in LEARNED:
        path = models / name
        if path.exists():
            path.unlink()
    if complete.exists():
        metadata = json.loads(complete.read_text())
        expected = sorted(name for name in LEARNED if name != "arbiter.joblib")
        actual = sorted(path.name for path in snapshot.iterdir() if path.is_file()) \
            if snapshot.exists() else []
        if (
            metadata.get("train_folds") != train_folds
            or metadata.get("fingerprint") != fingerprint
            or actual != expected
        ):
            raise RuntimeError(f"invalid cached component fit: {stage}")
        for name in actual:
            shutil.copy2(snapshot / name, models / name)
        return manifest
    call([sys.executable, "dev/train_name_lexicon.py", "--manifest", str(manifest),
          "--out", str(models / "name_token_lexicon.json"), "--truth", str(truth)])
    call([sys.executable, "dev/train_sponsor_policy.py", "--manifest", str(manifest),
          "--output", str(models / "sponsor_policy.json"), "--truth", str(truth), *fold_args])
    # This cache must be generated *after* the fold-local learned parsers are
    # fitted.  Subsetting a globally generated cache leaks upstream name and
    # sponsor behavior even if every downstream estimator is refitted.
    cache = stage / "raw-train.jsonl"
    env = os.environ.copy()
    env["MIB_OCR_CACHE"] = str(ocr_cache)
    call([sys.executable, "dev/train_evidence_models.py", "--manifest", str(manifest),
          "--cache", str(cache), "--truth", str(truth), "--workers", str(workers),
          "--rebuild", "--extract-only"], env=env)
    call([sys.executable, "dev/train_extraction_imputer.py", "--features-jsonl", str(cache),
          "--manifest", str(manifest), "--output", str(models / "extraction_imputer.joblib")])
    evidence_command = [sys.executable, "dev/fit_outer_evidence.py", "--cache", str(cache),
          "--manifest", str(manifest), *fold_args, "--out-dir", str(models),
          "--report", str(stage / "evidence-report.json")]
    if fixed_evidence_spec:
        evidence_command.extend(["--fixed-spec", fixed_evidence_spec])
    call(evidence_command)
    snapshot.mkdir(parents=True, exist_ok=True)
    expected = sorted(name for name in LEARNED if name != "arbiter.joblib")
    for name in expected:
        source = models / name
        if not source.exists():
            raise RuntimeError(f"component fit did not produce {name}")
        shutil.copy2(source, snapshot / name)
    temporary = complete.with_name(f".{complete.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps({
        "train_folds": train_folds,
        "train_n": len(ids),
        "artifacts": expected,
        "fingerprint": fingerprint,
    }, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, complete)
    return manifest


def component_stage(run_dir: Path, train_folds: list[int]) -> Path:
    key = "folds-" + "-".join(str(index) for index in train_folds)
    return run_dir / "_component-fits" / key


def dump_fold(fold_index: int, output: Path, workers: int, cache: Path) -> float:
    target_manifest = MANIFESTS / "nested_target.txt"
    target_manifest.write_text(FOLD_PATHS[fold_index].read_text())
    env = os.environ.copy()
    env["MIB_OCR_CACHE"] = str(cache)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    started = time.perf_counter()
    call([sys.executable, "dev/dump_state.py", "--split", "nested_target",
          "--out", str(temporary), "--workers", str(workers)], env=env)
    seconds = time.perf_counter() - started
    validate_state(temporary, fold_ids(fold_index), f"new fold {fold_index} state")
    os.replace(temporary, output)
    return seconds


def combine(paths: list[Path], fold_indexes: list[int], output: Path) -> None:
    expected = set().union(*(fold_ids(index) for index in fold_indexes))
    if len(expected) != 100 * len(fold_indexes):
        raise RuntimeError("combined folds are not disjoint")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    with temporary.open("w") as handle:
        for path in paths:
            handle.write(path.read_text())
    validate_state(temporary, expected, "combined inner OOF state")
    os.replace(temporary, output)


def record_timing(state_path: Path, row: dict) -> None:
    state_path.with_suffix(".runtime.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )


def load_timing(state_path: Path) -> dict | None:
    path = state_path.with_suffix(".runtime.json")
    return json.loads(path.read_text()) if path.exists() else None


def aggregate(result_paths: list[Path], timings: list[dict], pathway: str) -> dict:
    import statistics
    rows = [json.loads(path.read_text()) for path in result_paths]
    expected_runtime = pathway == "runtime"
    expected_majority = pathway == "runtime_majority"
    if any(bool(row.get("runtime_pathway")) != expected_runtime or
           bool(row.get("runtime_majority_pathway")) != expected_majority for row in rows):
        raise RuntimeError("cached nested result pathway does not match requested pathway")
    keys = ("extraction", "classification", "accuracy", "false_approvals",
            "legacy_total", "exact_path_total", "calibration_delta")
    summary = {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}
    summary["false_approvals"] = int(sum(int(row["false_approvals"]) for row in rows))
    for metric in ("legacy_total", "exact_path_total", "extraction", "classification",
                   "calibration_delta"):
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_fold_sd"] = statistics.pstdev(values)
        summary[f"{metric}_fold_min"] = min(values)
        summary[f"{metric}_fold_max"] = max(values)
    summary["legacy_calibration"] = sum(row["legacy_calibration"]["score"] for row in rows) / len(rows)
    summary["exact_path_calibration"] = sum(row["exact_path_calibration"]["score"] for row in rows) / len(rows)
    summary["calibration_improved_folds"] = sum(row["calibration_delta"] > 0 for row in rows)
    summary["outer_folds"] = len(rows)
    summary["decision_pathway"] = pathway
    summary["state_runtime_seconds"] = sum(t["seconds"] for t in timings)
    summary["state_runtime_seconds_per_pdf_warm_cache"] = summary["state_runtime_seconds"] / (100 * len(timings))
    summary["timings"] = timings
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ocr-cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--truth", type=Path, required=True,
                        help="DEV-only truth CSV; combined train_labels.csv is forbidden")
    parser.add_argument("--outer", type=int, action="append")
    parser.add_argument(
        "--pathway",
        choices=("runtime", "runtime_majority", "replace"),
        required=True,
        help="Exact final decision pathway to calibrate and score",
    )
    parser.add_argument("--fixed-evidence-spec",
                        choices=("gbc_d1", "gbc_d2", "gbc_d3", "logreg_c03", "extra_leaf5"))
    args = parser.parse_args()
    if args.truth.name == "train_labels.csv":
        raise SystemExit("combined train_labels.csv is forbidden; pass DEV800-only truth")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    outers = args.outer if args.outer is not None else list(range(8))
    timings: list[dict] = []
    result_paths: list[Path] = []
    for outer in outers:
        if outer not in range(8):
            raise SystemExit("--outer must be in 0..7")
        outer_dir = args.run_dir / f"outer-{outer}"
        outer_dir.mkdir(parents=True, exist_ok=True)
        training_folds = [f for f in range(8) if f != outer]
        oof_states = []
        for inner in training_folds:
            output = outer_dir / f"inner-{inner}-state.jsonl"
            if not output.exists():
                component_folds = [f for f in training_folds if f != inner]
                fit_components(component_folds, component_stage(args.run_dir, component_folds),
                               args.truth, args.workers, args.ocr_cache,
                               args.fixed_evidence_spec)
                seconds = dump_fold(inner, output, args.workers, args.ocr_cache)
                timing = {"outer": outer, "kind": "inner", "fold": inner, "seconds": seconds}
                record_timing(output, timing)
            else:
                validate_state(output, fold_ids(inner), f"cached inner fold {inner} state")
                timing = load_timing(output)
            if timing is not None:
                timings.append(timing)
            oof_states.append(output)
        train_state = outer_dir / "train-oof-state.jsonl"
        combine(oof_states, training_folds, train_state)
        held_state = outer_dir / "held-state.jsonl"
        if not held_state.exists():
            fit_components(training_folds, component_stage(args.run_dir, training_folds),
                           args.truth, args.workers, args.ocr_cache,
                           args.fixed_evidence_spec)
            seconds = dump_fold(outer, held_state, args.workers, args.ocr_cache)
            timing = {"outer": outer, "kind": "outer", "fold": outer, "seconds": seconds}
            record_timing(held_state, timing)
        else:
            validate_state(held_state, fold_ids(outer), f"cached outer fold {outer} state")
            timing = load_timing(held_state)
        if timing is not None:
            timings.append(timing)
        result = outer_dir / "result.json"
        command = [sys.executable, "dev/nested_arbiter_eval.py",
                   "--train-state", str(train_state), "--held-state", str(held_state),
                   "--held-manifest", str(FOLD_PATHS[outer]), "--output", str(result)]
        command.extend(["--truth", str(args.truth)])
        if args.pathway == "runtime":
            command.append("--runtime-pathway")
        elif args.pathway == "runtime_majority":
            command.append("--runtime-majority-pathway")
        for fold_index in training_folds:
            command.extend(["--train-fold-manifest", str(FOLD_PATHS[fold_index])])
        call(command)
        result_paths.append(result)
    all_result_paths = [args.run_dir / f"outer-{outer}" / "result.json" for outer in range(8)]
    if all(path.exists() for path in all_result_paths):
        all_timings = []
        for outer in range(8):
            outer_dir = args.run_dir / f"outer-{outer}"
            for path in sorted(outer_dir.glob("*.runtime.json")):
                all_timings.append(json.loads(path.read_text()))
        summary = aggregate(all_result_paths, all_timings, args.pathway)
        (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
