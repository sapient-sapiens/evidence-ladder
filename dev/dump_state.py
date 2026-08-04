#!/usr/bin/env python3
"""Dump the production arbiter state for a sanctioned DEV split.

FIT/DEV diagnostic and training-data builder.  The state comes from the exact
production call path, so what the arbiter is fitted on is what it sees at
runtime.  Never imported by the runtime.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHALLENGE = REPO.parent
sys.path.insert(0, str(REPO))


def state_for(case_id: str) -> dict:
    import solution as production

    pdf = CHALLENGE / "data" / "train" / f"{case_id}.pdf"
    state: dict = {}
    pred = production.prediction_for(pdf, state_out=state)
    state["case_id"] = case_id
    state["prediction"] = pred
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.split.startswith("hold"):
        raise SystemExit("dump_state.py is a FIT/DEV diagnostic; HOLD is aggregate-only")
    ids = (REPO / "dev" / "manifests" / f"{args.split}.txt").read_text().split()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with args.out.open("w", encoding="utf-8") as f:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(state_for, cid) for cid in ids]
            for fut in as_completed(futures):
                f.write(json.dumps(fut.result(), sort_keys=True) + "\n")
                f.flush()
                done += 1
                if done % 100 == 0:
                    print(f"{done}/{len(ids)}", flush=True)
    print(f"wrote {done} states to {args.out}")


if __name__ == "__main__":
    main()
