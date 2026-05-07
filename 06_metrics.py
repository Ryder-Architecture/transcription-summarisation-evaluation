#!/usr/bin/env python3
"""
Stage 06 — Compute deterministic per-model metrics from the saved
summaries / extractions / judgements.

Outputs results/metrics/{model}.json with:
  - exec_text_similarity: per-meeting ROUGE/BERTScore vs attendee notes (subset)
  - action_judge: per-meeting + per-model P/R/F1 + per-TP rubric means from the
    LLM judge's output
  - action_curated: per-meeting set P/R against curated user/notes items via
    sentence-transformers cosine match (subset of meetings with curated items)
  - dedup_judge: per-meeting + per-model dedup correctness
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from lib.io import (
    JUDGEMENTS_DIR,
    load_all_transcripts,
    load_extractions,
    load_judgements,
    load_summaries,
    save_metrics,
)
from lib.metrics import (
    action_metrics_from_judge,
    aggregate_action_meeting,
    aggregate_action_model,
    dedup_metrics_from_judge,
)


def _local_judge_name(judgements_dir: Path, candidate_model: str) -> str | None:
    """Find the local judge file for this candidate (there should be exactly one)."""
    safe = candidate_model.replace("/", "__")
    for p in judgements_dir.glob(f"local-*_{safe}.json"):
        return p.stem.split(f"_{safe}")[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 06: derived metrics per model.")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    summaries = load_summaries(args.model)
    extractions, dedups = load_extractions(args.model)

    if not summaries:
        print(f"No summaries for {args.model}.", file=sys.stderr)
        sys.exit(1)

    local_judge = _local_judge_name(JUDGEMENTS_DIR, args.model)
    if not local_judge:
        print(f"No local judgements for {args.model}. Run 04_judge_local.py first.", file=sys.stderr)
        sys.exit(1)

    judgements = load_judgements(local_judge, args.model)
    print(f"  Local judge:        {local_judge}")
    print(f"  Judgements loaded:  {len(judgements)}")

    # ---------- Action-item judge metrics -----------------------------------
    action_window_rows = action_metrics_from_judge(judgements)
    action_per_meeting = aggregate_action_meeting(action_window_rows)
    action_overall = aggregate_action_model(action_per_meeting)
    print(f"  Action windows judged: {len(action_window_rows)}")

    # ---------- Dedup metrics -----------------------------------------------
    dedup_per_meeting = dedup_metrics_from_judge(judgements)
    print(f"  Dedup judgements: {len(dedup_per_meeting)}")

    payload = {
        "model": args.model,
        "local_judge": local_judge,
        "action_judge": {
            "per_window": [dataclasses.asdict(r) for r in action_window_rows],
            "per_meeting": action_per_meeting,
            "overall": action_overall,
        },
        "dedup_judge": {
            "per_meeting": dedup_per_meeting,
        },
    }
    out_path = save_metrics(payload, args.model)
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
