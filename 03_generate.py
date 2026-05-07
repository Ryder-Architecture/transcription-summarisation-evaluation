#!/usr/bin/env python3
"""
Stage 03 — Run the carbon-copy summarisation pipeline on every cached transcript
for one candidate model. Save SummaryRecord + ExtractionRecord + DedupRecord per
transcript as JSON under results/summaries/ and results/extractions/.

Idempotent per (model, transcript_id): re-running with the same --model only
processes transcripts that don't yet have an executive summary record.

The first transcript in a fresh model run is flagged is_warmup=True so its tier-1
warm-up latency is excluded from timing aggregations downstream.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from tqdm import tqdm

from lib.io import (
    completed_transcripts_for_model,
    load_all_transcripts,
    load_extractions,
    load_summaries,
    save_extractions,
    save_summaries,
)
from lib.pipeline import DEFAULT_GEN_CONCURRENCY, summarise_meeting


async def run(args: argparse.Namespace) -> None:
    transcripts = load_all_transcripts(n=args.transcripts)
    if not transcripts:
        print("No cached transcripts found. Run 02_sample_transcripts.py first.", file=sys.stderr)
        sys.exit(1)

    completed = completed_transcripts_for_model(args.model)
    pending = [t for t in transcripts if t.transcript_id not in completed]
    if args.only:
        pending = [t for t in pending if t.transcript_id in args.only]

    if not pending:
        print(f"  {args.model}: all {len(transcripts)} transcripts already complete. Nothing to do.")
        return

    print(f"  {args.model}: {len(completed)} complete, {len(pending)} remaining")
    print(f"  vLLM endpoint:  {args.vllm_url}")
    print(f"  concurrency:    {args.gen_concurrency}")

    existing_summaries = load_summaries(args.model)
    existing_windows, existing_dedups = load_extractions(args.model)

    pbar = tqdm(pending, desc=f"  {args.model}", unit="meet")
    errors = 0
    is_first = len(completed) == 0  # first meeting overall → mark warmup
    t_start = time.time()

    for transcript in pbar:
        try:
            result = await summarise_meeting(
                transcript=transcript,
                model_id=args.model,
                vllm_url=args.vllm_url,
                gen_concurrency=args.gen_concurrency,
                is_first_meeting=is_first,
            )
        except Exception as e:  # noqa: BLE001
            errors += 1
            pbar.set_postfix(err=errors, last=str(e)[:30])
            continue

        is_first = False
        existing_summaries.extend(result.summaries)
        existing_windows.extend(result.extractions)
        if result.dedup is not None:
            existing_dedups.append(result.dedup)

        save_summaries(existing_summaries, args.model)
        save_extractions(existing_windows, existing_dedups, args.model)

        n_summaries = len(result.summaries)
        wall = sum(s.wall_time_s for s in result.summaries) + sum(
            x.wall_time_s for x in result.extractions
        )
        comp_tokens = sum(s.completion_tokens for s in result.summaries) + sum(
            x.completion_tokens for x in result.extractions
        )
        pbar.set_postfix(
            secs=n_summaries,
            ext=len(result.extractions),
            t=f"{wall:.0f}s",
            tok=comp_tokens,
            err=errors,
        )

    pbar.close()
    elapsed = time.time() - t_start
    print(
        f"  Done: {len(pending) - errors}/{len(pending)} transcripts ok, "
        f"{errors} errors, {elapsed:.0f}s wall."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 03: generate per-candidate-model summaries + extractions.")
    ap.add_argument("--model", required=True, help="HuggingFace-style model id loaded in vLLM.")
    ap.add_argument(
        "--vllm-url",
        default="http://localhost:11434/v1/chat/completions",
        help="vLLM /v1/chat/completions endpoint.",
    )
    ap.add_argument("--transcripts", type=int, default=None, help="Optional cap on # transcripts.")
    ap.add_argument("--only", nargs="+", metavar="TID", default=None, help="Run only these transcript IDs (e.g. --only t009).")
    ap.add_argument(
        "--gen-concurrency",
        type=int,
        default=DEFAULT_GEN_CONCURRENCY,
        help=f"Generation concurrency cap (default {DEFAULT_GEN_CONCURRENCY}).",
    )
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
