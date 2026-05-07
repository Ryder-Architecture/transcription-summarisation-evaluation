#!/usr/bin/env python3
"""
Stage 04 — Local Qwen3.6 judge over the saved summaries + extractions for one
candidate model.

Walks four kinds of records:
  * tier-1 + rollup summaries        → evaluate_summary_chunk
  * executive summaries              → evaluate_executive_summary (uses notes if present)
  * action-extraction windows        → evaluate_action_extraction
  * dedup records (one per transcript) → evaluate_dedup

Skips records already judged by this (judge, candidate) pair (idempotent resume).
Periodic save every 50 successful judgements for crash recovery.

The judge endpoint defaults to the same vLLM base URL — assumes you've loaded the
judge model in vLLM before running this stage.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys

from tqdm import tqdm

from lib.io import (
    JUDGEMENTS_DIR,
    existing_judgement_keys,
    load_all_transcripts,
    load_extractions,
    load_judgements,
    load_summaries,
    save_judgements,
)
from lib.judge import (
    evaluate_action_extraction,
    evaluate_dedup,
    evaluate_executive_summary,
    evaluate_summary_chunk,
)
from lib.pipeline import (
    SUMMARISATION_OVERLAP_TURNS,
    SUMMARISATION_TOKEN_BUDGET,
    EXTRACTION_OVERLAP_TURNS,
    EXTRACTION_TOKEN_BUDGET,
    chunk_transcript,
)


def _judge_name(judge_model: str) -> str:
    """Stable judge identifier used in filenames + judgement payloads."""
    short = judge_model.split("/")[-1]
    return f"local-{short}"


def _vllm_chat_to_base(url: str) -> str:
    """`http://host:port/v1/chat/completions` → `http://host:port/v1` for OpenAI client."""
    return url.rsplit("/chat/completions", 1)[0]


async def run(args: argparse.Namespace) -> None:
    transcripts = {t.transcript_id: t for t in load_all_transcripts()}
    if not transcripts:
        print("No cached transcripts. Run 02_sample_transcripts.py first.", file=sys.stderr)
        sys.exit(1)

    summaries = load_summaries(args.candidate_model)
    extractions, dedups = load_extractions(args.candidate_model)
    if not summaries and not extractions:
        print(f"No summaries/extractions for {args.candidate_model}. Run 03_generate.py first.", file=sys.stderr)
        sys.exit(1)

    judge_name = _judge_name(args.judge_model)
    base_url = _vllm_chat_to_base(args.vllm_url)
    print(f"  Judge:           {judge_name}  ({args.judge_model})")
    print(f"  Candidate:       {args.candidate_model}")
    print(f"  Base URL:        {base_url}")
    print(f"  Concurrency:     {args.judge_concurrency}")

    existing = load_judgements(judge_name, args.candidate_model)
    seen = existing_judgement_keys(judge_name, args.candidate_model)
    print(f"  Existing:        {len(existing)} judgements, skipping those keys")

    sem = asyncio.Semaphore(args.judge_concurrency)
    out: list[dict] = list(existing)

    # ---------- build the work queue ----------------------------------------
    tasks: list[tuple[str, dict]] = []

    # Build per-transcript window text caches (summary + extraction windows are deterministic)
    summary_window_cache: dict[str, dict[str, str]] = {}
    extraction_window_cache: dict[str, dict[int, str]] = {}
    full_transcript_cache: dict[str, str] = {}

    for tid, t in transcripts.items():
        sum_windows = chunk_transcript(
            t.turns,
            token_budget=SUMMARISATION_TOKEN_BUDGET,
            overlap_turns=SUMMARISATION_OVERLAP_TURNS,
        )
        ext_windows = chunk_transcript(
            t.turns,
            token_budget=EXTRACTION_TOKEN_BUDGET,
            overlap_turns=EXTRACTION_OVERLAP_TURNS,
        )
        # Index summary windows by their record_id suffix order
        # tier-1 record_ids are produced as "{tid}_tier1_w{idx:03d}"
        summary_window_cache[tid] = {
            f"{tid}_tier1_w{i:03d}": w.text for i, w in enumerate(sum_windows)
        }
        extraction_window_cache[tid] = {i: w.text for i, w in enumerate(ext_windows)}
        full_transcript_cache[tid] = "\n".join(
            (f"[{turn.speaker_name or 'Unknown'}]: {turn.content}") for turn in t.turns
        )

    # Summary judgements: tier-1 + rollup + exec
    for s in summaries:
        if s.error:
            continue
        if s.is_executive:
            kind = "exec"
            key = f"exec:{s.record_id}"
        else:
            kind = "summary"
            key = f"summary:{s.record_id}"
        if key in seen:
            continue

        if s.is_executive:
            source_text = full_transcript_cache.get(s.transcript_id, "")
            payload = dict(
                kind="exec",
                source_text=source_text,
                summary_title=s.title,
                summary_abstract=s.abstract,
                record_id=s.record_id,
                transcript_id=s.transcript_id,
                candidate_model=args.candidate_model,
                judge_name=judge_name,
                backend="local_qwen",
                judge_model=args.judge_model,
                base_url=base_url,
            )
        else:
            # For tier-1 we know the source window; for rollup tiers we don't
            # have a single source, so we use the full transcript as the source.
            if s.tier == 1:
                source_text = summary_window_cache.get(s.transcript_id, {}).get(s.record_id, "")
            else:
                source_text = full_transcript_cache.get(s.transcript_id, "")
            payload = dict(
                kind="summary",
                source_text=source_text,
                summary_title=s.title,
                summary_abstract=s.abstract,
                record_id=s.record_id,
                transcript_id=s.transcript_id,
                candidate_model=args.candidate_model,
                judge_name=judge_name,
                backend="local_qwen",
                judge_model=args.judge_model,
                base_url=base_url,
            )
        tasks.append((kind, payload))

    # Action-extraction judgements: per window
    for x in extractions:
        if x.error:
            continue
        key = f"action_extraction:{x.record_id}"
        if key in seen:
            continue
        source_text = extraction_window_cache.get(x.transcript_id, {}).get(x.window_index, "")
        tasks.append(
            (
                "action_extraction",
                dict(
                    source_text=source_text,
                    extracted_items=x.raw_items,
                    record_id=x.record_id,
                    transcript_id=x.transcript_id,
                    candidate_model=args.candidate_model,
                    judge_name=judge_name,
                    backend="local_qwen",
                    judge_model=args.judge_model,
                    base_url=base_url,
                ),
            )
        )

    # Dedup judgements: one per transcript
    # Reconstruct raw items per transcript by flattening that transcript's extractions.
    raw_by_transcript: dict[str, list[dict]] = {}
    for x in extractions:
        raw_by_transcript.setdefault(x.transcript_id, []).extend(x.raw_items or [])
    for d in dedups:
        if d.error:
            continue
        key = f"dedup:{d.transcript_id}_dedup"
        if key in seen:
            continue
        tasks.append(
            (
                "dedup",
                dict(
                    raw_items=raw_by_transcript.get(d.transcript_id, []),
                    groups=d.groups,
                    singletons=d.singletons,
                    transcript_id=d.transcript_id,
                    candidate_model=args.candidate_model,
                    judge_name=judge_name,
                    backend="local_qwen",
                    judge_model=args.judge_model,
                    base_url=base_url,
                ),
            )
        )

    if not tasks:
        print("  All target items already judged. Nothing to do.")
        return

    print(f"  Pending judgements: {len(tasks)}")
    pbar = tqdm(
        total=len(tasks),
        desc=f"  {judge_name} judging {args.candidate_model}",
        unit="judge",
    )
    errors = 0
    since_save = 0

    async def _run(kind: str, payload: dict):
        async with sem:
            if kind == "summary":
                return await evaluate_summary_chunk(**{k: v for k, v in payload.items() if k != "kind"})
            if kind == "exec":
                return await evaluate_executive_summary(**{k: v for k, v in payload.items() if k != "kind"})
            if kind == "action_extraction":
                return await evaluate_action_extraction(**payload)
            if kind == "dedup":
                return await evaluate_dedup(**payload)
            raise ValueError(f"unknown kind {kind}")

    coros = [_run(kind, payload) for kind, payload in tasks]
    for coro in asyncio.as_completed(coros):
        try:
            j = await coro
        except Exception as e:  # noqa: BLE001
            errors += 1
            pbar.set_postfix(err=errors, last=str(e)[:30])
            pbar.update(1)
            continue
        if j.error:
            errors += 1
        out.append(dataclasses.asdict(j))
        since_save += 1
        if since_save >= 50:
            save_judgements(out, judge_name, args.candidate_model)
            since_save = 0
        pbar.set_postfix(err=errors, done=len(out))
        pbar.update(1)

    pbar.close()
    save_judgements(out, judge_name, args.candidate_model)
    print(f"  Saved {len(out)} judgements ({errors} errors) → {JUDGEMENTS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 04: local Qwen judge.")
    ap.add_argument("--judge-model", required=True, help="HF-style judge model id loaded in vLLM.")
    ap.add_argument("--candidate-model", required=True, help="Candidate model whose outputs to judge.")
    ap.add_argument(
        "--vllm-url",
        default="http://localhost:11434/v1/chat/completions",
        help="vLLM /v1/chat/completions endpoint (the judge model must be loaded there).",
    )
    ap.add_argument("--judge-concurrency", type=int, default=8, help="Default 8.")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
