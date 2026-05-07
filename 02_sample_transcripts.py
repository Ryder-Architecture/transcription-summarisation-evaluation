#!/usr/bin/env python3
"""
Stage 02 — Sample transcripts from the CSV bundle into cached JSON.

Reads results/csv/* (produced by stage 01) and writes:
  - results/transcripts/{tid}.json      one per sampled transcript
  - results/transcripts/index.json      sampling metadata

Stratified sample of N (default 30) transcripts:
  10 short  (2,000-8,000 tokens)
  10 medium (8,000-15,000 tokens)
  10 long   (15,000-30,000 tokens)

Within each bucket we balance with-notes vs without-notes ~50/50.

Idempotent: skips work if results/transcripts/index.json already exists with N
entries. Use --force to rebuild.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from lib.io import (
    CSV_DIR,
    TRANSCRIPTS_DIR,
    CachedTranscript,
    TranscriptTurn,
    load_index,
    save_index,
    save_transcript,
)

SEED = 42
CHARS_PER_TOKEN = 4
MIN_CHUNKS = 10
MIN_TRANSCRIPT_TOKENS = 2_000
MAX_TRANSCRIPT_TOKENS = 22_000

LENGTH_BUCKETS = [
    ("short",   2_000,  6_000, 10),
    ("medium",  6_000, 14_000, 10),
    ("long",   14_000, 22_000, 10),
]


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _bucket(tokens: int) -> str | None:
    for name, lo, hi, _ in LENGTH_BUCKETS:
        if lo <= tokens < hi:
            return name
    return None


def _load_csvs(csv_dir: Path) -> dict[str, pd.DataFrame]:
    paths = {
        t: csv_dir / f"{t}.csv"
        for t in ("meeting", "transcript", "transcript_chunk")
    }
    missing = [t for t, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSVs: {missing}. Run 01_prepare_dataset.py first."
        )
    return {t: pd.read_csv(p) for t, p in paths.items()}


def _build_pool(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join CSVs into a per-transcript pool with chunk count + token count."""
    meetings = dfs["meeting"]
    transcripts = dfs["transcript"]
    chunks = dfs["transcript_chunk"]

    complete = meetings[meetings["status"] == "complete"][["id", "subject"]].rename(
        columns={"id": "meeting_id"}
    )

    chunk_stats = (
        chunks.groupby("transcript_id")
        .agg(chunk_count=("id", "count"), total_chars=("content", lambda s: s.fillna("").str.len().sum()))
        .reset_index()
    )
    chunk_stats["total_tokens"] = (chunk_stats["total_chars"] // CHARS_PER_TOKEN).astype(int)

    pool = (
        transcripts.rename(columns={"id": "transcript_id"})[["transcript_id", "meeting_id"]]
        .merge(complete, on="meeting_id")
        .merge(chunk_stats, on="transcript_id", how="left")
        .fillna({"chunk_count": 0, "total_tokens": 0})
    )
    pool = pool[pool["chunk_count"] >= MIN_CHUNKS]
    pool = pool[
        (pool["total_tokens"] >= MIN_TRANSCRIPT_TOKENS)
        & (pool["total_tokens"] <= MAX_TRANSCRIPT_TOKENS)
    ]
    pool["bucket"] = pool["total_tokens"].apply(_bucket)
    pool = pool.dropna(subset=["bucket"])

    return pool.reset_index(drop=True)


def _stratified_sample(pool: pd.DataFrame) -> list[dict]:
    rng = random.Random(SEED)
    chosen: list[dict] = []
    for name, _lo, _hi, target in LENGTH_BUCKETS:
        bucket_records = pool[pool["bucket"] == name].to_dict("records")
        rng.shuffle(bucket_records)
        picked = bucket_records[:target]
        if len(picked) < target:
            print(f"  WARNING: bucket '{name}' under-filled: {len(picked)}/{target}")
        chosen.extend(picked)
    return chosen


def _build_cached_transcript(
    row: dict,
    dfs: dict[str, pd.DataFrame],
    transcript_id: str,
) -> CachedTranscript:
    chunks = dfs["transcript_chunk"]
    tc_rows = chunks[chunks["transcript_id"] == row["transcript_id"]].copy()
    tc_rows = tc_rows.sort_values("order")
    turns = [
        TranscriptTurn(
            order=int(r["order"]),
            speaker_name=(str(r["speaker_name"]) if pd.notna(r["speaker_name"]) else None),
            content=str(r["content"]) if pd.notna(r["content"]) else "",
            start_time=str(r["start_time"]),
            end_time=str(r["end_time"]),
        )
        for _, r in tc_rows.iterrows()
    ]
    return CachedTranscript(
        transcript_id=transcript_id,
        meeting_id=int(row["meeting_id"]),
        bucket=str(row["bucket"]),
        total_tokens=int(row["total_tokens"]),
        turns=turns,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample transcripts from CSV bundle.")
    ap.add_argument("--n", type=int, default=30, help="Number of transcripts (default 30).")
    ap.add_argument("--csv-dir", type=Path, default=CSV_DIR, help=f"CSV dir (default {CSV_DIR}).")
    ap.add_argument("--force", action="store_true", help="Rebuild even if cached.")
    args = ap.parse_args()

    # Idempotent skip
    if not args.force:
        idx = load_index()
        if idx and len(idx) >= args.n:
            print(f"Cache present with {len(idx)} transcripts. Use --force to rebuild.")
            return

    print(f"Loading CSVs from {args.csv_dir}…")
    dfs = _load_csvs(args.csv_dir)
    pool = _build_pool(dfs)
    print(f"  Eligible transcripts: {len(pool)}")
    for name, lo, hi, target in LENGTH_BUCKETS:
        n = (pool["bucket"] == name).sum()
        print(f"    {name} ({lo}-{hi} tok): {n} pool (target {target})")

    chosen = _stratified_sample(pool)
    print(f"  Sampled: {len(chosen)} transcripts")

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    cached: list[CachedTranscript] = []
    for i, row in enumerate(chosen):
        tid = f"t{i:03d}"
        ct = _build_cached_transcript(row, dfs, tid)
        save_transcript(ct)
        cached.append(ct)

    save_index(
        cached,
        extra={
            "seed": SEED,
            "buckets": [
                {"name": name, "min": lo, "max": hi, "target": tgt}
                for name, lo, hi, tgt in LENGTH_BUCKETS
            ],
            "chars_per_token": CHARS_PER_TOKEN,
        },
    )
    print(f"  Wrote {len(cached)} transcripts to {TRANSCRIPTS_DIR}")
    print(f"  Wrote index → {TRANSCRIPTS_DIR / 'index.json'}")
    print()

    # Print summary table
    print("Sampled transcripts:")
    for ct in cached:
        print(
            f"  {ct.transcript_id}  bucket={ct.bucket:6}  tokens={ct.total_tokens:>6}  "
            f"turns={len(ct.turns):>4}"
        )


if __name__ == "__main__":
    main()
