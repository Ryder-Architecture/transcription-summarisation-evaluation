#!/usr/bin/env python3
"""
Stage 01 — Parse the transcription DB dump into a CSV bundle.

Inputs:
  - A pg_dump output file, either:
      * `--dump <file.sql>`     plain-text SQL with COPY ... FROM stdin blocks
      * `--dump <file.dump>`    pg_dump custom format → ask user to pre-extract via
                                pg_restore --data-only --file=... -F p
  - OR a directory of pre-extracted CSVs at `--csv-dir <dir>` (skips parsing)

Outputs (results/csv/):
  - meeting.csv
  - transcript.csv
  - transcript_chunk.csv
  - meeting_note.csv
  - participant.csv
  - action_items.csv          # for the curated baseline

This script never makes LLM calls. It's pure parse + describe.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

from lib.io import CSV_DIR

TARGET_TABLES = [
    "meeting",
    "transcript",
    "transcript_chunk",
    "meeting_note",
    "participant",
    "action_items",
]


# ---------------------------------------------------------------------------
# COPY block parser
# ---------------------------------------------------------------------------


def _parse_pg_value(token: str) -> str | None:
    """Decode a pg_dump COPY token into its string value (or None for \\N)."""
    if token == r"\N":
        return None
    # pg_dump escapes \, \t, \n, \r in COPY text format
    return (
        token.replace(r"\\", "\\")
        .replace(r"\t", "\t")
        .replace(r"\n", "\n")
        .replace(r"\r", "\r")
    )


COPY_HEADER = re.compile(
    r"^COPY\s+(?:transcription\.)?(?P<table>\w+)\s*\((?P<cols>[^)]+)\)\s+FROM\s+stdin;\s*$"
)


def parse_dump(dump_path: Path, out_dir: Path) -> dict[str, Path]:
    """Stream the dump file and write one CSV per target table.

    Returns a dict {table_name: written_path}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    fp = dump_path.open("r", encoding="utf-8", errors="replace")
    line_no = 0
    current_table: str | None = None
    current_cols: list[str] = []
    csv_writer = None
    csv_fp = None

    try:
        for line in fp:
            line_no += 1
            # End of COPY block
            if line.strip() == r"\." and current_table:
                if csv_fp is not None:
                    csv_fp.close()
                    csv_fp = None
                    csv_writer = None
                current_table = None
                current_cols = []
                continue

            # Inside an active COPY: parse the row
            if current_table and csv_writer is not None:
                # Strip trailing newline
                row_line = line.rstrip("\n")
                tokens = row_line.split("\t")
                values = [_parse_pg_value(t) for t in tokens]
                if len(values) != len(current_cols):
                    # Tolerate slight column mismatch — pad/truncate to header length
                    if len(values) < len(current_cols):
                        values.extend([None] * (len(current_cols) - len(values)))
                    else:
                        values = values[: len(current_cols)]
                csv_writer.writerow(values)
                continue

            # Look for new COPY header
            m = COPY_HEADER.match(line)
            if m:
                table = m.group("table")
                if table not in TARGET_TABLES:
                    continue
                cols = [c.strip().strip('"') for c in m.group("cols").split(",")]
                out_path = out_dir / f"{table}.csv"
                csv_fp = out_path.open("w", encoding="utf-8", newline="")
                csv_writer = csv.writer(csv_fp)
                csv_writer.writerow(cols)
                current_table = table
                current_cols = cols
                written[table] = out_path
                continue
    finally:
        if csv_fp is not None:
            csv_fp.close()
        fp.close()

    return written


# ---------------------------------------------------------------------------
# Sanity stats
# ---------------------------------------------------------------------------


def describe_pool(csv_dir: Path) -> None:
    """Print joined-pool stats so we can sanity-check before stage 02."""
    paths = {t: csv_dir / f"{t}.csv" for t in TARGET_TABLES}
    missing = [t for t, p in paths.items() if not p.exists()]
    if missing:
        print(f"  WARNING: missing CSVs for: {missing}", file=sys.stderr)
        return

    meetings = pd.read_csv(paths["meeting"])
    transcripts = pd.read_csv(paths["transcript"])
    chunks = pd.read_csv(paths["transcript_chunk"])
    notes = pd.read_csv(paths["meeting_note"])
    actions = pd.read_csv(paths["action_items"])

    print()
    print("=" * 70)
    print("CSV bundle stats")
    print("=" * 70)
    print(f"  meetings:           {len(meetings):>6}")
    print(f"    status==complete: {(meetings['status'] == 'complete').sum():>6}")
    print(f"  transcripts:        {len(transcripts):>6}")
    print(f"  transcript_chunks:  {len(chunks):>6}")
    print(f"  meeting_notes:      {len(notes):>6}")
    print(f"    with extracted_text: {notes['extracted_text'].notna().sum():>3}")
    print(f"  action_items:       {len(actions):>6}")
    if "source" in actions.columns:
        for src in ("pipeline", "user", "notes"):
            print(f"    source={src}: {(actions['source'] == src).sum():>4}")

    # Joined pool: complete meetings with at least one transcript chunk
    if len(meetings) and len(transcripts) and len(chunks):
        complete = meetings[meetings["status"] == "complete"][["id", "subject"]].rename(
            columns={"id": "meeting_id"}
        )
        chunk_counts = (
            chunks.groupby("transcript_id").size().rename("chunk_count").reset_index()
        )
        joined = (
            transcripts.rename(columns={"id": "transcript_id"})
            .merge(complete, on="meeting_id")
            .merge(chunk_counts, on="transcript_id", how="left")
            .fillna({"chunk_count": 0})
        )
        # Notes presence per meeting
        if len(notes):
            note_meetings = set(notes.dropna(subset=["extracted_text"])["meeting_id"].astype(int))
            joined["has_notes"] = joined["meeting_id"].astype(int).isin(note_meetings)
        else:
            joined["has_notes"] = False

        joined = joined[joined["chunk_count"] >= 10]
        print()
        print(f"  Eligible meetings (status=complete, >=10 chunks): {len(joined):>5}")
        print(f"    with attendee notes: {int(joined['has_notes'].sum()):>5}")
        print(f"    without notes:       {(~joined['has_notes']).sum():>5}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse transcription DB dump into CSVs.")
    ap.add_argument(
        "--dump",
        type=Path,
        help="Path to a pg_dump plain-text SQL file containing COPY blocks.",
    )
    ap.add_argument(
        "--csv-dir",
        type=Path,
        help="Path to a directory of already-extracted CSVs (skips parsing).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=CSV_DIR,
        help=f"Output directory for CSV bundle (default: {CSV_DIR}).",
    )
    args = ap.parse_args()

    if not args.dump and not args.csv_dir:
        ap.error("Provide either --dump or --csv-dir.")

    if args.csv_dir:
        if args.csv_dir.resolve() != args.out.resolve():
            args.out.mkdir(parents=True, exist_ok=True)
            for t in TARGET_TABLES:
                src = args.csv_dir / f"{t}.csv"
                if src.exists():
                    (args.out / f"{t}.csv").write_bytes(src.read_bytes())
        describe_pool(args.out)
        return

    print(f"Parsing dump: {args.dump}")
    written = parse_dump(args.dump, args.out)
    print(f"Wrote {len(written)} table(s):")
    for t, p in written.items():
        print(f"  {t}: {p}")

    describe_pool(args.out)


if __name__ == "__main__":
    main()
