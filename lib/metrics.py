"""
Deterministic per-model metrics for the transcription-only summarisation experiment.

Stage 06 calls these per model:
  - action_metrics_from_judge:  per-window TP/FP/FN aggregated to per-meeting and
                                per-model precision / recall / F1 plus per-TP rubric
                                means.
  - dedup_metrics_from_judge:   per-transcript dedup correctness from the judge.

This snapshot has no attendee notes and no user-curated action items, so
text-similarity-vs-notes and curated-baseline metrics are not computed. The
LLM judge is the sole source of action-item quality numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0


# ---------------------------------------------------------------------------
# Action-item P/R/F1 from the LLM judge's per-window output
# ---------------------------------------------------------------------------


@dataclass
class ActionWindowRow:
    transcript_id: str
    record_id: str
    tp: int
    fp: int
    fn: int


def action_metrics_from_judge(
    judgements: list[dict],
) -> list[ActionWindowRow]:
    """One row per action-extraction judgement (per window)."""
    rows: list[ActionWindowRow] = []
    for j in judgements:
        if j.get("kind") != "action_extraction":
            continue
        if j.get("error"):
            continue
        payload = j.get("payload") or {}
        matches = payload.get("matches") or []
        tp = len(matches)
        fp = len(payload.get("false_positive_indices") or [])
        fn = len(payload.get("false_negative_indices") or [])
        rows.append(
            ActionWindowRow(
                transcript_id=j["transcript_id"],
                record_id=j["record_id"],
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
    return rows


def aggregate_action_meeting(rows: Iterable[ActionWindowRow]) -> dict[str, dict]:
    """Roll per-window rows up to per-transcript aggregates."""
    by_meeting: dict[str, list[ActionWindowRow]] = {}
    for r in rows:
        by_meeting.setdefault(r.transcript_id, []).append(r)

    out: dict[str, dict] = {}
    for tid, items in by_meeting.items():
        tp = sum(r.tp for r in items)
        fp = sum(r.fp for r in items)
        fn = sum(r.fn for r in items)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        out[tid] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    return out


def aggregate_action_model(per_meeting: dict[str, dict]) -> dict:
    """Roll per-transcript aggregates to a single per-model summary."""
    if not per_meeting:
        return {}
    tp = sum(m["tp"] for m in per_meeting.values())
    fp = sum(m["fp"] for m in per_meeting.values())
    fn = sum(m["fn"] for m in per_meeting.values())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _f1(precision, recall)
    return {
        "meeting_count": len(per_meeting),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


# ---------------------------------------------------------------------------
# Dedup judge aggregation
# ---------------------------------------------------------------------------


def dedup_metrics_from_judge(judgements: list[dict]) -> dict[str, dict]:
    """One entry per transcript dedup judgement."""
    out: dict[str, dict] = {}
    for j in judgements:
        if j.get("kind") != "dedup":
            continue
        if j.get("error"):
            continue
        payload = j.get("payload") or {}
        correct = len(payload.get("correct_merges") or [])
        incorrect = len(payload.get("incorrect_merges") or [])
        missed = len(payload.get("missed_duplicate_pairs") or [])
        precision = _safe_div(correct, correct + incorrect)
        recall = _safe_div(correct, correct + missed)
        out[j["transcript_id"]] = {
            "correct_merges": correct,
            "incorrect_merges": incorrect,
            "missed_duplicate_pairs": missed,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    return out
