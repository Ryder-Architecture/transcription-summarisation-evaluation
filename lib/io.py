"""
Shared data classes, paths, and idempotent save/load helpers.

Every stage reads from earlier stages' output dirs and writes only to its own.
Loading respects partial files: append-after-each-transcript pattern means a
crashed run resumes by skipping the (model, transcript_id) keys it already has.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Filesystem layout — each stage writes to one and only one directory
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CSV_DIR = RESULTS_DIR / "csv"
TRANSCRIPTS_DIR = RESULTS_DIR / "transcripts"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
EXTRACTIONS_DIR = RESULTS_DIR / "extractions"
JUDGEMENTS_DIR = RESULTS_DIR / "judgements"
METRICS_DIR = RESULTS_DIR / "metrics"
ANALYSIS_PATH = RESULTS_DIR / "analysis.md"


def model_filename(model_id: str) -> str:
    """Convert a HuggingFace-style model id to a safe filesystem name.

    'Qwen/Qwen3.5-9B-AWQ' -> 'Qwen__Qwen3.5-9B-AWQ'
    """
    return model_id.replace("/", "__")


# ---------------------------------------------------------------------------
# Data classes — kept dataclass-based (not Pydantic) for cheap (de)serialisation
# ---------------------------------------------------------------------------


@dataclass
class TranscriptTurn:
    order: int
    speaker_name: str | None
    content: str
    start_time: str  # ISO 8601 string for JSON-friendliness
    end_time: str


@dataclass
class CachedTranscript:
    """A sampled transcript persisted under results/transcripts/{transcript_id}.json"""

    transcript_id: str
    meeting_id: int
    bucket: str  # "short" | "medium" | "long"
    total_tokens: int
    turns: list[TranscriptTurn] = field(default_factory=list)


@dataclass
class SummaryRecord:
    """One LLM call (tier 1, rollup tier, or executive)."""

    transcript_id: str
    record_id: str  # e.g. "t003_tier1_w005" / "t003_tier2_g002" / "t003_exec"
    model_name: str
    tier: int  # 0 = exec, 1 = leaf, 2+ = rollup
    is_executive: bool
    title: str
    abstract: str
    start_turn: int | None = None
    end_turn: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    ttft_s: float | None = None
    tpot_s: float | None = None
    is_warmup: bool = False
    error: str | None = None


@dataclass
class ExtractionRecord:
    """One extraction window (raw items, before dedup)."""

    transcript_id: str
    record_id: str  # e.g. "t003_xw002"
    model_name: str
    window_index: int
    start_turn: int
    end_turn: int
    raw_items: list[dict] = field(default_factory=list)  # ActionItemSchema dicts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    ttft_s: float | None = None
    tpot_s: float | None = None
    is_warmup: bool = False
    error: str | None = None


@dataclass
class DedupRecord:
    """One dedup call per transcript."""

    transcript_id: str
    model_name: str
    raw_count: int
    deduped_count: int
    deduped_indices: list[int] = field(default_factory=list)  # winners into flattened raw set
    groups: list[dict] = field(default_factory=list)  # {canonical_index, duplicate_indices}
    singletons: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_time_s: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Transcript JSON I/O (stage 02)
# ---------------------------------------------------------------------------


def save_transcript(t: CachedTranscript) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS_DIR / f"{t.transcript_id}.json"
    path.write_text(json.dumps(asdict(t), indent=2, default=_json_default))
    return path


def load_transcript(transcript_id: str) -> CachedTranscript | None:
    path = TRANSCRIPTS_DIR / f"{transcript_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["turns"] = [TranscriptTurn(**t) for t in data["turns"]]
    return CachedTranscript(**data)


def load_index() -> list[dict]:
    path = TRANSCRIPTS_DIR / "index.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("transcripts", [])
    return data


def save_index(transcripts: list[CachedTranscript], extra: dict | None = None) -> Path:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTS_DIR / "index.json"
    payload = {
        "transcripts": [
            {
                "transcript_id": t.transcript_id,
                "meeting_id": t.meeting_id,
                "bucket": t.bucket,
                "total_tokens": t.total_tokens,
                "turns": len(t.turns),
            }
            for t in transcripts
        ],
        **(extra or {}),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_all_transcripts(n: int | None = None) -> list[CachedTranscript]:
    """Load every cached transcript in index order; optionally limit to first N."""
    index = load_index()
    if n is not None:
        index = index[:n]
    out: list[CachedTranscript] = []
    for entry in index:
        t = load_transcript(entry["transcript_id"])
        if t is not None:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Summary / extraction / dedup I/O (stage 03)
# ---------------------------------------------------------------------------


def summaries_path(model_id: str) -> Path:
    return SUMMARIES_DIR / f"{model_filename(model_id)}.json"


def extractions_path(model_id: str) -> Path:
    return EXTRACTIONS_DIR / f"{model_filename(model_id)}.json"


def save_summaries(records: list[SummaryRecord], model_id: str) -> Path:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    path = summaries_path(model_id)
    path.write_text(json.dumps([asdict(r) for r in records], indent=2))
    return path


def load_summaries(model_id: str) -> list[SummaryRecord]:
    path = summaries_path(model_id)
    if not path.exists():
        return []
    return [SummaryRecord(**d) for d in json.loads(path.read_text())]


def save_extractions(
    records: list[ExtractionRecord],
    dedups: list[DedupRecord],
    model_id: str,
) -> Path:
    EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = extractions_path(model_id)
    path.write_text(json.dumps(
        {
            "windows": [asdict(r) for r in records],
            "dedup": [asdict(d) for d in dedups],
        },
        indent=2,
    ))
    return path


def load_extractions(model_id: str) -> tuple[list[ExtractionRecord], list[DedupRecord]]:
    path = extractions_path(model_id)
    if not path.exists():
        return [], []
    data = json.loads(path.read_text())
    windows = [ExtractionRecord(**d) for d in data.get("windows", [])]
    dedups = [DedupRecord(**d) for d in data.get("dedup", [])]
    return windows, dedups


def completed_transcripts_for_model(model_id: str) -> set[str]:
    """Transcripts that already have an executive summary — safe to skip on resume."""
    return {r.transcript_id for r in load_summaries(model_id) if r.is_executive}


# ---------------------------------------------------------------------------
# Judgement I/O (stages 04, 05)
# ---------------------------------------------------------------------------


def judgements_path(judge_name: str, model_id: str) -> Path:
    return JUDGEMENTS_DIR / f"{judge_name}_{model_filename(model_id)}.json"


def save_judgements(judgements: list[dict], judge_name: str, model_id: str) -> Path:
    JUDGEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = judgements_path(judge_name, model_id)
    path.write_text(json.dumps(judgements, indent=2))
    return path


def load_judgements(judge_name: str, model_id: str) -> list[dict]:
    path = judgements_path(judge_name, model_id)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def existing_judgement_keys(judge_name: str, model_id: str) -> set[str]:
    """Stable de-dup key for already-judged items: `<kind>:<record_id>`."""
    return {f"{j['kind']}:{j['record_id']}" for j in load_judgements(judge_name, model_id)}


# ---------------------------------------------------------------------------
# Metrics I/O (stage 06)
# ---------------------------------------------------------------------------


def metrics_path(model_id: str) -> Path:
    return METRICS_DIR / f"{model_filename(model_id)}.json"


def save_metrics(payload: dict, model_id: str) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = metrics_path(model_id)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_metrics(model_id: str) -> dict | None:
    path = metrics_path(model_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
