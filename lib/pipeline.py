"""
Self-contained reimplementation of the production summarisation pipeline for benchmarking.

Substitutions vs production:
  - The production HTTP inference wrapper is replaced by a direct streaming
    aiohttp client so we capture per-call TTFT, TPOT, wall-time, prompt/completion tokens.
  - Production's `outlines` library for action extraction is replaced by vLLM's
    `response_format` JSON schema mode (same guided-decoding path 06 uses).
  - `SummarisationCheckpoint` is omitted (production-only resume hook).
  - Concurrency cap is a single shared semaphore (default 16), exposed via
    --gen-concurrency in 03_generate.py.

Everything else — chunking, prompts, sampling parameters, hierarchical rollup,
parallel summarise+extract, dedup, executive synthesis, manual/combined aggregation
modes — matches production behaviour at the SHA pinned in lib/prompts.py.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime

import aiohttp
from json_repair import json_repair

from lib.io import (
    CachedTranscript,
    DedupRecord,
    ExtractionRecord,
    SummaryRecord,
    TranscriptTurn,
)
from lib.prompts import (
    ACTION_DEDUP_SYSTEM_PROMPT,
    ACTION_DEDUP_USER_PROMPT,
    ACTION_EXTRACTION_JSON_SCHEMA,
    ACTION_EXTRACTION_SYSTEM_PROMPT,
    ACTION_EXTRACTION_USER_PROMPT,
    CHUNK_SUMMARISATION_SYSTEM_PROMPT,
    CHUNK_SUMMARISATION_USER_PROMPT,
    DEDUP_JSON_SCHEMA,
    ROLLUP_SUMMARISATION_SYSTEM_PROMPT,
    ROLLUP_SUMMARISATION_USER_PROMPT,
    SUMMARY_AGGREGATION_SYSTEM_PROMPT,
    SUMMARY_AGGREGATION_USER_PROMPT,
    SUMMARY_JSON_SCHEMA,
)

# ---------------------------------------------------------------------------
# Production sampling/chunking constants — frozen from src/config.py defaults
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # matches transcript_chunker.estimate_tokens

SUMMARISATION_TOKEN_BUDGET = 6000
SUMMARISATION_OVERLAP_TURNS = 5
EXTRACTION_TOKEN_BUDGET = 6000
EXTRACTION_OVERLAP_TURNS = 5
ROLLUP_TOKEN_BUDGET = 2000
MIN_SUMMARISATION_TOKENS = 100

SUMMARISATION_TEMPERATURE = 0.3
SUMMARISATION_MAX_TOKENS = 4096
ROLLUP_TEMPERATURE = 0.3
ROLLUP_MAX_TOKENS = 4096
EXECUTIVE_TEMPERATURE = 0.0
EXECUTIVE_MAX_TOKENS = 4096
EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_MAX_TOKENS = 2048
DEDUP_TEMPERATURE = 0.0
DEDUP_MAX_TOKENS = 4096

SUMMARISATION_MIN_WORDS = 100
SUMMARISATION_MAX_WORDS = 200
ROLLUP_MIN_WORDS = 100
ROLLUP_MAX_WORDS = 200
EXECUTIVE_MIN_WORDS = 300
EXECUTIVE_MAX_WORDS = 400

DEFAULT_GEN_CONCURRENCY = 16


# ---------------------------------------------------------------------------
# Transcript windowing — port of src/services/transcription/transcript_chunker.py
# ---------------------------------------------------------------------------


@dataclass
class TranscriptWindow:
    text: str
    start_turn: int
    end_turn: int
    start_time: datetime
    end_time: datetime
    token_count: int


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _format_turn(turn: TranscriptTurn) -> str:
    speaker = turn.speaker_name or "Unknown"
    return f"[{speaker}]: {turn.content}"


def chunk_transcript(
    turns: list[TranscriptTurn],
    token_budget: int = SUMMARISATION_TOKEN_BUDGET,
    overlap_turns: int = SUMMARISATION_OVERLAP_TURNS,
) -> list[TranscriptWindow]:
    if not turns:
        return []

    sorted_turns = sorted(turns, key=lambda t: t.order)
    formatted = [_format_turn(t) for t in sorted_turns]

    windows: list[TranscriptWindow] = []
    start_idx = 0

    while start_idx < len(sorted_turns):
        accumulated = 0
        end_idx = start_idx
        while end_idx < len(sorted_turns):
            turn_tokens = estimate_tokens(formatted[end_idx]) + 1  # newline
            if accumulated + turn_tokens > token_budget and end_idx > start_idx:
                break
            accumulated += turn_tokens
            end_idx += 1

        window_text = "\n".join(formatted[start_idx:end_idx])
        windows.append(
            TranscriptWindow(
                text=window_text,
                start_turn=sorted_turns[start_idx].order,
                end_turn=sorted_turns[end_idx - 1].order,
                start_time=_parse_dt(sorted_turns[start_idx].start_time),
                end_time=_parse_dt(sorted_turns[end_idx - 1].end_time),
                token_count=estimate_tokens(window_text),
            )
        )
        if end_idx >= len(sorted_turns):
            break
        start_idx = max(end_idx - overlap_turns, start_idx + 1)

    return windows


def _parse_dt(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Streaming LLM call — port of 06/benchmark.py call_llm_streaming
# ---------------------------------------------------------------------------


@dataclass
class StreamingResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    wall_time_s: float
    ttft_s: float | None
    tpot_s: float | None
    error: str | None = None


async def _call_llm_streaming(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
) -> StreamingResult:
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    t_start = time.perf_counter()
    t_first_chunk: float | None = None
    chunk_times: list[float] = []
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return StreamingResult(
                    text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    wall_time_s=time.perf_counter() - t_start,
                    ttft_s=None,
                    tpot_s=None,
                    error=f"HTTP {resp.status}: {body[:500]}",
                )

            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line or not line.startswith(b"data: "):
                    continue
                data_str = line[6:].decode("utf-8", errors="replace")
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                now = time.perf_counter()
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        if t_first_chunk is None:
                            t_first_chunk = now
                        chunk_times.append(now)
                        chunks.append(content)
                if usage := data.get("usage"):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        wall_time = time.perf_counter() - t_start
        ttft = (t_first_chunk - t_start) if t_first_chunk is not None else None
        tpot = None
        if len(chunk_times) > 1:
            intervals = [
                chunk_times[i] - chunk_times[i - 1]
                for i in range(1, len(chunk_times))
            ]
            tpot = sum(intervals) / len(intervals)

        text = "".join(chunks)
        if "<think>" in text:
            parts = text.split("</think>")
            text = parts[-1] if len(parts) > 1 else text
        text = text.strip()

        return StreamingResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_time_s=wall_time,
            ttft_s=ttft,
            tpot_s=tpot,
        )
    except Exception as e:
        return StreamingResult(
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            wall_time_s=time.perf_counter() - t_start,
            ttft_s=None,
            tpot_s=None,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Pipeline result types
# ---------------------------------------------------------------------------


@dataclass
class MeetingResult:
    summaries: list[SummaryRecord] = field(default_factory=list)
    extractions: list[ExtractionRecord] = field(default_factory=list)
    dedup: DedupRecord | None = None


# ---------------------------------------------------------------------------
# Per-LLM-call helpers (one per stage)
# ---------------------------------------------------------------------------


async def _summarise_chunk(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    window: TranscriptWindow,
    window_index: int,
    model_id: str,
    vllm_url: str,
    is_warmup: bool = False,
) -> SummaryRecord:
    sys_prompt = CHUNK_SUMMARISATION_SYSTEM_PROMPT.format(
        min_words=SUMMARISATION_MIN_WORDS,
        max_words=SUMMARISATION_MAX_WORDS,
    )
    user_prompt = CHUNK_SUMMARISATION_USER_PROMPT.format(chunk=window.text)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": SUMMARISATION_MAX_TOKENS,
        "temperature": SUMMARISATION_TEMPERATURE,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        result = await _call_llm_streaming(session, vllm_url, payload)

    title, abstract = _parse_title_abstract(result.text)
    return SummaryRecord(
        transcript_id=transcript_id,
        record_id=f"{transcript_id}_tier1_w{window_index:03d}",
        model_name=model_id,
        tier=1,
        is_executive=False,
        title=title,
        abstract=abstract,
        start_turn=window.start_turn,
        end_turn=window.end_turn,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        wall_time_s=result.wall_time_s,
        ttft_s=result.ttft_s,
        tpot_s=result.tpot_s,
        is_warmup=is_warmup,
        error=result.error,
    )


async def _rollup_summaries(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    children: list[SummaryRecord],
    tier: int,
    group_index: int,
    source_excerpts: str,
    model_id: str,
    vllm_url: str,
) -> SummaryRecord:
    sections = "\n".join(
        f"Section Title: {c.title}\nSection Abstract: {c.abstract}" for c in children
    )
    sys_prompt = ROLLUP_SUMMARISATION_SYSTEM_PROMPT.format(
        min_words=ROLLUP_MIN_WORDS,
        max_words=ROLLUP_MAX_WORDS,
    )
    user_prompt = ROLLUP_SUMMARISATION_USER_PROMPT.format(
        sections=sections,
        source_excerpts=source_excerpts,
    )
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": ROLLUP_MAX_TOKENS,
        "temperature": ROLLUP_TEMPERATURE,
        "response_format": SUMMARY_JSON_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        result = await _call_llm_streaming(session, vllm_url, payload)

    title, abstract = _parse_title_abstract(result.text)
    start_turns = [c.start_turn for c in children if c.start_turn is not None]
    end_turns = [c.end_turn for c in children if c.end_turn is not None]
    return SummaryRecord(
        transcript_id=transcript_id,
        record_id=f"{transcript_id}_tier{tier}_g{group_index:03d}",
        model_name=model_id,
        tier=tier,
        is_executive=False,
        title=title,
        abstract=abstract,
        start_turn=min(start_turns) if start_turns else None,
        end_turn=max(end_turns) if end_turns else None,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        wall_time_s=result.wall_time_s,
        ttft_s=result.ttft_s,
        tpot_s=result.tpot_s,
        error=result.error,
    )


async def _synthesise_executive_call(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    combined_context: str,
    model_id: str,
    vllm_url: str,
) -> SummaryRecord:
    word_vars = {"min_words": EXECUTIVE_MIN_WORDS, "max_words": EXECUTIVE_MAX_WORDS}
    sys_prompt = SUMMARY_AGGREGATION_SYSTEM_PROMPT.format(**word_vars)
    user_prompt = SUMMARY_AGGREGATION_USER_PROMPT.format(combined_context=combined_context, **word_vars)

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": EXECUTIVE_MAX_TOKENS,
        "temperature": EXECUTIVE_TEMPERATURE,
        "response_format": SUMMARY_JSON_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        result = await _call_llm_streaming(session, vllm_url, payload)

    title, abstract = _parse_title_abstract(result.text)
    return SummaryRecord(
        transcript_id=transcript_id,
        record_id=f"{transcript_id}_exec",
        model_name=model_id,
        tier=0,
        is_executive=True,
        title=title,
        abstract=abstract,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        wall_time_s=result.wall_time_s,
        ttft_s=result.ttft_s,
        tpot_s=result.tpot_s,
        error=result.error,
    )


async def _extract_window(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    window: TranscriptWindow,
    window_index: int,
    model_id: str,
    vllm_url: str,
) -> ExtractionRecord:
    sys_prompt = ACTION_EXTRACTION_SYSTEM_PROMPT
    user_prompt = ACTION_EXTRACTION_USER_PROMPT.format(chunk=window.text)
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": EXTRACTION_MAX_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "response_format": ACTION_EXTRACTION_JSON_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        result = await _call_llm_streaming(session, vllm_url, payload)

    raw_items: list[dict] = []
    error = result.error
    if not error and result.text:
        try:
            parsed = json_repair.loads(result.text)
            raw_items = parsed.get("action_items", []) if isinstance(parsed, dict) else []
        except Exception as e:  # noqa: BLE001
            error = f"json parse failed: {e}"

    return ExtractionRecord(
        transcript_id=transcript_id,
        record_id=f"{transcript_id}_xw{window_index:03d}",
        model_name=model_id,
        window_index=window_index,
        start_turn=window.start_turn,
        end_turn=window.end_turn,
        raw_items=raw_items,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        wall_time_s=result.wall_time_s,
        ttft_s=result.ttft_s,
        tpot_s=result.tpot_s,
        error=error,
    )


async def _dedup_call(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    raw_items: list[dict],
    model_id: str,
    vllm_url: str,
) -> DedupRecord:
    if len(raw_items) <= 1:
        return DedupRecord(
            transcript_id=transcript_id,
            model_name=model_id,
            raw_count=len(raw_items),
            deduped_count=len(raw_items),
            deduped_indices=list(range(len(raw_items))),
            singletons=list(range(len(raw_items))),
        )

    items_for_llm = [
        {
            "index": i,
            "text": item.get("text", ""),
            "attributed_to": item.get("attributed_to"),
            "deadline": item.get("deadline"),
        }
        for i, item in enumerate(raw_items)
    ]
    sys_prompt = ACTION_DEDUP_SYSTEM_PROMPT
    user_prompt = ACTION_DEDUP_USER_PROMPT.format(
        items_json=json.dumps(items_for_llm, indent=2)
    )
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": DEDUP_MAX_TOKENS,
        "temperature": DEDUP_TEMPERATURE,
        "response_format": DEDUP_JSON_SCHEMA,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        result = await _call_llm_streaming(session, vllm_url, payload)

    error = result.error
    groups: list[dict] = []
    singletons: list[int] = []
    deduped: list[int] = []
    if not error and result.text:
        try:
            parsed = json_repair.loads(result.text)
            if isinstance(parsed, dict):
                groups = parsed.get("groups", []) or []
                singletons = parsed.get("singletons", []) or []
                winners = {g.get("canonical_index") for g in groups if isinstance(g, dict)}
                winners.update(int(i) for i in singletons)
                deduped = sorted(int(i) for i in winners if isinstance(i, int) and 0 <= i < len(raw_items))
        except Exception as e:  # noqa: BLE001
            error = f"json parse failed: {e}"

    if not deduped and not error:
        # Fallback: dedup parse returned empty winners — keep all items
        deduped = list(range(len(raw_items)))
        singletons = deduped

    return DedupRecord(
        transcript_id=transcript_id,
        model_name=model_id,
        raw_count=len(raw_items),
        deduped_count=len(deduped),
        deduped_indices=deduped,
        groups=groups,
        singletons=singletons,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        wall_time_s=result.wall_time_s,
        error=error,
    )


# ---------------------------------------------------------------------------
# Hierarchical orchestration — port of SummarisationService private methods
# ---------------------------------------------------------------------------


def _combined_size(records: list[SummaryRecord]) -> int:
    return sum(estimate_tokens(r.abstract or "") for r in records)


def _group_by_budget(
    records: list[SummaryRecord], budget: int
) -> list[list[SummaryRecord]]:
    groups: list[list[SummaryRecord]] = []
    current: list[SummaryRecord] = []
    current_size = 0
    for r in records:
        size = estimate_tokens(r.abstract or "")
        if current_size + size > budget and current:
            groups.append(current)
            current = []
            current_size = 0
        current.append(r)
        current_size += size
    if current:
        groups.append(current)
    return groups


def _extract_source_excerpts(
    windows: list[TranscriptWindow],
    chunks: list[SummaryRecord],
    max_chars: int = 2000,
) -> str:
    if not windows or not chunks:
        return ""

    start_turns = [c.start_turn for c in chunks if c.start_turn is not None]
    end_turns = [c.end_turn for c in chunks if c.end_turn is not None]
    if not start_turns or not end_turns:
        return ""

    range_start = min(start_turns)
    range_end = max(end_turns)
    relevant = [w for w in windows if w.start_turn <= range_end and w.end_turn >= range_start]
    if not relevant:
        return ""

    excerpts: list[str] = []
    total = 0
    for w in relevant:
        lines = w.text.split("\n")
        selected = lines[:2]
        if len(lines) > 4:
            selected.append("...")
        if len(lines) > 2:
            selected.extend(lines[-2:])
        snippet = "\n".join(selected)
        if total + len(snippet) > max_chars:
            break
        excerpts.append(snippet)
        total += len(snippet)
    return "\n---\n".join(excerpts)


async def _hierarchical_summarise(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    windows: list[TranscriptWindow],
    model_id: str,
    vllm_url: str,
) -> tuple[list[SummaryRecord], list[SummaryRecord]]:
    """Tier-1 in parallel; recursively roll up until under ROLLUP_TOKEN_BUDGET.

    Returns (all_chunks, final_tier).
    """
    tier1_results = await asyncio.gather(
        *[
            _summarise_chunk(
                session,
                sem,
                transcript_id=transcript_id,
                window=w,
                window_index=i,
                model_id=model_id,
                vllm_url=vllm_url,
                is_warmup=False,
            )
            for i, w in enumerate(windows)
        ],
        return_exceptions=False,
    )

    all_chunks: list[SummaryRecord] = list(tier1_results)
    current = list(tier1_results)
    tier_num = 1

    while _combined_size(current) > ROLLUP_TOKEN_BUDGET:
        tier_num += 1
        groups = _group_by_budget(current, ROLLUP_TOKEN_BUDGET)
        next_tier = await asyncio.gather(
            *[
                _rollup_summaries(
                    session,
                    sem,
                    transcript_id=transcript_id,
                    children=group,
                    tier=tier_num,
                    group_index=gi,
                    source_excerpts=_extract_source_excerpts(windows, group),
                    model_id=model_id,
                    vllm_url=vllm_url,
                )
                for gi, group in enumerate(groups)
            ],
            return_exceptions=False,
        )
        if not next_tier:
            break
        all_chunks.extend(next_tier)
        current = list(next_tier)

    return all_chunks, current


async def _extract_and_deduplicate(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    *,
    transcript_id: str,
    windows: list[TranscriptWindow],
    model_id: str,
    vllm_url: str,
) -> tuple[list[ExtractionRecord], DedupRecord]:
    raw_records = await asyncio.gather(
        *[
            _extract_window(
                session,
                sem,
                transcript_id=transcript_id,
                window=w,
                window_index=i,
                model_id=model_id,
                vllm_url=vllm_url,
            )
            for i, w in enumerate(windows)
        ],
        return_exceptions=False,
    )

    flattened: list[dict] = []
    for rec in raw_records:
        flattened.extend(rec.raw_items or [])

    dedup = await _dedup_call(
        session,
        sem,
        transcript_id=transcript_id,
        raw_items=flattened,
        model_id=model_id,
        vllm_url=vllm_url,
    )
    return list(raw_records), dedup


def _build_synthesis_context(
    summaries: list[SummaryRecord],
    deduped_items: list[dict],
) -> str:
    parts = [
        f"Section Title: {s.title}\nSection Abstract: {s.abstract}"
        for s in summaries
        if not s.is_executive
    ]
    context = "\n".join(parts)

    if deduped_items:
        action_lines = []
        for item in deduped_items:
            line = f"- {item.get('text', '')}"
            if item.get("attributed_to"):
                line += f" (Owner: {item['attributed_to']})"
            if item.get("deadline"):
                line += f" [Deadline: {item['deadline']}]"
            action_lines.append(line)
        context += "\n\nAction Items:\n" + "\n".join(action_lines)

    return context


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def summarise_meeting(
    transcript: CachedTranscript,
    model_id: str,
    vllm_url: str,
    *,
    gen_concurrency: int = DEFAULT_GEN_CONCURRENCY,
    is_first_meeting: bool = False,
) -> MeetingResult:
    """Run the carbon-copy pipeline on one transcript.

    Mirrors `SummarisationService.run()` step-for-step. Returns a MeetingResult
    with all summary records (tier 1, rollups, executive), all extraction window
    records, and the dedup record.

    `is_first_meeting=True` flags the very first tier-1 call as `is_warmup=True`
    so warmup latency is excluded from timing aggregations downstream.
    """
    sem = asyncio.Semaphore(gen_concurrency)

    summary_windows = chunk_transcript(
        transcript.turns,
        token_budget=SUMMARISATION_TOKEN_BUDGET,
        overlap_turns=SUMMARISATION_OVERLAP_TURNS,
    )
    extraction_windows = chunk_transcript(
        transcript.turns,
        token_budget=EXTRACTION_TOKEN_BUDGET,
        overlap_turns=EXTRACTION_OVERLAP_TURNS,
    )

    async with aiohttp.ClientSession() as session:
        # Run summarisation + extraction in parallel — production behaviour
        (all_tier_chunks, final_tier), (extractions, dedup) = await asyncio.gather(
            _hierarchical_summarise(
                session,
                sem,
                transcript_id=transcript.transcript_id,
                windows=summary_windows,
                model_id=model_id,
                vllm_url=vllm_url,
            ),
            _extract_and_deduplicate(
                session,
                sem,
                transcript_id=transcript.transcript_id,
                windows=extraction_windows,
                model_id=model_id,
                vllm_url=vllm_url,
            ),
        )

        # Mark first call as warmup retroactively (only for first meeting in a model run)
        if is_first_meeting and all_tier_chunks:
            all_tier_chunks[0].is_warmup = True

        deduped_items = [extractions_for_dedup_winner(extractions, idx) for idx in dedup.deduped_indices]
        deduped_items = [d for d in deduped_items if d is not None]

        context = _build_synthesis_context(final_tier, deduped_items)
        executive = await _synthesise_executive_call(
            session,
            sem,
            transcript_id=transcript.transcript_id,
            combined_context=context,
            model_id=model_id,
            vllm_url=vllm_url,
        )
        all_tier_chunks.append(executive)

    return MeetingResult(
        summaries=all_tier_chunks,
        extractions=extractions,
        dedup=dedup,
    )


def extractions_for_dedup_winner(
    extractions: list[ExtractionRecord],
    flat_index: int,
) -> dict | None:
    """Translate a flat winner index back to its raw item dict."""
    cursor = 0
    for rec in extractions:
        n = len(rec.raw_items or [])
        if flat_index < cursor + n:
            return rec.raw_items[flat_index - cursor]
        cursor += n
    return None


def _parse_title_abstract(raw_text: str) -> tuple[str, str]:
    """Parse the JSON title/abstract response, with json_repair fallback."""
    if not raw_text:
        return "", ""
    try:
        parsed = json_repair.loads(raw_text)
        if isinstance(parsed, dict):
            title = str(parsed.get("title") or "")
            abstract = str(parsed.get("abstract") or "")
            return title, abstract
    except Exception:  # noqa: BLE001
        pass
    return "", raw_text  # fall back to dumping raw text into abstract
