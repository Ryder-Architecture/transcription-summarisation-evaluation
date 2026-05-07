"""
LLM-as-judge evaluation for the transcription summarisation pipeline.

Four judge functions:
  1. evaluate_summary_chunk      — reference-free, 4 dims (Faithfulness, Completeness,
                                   Conciseness, Clarity). Used for tier-1 + rollup
                                   summaries.
  2. evaluate_executive_summary  — adds Notes Alignment when attendee notes exist.
  3. evaluate_action_extraction  — set-based: judge synthesises gold commitments,
                                   matches extracted items (TP/FP/FN for F1).
  4. evaluate_dedup              — judges merge correctness over the raw → deduped delta.

Two backends (interchangeable per call):
  * local_qwen   — vLLM OpenAI-compatible endpoint, JSON-schema guided decoding
  * anthropic    — Claude Haiku via the Anthropic API (cross-check)

All calls use temperature=0.1 for consistency. Per-call retry with exponential
backoff on transient errors. Concurrency cap is enforced by the caller (stage 04/05),
not here.

Prompt design basis:
  - Prometheus (Kim et al. 2024, ICLR): customised score rubrics
  - G-Eval (Liu et al. 2023, EMNLP): numbered evaluation steps + chain-of-thought
  - Validated 0-5 scale: arxiv:2601.03444 (ICC 0.853)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Shared judge prompts — adapted from 06/judge.py for the transcription domain
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a fair judge tasked with evaluating the quality \
of meeting summaries and extracted action items. Assess each output strictly based \
on the given score rubric, not evaluating in general."""

# Reference-free 4-dim judge — used for tier-1 chunks, rollups, and exec
# summaries when no attendee notes exist.
SUMMARY_JUDGE_USER_TEMPLATE = """## Task

Evaluate the following meeting-transcript summary against its source text on four \
dimensions. For each dimension, follow the evaluation steps, then assign a score \
strictly based on the rubric.

## Source Transcript Window

{source_text}

## Summary to Evaluate

Title: {summary_title}

Abstract: {summary_abstract}

## Evaluation Criteria

### 1. Faithfulness

Evaluation steps:
1. Identify all factual claims made in the summary (named participants, decisions, \
numbers, dates, project names).
2. For each claim, check whether it is directly supported by the source transcript.
3. Note any hallucinated facts, invented attendees, or unsupported inferences.
4. Assign a score based on the rubric below.

[Is every claim in the summary traceable to the source transcript?]
Score 1: The summary contains multiple fabricated facts or directly contradicts the source.
Score 2: The summary has several unsupported claims or significant inaccuracies.
Score 3: The summary is mostly accurate but contains minor unsupported inferences or imprecise claims.
Score 4: The summary is accurate with only one negligible imprecision that does not mislead.
Score 5: Every claim is directly and precisely supported by the source transcript.

### 2. Completeness

Evaluation steps:
1. Identify the key information in the source: decisions reached, action commitments, \
named participants, numerical results, project names, key topics discussed.
2. Check which of these appear in the summary.
3. Note any important omissions.
4. Assign a score based on the rubric below.

[Does the summary preserve the key information from the source?]
Score 1: The summary misses most important information and captures only trivial details.
Score 2: The summary captures the general topic but omits several key decisions or commitments.
Score 3: The summary covers the main idea and some key details but misses notable specifics.
Score 4: The summary captures all major points with only minor secondary details omitted.
Score 5: All important information is preserved, including key decisions, names, dates, and numbers.

### 3. Conciseness

Evaluation steps:
1. Check for redundant phrases, repeated information, or unnecessary filler.
2. Assess whether the summary length is appropriate for the source content.
3. Note any information that adds no value.
4. Assign a score based on the rubric below.

[Is the summary appropriately compressed without redundancy?]
Score 1: Extremely redundant, padded with filler, or absurdly short/long for the content.
Score 2: Contains noticeable redundancy or is clearly too verbose for the information conveyed.
Score 3: Mostly concise but has some unnecessary repetition or could be tighter.
Score 4: Well-compressed with only minimal wordiness that does not detract.
Score 5: Tight and efficient — no redundancy, every sentence earns its place.

### 4. Clarity

Evaluation steps:
1. Read the summary for logical flow and sentence structure.
2. Check for grammatical errors, awkward phrasing, or ambiguous references.
3. Assess whether a reader unfamiliar with the source could follow the summary.
4. Assign a score based on the rubric below.

[Is the summary well-written, professional, and easy to understand?]
Score 1: Incoherent, poorly structured, or unreadable.
Score 2: Understandable but contains multiple grammatical issues or confusing passages.
Score 3: Readable and mostly clear but with some awkward phrasing or structural issues.
Score 4: Well-written and well-organized with only minor stylistic imperfections.
Score 5: Clear, professional prose with excellent structure and logical flow.

## Instructions

For each dimension, write your reasoning following the evaluation steps. Then \
provide your final scores. You MUST end your response with a JSON block in \
exactly this format:
```json
{{"faithfulness": <int>, "completeness": <int>, "conciseness": <int>, "clarity": <int>}}
```"""


EXEC_JUDGE_USER_TEMPLATE = """## Task

Evaluate the following executive meeting summary against the full transcript and \
the attendee's own notes on five dimensions. For each dimension, follow the \
evaluation steps, then assign a score strictly based on the rubric.

## Full Transcript

{source_text}

## Executive Summary to Evaluate

Title: {summary_title}

Abstract: {summary_abstract}

## Attendee Notes (Reference)

{gold_notes}

## Evaluation Criteria

### 1. Faithfulness

Evaluation steps:
1. Identify all factual claims made in the executive summary.
2. For each claim, check whether it is directly supported by the transcript.
3. Note any hallucinated facts, invented attendees, or unsupported inferences.
4. Assign a score based on the rubric below.

[Is every claim in the summary traceable to the source transcript?]
Score 1: The summary contains multiple fabricated facts or directly contradicts the source.
Score 2: The summary has several unsupported claims or significant inaccuracies.
Score 3: The summary is mostly accurate but contains minor unsupported inferences or imprecise claims.
Score 4: The summary is accurate with only one negligible imprecision that does not mislead.
Score 5: Every claim is directly and precisely supported by the source transcript.

### 2. Completeness

Evaluation steps:
1. Identify key meeting information: decisions, commitments, participants, numbers, dates, project names, key topics.
2. Check which of these appear in the executive summary.
3. Note any important omissions.
4. Assign a score based on the rubric below.

[Does the summary preserve the key information from the source?]
Score 1: The summary misses most important information and captures only trivial details.
Score 2: The summary captures the general topic but omits several key decisions or commitments.
Score 3: The summary covers the main idea and some key details but misses notable specifics.
Score 4: The summary captures all major points with only minor secondary details omitted.
Score 5: All important information is preserved, including key decisions, names, dates, and numbers.

### 3. Conciseness

Evaluation steps:
1. Check for redundant phrases, repeated information, or unnecessary filler.
2. Assess whether the summary length is appropriate for the meeting's content.
3. Note any information that adds no value.
4. Assign a score based on the rubric below.

[Is the executive summary appropriately compressed without redundancy?]
Score 1: Extremely redundant, padded with filler, or absurdly short/long for the content.
Score 2: Contains noticeable redundancy or is clearly too verbose for the information conveyed.
Score 3: Mostly concise but has some unnecessary repetition or could be tighter.
Score 4: Well-compressed with only minimal wordiness that does not detract.
Score 5: Tight and efficient — no redundancy, every sentence earns its place.

### 4. Clarity

Evaluation steps:
1. Read the executive summary for logical flow and sentence structure.
2. Check for grammatical errors, awkward phrasing, or ambiguous references.
3. Assess whether a reader unfamiliar with the meeting could follow the summary.
4. Assign a score based on the rubric below.

[Is the summary well-written, professional, and easy to understand?]
Score 1: Incoherent, poorly structured, or unreadable.
Score 2: Understandable but contains multiple grammatical issues or confusing passages.
Score 3: Readable and mostly clear but with some awkward phrasing or structural issues.
Score 4: Well-written and well-organized with only minor stylistic imperfections.
Score 5: Clear, professional prose with excellent structure and logical flow.

### 5. Notes Alignment

Evaluation steps:
1. Read the attendee's notes — they reflect what the attendee considered important enough to write down.
2. Compare the executive summary's emphasis, decisions, and topics to the notes.
3. Note items the attendee flagged that are missing or under-emphasised in the summary.
4. Assign a score based on the rubric below.

[Does the executive summary capture what the attendee considered important?]
Score 1: Completely different focus — summary and attendee notes discuss different aspects.
Score 2: Some overlap but misses items the attendee called out as important.
Score 3: Covers the same topics and some flagged items, but has notable gaps.
Score 4: Captures most attendee-flagged items with minor differences in emphasis.
Score 5: Excellent alignment — same emphasis, decisions, and priorities as the attendee notes.

## Instructions

For each dimension, write your reasoning following the evaluation steps. Then \
provide your final scores. You MUST end your response with a JSON block in \
exactly this format:
```json
{{"faithfulness": <int>, "completeness": <int>, "conciseness": <int>, "clarity": <int>, "alignment": <int>}}
```"""


EXEC_SUMMARY_JUDGE_USER_TEMPLATE = """## Task

Evaluate the following executive meeting summary against the full meeting \
transcript on four dimensions. The executive summary was generated from \
section-level abstracts (not directly from this transcript), so it may \
omit minor details — but every claim it makes must be traceable here.

## Full Meeting Transcript

{source_text}

## Executive Summary to Evaluate

Title: {summary_title}

Abstract: {summary_abstract}

## Evaluation Criteria

### 1. Faithfulness

Evaluation steps:
1. Identify all factual claims made in the executive summary (named participants, decisions, \
numbers, dates, project names).
2. For each claim, check whether it is directly supported by the full transcript.
3. Note any hallucinated facts, invented attendees, or unsupported inferences.
4. Assign a score based on the rubric below.

[Is every claim in the executive summary traceable to the full transcript?]
Score 1: The summary contains multiple fabricated facts or directly contradicts the transcript.
Score 2: The summary has several unsupported claims or significant inaccuracies.
Score 3: The summary is mostly accurate but contains minor unsupported inferences or imprecise claims.
Score 4: The summary is accurate with only one negligible imprecision that does not mislead.
Score 5: Every claim is directly and precisely supported by the full transcript.

### 2. Completeness

Evaluation steps:
1. Identify the key information in the full meeting: decisions reached, action commitments, \
named participants, numerical results, project names, key topics discussed.
2. Check which of these appear in the executive summary.
3. Note any important omissions that a reader would need to understand the meeting.
4. Assign a score based on the rubric below.

[Does the executive summary preserve the key information from the full meeting?]
Score 1: The summary misses most important information and captures only trivial details.
Score 2: The summary captures the general topic but omits several key decisions or commitments.
Score 3: The summary covers the main idea and some key details but misses notable specifics.
Score 4: The summary captures all major points with only minor secondary details omitted.
Score 5: All important information is preserved, including key decisions, names, dates, and numbers.

### 3. Conciseness

Evaluation steps:
1. Check for redundant phrases, repeated information, or unnecessary filler.
2. Assess whether the summary length is appropriate for the meeting's content.
3. Note any information that adds no value.
4. Assign a score based on the rubric below.

[Is the executive summary appropriately compressed without redundancy?]
Score 1: Extremely redundant, padded with filler, or absurdly short/long for the content.
Score 2: Contains noticeable redundancy or is clearly too verbose for the information conveyed.
Score 3: Mostly concise but has some unnecessary repetition or could be tighter.
Score 4: Well-compressed with only minimal wordiness that does not detract.
Score 5: Tight and efficient — no redundancy, every sentence earns its place.

### 4. Clarity

Evaluation steps:
1. Read the executive summary for logical flow and sentence structure.
2. Check for grammatical errors, awkward phrasing, or ambiguous references.
3. Assess whether a reader unfamiliar with the meeting could follow the summary.
4. Assign a score based on the rubric below.

[Is the executive summary well-written, professional, and easy to understand?]
Score 1: Incoherent, poorly structured, or unreadable.
Score 2: Understandable but contains multiple grammatical issues or confusing passages.
Score 3: Readable and mostly clear but with some awkward phrasing or structural issues.
Score 4: Well-written and well-organized with only minor stylistic imperfections.
Score 5: Clear, professional prose with excellent structure and logical flow.

## Instructions

For each dimension, write your reasoning following the evaluation steps. Then \
provide your final scores. You MUST end your response with a JSON block in \
exactly this format:
```json
{{"faithfulness": <int>, "completeness": <int>, "conciseness": <int>, "clarity": <int>}}
```"""


ACTION_EXTRACTION_JUDGE_USER_TEMPLATE = """## Task

You are evaluating an action-item extraction system on a transcript window. Three steps:

1. Read the transcript window carefully and enumerate the GOLD COMMITMENTS — every \
explicit or implicit commitment to do something. For each, capture: text, owner \
(or null), deadline (or null), explicit vs implicit.
2. For each EXTRACTED ITEM, decide whether it matches one of your gold commitments \
(true positive) or has no match in the transcript (false positive — hallucinated). \
A match is correct if the same commitment is being described, even with different \
wording, owner phrasing, or paraphrase. Be GENEROUS in matching paraphrases: if \
two items describe the same outcome by the same person, they are the same commitment.
3. For each gold commitment with no matching extracted item, count it as a false negative.

GRANULARITY:
  - If the model split ONE gold commitment into multiple extracted items, the FIRST \
matching item is a TP; the others should ALSO be matched to the same gold_index (it \
is fine to have multiple matches sharing one gold_index). Do NOT mark the additional \
splits as false positives.
  - If the model merged multiple commitments into one extracted item, match it to \
the most-related gold and treat the unmatched golds as false negatives.

A commitment is something a person agreed (explicitly or implicitly) to do. \
Discussion topics, opinions, and statements of fact are NOT commitments and should \
not appear in the gold list.

## Transcript Window

{source_text}

## Extracted Items

{extracted_json}

## Instructions

Reason through the transcript first to enumerate gold commitments. Then walk through \
the extracted items and assign matches. End your response with a JSON block in \
exactly this format:
```json
{{
  "gold_commitments": [
    {{"text": "...", "attributed_to": "..."|null, "deadline": "..."|null, "explicit": true|false}}
  ],
  "matches": [
    {{"extracted_index": <int>, "gold_index": <int>}}
  ],
  "false_positive_indices": [<int>],
  "false_negative_indices": [<int>]
}}
```"""


DEDUP_JUDGE_USER_TEMPLATE = """## Task

You are evaluating a deduplication system that was run over the action items \
extracted from overlapping transcript windows. Decide whether each merge decision was correct.

## Raw Items (with indices)

{raw_json}

## Dedup Result

Groups (each item: canonical_index merged with duplicate_indices):
{groups_json}

Singletons (kept separately, not deduplicated):
{singletons_json}

Two items are duplicates if they describe the SAME commitment by the SAME person, \
even with different wording. Items about different actions, even by the same person, \
are NOT duplicates. Items about the same topic but different actions are NOT duplicates.

## Instructions

For each merge group, decide whether it is a CORRECT merge (true positive — items \
were genuinely duplicates) or an INCORRECT merge (false positive — distinct items \
were wrongly conflated).

Then look across the singletons and groups for missed duplicates — pairs of items \
that describe the same commitment but were not merged.

End your response with a JSON block in exactly this format:
```json
{{
  "correct_merges": [<group_position>],
  "incorrect_merges": [<group_position>],
  "missed_duplicate_pairs": [[<raw_index_a>, <raw_index_b>]]
}}
```
Where <group_position> is the 0-based index of the group in the dedup result above."""


# ---------------------------------------------------------------------------
# JSON schemas for vLLM guided decoding
# ---------------------------------------------------------------------------

SUMMARY_SCORES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "summary_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "faithfulness_reasoning": {"type": "string"},
                "completeness_reasoning": {"type": "string"},
                "conciseness_reasoning": {"type": "string"},
                "clarity_reasoning": {"type": "string"},
                "faithfulness": {"type": "integer"},
                "completeness": {"type": "integer"},
                "conciseness": {"type": "integer"},
                "clarity": {"type": "integer"},
            },
            "required": [
                "faithfulness_reasoning", "completeness_reasoning",
                "conciseness_reasoning", "clarity_reasoning",
                "faithfulness", "completeness", "conciseness", "clarity",
            ],
            "additionalProperties": False,
        },
    },
}

EXEC_SCORES_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "exec_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "faithfulness_reasoning": {"type": "string"},
                "completeness_reasoning": {"type": "string"},
                "conciseness_reasoning": {"type": "string"},
                "clarity_reasoning": {"type": "string"},
                "alignment_reasoning": {"type": "string"},
                "faithfulness": {"type": "integer"},
                "completeness": {"type": "integer"},
                "conciseness": {"type": "integer"},
                "clarity": {"type": "integer"},
                "alignment": {"type": "integer"},
            },
            "required": [
                "faithfulness_reasoning", "completeness_reasoning",
                "conciseness_reasoning", "clarity_reasoning", "alignment_reasoning",
                "faithfulness", "completeness", "conciseness", "clarity", "alignment",
            ],
            "additionalProperties": False,
        },
    },
}

ACTION_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "action_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "gold_commitments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "attributed_to": {"type": ["string", "null"]},
                            "deadline": {"type": ["string", "null"]},
                            "explicit": {"type": "boolean"},
                        },
                        "required": ["text", "attributed_to", "deadline", "explicit"],
                        "additionalProperties": False,
                    },
                },
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "extracted_index": {"type": "integer"},
                            "gold_index": {"type": "integer"},
                        },
                        "required": ["extracted_index", "gold_index"],
                        "additionalProperties": False,
                    },
                },
                "false_positive_indices": {"type": "array", "items": {"type": "integer"}},
                "false_negative_indices": {"type": "array", "items": {"type": "integer"}},
            },
            "required": [
                "gold_commitments", "matches",
                "false_positive_indices", "false_negative_indices",
            ],
            "additionalProperties": False,
        },
    },
}

DEDUP_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "dedup_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correct_merges": {"type": "array", "items": {"type": "integer"}},
                "incorrect_merges": {"type": "array", "items": {"type": "integer"}},
                "missed_duplicate_pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
            "required": ["correct_merges", "incorrect_merges", "missed_duplicate_pairs"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Result dataclasses (serialised to per-judge JSON)
# ---------------------------------------------------------------------------


@dataclass
class JudgementOut:
    """Common fields for any judgement; written to results/judgements/...json."""

    kind: str  # "summary" | "exec" | "action_extraction" | "dedup"
    record_id: str
    transcript_id: str
    candidate_model: str
    judge_name: str
    payload: dict  # the parsed scores / structured judgement
    raw_response: str
    error: str | None = None


def _judgement_dict(j: JudgementOut) -> dict:
    return asdict(j)


# ---------------------------------------------------------------------------
# Backend abstraction — local Qwen (vLLM) or Anthropic Haiku
# ---------------------------------------------------------------------------


JUDGE_TEMPERATURE = 0.1
JUDGE_MAX_TOKENS = 4096
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 2.0


async def _retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
    raise last_err  # type: ignore[misc]


def _local_client(base_url: str):
    import openai
    key = f"_openai_{base_url}"
    if not hasattr(_local_client, key):
        setattr(
            _local_client,
            key,
            openai.AsyncOpenAI(
                base_url=base_url,
                api_key="not-needed",
                timeout=httpx.Timeout(connect=30, read=900, write=30, pool=30),
            ),
        )
    return getattr(_local_client, key)


def _anthropic_client():
    import anthropic
    if not hasattr(_anthropic_client, "_instance"):
        _anthropic_client._instance = anthropic.AsyncAnthropic()
    return _anthropic_client._instance


async def _call_local_qwen(
    *,
    base_url: str,
    model: str,
    user_prompt: str,
    response_schema: dict | None,
) -> str:
    client = _local_client(base_url)
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if response_schema is not None:
        kwargs["response_format"] = response_schema
    resp = await _retry(client.chat.completions.create, **kwargs)
    return resp.choices[0].message.content or ""


async def _call_anthropic(
    *,
    model: str,
    user_prompt: str,
) -> str:
    client = _anthropic_client()
    resp = await _retry(
        client.messages.create,
        model=model,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Prompt-truncation helper
# ---------------------------------------------------------------------------


def _truncate_source(source: str, limit_chars: int = 90_000) -> str:
    """Anthropic + most local Qwens cap context around 100K chars; trim long inputs."""
    if len(source) <= limit_chars:
        return source
    return source[:limit_chars] + "\n\n[... truncated ...]"


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Find the last JSON object in a judge response, with json_repair fallback."""
    if not text:
        return None
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = list(blocks)
    if not candidates:
        # Fallback: any JSON object in the text
        candidates = re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.DOTALL)
    for raw in reversed(candidates):
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
    # Last resort: json_repair on the whole text
    try:
        from json_repair import json_repair  # local import to keep top-level cheap
        parsed = json_repair.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:  # noqa: BLE001
        return None
    return None


def _clamp_score(v: Any) -> int:
    try:
        return max(1, min(5, int(v)))
    except (TypeError, ValueError):
        return 1


# ---------------------------------------------------------------------------
# Public judge functions — keyword-only, backend-agnostic
# ---------------------------------------------------------------------------


async def evaluate_summary_chunk(
    *,
    source_text: str,
    summary_title: str,
    summary_abstract: str,
    record_id: str,
    transcript_id: str,
    candidate_model: str,
    judge_name: str,
    backend: str,  # "local_qwen" | "anthropic"
    judge_model: str,
    base_url: str | None = None,
) -> JudgementOut:
    user_prompt = SUMMARY_JUDGE_USER_TEMPLATE.format(
        source_text=_truncate_source(source_text),
        summary_title=summary_title,
        summary_abstract=summary_abstract,
    )
    return await _run_judge(
        kind="summary",
        record_id=record_id,
        transcript_id=transcript_id,
        candidate_model=candidate_model,
        judge_name=judge_name,
        backend=backend,
        judge_model=judge_model,
        base_url=base_url,
        user_prompt=user_prompt,
        schema=SUMMARY_SCORES_SCHEMA,
        post_parse=_summary_payload,
    )


async def evaluate_executive_summary(
    *,
    source_text: str,
    summary_title: str,
    summary_abstract: str,
    record_id: str,
    transcript_id: str,
    candidate_model: str,
    judge_name: str,
    backend: str,
    judge_model: str,
    base_url: str | None = None,
) -> JudgementOut:
    """Judge an exec summary against the full transcript using the exec-specific
    template (correct 'Full Meeting Transcript' framing, not 'Source Transcript Window').
    Uses the same 4-dim rubric as tier-1 — no Notes Alignment since the experiment
    has no attendee notes."""
    user_prompt = EXEC_SUMMARY_JUDGE_USER_TEMPLATE.format(
        source_text=_truncate_source(source_text),
        summary_title=summary_title,
        summary_abstract=summary_abstract,
    )
    return await _run_judge(
        kind="exec",
        record_id=record_id,
        transcript_id=transcript_id,
        candidate_model=candidate_model,
        judge_name=judge_name,
        backend=backend,
        judge_model=judge_model,
        base_url=base_url,
        user_prompt=user_prompt,
        schema=SUMMARY_SCORES_SCHEMA,
        post_parse=_summary_payload,
    )


async def evaluate_action_extraction(
    *,
    source_text: str,
    extracted_items: list[dict],
    record_id: str,
    transcript_id: str,
    candidate_model: str,
    judge_name: str,
    backend: str,
    judge_model: str,
    base_url: str | None = None,
) -> JudgementOut:
    user_prompt = ACTION_EXTRACTION_JUDGE_USER_TEMPLATE.format(
        source_text=_truncate_source(source_text),
        extracted_json=json.dumps(extracted_items, indent=2),
    )
    return await _run_judge(
        kind="action_extraction",
        record_id=record_id,
        transcript_id=transcript_id,
        candidate_model=candidate_model,
        judge_name=judge_name,
        backend=backend,
        judge_model=judge_model,
        base_url=base_url,
        user_prompt=user_prompt,
        schema=ACTION_JUDGE_SCHEMA,
        post_parse=_action_payload,
    )


async def evaluate_dedup(
    *,
    raw_items: list[dict],
    groups: list[dict],
    singletons: list[int],
    transcript_id: str,
    candidate_model: str,
    judge_name: str,
    backend: str,
    judge_model: str,
    base_url: str | None = None,
) -> JudgementOut:
    user_prompt = DEDUP_JUDGE_USER_TEMPLATE.format(
        raw_json=json.dumps(
            [
                {"index": i, "text": x.get("text"), "attributed_to": x.get("attributed_to"),
                 "deadline": x.get("deadline")}
                for i, x in enumerate(raw_items)
            ],
            indent=2,
        ),
        groups_json=json.dumps(groups, indent=2),
        singletons_json=json.dumps(singletons, indent=2),
    )
    return await _run_judge(
        kind="dedup",
        record_id=f"{transcript_id}_dedup",
        transcript_id=transcript_id,
        candidate_model=candidate_model,
        judge_name=judge_name,
        backend=backend,
        judge_model=judge_model,
        base_url=base_url,
        user_prompt=user_prompt,
        schema=DEDUP_JUDGE_SCHEMA,
        post_parse=_dedup_payload,
    )


# ---------------------------------------------------------------------------
# Shared judge runner
# ---------------------------------------------------------------------------


async def _run_judge(
    *,
    kind: str,
    record_id: str,
    transcript_id: str,
    candidate_model: str,
    judge_name: str,
    backend: str,
    judge_model: str,
    base_url: str | None,
    user_prompt: str,
    schema: dict,
    post_parse,
) -> JudgementOut:
    raw = ""
    error: str | None = None
    payload: dict = {}
    try:
        if backend == "local_qwen":
            assert base_url, "local_qwen backend requires base_url"
            raw = await _call_local_qwen(
                base_url=base_url,
                model=judge_model,
                user_prompt=user_prompt,
                response_schema=schema,
            )
        elif backend == "anthropic":
            raw = await _call_anthropic(model=judge_model, user_prompt=user_prompt)
        else:
            raise ValueError(f"unknown backend: {backend}")
        parsed = _extract_json(raw)
        if parsed is None:
            error = "no JSON object found in judge response"
        else:
            payload = post_parse(parsed)
    except Exception as e:  # noqa: BLE001
        error = f"{type(e).__name__}: {e}"

    return JudgementOut(
        kind=kind,
        record_id=record_id,
        transcript_id=transcript_id,
        candidate_model=candidate_model,
        judge_name=judge_name,
        payload=payload,
        raw_response=raw,
        error=error,
    )


# ---------------------------------------------------------------------------
# Per-kind payload normalisers
# ---------------------------------------------------------------------------


def _summary_payload(parsed: dict) -> dict:
    return {
        "faithfulness": _clamp_score(parsed.get("faithfulness")),
        "completeness": _clamp_score(parsed.get("completeness")),
        "conciseness": _clamp_score(parsed.get("conciseness")),
        "clarity": _clamp_score(parsed.get("clarity")),
        "reasoning": {
            k: parsed.get(f"{k}_reasoning") for k in ("faithfulness", "completeness", "conciseness", "clarity")
        },
    }


def _exec_payload(parsed: dict) -> dict:
    base = _summary_payload(parsed)
    base["alignment"] = _clamp_score(parsed.get("alignment"))
    base["reasoning"]["alignment"] = parsed.get("alignment_reasoning")
    return base


def _action_payload(parsed: dict) -> dict:
    matches = []
    for m in parsed.get("matches", []) or []:
        try:
            matches.append({
                "extracted_index": int(m.get("extracted_index")),
                "gold_index": int(m.get("gold_index")),
            })
        except (TypeError, ValueError):
            continue
    return {
        "gold_commitments": parsed.get("gold_commitments", []) or [],
        "matches": matches,
        "false_positive_indices": [int(i) for i in (parsed.get("false_positive_indices") or []) if isinstance(i, int)],
        "false_negative_indices": [int(i) for i in (parsed.get("false_negative_indices") or []) if isinstance(i, int)],
    }


def _dedup_payload(parsed: dict) -> dict:
    return {
        "correct_merges": [int(i) for i in (parsed.get("correct_merges") or []) if isinstance(i, int)],
        "incorrect_merges": [int(i) for i in (parsed.get("incorrect_merges") or []) if isinstance(i, int)],
        "missed_duplicate_pairs": [
            [int(p[0]), int(p[1])]
            for p in (parsed.get("missed_duplicate_pairs") or [])
            if isinstance(p, list) and len(p) == 2
        ],
    }


# ---------------------------------------------------------------------------
# Anthropic Haiku model id (cross-check)
# ---------------------------------------------------------------------------

ANTHROPIC_HAIKU_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_JUDGE_NAME = "claude-haiku-api"
