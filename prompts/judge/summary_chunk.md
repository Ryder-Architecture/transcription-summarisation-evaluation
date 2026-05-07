# Judge — Summary Chunk

Used to evaluate tier-1 chunk summaries and rollup summaries. Reference-free: scored against the source window only. Temperature 0.1.

Prompt design basis: Prometheus (Kim et al. 2024, ICLR) score rubrics; G-Eval (Liu et al. 2023, EMNLP) numbered steps + chain-of-thought; validated 0–5 scale (arxiv:2601.03444, ICC 0.853).

## System

```
You are a fair judge tasked with evaluating the quality of meeting summaries
and extracted action items. Assess each output strictly based on the given
score rubric, not evaluating in general.
```

## User

```
## Task

Evaluate the following meeting-transcript summary against its source text on four
dimensions. For each dimension, follow the evaluation steps, then assign a score
strictly based on the rubric.

## Source Transcript Window

{source_text}

## Summary to Evaluate

Title: {summary_title}

Abstract: {summary_abstract}

## Evaluation Criteria

### 1. Faithfulness

Evaluation steps:
1. Identify all factual claims made in the summary (named participants, decisions,
   numbers, dates, project names).
2. For each claim, check whether it is directly supported by the source transcript.
3. Note any hallucinated facts, invented attendees, or unsupported inferences.
4. Assign a score based on the rubric below.

[Is every claim in the summary traceable to the source transcript?]
Score 0: The summary is entirely fabricated or completely unrelated to the source.
Score 1: The summary contains multiple fabricated facts or directly contradicts the source.
Score 2: The summary has several unsupported claims or significant inaccuracies.
Score 3: The summary is mostly accurate but contains minor unsupported inferences or imprecise claims.
Score 4: The summary is accurate with only one negligible imprecision that does not mislead.
Score 5: Every claim is directly and precisely supported by the source transcript.

### 2. Completeness

Evaluation steps:
1. Identify the key information in the source: decisions reached, action commitments,
   named participants, numerical results, project names, key topics discussed.
2. Check which of these appear in the summary.
3. Note any important omissions.
4. Assign a score based on the rubric below.

[Does the summary preserve the key information from the source?]
Score 0: The summary contains no useful information from the source.
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
Score 0: Entirely unusable — incoherent fragments or a single word with no meaningful content.
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
Score 0: Completely unreadable — random characters, no recognisable sentences.
Score 1: Incoherent, poorly structured, or unreadable.
Score 2: Understandable but contains multiple grammatical issues or confusing passages.
Score 3: Readable and mostly clear but with some awkward phrasing or structural issues.
Score 4: Well-written and well-organized with only minor stylistic imperfections.
Score 5: Clear, professional prose with excellent structure and logical flow.

## Instructions

For each dimension, write your reasoning following the evaluation steps. Then
provide your final scores. You MUST end your response with a JSON block in
exactly this format:
```json
{"faithfulness": <int>, "completeness": <int>, "conciseness": <int>, "clarity": <int>}
```
```
