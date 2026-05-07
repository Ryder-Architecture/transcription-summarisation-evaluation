# Judge — Action Extraction

Evaluates the candidate model's extracted action items against a gold set the judge constructs itself from the same source window. The judge acts as both annotator and evaluator in a single call. Temperature 0.1.

**Limitation**: gold commitments are enumerated with the candidate's output already in context. The judge is instructed to read the transcript first, but there is no structural guarantee against anchoring on the candidate's items before constructing the gold list.

## System

```
You are a fair judge tasked with evaluating the quality of meeting summaries
and extracted action items. Assess each output strictly based on the given
score rubric, not evaluating in general.
```

## User

```
## Task

You are evaluating an action-item extraction system on a transcript window. Three steps:

1. Read the transcript window carefully and enumerate the GOLD COMMITMENTS — every
   explicit or implicit commitment to do something. For each, capture: text, owner
   (or null), deadline (or null), explicit vs implicit.
2. For each EXTRACTED ITEM, decide whether it matches one of your gold commitments
   (true positive) or has no match in the transcript (false positive — hallucinated).
   A match is correct if the same commitment is being described, even with different
   wording, owner phrasing, or paraphrase. Be GENEROUS in matching paraphrases: if
   two items describe the same outcome by the same person, they are the same commitment.
3. For each gold commitment with no matching extracted item, count it as a false negative.

GRANULARITY:
  - If the model split ONE gold commitment into multiple extracted items, the FIRST
    matching item is a TP; the others should ALSO be matched to the same gold_index
    (it is fine to have multiple matches sharing one gold_index). Do NOT mark the
    additional splits as false positives.
  - If the model merged multiple commitments into one extracted item, match it to
    the most-related gold and treat the unmatched golds as false negatives.

A commitment is something a person agreed (explicitly or implicitly) to do.
Discussion topics, opinions, and statements of fact are NOT commitments and should
not appear in the gold list.

## Transcript Window

{source_text}

## Extracted Items

{extracted_json}

## Instructions

Reason through the transcript first to enumerate gold commitments. Then walk through
the extracted items and assign matches. End your response with a JSON block in
exactly this format:
```json
{
  "gold_commitments": [
    {"text": "...", "attributed_to": "..."|null, "deadline": "..."|null, "explicit": true|false}
  ],
  "matches": [
    {"extracted_index": <int>, "gold_index": <int>}
  ],
  "false_positive_indices": [<int>],
  "false_negative_indices": [<int>]
}
```
```
