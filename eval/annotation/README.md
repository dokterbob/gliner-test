# Gold v3 annotation record

Files in this directory (all produced for `eval/gold.jsonl` v3; guidelines in `docs/annotation-guidelines.md`):

| file | what |
|---|---|
| `gold-v2.1.jsonl` | frozen copy of the previous gold (84 sentences, 49 relations). Never shown to annotators. |
| `passages.jsonl` | 40 new passages (10 each en/nl/de/fr) written by Haiku writers; texts only |
| `items.jsonl`, `batches/` | 124 annotation items (84 old texts + 40 passages) and the 5 shuffled batches |
| `raw/` | raw annotator outputs `bN-A1.txt` / `bN-A2.txt`, their aligned records (`*.aligned.jsonl`) and alignment issue logs (`*.issues.json`, all empty) |
| `agreement.json` | inter-annotator agreement (A1 vs A2) |
| `merged.jsonl` | agreed items plus keyed disagreements |
| `decisions.json` | adjudication decisions (accepted disagreement keys, with a note each) |
| `adjudicated.jsonl` | adjudicated annotation before pooling (101 pairs) |
| `adjudicated-pooled.jsonl`, `pool-extra.json` | adjudicated annotation plus accepted pool pairs (103 pairs); input of `gold_tools.py build` |
| `pool-candidates.jsonl`, `pool-review.jsonl` | pooling candidates (CoreML fp32 relation score >= 0.3, not in gold) and the reviewer verdicts |

## Method

1. Two independent Haiku annotators (A1, A2) per batch of about 25 items. They saw the guidelines and the
   texts only (no predictions, no old labels) and returned surface strings with an occurrence index.
   `scripts/gold_tools.py parse` aligned them to char offsets; nothing failed to align.
2. `gold_tools.py merge` accepted every entity and pair on which A1 and A2 agree exactly (span, type,
   category, direction, qualifiers). The 36 disagreements (18 items) were resolved against the
   guidelines in `decisions.json`. No tie-break annotator was needed.
3. Pooling: CoreML fp32 (schema v2.2, relation threshold 0.3) predictions that the gold lacked were
   reviewed by Haiku against the guidelines. Accepted pairs carry `"source": "pool"`.

## Inter-annotator agreement (A1 vs A2, before adjudication, 124 items)

A1 and A2 are the same model with the same prompt, so this overstates the agreement a second
human or a different model would reach. It measures prompt/guideline clarity, not truth.

Entities (exact span and type): P=0.981 R=0.975 **F1=0.978**
(462 of 474/471); span only F1=0.980.

Pairs (category and endpoint spans; undirected pairs order-free, `place` ordered):

| category | n(A1) | n(A2) | agree | P | R | F1 |
|---|---|---|---|---|---|---|
| all | 100 | 93 | 89 | 0.957 | 0.890 | 0.922 |
| org_org | 12 | 12 | 11 | 0.917 | 0.917 | 0.917 |
| per_org | 39 | 37 | 35 | 0.946 | 0.897 | 0.921 |
| per_per | 24 | 22 | 22 | 1.000 | 0.917 | 0.957 |
| place | 25 | 22 | 21 | 0.955 | 0.840 | 0.894 |

Pair F1 including the qualifier set is identical to the numbers above: wherever the two annotators agreed on
a pair they also agreed on all qualifiers.

## Pooling

CoreML fp32 on all 124 items (schema v2.2, relation threshold 0.3 via `--relation-threshold`, under the model lock).
Texts that overflow the 128-subword bucket (30 passages) were split at sentence boundaries and the
predictions merged with shifted offsets, so pairs that span chunks are not proposed. 83 predicted
pairs scored >= 0.3, were not in the adjudicated gold, fitted a category, and got a Haiku review
(`raw/pool-review-*.txt`, `pool-review.jsonl`). The reviewer accepted 3; the adjudicator kept 2:
`Idris resident_of Antwerp` (gold-p-en-06) and `Pelgrim Techniek located_in Groningen` (gold-p-nl-09,
full-name alias); the third (second `Hana` mention -> Idris, family) is a duplicate of a pair already in
gold and was rejected. The low yield (2 of 83) shows the annotators were near-complete on this ontology.

## Reproduce

```
uv run python scripts/gold_tools.py items                       # items + batches
uv run python scripts/gold_tools.py parse raw/b1-A1.txt --batch batches/batch-1.jsonl --out raw/b1-A1.aligned.jsonl
uv run python scripts/gold_tools.py agreement raw/all-A1.aligned.jsonl raw/all-A2.aligned.jsonl --out agreement.json
uv run python scripts/gold_tools.py merge --a raw/b?-A1.aligned.jsonl --b raw/b?-A2.aligned.jsonl
uv run python scripts/gold_tools.py adjudicate [--extra pool-extra.json] --out adjudicated-pooled.jsonl
uv run python scripts/gold_tools.py build --adjudicated eval/annotation/adjudicated-pooled.jsonl
uv run python scripts/gold_tools.py validate eval/gold.jsonl
```
