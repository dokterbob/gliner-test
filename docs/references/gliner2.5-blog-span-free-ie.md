# GLiNER2.5 launch blog post — reference notes

- **Source:** Mary Newhauser & Urchade Zaratiana (Fastino Labs), *Introducing GLiNER2.5: Efficient
  Span-Free Information Extraction with Schema-Driven Interface*,
  <https://fastino.ai/blog/gliner2-5-span-free-information-extraction>
- **Dates:** the byline says 2026-08-24; the page metadata says 2026-11-06. Retrieved 2026-11-09.
- **Status:** these are notes on the post's technical substance, written in our own words. The post
  itself is © Fastino Labs and is not reproduced here. Read the original for the full text.

## Release

- Three open-weight variants under Apache-2.0: base (0.2B), multi (0.3B; this repo uses
  `fastino/gliner2.5-multi-v1`) and small (74M).

## Architecture

- **Span-free, boundary-based.** The model scores start and end boundaries directly instead of
  enumerating spans, so inference cost is linear in document length.
- **Sparse proposals.** For each query, a proposal stage keeps the most promising start and end
  boundaries and pairs them. Span width is unlimited; GLiNER2 had a cap of about 12 words.
- **Reranking.** A reranking head scores the proposed candidates.
- **Shared pool.** Relation candidates come from the **same candidate pool** as entities, not from a
  separate path. This repo's `models/boundary/pool.py` holds 192 spans with 8 per query, so entity
  and relation-role queries compete for slots.
- The post does not give top-k or pool sizes.

## Training

- The training data is synthetic, made with the Fastino Data Agent. It covers a matrix of domains
  and tasks (NER, relations, classification, JSON), with multi-task schemas and several languages.
- **Training sequences run up to 4,096 words.** The library chunks longer input into overlapping
  word chunks, remaps offsets and deduplicates. The post does not give default chunk or overlap
  sizes.
- Our CoreML export uses 64-word windows (median 26 words), far below the training length. See
  `docs/relation-accuracy.md`, which lists this as an open lever.

## Joint entity–relation extraction

- Entities and relations are decoded together as one globally decoded graph. All candidates are
  scored in one forward pass, then beam search assembles the best-scoring combination that satisfies
  the declared rules.
- Example rules:
  - typed relations;
  - cardinality limits, e.g. `unique_head=True` (one employer per person);
  - `no_self_loops()`;
  - hierarchical relations must form trees;
  - relations may only reference entities that are in the output.

## Other capabilities

- Constrained classification with `implies`/`excludes` rules. It raises an error if no valid
  assignment exists.
- Span attributes, such as entity sentiment, are decoded in the same pass.

## Benchmarks

- Zero-shot macro-F1 is reported over 16 datasets, covering classification and NER only.
- **No relation-extraction benchmark is published.**
- Headline numbers, multi model:

  | | GLiNER2.5 | GLiNER2 |
  |---|---|---|
  | Overall | 56.17 | 56.09 |
  | Extraction | 46.40 | 47.56 |

- Some datasets regress against GLiNER2, e.g. CrossNER politics at 55.26 vs 62.47.

## Confidence

- Confidence values appear in the examples, and thresholds can be set.
- The post does not say whether the scores are calibrated. We measure that in
  `docs/relation-accuracy.md`.
