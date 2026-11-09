"""Split localdb corpus chunks into windows that fit the CoreML fixed-shape bucket.

The exported CoreML extraction bucket (see the model card and
``gliner2-multi-coreml/config.json``) allows at most:
  - 128 combined subword tokens (schema prompt + text), L128
  - 64 text words, W64
  - 8 extraction queries, Q8

Our shared OntoNotes schema (``schema/ontonotes.json``) has 8 entity types and
4 relation types. A naive single call with ``Schema().entities(...).relations(...)``
would need 8 + 4*2 = 16 queries -- double the Q8 budget (each relation type
consumes 2 queries: one for its head role, one for its tail role). We
therefore run extraction in **two passes** per window (see
``scripts/run_coreml.py``):
  - an entity pass:   ``Schema().entities([8 OntoNotes labels])``   -> 8 queries
  - a relation pass:  ``Schema().relations([4 relation names])``    -> 8 queries

Each pass has its own schema-prompt token cost, measured with the model's own
tokenizer (empty text, see ``measure_schema_cost``):
  - entity-schema prompt alone:   42 subword tokens
  - relation-schema prompt alone: 65 subword tokens

The relation pass is the tighter budget. Windows are sized so that a single
window's text fits *both* passes' 128-token budget, with a safety margin to
absorb small non-additive tokenizer effects observed at window boundaries
(merging across the schema/text join can cost a token or two beyond the
"schema tokens + raw text tokens" sum).

Entity-description strings in the schema (e.g. "Companies, agencies,
institutions, etc.") are intentionally NOT sent to the model: appending them
to every entity label blew the entity-schema prompt alone to 142 tokens,
already over the 128-token *total* budget. Label text only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

WORD_RE = re.compile(r"\S+")

TOTAL_BUDGET = 128
MAX_TEXT_WORDS = 64
SCHEMA_SAFETY_MARGIN = 6  # subword tokens, covers non-additive tokenizer effects
OVERLAP_FRACTION = 0.25  # fraction of a window's words repeated in the next window


def words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Whitespace-delimited words with char offsets into ``text``."""
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def load_runtime_pieces(model_dir: Path):
    """Import the vendored runtime modules and build the two pass schemas."""
    sys.path.insert(0, str(model_dir))
    from gliner2 import Schema
    from gliner2.training.trainer import ExtractorCollator
    from preprocessing import load_processor

    processor = load_processor(str(model_dir / "tokenizer"))
    collator = ExtractorCollator(processor, is_training=False, max_len=None, architecture="boundary")
    return processor, collator, Schema


def build_schemas(schema_cfg: dict, Schema):
    ent_labels = [e["label"] for e in schema_cfg["entities"]]
    rel_names = [r["name"] for r in schema_cfg["relations"]]
    ent_schema = Schema().entities(ent_labels).build()
    rel_schema = Schema().relations(rel_names).build()
    return ent_schema, rel_schema


def measure_schema_cost(collator, ent_schema, rel_schema) -> tuple[int, int]:
    """Real (measured, not guessed) subword-token cost of each schema prompt alone."""
    ent_tokens = collator([("", ent_schema)]).input_ids.shape[1]
    rel_tokens = collator([("", rel_schema)]).input_ids.shape[1]
    return int(ent_tokens), int(rel_tokens)


def fits_bucket(collator, schema, text: str) -> bool:
    batch = collator([(text, schema)])
    return batch.input_ids.shape[1] <= TOTAL_BUDGET and batch.text_word_mask.shape[1] <= MAX_TEXT_WORDS


def window_chunk(
    text: str,
    processor,
    collator,
    ent_schema,
    rel_schema,
    text_token_budget: int,
) -> list[tuple[int, int, str]]:
    """Greedily pack ``text`` into overlapping (char_start, char_end, window_text) windows."""
    words = words_with_offsets(text)
    n = len(words)
    if n == 0:
        return []

    windows: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        last_good = i
        for j in range(i, n):
            candidate = text[words[i][1] : words[j][2]]
            raw_len = len(processor.tokenizer(candidate, add_special_tokens=False)["input_ids"])
            if raw_len > text_token_budget:
                break
            last_good = j
        j = last_good
        window_text = text[words[i][1] : words[j][2]]

        # Exact validation against both real passes; shrink on any tokenizer
        # boundary surprise (join effects, bucket's word-count dimension, etc.).
        while j > i and not (fits_bucket(collator, ent_schema, window_text) and fits_bucket(collator, rel_schema, window_text)):
            j -= 1
            window_text = text[words[i][1] : words[j][2]]

        windows.append((words[i][1], words[j][2], window_text))

        if j >= n - 1:
            break
        window_word_count = j - i + 1
        overlap_words = max(1, int(window_word_count * OVERLAP_FRACTION))
        next_i = max(i + 1, j + 1 - overlap_words)
        i = next_i
    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--model-dir", type=Path, default=None, help="defaults to .cache/coreml/<revision>")
    ap.add_argument("--out", type=Path, default=Path("data/windows.jsonl"))
    args = ap.parse_args()

    if args.model_dir is None:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from download_models import REVISION

        from gliner_test.compiled_models import CACHE_ROOT

        args.model_dir = CACHE_ROOT / REVISION

    processor, collator, Schema = load_runtime_pieces(args.model_dir)
    schema_cfg = json.loads(args.schema.read_text())
    ent_schema, rel_schema = build_schemas(schema_cfg, Schema)
    ent_tokens, rel_tokens = measure_schema_cost(collator, ent_schema, rel_schema)
    tighter_schema_cost = max(ent_tokens, rel_tokens)
    text_token_budget = TOTAL_BUDGET - tighter_schema_cost - SCHEMA_SAFETY_MARGIN
    print(
        f"entity-schema tokens={ent_tokens}, relation-schema tokens={rel_tokens}, "
        f"text token budget (shared, with margin {SCHEMA_SAFETY_MARGIN})={text_token_budget}"
    )

    n_chunks = 0
    n_windows = 0
    word_counts = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus.open() as fin, args.out.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"] or ""
            n_chunks += 1
            for start, end, wtext in window_chunk(text, processor, collator, ent_schema, rel_schema, text_token_budget):
                n_windows += 1
                word_counts.append(len(wtext.split()))
                out_rec = {
                    "window_id": f"{rec['chunk_id']}:{start}-{end}",
                    "chunk_id": rec["chunk_id"],
                    "char_start": start,
                    "char_end": end,
                    "text": wtext,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    word_counts.sort()
    median_words = word_counts[len(word_counts) // 2] if word_counts else 0
    print(f"chunks={n_chunks} windows={n_windows} median_words/window={median_words} -> {args.out}")


if __name__ == "__main__":
    main()
