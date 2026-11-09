# gliner-test

A test drive of **GLiNER2.5** (`fastino/gliner2.5-multi-v1`, 287M params, mDeBERTa-v3) for named
entity and relation extraction. It feeds the decision on integrating NER into
[localdb](../localdb). The test covers three engines on the same inputs:

| Engine                       | What                                                                                                      | Runner                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------- |
| **CoreML** (Python)          | [FluidInference/gliner2-5-multi-coreml](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) fixed-shape export | `scripts/run_coreml.py`   |
| **gliner25-rs** (Rust)       | [dariofinardi/gliner25-rs](https://github.com/dariofinardi/gliner25-rs) on ONNX Runtime, weights `jugaadsrl/gliner2.5-multi-v1-onnx` | `rust/` (`gliner-rust-eval`) |
| **PyTorch** (reference)      | native `gliner2` 2.0.0, independent decoding and `JointIE`                                                | `scripts/run_torch.py`, `scripts/run_torch_jointie.py` |

Inputs are prose chunks exported read-only from the localdb corpus, and `eval/gold.jsonl`
(gold v3): 124 hand-labelled items in English, Dutch, German and French. These are 84 sentences
plus 40 localdb-style passages (emails, notes, chats, technical docs), with 103 entity pairs and a
fixed dev/test split. Claude Code subagents double-annotated them blind and adjudicated them by
the rules in [`docs/annotation-guidelines.md`](docs/annotation-guidelines.md). Inter-annotator
pair F1 is 0.92, but both annotators were the same model, so that figure is optimistic.

## Schema

`schema/ontonotes.json` (v2.2) is the single source of truth for every engine.

- **Entities:** 8 OntoNotes 5.0 types (PERSON, ORG, GPE, LOC, FAC, NORP, DATE, TIME).
- **Relations:** 8 ACE 2005 types mapped to Wikidata properties (`employee_of`, `member_of`,
  `founder_of`, `subsidiary_of`, `family`, `business_associate`, `personal_relation`,
  `located_in`, plus `resident_of`, which is used only as a qualifier).
- **Pair categories (v2.2):** the output is pair-centric. It answers *which entities are related,
  with a confidence*; the relation name is a secondary qualifier. There are four categories:
  - `place`: {PERSON, ORG, FAC, GPE, LOC} → {GPE, LOC, FAC}, directed;
  - `per_per`: PERSON ↔ PERSON;
  - `org_org`: ORG ↔ ORG;
  - `per_org`: PERSON ↔ ORG.

Rationale is in [`docs/relation-ontology.md`](docs/relation-ontology.md). The CoreML bucket allows
≤128 subword tokens, ≤64 words and ≤8 queries per call, with ≤4 relation types. Every window
therefore runs **three passes**: entities, `affiliation` and `social_place`. All engines run the
same passes.

## Findings

Measured on a macOS 26.5 VM (M4 Max virtual, 4 cores, 8 GB RAM, **no Neural Engine**), CPU only.
Details are in [`docs/coreml-findings.md`](docs/coreml-findings.md) and
[`docs/cross-validation.md`](docs/cross-validation.md).

- **Fidelity.** CoreML fp32 matches PyTorch exactly on the full corpus: F1 1.000 on 1835 entities
  and 328 relations. CoreML fp16-w8 agrees with fp32 at 0.966 entity F1 and 0.942 relation F1.
  gliner25-rs agrees with PyTorch at 0.980 entity F1. That needed a byte→char offset conversion,
  because gliner25-rs spans are byte offsets.
- **Accuracy on the hand-labelled set.**
  - **Entities:** micro-F1 0.68 with the v2 labels, the same for every engine. The sharper GPE/LOC
    prompts in v2.1 raise it to 0.80.
  - **Relations, as typed triples** (exact name + head + tail), were weak on the old 84-sentence
    gold: F1 0.32 for PyTorch/CoreML and 0.27 for gliner25-rs.
  - **Relations, as scored pairs.** The question localdb actually asks is which entities are
    related. On the gold v3 **test** split (45 pairs), CoreML fp32 with the post-processing
    pipeline reaches **P 0.62, R 0.80, F1 0.70** (36 tp / 22 fp / 9 fn). AUPRC is 0.69 and
    ECE 0.17. The relation name is right on 29 of 36 matched pairs. Full details are in
    [`docs/relation-accuracy.md`](docs/relation-accuracy.md).

    | Category | Test pairs | F1 | AUPRC |
    | --- | --- | --- | --- |
    | `per_org` | 17 | 0.78 | 0.91 |
    | `org_org` | 5 | 0.89 | 0.80 |
    | `per_per` | 11 | 0.61 | 0.71 |
    | `place` | 12 | 0.52 | 0.36 |

    - **The pipeline**, all in `src/gliner_test/relations.py` and pure arithmetic, so it ports to
      Rust:
      1. a type gate that snaps endpoints to entity spans and picks a category;
      2. collapse to one score per pair, taking the max over the category's own relation names;
      3. NMS over nested-span duplicates;
      4. per-category temperature calibration;
      5. one global threshold of 0.5.
    - **What didn't help:**
      - per-category prompt passes;
      - lower argument thresholds;
      - coarse "related to" prompts;
      - relation descriptions;
      - noisy-or collapse;
      - per-category thresholds;
      - isotonic calibration.
    - **Caveats:**
      - Small samples: one pair is 8–20 points inside a category.
      - No threshold reliably reaches 0.8 precision.
      - `place` is the weak category.
  - **Engines** on the same pipeline (test F1):
    - CoreML fp16-w8: 0.71;
    - CoreML fp32 = PyTorch: 0.70;
    - gliner25-rs: 0.61;
    - JointIE: 0.57.

    gliner25-rs has no learned pair scorer, because the ONNX export ships no relation weights;
    its score is `min(head, tail)`. Post-processing lifts it from 0.47 to 0.61 but can't close
    the ranking gap (AUPRC 0.56 vs 0.69).
  - **JointIE** ranks no better than independent decoding, has much lower precision, and gliner2
    2.1.0 doesn't change that. `symmetric=True` still returns nothing; use `directed=False`
    instead. It isn't worth porting.
  - **Windows:** only 4 of 103 gold pairs cross a 64-word window boundary, and torch on unsplit
    passages is no better than on windows.
- **Speed** (three passes per window):

  | Engine | p50 per window | Throughput |
  | --- | --- | --- |
  | CoreML fp32, VM CPU | 115 ms | 219 words/s |
  | CoreML fp16-w8, VM CPU | 69 ms | 367 words/s, 1.07 GB RSS |
  | CoreML fp32, M4 Max GPU | 27 ms | 900 words/s, 2.2 GB RSS |
  | CoreML fp16-w8, M4 Max ANE | 38 ms | 663 words/s, 0.87 GB RSS |
  | gliner25-rs fp32 | ~183 ms | — |
  | PyTorch fp32 | ~141 ms | — |

  The compute-unit setting makes no difference in the VM, where every op runs on the CPU. On the
  bare-metal M4 Max it does:
  - The GPU is fastest, and there fp16-w8 is no quicker than fp32.
  - The ANE uses the least memory.
  - `all` is slower than pinning either one.

  See [`docs/coreml-findings.md`](docs/coreml-findings.md#bare-metal-benchmark-m4-max-300-window-sample-schema-v2).
- **On localdb's technical docs**, the model finds real product and organisation names well. It
  over-tags code identifiers and generic technical nouns (`CLI`, `id`, `MarkdownSplitter`) as
  ORG/PERSON. Relations inherit this:
  - over the 3450-window corpus, only 20 document-level pairs pass the 0.5 threshold;
  - in a 40-pair hand-checked sample, only 2 of the 13 pairs at ≥ 0.5 are correct.

  On technical text the entity layer, not the pair scorer, is the bottleneck.
- **Implications for localdb.**
  - **Entities:** gliner25-rs is fit for purpose, once byte offsets are converted to char offsets.
  - **Relations:**
    - Port the pair pipeline from `relations.py`, in the order given in
      [`docs/relation-accuracy.md`](docs/relation-accuracy.md#recommendations-for-localdb-rust-port).
    - Show pairs with their calibrated score, and expose the threshold as a user knob.
    - On technical docs, filter code-like entities before trusting pairs.
    - Pair ranking as good as CoreML's in pure Rust needs an ONNX export that includes the
      relation-scorer fragment, or a CoreML path on macOS.

## Setup

Requires macOS on Apple Silicon (for CoreML), [uv](https://docs.astral.sh/uv/), and for the Rust
harness rustup plus Homebrew ONNX Runtime ≥ 1.23.

```sh
uv sync
uv run python scripts/export_corpus.py          # localdb → data/corpus.jsonl (read-only)
uv run python -m gliner_test.windowing          # → data/windows.jsonl, sized to the CoreML bucket
uv run python scripts/download_models.py        # CoreML export → HF cache, pinned revision
uv run python scripts/compile_models.py         # hard-link + compile once → .cache/coreml/<rev>/
```

**Why compile once.** Loading an `.mlpackage` makes CoreML recompile it into the per-user temp
directory on the *boot* volume at every process start, whatever `TMPDIR` says. The FP32 feature
package needs about 1.2 GB there. `compile_models.py` builds `.mlmodelc` once and the runtime
loads those, so later runs write nothing to the boot volume. It hard-links the HF cache blobs
because CoreML rejects the cache's symlinks.

## Running

```sh
# CoreML: full corpus or gold set
uv run python scripts/run_coreml.py --precision fp16-w8 --compute-units cpu \
  --out data/preds/coreml-fp16-w8-cpu.jsonl
uv run python scripts/run_coreml.py --windows eval/gold.jsonl --precision fp32 --compute-units cpu \
  --out data/preds/gold/coreml-fp32-cpu.jsonl

# PyTorch reference, independent and JointIE
uv run python scripts/run_torch.py --windows eval/gold.jsonl --out data/preds/gold/torch-fp32.jsonl
uv run python scripts/run_torch_jointie.py --windows eval/gold.jsonl --out data/preds/gold/torch-jointie.jsonl

# gliner25-rs (Rust / ONNX Runtime, CPU)
brew install onnxruntime
hf download jugaadsrl/gliner2.5-multi-v1-onnx --include "fp32_25/*" --local-dir models/gliner2.5-multi-v1-onnx
(cd rust && cargo build --release)
ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
  rust/target/release/gliner-rust-eval --schema schema/ontonotes.json \
  --models-dir models/gliner2.5-multi-v1-onnx --windows eval/gold.jsonl \
  --out data/preds/gold/rust-ort-fp32.jsonl --device cpu --precision fp32

# Compare: engine agreement vs a reference, plus gold P/R/F1 per label
uv run python scripts/compare.py --windows data/windows.jsonl --preds-dir data/preds \
  --gold eval/gold.jsonl --reference torch-fp32 --report data/report.md
```

Relation pairs: rerun with every pair kept, post-process, then score on the gold test split.
The final pipeline's full command list is in `eval/experiments/final_runs.sh` and
`eval/experiments/make_pairs.py`.

```sh
uv run python scripts/gold_windows.py --help    # split gold into bucket-sized windows, merge back
uv run python scripts/run_coreml.py --windows data/gold-windows.jsonl --precision fp32 \
  --compute-units cpu --relation-threshold 0.0 --out …
uv run python -m gliner_test.relations --preds data/preds/gold/coreml-fp32-cpu.jsonl \
  --schema schema/ontonotes.json --score-names qualifiers --collapse max --nms \
  --calibration eval/calibration/coreml-fp32-cpu.json --out data/pairs/coreml-fp32-cpu.jsonl
uv run python scripts/compare.py --gold eval/gold.jsonl --split test --no-raw \
  --pairs-dir data/pairs-all --pairs-only --threshold 0.5 --engine coreml-fp32-cpu
uv run pytest
```

All engines write the same JSONL format: one line per window, with char offsets relative to the
window text and OntoNotes tags as labels.

```json
{"window_id": "…", "entities": [{"label", "text", "start", "end", "score"}],
 "relations": [{"name", "head": {"text", "start", "end"}, "tail": {…}, "score"}], "latency_ms": 0.0}
```

## Benchmarking on bare metal (ANE)

```sh
# Per-op device placement. The W8 FP16 feature graph should land mostly on the ANE
# (the model card reports 620/652 ops on an M5 Pro; the M4 Max host measured 582/651;
# this VM shows 0/651).
uv run python scripts/dump_compute_plan.py --compute-units ane \
  --package gliner2_multi_extraction_features_w8_embedding_linear_fp16_L128_W64_Q8

# Full grid
for units in cpu gpu ane all; do for prec in fp32 fp16-w8 adaptive; do
  uv run python scripts/bench.py --precision "$prec" --compute-units "$units" \
    --sample-size 300 --out "data/bench/host-${prec}-${units}.json"
done; done
```

## Layout

```
schema/ontonotes.json      shared label schema (entities, relation passes, pair categories)
schema/ontonotes-v2.1.json frozen v2.1 schema (A/B baseline)
src/gliner_test/           windowing, compiled-model cache, relation pair post-processor
tests/                     unit tests for the pair post-processor
scripts/                   export, download/compile, runners, bench, compare, quality review
rust/                      gliner25-rs harness (standalone Cargo workspace)
eval/gold.jsonl            gold v3: 124 hand-labelled items (EN/NL/DE/FR), dev/test split
eval/annotation/           guidelines' raw annotations, agreement, adjudication, frozen gold v2.1
eval/experiments/          relation experiment configs, results, corpus sample review
eval/calibration/          per-engine temperature calibration tables
docs/                      findings, cross-validation, relation ontology
docs/references/           third-party papers and docs (own licenses, see its README)
third_party/               pinned submodules: GLiNER2 v2.0.0, gliner25-rs 580c57b (0.5.6)
data/, models/, .cache/    generated, gitignored
```

## License

This project's own code, data and docs are licensed under
[AGPL-3.0-or-later](LICENSE). The material in `docs/references/` keeps its original licenses, listed
in [`docs/references/README.md`](docs/references/README.md). Model weights are downloaded at run
time, not distributed here. GLiNER2.5 and both exports are Apache-2.0.

The upstream sources are pinned as git submodules, both Apache-2.0. Fetch them with
`git submodule update --init`.
- `third_party/GLiNER2`: tag `v2.0.0`, the version this project runs. `v2.1.0` uses the same
  weights; its JointIE was tested in an isolated environment and changes nothing material.
- `third_party/gliner25-rs`: commit `580c57b`, crate 0.5.6. The upstream repo has no tags.
