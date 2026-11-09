# gliner-test

A test drive of **GLiNER2.5** (`fastino/gliner2.5-multi-v1`, 287M params, mDeBERTa-v3) for named
entity and relation extraction. It feeds the decision on integrating NER into
[localdb](../localdb). The test covers three engines on the same inputs:

| Engine                       | What                                                                                                      | Runner                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------- |
| **CoreML** (Python)          | [FluidInference/gliner2-5-multi-coreml](https://huggingface.co/FluidInference/gliner2-5-multi-coreml) fixed-shape export | `scripts/run_coreml.py`   |
| **gliner25-rs** (Rust)       | [dariofinardi/gliner25-rs](https://github.com/dariofinardi/gliner25-rs) on ONNX Runtime, weights `jugaadsrl/gliner2.5-multi-v1-onnx` | `rust/` (`gliner-rust-eval`) |
| **PyTorch** (reference)      | native `gliner2` 2.0.0, independent decoding and `JointIE`                                                | `scripts/run_torch.py`, `scripts/run_torch_jointie.py` |

Inputs are prose chunks exported read-only from the localdb corpus and `eval/gold.jsonl`, a
hand-labelled set of 84 sentences in English, Dutch, German and French.

## Schema

`schema/ontonotes.json` (v2.1) is the single source of truth for every engine.

- **Entities:** 8 OntoNotes 5.0 types (PERSON, ORG, GPE, LOC, FAC, NORP, DATE, TIME).
- **Relations:** 8 ACE 2005 types mapped to Wikidata properties (`employee_of`, `member_of`,
  `founder_of`, `subsidiary_of`, `family`, `business_associate`, `personal_relation`,
  `located_in`).

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
  - **Relations** are the weak spot (F1, raw → with type filter):

    | Engine | Raw | With type filter |
    | --- | --- | --- |
    | PyTorch / CoreML | 0.32 | 0.41 |
    | gliner25-rs | 0.27 | 0.40 |
    | JointIE | 0.375 | — |

  - **gliner25-rs** has no learned relation scorer, because the ONNX export ships no relation
    weights.
  - **JointIE** returns nothing for symmetric PERSON↔PERSON relations (a gliner2 2.0.0 bug).
- **Speed** (CPU, three passes per window):

  | Engine | p50 per window | Throughput |
  | --- | --- | --- |
  | CoreML fp32 | 115 ms | 219 words/s |
  | CoreML fp16-w8 | 69 ms | 367 words/s, 1.07 GB RSS |
  | gliner25-rs fp32 | ~183 ms | — |
  | PyTorch fp32 | ~141 ms | — |

  The compute-unit setting makes no difference in the VM, where every op runs on the CPU. The ANE
  numbers still have to come from bare metal (see below).
- **On localdb's technical docs**, the model finds real product and organisation names well. It
  over-tags code identifiers and generic technical nouns (`CLI`, `id`, `MarkdownSplitter`) as
  ORG/PERSON.
- **Implications for localdb.**
  - **Entities:** gliner25-rs is fit for purpose, once byte offsets are converted to char offsets.
  - **Relations:**
    - Always apply the schema's `head`/`tail` type filter, and treat relations as low-confidence
      suggestions.
    - Full relation fidelity in pure Rust needs an ONNX export that includes the relation-scorer
      fragment, or a CoreML path on macOS.
    - JointIE's typed graph decoding doesn't depend on the backend. It could be reimplemented in
      Rust on top of the raw scores.

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

All engines write the same JSONL format: one line per window, with char offsets relative to the
window text and OntoNotes tags as labels.

```json
{"window_id": "…", "entities": [{"label", "text", "start", "end", "score"}],
 "relations": [{"name", "head": {"text", "start", "end"}, "tail": {…}, "score"}], "latency_ms": 0.0}
```

## Benchmarking on bare metal (ANE)

```sh
# Per-op device placement. The W8 FP16 feature graph should land mostly on the ANE
# (the model card reports 620/652 ops on an M5 Pro; this VM shows 0/651).
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
schema/ontonotes.json      shared label schema (entities + relation passes)
src/gliner_test/           windowing, compiled-model cache
scripts/                   export, download/compile, runners, bench, compare, quality review
rust/                      gliner25-rs harness (standalone Cargo workspace)
eval/gold.jsonl            hand-labelled sentences (EN/NL/DE/FR)
docs/                      findings, cross-validation, relation ontology
docs/references/           third-party papers and docs (own licenses, see its README)
data/, models/, .cache/    generated, gitignored
```

## License

This project's own code, data and docs are licensed under
[AGPL-3.0-or-later](LICENSE). The material in `docs/references/` keeps its original licenses, listed
in [`docs/references/README.md`](docs/references/README.md). Model weights are downloaded at run
time, not distributed here. GLiNER2.5 and both exports are Apache-2.0.
