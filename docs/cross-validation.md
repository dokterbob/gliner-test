# GLiNER2.5 cross-validation: gliner25-rs (Rust/ONNX Runtime) vs. torch vs. gold

This is the independent cross-validation leg of the GLiNER2.5 test drive. It
checks the CoreML export's results against two more engines and a small hand
gold set: **gliner25-rs** (native Rust on ONNX Runtime, crates.io
`gliner25-rs` 0.5.6, weights `jugaadsrl/gliner2.5-multi-v1-onnx`) and the
native PyTorch reference (`fastino/gliner2.5-multi-v1` via `gliner2` 2.0.0,
the version the CoreML runtime pins). Labels, relation names and the shared
threshold all come from `schema/ontonotes.json`; nothing is hard-coded.

## What was built

- `rust/` — a standalone Cargo workspace (own `Cargo.toml`/`Cargo.lock`,
  binary `gliner-rust-eval`) that reads `schema/ontonotes.json` and a windows
  JSONL and writes predictions in the shared format, with both entities and
  relations. Source: `rust/src/main.rs`.
- `scripts/run_torch.py` — the native PyTorch `gliner2.5-multi-v1` reference
  over the same windows, CPU (and an `mps` attempt with CPU fallback).
- `eval/gold.jsonl` — 54 hand-written sentences across English (30), Dutch
  (8), German (8) and French (8), covering all 8 OntoNotes labels and all 4
  relations, plus the requested tricky cases: dates vs. version strings
  (`version 2.5.3`, `v1.23.0`), ISO datetimes (`2026-09-06T15:34:07Z`),
  relative times (`Tuesday at 3pm`), GPE-as-actor sentences (`Amsterdam
  signed...`), and technical-doc-style sentences resembling the corpus.
- `scripts/compare.py` — pairwise agreement vs. a reference engine (exact +
  overlap span P/R/F1 per label, score-diff stats on exact matches), gold-set
  P/R/F1 per engine, and a ranked disagreement list.

## ORT setup and CoreML EP status

- ONNX Runtime: Homebrew `onnxruntime` 1.31.1 (`brew install onnxruntime`),
  `ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib`.
  gliner25-rs's `ort` dependency uses the `load-dynamic` feature (no
  `download-binaries`), so a system ONNX Runtime is the natural fit — no
  extra linking/build-script work, and Homebrew's 1.31.1 clears the crate's
  documented 1.23+ floor (older builds "run correctly and then segfault at
  process exit").
- **Pin `ort` to exactly `=2.0.0-rc.13` in `rust/Cargo.toml`.** gliner25-rs
  0.5.6's own manifest requires `ort = "2.0.0-rc.13"`, but Cargo's resolver
  will happily pick rc.15 to satisfy that (pre-release ranges widen upward),
  and rc.14/rc.15 made breaking API changes (`Session::builder()` gained a
  required `&Environment` argument, `EnvironmentBuilder::commit()` was
  renamed, `AllocationDevice` grew a lifetime param) that gliner25-rs's own
  source does not compile against. Without the exact pin the build fails
  inside the dependency, not in our code. This is worth flagging upstream.
- **CoreML EP registers and runs, but is not a clear win here, and is
  operationally fragile on a disk-constrained Mac.** `GLINER2_DEVICE=coreml`
  resolves to `ep::CoreML::default().build()` (confirmed present:
  `_OrtSessionOptionsAppendExecutionProvider_CoreML` is exported from
  Homebrew's dylib). A run against the fp32 export completed without
  crashing (707 ms single-window latency, vs. ~100 ms on CPU — no benefit,
  consistent with this VM having **no Neural Engine**, CPU+GPU only). A
  second attempt failed with a CoreML *compiler* I/O error because the
  shared VM's root volume filled up — CoreML EP compiles each session's graph
  to an `.mlmodelc` under `/var/folders/.../T` on the **system** volume (not
  the data volume `models/` lives on) on every load, with no persistent
  compile cache. On this VM that competed directly with the CoreML agent's
  own `coremltools` compilation for the same scratch space (see below) and
  twice drove `/` to 100% full. For localdb this is the real integration
  risk, more than raw speed: a CoreML EP path needs headroom on the system
  volume and should not be assumed free once the model is already on disk.
  Also, `provider_has_device_memory()` in `runtime.rs` does not special-case
  CoreML, so the engine never attempts `IoBinding` for it — every CoreML run
  goes through the `Standard` host-memory transport, which is the right
  call for Apple's unified memory but means CoreML gets none of the
  IOBinding speedup CUDA/ROCm get in this crate.
- **Shared disk exhaustion was a recurring blocker**, not specific to our
  work: multiple plain `Bash` calls (not just CoreML-EP calls) failed with
  `ENOSPC` on the root volume while the CoreML agent's own model
  compilation ran concurrently. Root cause looks to be `coremltools`/Core ML
  compiler scratch space on `/`, not anything either agent wrote under
  `models/`, `data/`, or `~/.cargo` (all on the separate `/Volumes/User Home`
  volume). Flagging for awareness; not something either agent should try to
  clean up unilaterally on a shared VM.

## Engine agreement vs. torch (corpus sample, `data/windows.jsonl`)

`data/windows.jsonl` (3450 windows) only appeared mid-session. Running the
full file through two from-scratch model loads plus inference would not fit
the "keep locked runs to a few minutes" rule, so both engines ran on a fixed,
deterministic systematic sample (every 17th window, 203 windows, same
window IDs for both engines — `awk 'NR%17==2'`). Predictions:
`data/preds/rust-ort-fp32.jsonl`, `data/preds/torch-fp32.jsonl`.

**Exact span match** (203 shared windows), rust-ort-fp32 vs. torch-fp32:

| label | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| DATE | 5 | 2 | 2 | 0.714 | 0.714 | 0.714 |
| FAC | 9 | 8 | 7 | 0.529 | 0.562 | 0.545 |
| GPE | 0 | 1 | 0 | 0.000 | 0.000 | 0.000 |
| LOC | 19 | 5 | 3 | 0.792 | 0.864 | 0.826 |
| ORG | 119 | 33 | 25 | 0.783 | 0.826 | 0.804 |
| PERSON | 27 | 5 | 3 | 0.844 | 0.900 | 0.871 |
| TIME | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| **micro-all** | 185 | 54 | 40 | 0.774 | 0.822 | 0.797 |

Score diff on exact matches (rust − torch): mean **+0.0085**, mean|diff|
0.0177, stdev 0.0336, n=185 — i.e. essentially the same probabilities, with a
small, consistent positive bias for the Rust/ONNX path. That bias is well
within fp32 ONNX-vs-torch numerical noise and is *not* where the disagreement
comes from.

**Overlap match**, same pair: micro-F1 **0.944** (P=0.916, R=0.973) — far
higher than exact-match's 0.797. So most of the "disagreement" is boundary
disagreement (off-by-a-token span edges on things like `"gate"` vs.
`"Gate-skips"`, `"source list"` vs. no match), not label or presence
disagreement. **Relations** (exact name+head_text+tail_text):
tp=6 fp=38 fn=1, P=0.136 R=0.857 F1=0.235 — recall is fine but precision is
poor, because `pair_relations` in gliner25-rs does a full cross-product of
every extracted head-role span with every tail-role span for a relation name
(see "Integration assessment" below): on technical-doc text with several
ORGs/PERSONs in a window this produces many spurious pairs that torch's
`extract_relations` (same cross-product, scored independently per role) also
produces, but not identically, since the head/tail role queries pick
slightly different spans between the two engines.

## Gold-set P/R/F1 (`eval/gold.jsonl`, 54 sentences)

Both of our engines ran on the full gold set
(`data/preds/gold/{rust-ort-fp32,torch-fp32}.jsonl`). The CoreML-on-gold
attempt (`scripts/run_coreml.py --windows eval/gold.jsonl`, which does accept
an arbitrary windows file) failed with a Core ML compiler `ENOSPC` from the
shared-disk issue above, and was not retried — the resulting empty output
file was deleted rather than counted as "0 predictions".

| engine | match | micro P | micro R | micro F1 |
|---|---|---|---|---|
| rust-ort-fp32 | exact | 0.407 | 0.498 | 0.448 |
| rust-ort-fp32 | overlap | 0.617 | 0.754 | 0.678 |
| torch-fp32 | exact | 0.569 | 0.690 | 0.624 |
| torch-fp32 | overlap | 0.630 | 0.764 | 0.690 |

The exact-match gap between the two engines (0.448 vs. 0.624) almost
disappears under overlap match (0.678 vs. 0.690) — again a boundary story,
not a semantic one, and this time it is mostly the Rust path drawing
*slightly* different candidate-span edges than torch on multi-word spans
("Port Authority of Rotterdam"-style names), which the 203-window exact-match
numbers above already hinted at.

**Relations on gold:** rust tp=13 fp=100 fn=3 (P=0.115 R=0.812 F1=0.202);
torch tp=12 fp=55 fn=4 (P=0.179 R=0.750 F1=0.289). Same story, worse on a
small set: recall is good (the model usually finds the right head and tail
spans), precision is bad (the naive cross-product invents pairs the schema's
own `head`/`tail` OntoNotes-type constraints in `schema/ontonotes.json` would
rule out, e.g. pairing `founded_by`'s head role with every ORG in the
sentence rather than just the one adjacent to the matched tail). **Neither
engine filters by those type constraints today** — see below.

**The clearest systematic miss, present in *both* engines equally:** low
**LOC** precision (rust fp=44, torch fp=41 exact-match) paired with
depressed **GPE** recall (fn=23 / fn=17). The `location` prompt and the
`country, city or state` prompt compete for the same spans in
`schema/ontonotes.json`'s wording, and the model frequently tags a GPE gold
span (e.g. "Veldhoven", "Marseille") as `location` instead of
`country, city or state`, or as both. This is a schema-wording issue, not a
cross-engine one — both engines read the same prompts and make the same
call. Worth revisiting the `label` text for GPE/LOC in `schema/ontonotes.json`
if GPE/LOC precision matters for localdb.

## Top disagreements (rust-ort vs. torch, corpus sample)

- `"...open that store for source list or index at all..."` — only torch:
  `ORG:"source list"@0.60`, `ORG:"index"@0.59`. Both low-confidence, both
  arguably wrong (common nouns, not organizations) — a false-positive torch
  makes that rust-ort happens not to.
- `"...Gate-skips never touch..."` — rust: `FAC:"Gate-skips"@0.58`; torch:
  `FAC:"gate"@0.56`. Same underlying span, different boundary — tokenization/
  candidate-pool differences pick a different half-open `[start,end)` for a
  hyphenated compound.
- `"...a 304 whose metadata write failed..."` — both engines tag `"304"` as
  `ORG` at nearly identical confidence (0.53 vs. 0.52) — an agreed-upon
  shared error mode (an HTTP status code is not an organization), useful to
  know is not a Rust-vs-Python artifact but a model behavior to filter in
  localdb's own post-processing if it matters.

Full disagreement list, timing sidecars and raw predictions are in
`data/preds/` (gitignored; not committed — regenerate with the commands
below).

## Latency / resource comparison (CPU, fp32, this VM: M4 Max VM, 4 cores, 8 GB RAM, no ANE)

| engine | windows | load ms | p50 ms/window | words/s | peak RSS |
|---|---|---|---|---|---|
| rust-ort (gold, 54) | 54 | 2795 | 98.9 | 151 | 2076 MB |
| rust-ort (sample, 203) | 203 | 1926 | 111.1 | 220 | 2742 MB |
| torch (gold, 54) | 54 | 7795 | 70.8 | 215 | 2967 MB |
| torch (sample, 203) | 203 | 7764 | 78.9 | 312 | 3264 MB |

Torch loads ~3–4x slower than the Rust/ONNX path (safetensors + Python
import overhead vs. a handful of `mmap`'d ONNX sessions) but has lower
per-window latency and higher throughput once warm, and somewhat higher peak
RSS. Rust-ort's `Standard` execution mode, 4 intra-op threads, and the same
fp32 weights as torch, so this is a CPU-kernel-efficiency comparison (ORT vs.
PyTorch/eager-DeBERTa — note the CPU run also warns
`attn_implementation='sdpa'` is rejected and falls back to `eager` for
DebertaV2, which likely costs torch some of that gap back on longer inputs
even though it still wins here). Both comfortably fit the 8 GB budget loaded
one at a time, per the shared-machine rule.

## Integration assessment: gliner25-rs for localdb

**API fit:** good. `BoundaryEngine::extract`/`extract_with` map directly onto
`schema/ontonotes.json`'s shape (`SchemaTask::Entities`/`SchemaTask::Relations`
with `head`/`tail` roles mirror gliner2's own `.entities()`/`.relations()`
schema builder almost one-to-one), spans are half-open and byte-indexed into
the original text, and `BoundaryManifest` self-describes buckets/pool
size/overlap policy so a caller never has to hard-code them. `extract_long`
handles documents past the largest bucket with word-window chunking and
offset remapping — directly useful for localdb's longer chunks, not just the
≤64-word CoreML-bucket windows used here.

**Dependency weight:** light for a Rust project already using `ort` or
willing to. No Python at build or run time (the README's main selling
point). Needs a system ONNX Runtime ≥1.23 (Homebrew is fine) because the
crate hard-pins `load-dynamic`; depends on `tokenizers`, `ndarray`, `hf-hub`
(optional, for `or_download`), `regex` — all common, no surprises. The one
real friction point is the **`ort` rc.13 pin** above: it will silently break
on a plain `cargo update` until gliner25-rs republishes against a newer
`ort`, so a localdb integration should pin it explicitly too and watch for
upstream movement.

**Relations support:** present (`families` feature, on by default) but
**thin**: `pair_relations` is a bare cross-product of every head-role span
with every tail-role span sharing a relation name, with no joint score and
no use of the schema's head/tail OntoNotes-type constraints. This is a
faithful port of gliner2's own `extract_relations` (also tuples, also no
joint score, confirmed via the Python docs), so it is not a Rust-specific
gap — but it means **any production use of relations needs a
post-filter/re-ranker on top of both engines**, e.g. using
`schema/ontonotes.json`'s own `relations[].head`/`.tail` OntoNotes-category
lists to drop pairs whose endpoints aren't the expected entity types, and
probably a real pair score (we used `min(head.score, tail.score)` for the
shared format here, documented in both `rust/src/main.rs` and
`scripts/run_torch.py`, purely as an evaluation convenience).

**Gaps / open items for localdb:**
1. Pin `ort` to the exact rc gliner25-rs was built against (see above) —
   otherwise `cargo build` breaks on a routine dependency update.
2. CoreML EP is usable but not obviously worth it without an ANE, and its
   per-load compile step eats system-volume scratch space that a
   disk-constrained Mac (or this VM) may not have spare; budget for that or
   pre-warm/cache compiled graphs before relying on it.
3. Relation precision needs a type-constraint post-filter; ship one before
   trusting `works_for`/`located_in`/`founded_by`/`occurred_on` output
   directly.
4. GPE vs. LOC boundary is schema-wording-sensitive, not engine-sensitive —
   worth tuning `schema/ontonotes.json`'s label text if localdb needs that
   distinction to be reliable.
5. gliner25-rs's exact-match spans disagree with torch more than expected
   (0.797 micro-F1 corpus, 0.448 on gold) while overlap-match agreement is
   excellent (0.944 / 0.678) — fine for "does localdb's NER find the right
   stretch of text", worth a second look before relying on exact character
   offsets for anything precision-sensitive (e.g. auto-linking spans back
   into source documents).

## Reproducing

```bash
# Rust harness (needs Homebrew onnxruntime; wrap heavy runs in lockf per the shared-machine rule)
cd rust && cargo build --release
ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
  ./target/release/gliner-rust-eval \
  --schema ../schema/ontonotes.json --models-dir ../models/gliner2.5-multi-v1-onnx \
  --windows ../eval/gold.jsonl --out ../data/preds/gold/rust-ort-fp32.jsonl \
  --device cpu --precision fp32

# PyTorch reference
uv run python scripts/run_torch.py --windows eval/gold.jsonl \
  --out data/preds/gold/torch-fp32.jsonl --device cpu

# Comparison report
uv run python scripts/compare.py --windows data/windows.jsonl \
  --preds-dir data/preds --gold eval/gold.jsonl --reference torch-fp32 \
  --report /tmp/cross-validation-report.md
```

Model weights for gliner25-rs: `hf download jugaadsrl/gliner2.5-multi-v1-onnx
--include "fp32_25/*" --local-dir models/gliner2.5-multi-v1-onnx` (gitignored,
not committed).
