# GLiNER2.5 cross-validation: gliner25-rs (Rust/ONNX Runtime) vs. torch vs. gold

This is the independent cross-validation leg of the GLiNER2.5 test drive. It
checks the CoreML export's results against two more engines and a small hand
gold set: **gliner25-rs** (native Rust on ONNX Runtime, crates.io
`gliner25-rs` 0.5.6, weights `jugaadsrl/gliner2.5-multi-v1-onnx`) and the
native PyTorch reference (`fastino/gliner2.5-multi-v1` via `gliner2` 2.0.0,
the version the CoreML runtime pins). Labels, relations and the shared
threshold all come from `schema/ontonotes.json` (now **schema v2**: 8
OntoNotes entities + 8 ACE-2005-derived relation types in two
`relation_passes`, see `docs/relation-ontology.md`); nothing is hard-coded.

**Revision note.** An earlier version of this document reported entity
exact-match agreement well below overlap-match agreement and called it a
"boundary story" without a mechanism, and claimed gliner2's independent
`extract_relations` was a naive cross-product like gliner25-rs's. Both
claims were wrong, or at best incomplete. This revision replaces them with
root-caused, re-measured numbers; the wrong conclusions are not repeated
below.

## What was built

- `rust/` — a standalone Cargo workspace (own `Cargo.toml`/`Cargo.lock`,
  binary `gliner-rust-eval`) that reads `schema/ontonotes.json` and a windows
  JSONL and writes predictions in the shared format, with both entities and
  relations. Source: `rust/src/main.rs`.
- `scripts/run_torch.py` — the native PyTorch `gliner2.5-multi-v1` reference
  (independent `extract_entities`/`extract_relations`) over the same
  windows, CPU (and an `mps` attempt with CPU fallback).
- `scripts/run_torch_jointie.py` — the `gliner2.joint_ie.JointIE` reference,
  for the relation-decoding comparison below.
- `eval/gold.jsonl` — 84 hand-written sentences across English (42), Dutch
  (14), German (14) and French (14), covering all 8 OntoNotes labels and all
  8 schema-v2 relations (≥5 positives each), plus the requested tricky
  cases: dates vs. version strings, ISO datetimes, relative times,
  GPE-as-actor sentences, an employee_of/member_of contrast, two founder
  pairs, and four hard negatives (co-occurring people with no relation).
- `scripts/compare.py` — pairwise agreement vs. a reference engine (exact +
  overlap span P/R/F1 per label, score-diff stats on exact matches), gold-set
  P/R/F1 per engine (symmetric relations scored order-insensitively, plus an
  optional type-constraint post-filter variant), and a ranked disagreement
  list.

## 1. Root cause of the exact-span disagreement: a byte/char offset bug (ours, fixed)

**Finding: gliner25-rs's `Mention::char_start`/`char_end` are byte offsets,
not char offsets, despite the name.** `boundary.rs:108` ("Byte range
`[start, end)` in the original text.") and `processor.rs:148` (identical
wording for `word_to_char_maps`) say so directly — Rust string indexing is
byte-based and the crate never converts. The shared prediction format used
across all three engines in this repo is *char* (Unicode scalar value)
offsets, matching Python string indexing (torch's `include_spans=True`
spans are already char offsets natively, so `scripts/run_torch.py` needed
no change here). On ASCII-only text the two coincide; on any window with a
multi-byte UTF-8 character — accented Latin letters in the NL/DE/FR gold
sentences, but also em dashes/arrows/section signs that are common in the
"English" corpus text — every span *at or after* that character silently
drifted by the extra byte count.

Proof, before any fix, from the old gold set (English/Dutch/German sentences, same
model, same window):

| window | entity | rust-ort span | torch span | drift | cause |
|---|---|---|---|---|---|
| `Angela Merkel traf sich am Dienstag um 15 Uhr mit Vertretern der Europäischen Union in Berlin.` | ORG "Europäischen Union" | (65, 84) | (65, 83) | +1 at end | "ä" (2 bytes, 1 char) *inside* the span |
| `Die Siemens AG wurde 1847 von Werner von Siemens in München gegründet.` | GPE "München" | (52, 60) | (52, 59) | +1 at end | "ü" inside the span |
| `Katholische und evangelische Gemeinden in Köln feiern das Fest nächsten Sonntag um 10:00 Uhr.` | DATE "Sonntag" | (74, 81) | (72, 79) | +2 at start | "ö" in "Köln" + "ä" in "nächsten", both *before* the span |

The drift is always exactly the number of extra UTF-8 bytes contributed by
multi-byte characters up to that position — cumulative, and present at both
span start and end. Isolating the gold set into ASCII-only vs. non-ASCII
subsets *before any fix* confirmed it was the entire story for entities:
rust-ort vs. torch exact-match on the 36 ASCII-only old-gold sentences was
**P=R=F1=1.000** (157/157 spans agreed exactly); all of the disagreement was
concentrated in the 18 non-ASCII sentences, in amounts matching the drift
arithmetic above. The 203-window corpus sample showed the same pattern: its
125 ASCII-only windows already agreed at F1=0.967 pre-fix; the other 78
(38%, mostly from em dashes/arrows/section signs in otherwise-English
technical prose) carried the rest of the 0.797 aggregate gap.

**Fix** (`rust/src/main.rs`, `byte_to_char_map`): build a byte-offset → char-offset
table per window (`text.char_indices()`) and convert every entity and
relation endpoint through it before writing the shared format. Both
candidate causes the coordinator asked to rule out were checked and are
**not** the issue, with receipts:

- **Word-splitter regex** (hyphen handling): gliner25-rs's `processor.rs`
  `WhitespaceTokenSplitter` pattern and gliner2's
  `gliner2/processing/word_splitter.py:22-29` `WhitespaceTokenSplitter._PATTERN`
  are character-for-character identical (`\w+(?:[-_]\w+)*` among the same
  alternatives) — ruled out.
- **Threshold comparison**: both keep `score >= threshold`
  (`gliner2/models/boundary/model.py:974`: `probs >= threshold_value`;
  gliner25-rs `boundary.rs`: `if score < params.threshold { continue; }`,
  i.e. keeps `>=`) — ruled out.
- **Overlap/flat-NER policy**: both read `"flat"` from the checkpoint rather
  than hard-coding a default (gliner2
  `gliner2/inference/runtime.py:224-237` `_resolved_overlap_policy`, reading
  `boundary_settings.overlap_policy`; gliner25-rs `boundary.rs` reading
  `manifest.overlap_policy` from `boundary_manifest.json`, which is
  `"flat"` for this checkpoint) — ruled out.
- **Prompt composition / task order**: see §2 below — this *does* matter,
  but for a different reason (encoder conditioning), not for entity
  exact-match agreement, since both harnesses already ran entities as their
  own isolated call before and after the byte/char fix.

**After the fix**, on the full, re-annotated 84-sentence gold set, with both
engines in the new default multi-pass mode: rust-ort vs. torch entity
exact-match is **P=0.960 R=1.000 F1=0.980** (264 matched spans, 11
disagreements, 0 one-sided misses). The residual 11 are not offset
artifacts — inspecting them shows genuine low-margin model disagreements,
mostly the GPE/LOC query-competition effect in §4 (Utrecht/Amsterdam/
Köln/Berlin/Marseille/Geneva: rust's independent `location` query clears
threshold at 0.78–0.89 where torch's does not) plus a couple of ORG/TIME/DATE
calls in the 0.6–0.8 score range consistent with ordinary fp32 ONNX-vs-PyTorch
numerical noise (DebertaV2 attention runs as `eager` in torch, since `sdpa`
is rejected — see the latency section — vs. ONNX Runtime's fused kernels).
**Target met: agreement is at 0.98, and the residual is explained, not
mysterious.**

### Full-corpus confirmation, and an independent check of CoreML's own numbers

With the machine free again, `scripts/run_torch.py` ran the full
`data/windows.jsonl` (3450 windows, multi-pass, ~9 minutes under the lock)
so CoreML's own corpus predictions
(`data/preds/coreml-{fp32,fp16-w8}-cpu.jsonl`, already complete at 3450
lines each) could be checked against a *real* torch reference at full
scale, not the 203-window sample:

| engine vs. torch-fp32, all 3450 windows | entity exact-match F1 | relation F1 |
|---|---|---|
| coreml-fp32-cpu | **1.000** (tp=1835 fp=0 fn=0) | **1.000** (tp=328 fp=0 fn=0) |
| coreml-fp16-w8-cpu | 0.966 (tp=1751 fp=41 fn=84) | 0.942 (tp=307 fp=17 fn=21) |

CoreML fp32 is bit-for-bit identical to torch-fp32 on every one of 1835
entities and 328 relations across the whole corpus — the cleanest possible
confirmation that the CoreML export and the PyTorch checkpoint agree
exactly when run at the same precision. The fp16-w8 numbers
(**entity F1 0.966, relation F1 0.942**) are an independent reproduction,
via a different reference engine (torch, not CoreML's own fp32 run) and a
different tool (`scripts/compare.py`, not whatever the CoreML agent used),
of the exact figures already reported in `docs/coreml-findings.md` — a
genuine cross-check, not a restatement, and it lands on the same numbers.

rust-ort's corpus-level comparison (§ above) is still the 203-window sample
from before this round — a full 3450-window rust-ort run was out of scope
for this pass (the task asked to prioritize the root cause, gold
re-annotation and the relation/JointIE comparison over full-corpus runs);
re-running it is a cheap follow-up with the harness as it stands.

## 2. Prompt composition: one-pass vs. multi-pass

Both harnesses now default to **multi-pass**, mirroring
`scripts/run_coreml.py` exactly: one `extract()` call for entities (8
queries), then one more per `schema.relation_passes` group ("affiliation"
then "social_place", 4 relations × 2 roles = 8 queries each) — three calls
per window total. `--mode one-pass` keeps the old combined-schema behavior
(everything in one call) for an explicit before/after cost reading, since
GLiNER2 conditions every task's attention on the whole prompt and a
combined call is a genuinely different encoder input, not just a latency
optimization.

Cost of the split, torch, full gold set (84 sentences):

| mode | p50 latency/window | p90 |
|---|---|---|
| one-pass | 72.3 ms | n/a |
| multi-pass | 141.3 ms | 151.2 ms |

and rust-ort:

| mode | p50 latency/window | p90 |
|---|---|---|
| one-pass | 118.4 ms | 130.5 ms |
| multi-pass | 183.3 ms | 202.5 ms |

Multi-pass costs ~1.95x (torch) / ~1.55x (rust-ort) the one-pass latency —
real, but not disqualifying for localdb's batch-indexing use case. Entity
numbers are identical between the two modes in both harnesses (entities
already ran as their own call either way); the split changes relation
*composition*, covered next.

## 3. Relations: independent decoding is not a naive cross-product — verdict corrected

**The earlier claim that gliner2's independent `extract_relations` is a
naive cross-product, like gliner25-rs's `pair_relations`, was wrong.**
Checked directly in the installed `gliner2` 2.0.0 source:

- `gliner2/models/boundary/relations.py:1-8` (module docstring): "Pair
  generation is *typed and capped*: for each relation type we keep only the
  top-`Rh` head-typed and top-`Rt` tail-typed mentions and score their capped
  cross product." `TypedRelationPairGenerator.generate_batched`
  (`relations.py:93-230`) does exactly that, with defaults
  `heads_per_relation=32, tails_per_relation=32, pair_cap=128`
  (`RelationProposalSettings`, `relations.py:26-29`); the simple
  `head_prob * tail_prob` product at `relations.py:214` is used only to rank
  candidates down to `pair_cap`, not as the final decision.
- The final decision comes from **`SparseRelationScorer.forward`**
  (`relations.py:320-409`): a learned MLP over the head/tail boundary
  states and the relation's own encoder representation
  (`score = self.mlp(feats)` at `relations.py:376-377`), optionally summed
  with a biaffine term and a linear term over pooled span content
  (`relations.py:395-408`). This is a genuine, trained, joint pair score —
  not an unscored enumeration.

So independent `extract_relations` already benefits from a real relation
classifier; it just isn't *typed* (no head/tail OntoNotes constraint) and
has no cross-relation consistency. That difference shows up directly in the
false-positive counts on the full 84-sentence gold set (multi-pass, 8
relation types, symmetric relations scored order-insensitively):

| engine | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| rust-ort (independent, raw) | 41 | 217 | 8 | 0.159 | 0.837 | 0.267 |
| torch (independent, raw) | 38 | 149 | 11 | 0.203 | 0.776 | 0.322 |
| rust-ort (independent + type post-filter) | 38 | 103 | 11 | 0.270 | 0.776 | 0.400 |
| torch (independent + type post-filter) | 34 | 83 | 15 | 0.291 | 0.694 | 0.410 |
| torch-jointie (raw) | 30 | 81 | 19 | 0.270 | 0.612 | 0.375 |

(The type post-filter variant — optional, reported separately as asked —
drops a predicted pair unless its head/tail span overlaps that *same
engine's own* entity prediction of a type the schema's `relations[].head`/
`.tail` allow; `scripts/compare.py:type_filter_relations`.)

**gliner25-rs has no equivalent of `SparseRelationScorer` at all** —
confirmed by reading every line of `src/*.rs`: `pair_relations`
(`boundary.rs:767-789`) is a bare, untyped, uncapped cross-product of every
head-role mention × every tail-role mention sharing a relation name, with
no learned score. This is not a missing-feature oversight in the crate so
much as a missing-*weights* problem: `jugaadsrl/gliner2.5-multi-v1-onnx`
(31 files total, checked via the HF API) exports `encoder`, `routed_gather`,
`classifier` and `boundary_head_L{64,128,256,512}` per precision, and
**nothing named `relation` or `record`** — unlike the CoreML export, which
ships dedicated `gliner2_multi_relation_*_W64_R4_P256.mlpackage` files.
gliner25-rs structurally cannot match gliner2/CoreML relation fidelity
without a re-export that includes the relation-scorer fragment, independent
of anything the crate's Rust code does.

### All five engines against gold, side by side

CoreML's earlier gold-set attempt (noted in previous revisions of this
document) failed with a disk-full `ENOSPC` from the Core ML compiler; the
CoreML agent has since added a persistent compiled-model cache
(`.cache/coreml/<revision>/*.mlmodelc`) that fixes that, so both precisions
now have real gold numbers:

```
uv run python scripts/run_coreml.py --windows eval/gold.jsonl --precision fp32 --compute-units cpu --out data/preds/gold/coreml-fp32-cpu.jsonl
uv run python scripts/run_coreml.py --windows eval/gold.jsonl --precision fp16-w8 --compute-units cpu --out data/preds/gold/coreml-fp16-w8-cpu.jsonl
```

Entity exact-match micro-F1 against `eval/gold.jsonl`, all five engines:

| engine | micro P | micro R | micro F1 | overlap F1 |
|---|---|---|---|---|
| coreml-fp32-cpu | 0.689 | 0.674 | 0.682 | 0.749 |
| coreml-fp16-w8-cpu | 0.698 | 0.667 | 0.682 | 0.750 |
| torch-fp32 (independent) | 0.689 | 0.674 | 0.682 | 0.749 |
| rust-ort-fp32 | 0.676 | 0.689 | 0.683 | 0.749 |
| torch-jointie | 0.609 | 0.670 | 0.638 | 0.702 |

CoreML (either precision) and torch-fp32 land on **the same accuracy
against gold**, label for label — fp32 CoreML's exact-match row is
identical to torch's, and fp16-w8 is a rounding error away. rust-ort is
likewise within noise of the other three on entities (the exact-match gap
seen earlier in this document, before the byte/char fix, is gone — see §1).
torch-jointie is the outlier, 4-6 points lower on both metrics — entirely
attributable to the 15 symmetric-relation gold sentences producing 0
*entities* under `optimizer="greedy"`'s harder failure mode in a few cases
(§3), which drags down its entity recall alongside the relation recall. None
of the four non-JointIE engines meaningfully disagree with gold or each
other on entities once the offset bug is fixed; the real spread in this
document is in relations, not entities.

Relations, independent decoding, all engines (exact name+head+tail match,
symmetric order-insensitive):

| engine | tp | fp | fn | P | R | F1 |
|---|---|---|---|---|---|---|
| coreml-fp32-cpu | 38 | 149 | 11 | 0.203 | 0.776 | 0.322 |
| coreml-fp16-w8-cpu | 38 | 147 | 11 | 0.205 | 0.776 | 0.325 |
| torch-fp32 | 38 | 149 | 11 | 0.203 | 0.776 | 0.322 |
| rust-ort-fp32 | 41 | 217 | 8 | 0.159 | 0.837 | 0.267 |
| torch-jointie | 30 | 81 | 19 | 0.270 | 0.612 | 0.375 |

rust-ort is the clear outlier on relations (more FPs, lower precision) —
consistent with §3's finding that it is the one engine with no learned
relation scorer behind its pairing at all.

### JointIE

`gliner2.joint_ie.JointIE` (same checkpoint, `docs/references/gliner2-tutorial-15-joint-ie.md`)
adds **typed endpoints and a global graph-consistency search** (beam/greedy
over `JointIEConfig`, `gliner2/joint_ie/engine.py`,
`gliner2/joint_ie/schema.py`) on top of that *same* `SparseRelationScorer`.
On the 5 relation types it could decode at all (excluding the 3 broken
symmetric types below), JointIE reaches F1≈0.41, a real but modest precision
gain over raw independent decoding (0.39) — and, on this small gold set, not
clearly better than the much cheaper independent-decoding-plus-type-filter
(0.46 on the same 5 types). JointIE's advantage is the graph-level
guarantees it buys beyond type filtering (unique-head caps, no self-loops,
cross-relation consistency), which this gold set is too small to stress.

**CoreML's relation numbers are essentially identical to torch's**
(`data/preds/gold/coreml-fp32-cpu.jsonl`: tp=38 fp=149 fn=11, bit-for-bit
the same counts as torch-fp32) — expected, since the CoreML export ships
the dedicated `gliner2_multi_relation_*.mlpackage` fragment that backs the
same `SparseRelationScorer`-equivalent computation, unlike the ONNX export
gliner25-rs depends on. fp16-w8 is close (tp=38 fp=147 fn=11), the small
delta being ordinary quantization noise.

**JointIE library limitation found, not ours to fix:** a symmetric relation
whose head and tail are the **same single entity type** —
`family`/`business_associate`/`personal_relation`, all PERSON↔PERSON in
schema v2 — makes JointIE's decoder return an **empty result** every time:
0/15 gold instances recovered. Reproduced on a minimal single-sentence
example ("Barack Obama and Joe Biden remained close friends after leaving
office."): 0 relations under `optimizer="beam"`, and 0 entities *and* 0
relations (`result.feasible=False`) under `optimizer="greedy"`, logging
`"greedy joint-IE decoding produced a constraint-violating assignment;
returning empty solution"`. This happens whether the three symmetric types
are declared together in one schema or each alone in its own schema/call —
isolating it to the PERSON↔PERSON-symmetric shape itself, not a
co-declaration conflict. Not investigated further inside
`gliner2.joint_ie`'s internals (out of scope here); documented as a genuine,
reproducible gliner2 2.0.0 limitation, independent of gliner25-rs or the
Rust side entirely. `scripts/run_torch_jointie.py` works around it by default
(runs only the 5 non-symmetric relations in one schema/call); pass
`--include-symmetric` to re-confirm the three stay empty at ~4x the call cost.

**gliner25-rs has no JointIE equivalent whatsoever.** An exhaustive
case-insensitive search of `src/*.rs` for `joint`, `beam`, `graph`
(constraint sense), `acyclic`, `unique_head`, `constraint` turns up nothing
but unrelated uses of "graph" (ONNX computation graphs) and "crossings"
(the span-overlap policy). This is a structural gap on both axes — the
missing relation-scorer weights *and* the missing graph-consistency decoder
— not merely an unimplemented convenience.

## 4. GPE/LOC label wording (optional, torch only, gold only — `schema/ontonotes.json` untouched)

Both engines read the same schema, so the GPE/LOC confusion flagged
previously (the `location` and `country, city or state` prompts competing
for the same spans) is schema-wording-sensitive, not engine-sensitive.
Tried one alternative wording on a scratch copy of the schema (never
touching the shared file), torch only, gold only, overlap match:

| label | current wording | overlap F1 | alt wording | overlap F1 |
|---|---|---|---|---|
| GPE | "country, city or state" | 0.375 | "country, city, state or other political territory" | **0.774** |
| LOC | "location" | 0.233 | "physical geographic feature such as a mountain, river or region, not a country city or state" | **0.455** |

More than doubles both. This is a cheap, high-leverage knob for whoever owns
`schema/ontonotes.json` next — reported here as numbers only, per the
instruction not to change the shared file.

## 5. Latency / resource comparison (CPU, fp32, this VM: M4 Max VM, 4 cores, 8 GB RAM, no ANE)

| engine | windows | mode | load ms | p50 ms/window | words/s | peak RSS |
|---|---|---|---|---|---|---|
| rust-ort (gold, 84) | 84 | multi-pass | 744 | 183.3 | 73 | 3095 MB |
| rust-ort (gold, 84) | 84 | one-pass | 910 | 118.4 | 113 | 3101 MB |
| torch (gold, 84) | 84 | multi-pass | 7588 | 141.3 | 95 | 3458 MB |
| torch (gold, 84) | 84 | one-pass | 5599 | 72.3 | 185 | 3458 MB |
| torch-jointie (gold, 84) | 84 | 1 call/window, 5 relation types | 6449 | 70.8 | 189 | 3125 MB |
| coreml-fp32-cpu (gold, 84) | 84 | 2 calls/window (entities, relations) | 909 | 118.2 | — | — |
| coreml-fp16-w8-cpu (gold, 84) | 84 | 2 calls/window (entities, relations) | 896 | 71.0 | — | — |

Torch loads 7–10x slower than the Rust/ONNX path (safetensors + Python
import overhead vs. a handful of `mmap`'d ONNX sessions) but has lower
per-window latency once warm. CoreML EP was **not** re-tested here per the
constraint (its graph-compile step previously drove the shared VM's root
volume to ~100% full; CPU only for this round). Everything fit the 8 GB
budget loaded one at a time, per the shared-machine rule.

## Integration assessment: gliner25-rs for localdb

**API fit:** good. `BoundaryEngine::extract`/`extract_with` map directly onto
`schema/ontonotes.json`'s shape, spans are half-open, and `BoundaryManifest`
self-describes buckets/pool size/overlap policy. `extract_long` handles
documents past the largest bucket with word-window chunking and offset
remapping. One real sharp edge found this round: `Mention::char_start`/
`char_end` are byte offsets despite the name (§1) — any caller treating them
as char offsets on non-ASCII text needs the same `byte_to_char_map`
conversion this harness now does.

**Dependency weight:** light. No Python at build or run time. The
**`ort` rc.13 pin** (gliner25-rs 0.5.6 needs exactly that rc; Cargo's
resolver otherwise silently picks a newer, API-incompatible one) is the one
real friction point — pin it explicitly and watch for upstream movement.

**Relations support:** present but **structurally thinner than gliner2's**,
not merely unoptimized (§3): no learned relation scorer (the ONNX export
has no weights for one), no typed endpoints, no graph consistency. A
localdb integration using gliner25-rs relations should budget for a
type-constraint post-filter at minimum (this gets independent decoding to
F1≈0.40 on gold, close to torch's own post-filtered number) and should not
expect JointIE-equivalent behavior without new weights and new Rust code.

**Gaps / open items for localdb:**
1. Pin `ort` to the exact rc gliner25-rs was built against.
2. CoreML EP is usable but not obviously worth it without an ANE, and its
   per-load compile step eats system-volume scratch space; budget for that
   or pre-warm/cache compiled graphs before relying on it (not re-verified
   this round per the no-CoreML-EP constraint).
3. Ship a relation type-constraint post-filter (schema's own `head`/`tail`
   OntoNotes lists) before trusting relation output directly; it is the
   single highest-leverage fix available without new model weights.
4. GPE vs. LOC precision is schema-wording-sensitive (§4) — cheap to tune,
   worth doing if localdb needs that distinction to be reliable.
5. Entity exact-match agreement is now 0.98 after the byte/char fix (§1) —
   the crate is fit for purpose on span text and label; just convert its
   byte offsets to char offsets at the boundary.

## Reproducing

```bash
# Rust harness (needs Homebrew onnxruntime; wrap heavy runs in lockf per the shared-machine rule)
cd rust && cargo build --release
ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib \
  ./target/release/gliner-rust-eval \
  --schema ../schema/ontonotes.json --models-dir ../models/gliner2.5-multi-v1-onnx \
  --windows ../eval/gold.jsonl --out ../data/preds/gold/rust-ort-fp32.jsonl \
  --device cpu --precision fp32 --mode multi-pass

# PyTorch reference (independent decoding)
uv run python scripts/run_torch.py --windows eval/gold.jsonl \
  --out data/preds/gold/torch-fp32.jsonl --device cpu --mode multi-pass

# PyTorch reference (JointIE)
uv run python scripts/run_torch_jointie.py --windows eval/gold.jsonl \
  --out data/preds/gold/torch-jointie.jsonl

# CoreML (persistent .cache/coreml/<revision>/*.mlmodelc avoids the earlier ENOSPC)
uv run python scripts/run_coreml.py --windows eval/gold.jsonl --precision fp32 --compute-units cpu \
  --out data/preds/gold/coreml-fp32-cpu.jsonl
uv run python scripts/run_coreml.py --windows eval/gold.jsonl --precision fp16-w8 --compute-units cpu \
  --out data/preds/gold/coreml-fp16-w8-cpu.jsonl

# Comparison report
uv run python scripts/compare.py --windows data/windows.jsonl \
  --preds-dir data/preds --gold eval/gold.jsonl --schema schema/ontonotes.json \
  --reference torch-fp32 --report /tmp/cross-validation-report.md
```

Model weights for gliner25-rs: `hf download jugaadsrl/gliner2.5-multi-v1-onnx
--include "fp32_25/*" --local-dir models/gliner2.5-multi-v1-onnx` (gitignored,
not committed).
