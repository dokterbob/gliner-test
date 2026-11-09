# Relation accuracy: decoding and post-processing experiments

Goal: **which entities are related, with a confidence score**. The relation name is a secondary qualifier.
Scope is decoding, post-processing and evaluation only (no fine-tuning, no new export).

All numbers are reproducible from committed configs (`eval/experiments/*.json`, one per experiment; results in
`eval/experiments/results.jsonl` / `results.md`) plus the commands below. Raw model outputs live in `data/` (not committed);
`eval/experiments/final_runs.sh` regenerates them.

**Read the numbers with the sample size in mind.** Gold v3 has 103 pairs: dev 58, test 45 (place 12, per_per 11, org_org 5,
per_org 17 on test). One pair is 1.7 points of overall recall on test and 8 to 20 points inside a category. Differences of 1 to 2
pairs are noise; counts (tp/fp/fn) are given next to every rate.

## Summary

| | |
|---|---|
| Best pipeline | 2 relation passes (v2.1 grouping) + entity pass; max over the category's qualifier-name scores; type gate; overlap NMS; per-category **temperature** calibration; **one global threshold 0.5** on the calibrated score |
| Test (45 pairs), CoreML fp32 | P 0.62, R 0.80, F1 0.70 (36/22/9), AUPRC 0.69, ECE 0.17, qualifier top-1 26/36 |
| Raw model, same engine | F1 0.62 (41/47/4 at 0.5), AUPRC 0.61 |
| Weakest category | `place` (test F1 0.52, AUPRC 0.36) |
| Per-category thresholds | Did not help: no gain in 5-fold dev CV, worse on test. Not adopted |
| Precision >= 0.8 | Reachable on dev only for 3 of 4 categories, and did not transfer to test (overall P 0.65 at that point) |
| JointIE decoder | No better ranking, better qualifier accuracy, much worse precision. Not worth porting |
| Window cost | 4 of 103 gold pairs (3.9%) cross a window boundary; unsplit torch is not better than windowed torch |
| Rust (no pair scorer) | F1 0.47 raw to 0.61 after post-processing; the pair scorer is still missing (AUPRC 0.56 vs 0.69) |

## Headline command

```sh
uv run python scripts/compare.py --gold eval/gold.jsonl --split test --preds-dir data/no-raw \
  --pairs-dir data/pairs-all --pairs-only --threshold 0.5 --engine coreml-fp32-cpu
```

`data/pairs-all/<engine>.jsonl` holds every scored pair after the final pipeline (calibrated scores, no thresholding), so
AUPRC and ECE are over the full ranking and `--threshold 0.5` is the production operating point. `--preds-dir data/no-raw`
(an empty path) keeps compare.py from also scoring the raw prediction files. The pair files are built by:

```sh
uv run python eval/experiments/mk_final.py      # dev-only tuning of the p80 thresholds -> eval/experiments/final-*.json
uv run python eval/experiments/make_pairs.py    # gliner_test.relations CLI -> data/pairs, pairs-p80, pairs-all, pairs-percat
uv run python eval/experiments/report.py --split test --engines coreml-fp32-cpu   # tables below (both operating points)
```

## Setup

- **Gold v3** (`eval/gold.jsonl`): 124 items (84 sentences, 40 passages; en/nl/de/fr), 103 pairs, split dev 62 / test 62 items.
  Matching: endpoint text equality or char overlap, one-to-one, best score per pair, category from predicted entity types.
- **Windowing** (`scripts/gold_windows.py`): `windowing.window_chunk` with the corpus budgets, against the current schema's
  worst relation pass and the entity pass (entity prompt 59 tokens, worst relation pass 60, text budget 62). 124 items give
  157 windows (93 items 1 window, 29 items 2, 2 items 3). Predictions are merged back to the parent item with
  parent-relative offsets (entities dedup on span+label, relations on head/tail span+name, max score, summed latency).
  Windows are re-derived per schema variant, because a longer relation prompt shrinks the windows.
- **Protocol:** every model configuration runs once with `--relation-threshold 0.0`; collapse, calibration and thresholds are
  offline (`eval/experiments/exp.py`). Tuning used **dev only**. Test was opened for the final confirmation, with one
  disclosed exception (see "Threshold selection").
- **Baseline:** v2.1 grouping (`schema/ontonotes-v2.1.json`), raw scores, max collapse, type gate, global threshold 0.5,
  CoreML fp32 on CPU.

## Dev results per step

Overall = untyped pairs. "best-global" = F1 at the best single threshold on a 0.05 grid (what the ranking supports without
per-category tuning). Latency is the mean CoreML fp32 CPU time per window (all passes) on the gold windows. n(dev) = 58 pairs.

| step | change | P / R / F1 at 0.5 | macro F1 | AUPRC | best-global F1 | ms/win | kept? |
|---|---|---|---|---|---|---|---|
| 01a | **baseline** v2.1 grouping (3 calls) | .641 / .707 / .672 | .679 | .576 | .703 | 120 | base |
| 01b | v2.2 grouping (4 calls) | .600 / .672 / .634 | .635 | .498 | .667 | 160 (+33%) | no |
| 02 | `--arg-threshold` 0.1 | .545 / .724 / .622 | .640 | .571 | .673 | 120 | no |
| 02 | `--arg-threshold` 0.05 | .424 / .724 / .535 | .527 | .545 | .622 | 121 | no |
| 03a | coarse-only prompts (located in / affiliated with / partner of / related to), 2 calls | .774 / .414 / .539 | .439 | .446 | .633 | 79 | no |
| 03e | coarse-only, wording b (based in / works for / affiliated with / relationship) | .780 / .552 / .646 | .622 | .513 | .660 | 78 | no (see notes) |
| 03b/c | coarse + fine (4 calls), score = coarse only / max over all | same as 03a / .581 / .741 / .652 | .439 / .658 | .446 / .564 | .633 / .696 | 159 | no |
| 04 | v2.2 + short relation descriptions (vs 01b) | .578 / .638 / .607 | .605 | .476 | .612 | 155 | no |
| 05a | noisy-or collapse over all names | .558 / .741 / .637 | .632 | .613 | .667 | 120 | no (see notes) |
| 05b | max over the category's **qualifier names only** | .678 / .690 / .684 | .690 | .577 | .710 | 120 | **yes** |
| 05e | + overlap **NMS** of nested-span duplicates | .727 / .690 / .708 | .708 | .580 | .710 | 120 | **yes** |
| 05f | noisy-or over qualifiers + NMS | .678 / .690 / .684 | .690 | .606 | .684 | 120 | no |
| 07 | + **temperature** calibration (dev fit), threshold 0.5 | .727 / .690 / .708 | .708 | **.642** | .708 | 120 | **yes** |
| 07 | isotonic calibration instead | .783 / .621 / .692 | .682 | .662 | .726 | 120 | no (logloss 2x worse) |

Notes per step:

1. **Grouping.** v2.2's per-category passes lose on every dev metric and cost one extra call (+33% latency). The final schema
   uses the v2.1 grouping (`affiliation`, `social_place`); `resident_of` is `unassigned` (not sent to the model). It never
   fired in v2.1 either, and v2.2 did not recover more place pairs (5 vs 6 tp of 16).
   Prompt **order inside a pass matters**: re-running with the final schema's relation order gave different raw scores
   (56 of 124 items differ) but the same dev numbers (F1 .708, AUPRC .576 vs .580, row 09).
2. **arg-threshold.** The default 0.2 is best. Lower values add candidate endpoints that mostly become false positives
   (per_org fp 13 to 32).
3. **Coarse prompts.** Cheaper (2 calls, 79 ms) but the coarse score ranks worse (AUPRC .45 to .51 vs .58). Wording matters a lot
   (03a vs 03e: F1 .54 vs .65). Coarse + fine adds no information beyond the fine scores. Each coarse pass costs 60 to 62 prompt
   tokens (within the 65 limit; 62 shrinks windows by one token). Not adopted for accuracy, but 03e is a viable **latency lever** (-35%)
   if a 0.07 F1 drop is acceptable.
4. **Descriptions.** The 4 short descriptions pushed passes to 57 to 60 tokens and lowered all dev metrics vs v2.2 without them.
5. **Collapse.** Noisy-or raises AUPRC by .03 to .04 but lowers F1 at every operating point I tried (best-global .667 vs .703)
   and inflates scores (median TP .95, median FP .43); with calibration it did not win the per-category or the global F1 either.
   The AUPRC gain is about 2 pairs of ranking and is within noise, so max is kept. Scoring a pair by its **qualifier names only**
   removes the cross-category evidence (a location relation raising a person pair), +.012 F1, +.011 macro.
   NMS (not in the brief) fixes the duplicate caused by nested entity spans ("Grünen-Fraktion" vs "Grünen-Fraktion im Bundestag"):
   per_org fp 13 to 10, +.024 F1.
6. **Thresholds.** See below.
7. **Calibration.** Fitted on dev only, 5-fold cross-fitted dev ECE(10): raw .162, isotonic .103 (log-loss 1.00), temperature .098
   (log-loss .495). Isotonic overfits (hard 0/1 steps); temperature is 2 parameters per category and monotone.
   **Test ECE(10), overall:** raw .196, isotonic .198, temperature .169. Test AUPRC: raw .652, temperature .692. Temperature calibration
   mainly helps by putting categories on one scale (pooled AUPRC .576 to .642 on dev); it does not make the scores trustworthy
   probabilities (ECE .17 on test, place .29, per_per .22).

### Threshold selection (steps 6 and 7)

Per-category thresholds that maximise dev F1 (on calibrated scores): place .55, per_per .70, org_org .15, per_org .65.
- In-sample dev: F1 .724, macro .728.
- **5-fold CV on dev** (`exp.py cvthr`): per-category thresholds F1 .673 = global tuned threshold .673 < fixed 0.5 .708.
- **Test** (`final-coreml-fp32-cpu-percat`): F1 .617 (29/20/16) vs .699 (36/22/9) for the fixed 0.5.

So they were dropped. Disclosure: I first committed them to the schema (724dc26), then saw the test drop and reverted
(`schema/ontonotes.json` thresholds are `null` again, i.e. global 0.5). The reason to prefer 0.5 stands on dev alone (the CV above),
but the test split did inform the timing, so treat the test numbers as lightly optimistic.

**Precision >= 0.8 operating point** (lowest dev threshold per category with dev P >= 0.8: place none, per_per .40, org_org .10, per_org .65):
dev overall P .82 / R .55; **on test overall P .65 / R .58**, so the target does not transfer with 45 pairs, and `place` never reaches 0.8
precision on dev (its best-scored false positives are 0.88 to 0.97, e.g. "NATO in Brussel" which gold does not list). Do not
promise 0.8 precision from thresholds alone.

## Final configuration

- Schema `schema/ontonotes.json`: entity pass + `affiliation` (employee_of, member_of, founder_of, subsidiary_of) + `social_place`
  (family, business_associate, personal_relation, located_in) = **3 calls per window**, prompts as in the schema, `--relation-threshold 0.0`,
  `--arg-threshold` 0.2 (default), `--pair-cap` 64.
- Post-processing (`gliner_test.relations`): snap endpoints to entity types, drop pairs fitting no category (type gate), collapse
  per (category, endpoints) with **max** over the category's own qualifier names (`--score-names qualifiers`), **NMS**
  (`--nms`), temperature calibration (`--calibration eval/calibration/<engine>.json`), threshold **0.5**, document aggregation by
  max (`--aggregate`) for corpus use.
- Temperatures T (score' = sigmoid(logit(score) / T), clamp 1e-6), fitted on dev:

  | engine | global | place | per_per | org_org | per_org |
  |---|---|---|---|---|---|
  | coreml-fp32 (= torch) | 0.949 | 1.895 | 0.505 | 0.384 | 1.154 |
  | coreml-fp16-w8 | 1.047 | 2.618 | 0.534 | 0.429 | 1.118 |

  Place is over-confident (T 1.9 to 2.6), per_per and org_org under-confident (T 0.4 to 0.5). The org_org value rests on 11 to 12 dev samples.
- Schema edits made (commit 724dc26; thresholds reverted to null inside 62aaa44): relation `pass` fields, the `relation_passes` list
  (`["affiliation", "social_place"]`; needed for the pass change), `$comment`; `pair_categories[].threshold` stays `null`.

## Test results (45 pairs; counts tp/fp/fn)

Final pipeline, CoreML fp32 CPU, split=test, pool items included. F1 point = global 0.5 on the calibrated score.
P>=0.8 point = per-category dev-tuned thresholds. AUPRC and ECE(10) over all scored pairs. Macro F1 .701 (F1 point), .530 (P>=0.8 point).
Qualifier top-1 on matched pairs: 26/36 = 0.72.

| category | gold | F1 point (tp/fp/fn, P, R, F1) | P>=0.8 point | AUPRC | ECE |
|---|---|---|---|---|---|
| overall | 45 | 36/22/9, P .62, R .80, F1 .70 | 26/14/19, P .65, R .58, F1 .61 | 0.692 | 0.169 |
| place | 12 | 7/8/5, P .47, R .58, F1 .52 | no pairs kept (threshold 0.95) | 0.357 | 0.286 |
| per_per | 11 | 8/7/3, P .53, R .73, F1 .61 | 8/7/3, same | 0.706 | 0.218 |
| org_org | 5 | 4/0/1, P 1.00, R .80, F1 .89 | 4/2/1, P .67, R .80, F1 .73 | 0.800 | 0.149 |
| per_org | 17 | 16/8/1, P .67, R .94, F1 .78 | 14/5/3, P .74, R .82, F1 .78 | 0.905 | 0.163 |

Without the 2 pool-sourced gold pairs (one place pair in test, one in dev; `report.py --exclude-pool`): test overall 35/23/9, P .60, R .80, F1 .69;
place 6/9/5 (F1 .46, gold 11); macro F1 .687; other categories unchanged. The pool pair is the one place pair the model got right. Dev for comparison (58 pairs): F1 point overall 40/15/18, P .73, R .69, F1 .71; AUPRC .642; ECE .132.
Dev to test: per_per fell from F1 .75 to .61 and place from .48 to .52 (up); per_org .77 to .78; all moves are 1 to 3 pairs.

## Engine comparison (test, F1 point; same windows, same pipeline, each engine calibrated on its own dev scores)

| engine | overall tp/fp/fn | P / R / F1 | AUPRC | ECE | macro F1 | qualifier top-1 | ms/window |
|---|---|---|---|---|---|---|---|
| coreml-fp32 CPU | 36/22/9 | .62 / .80 / .70 | .692 | .169 | .701 | 26/36 | 122 |
| coreml-fp16-w8 CPU | 37/22/8 | .63 / .82 / .71 | .715 | .156 | .715 | 27/37 | 72 |
| torch fp32 | 36/22/9 | .62 / .80 / .70 | .692 | .169 | .701 | 26/36 | 156 |
| torch JointIE (2.0.0, undirected) | 42/61/3 | .41 / .93 / .57 | .661 | .278 | .594 | 38/42 | 160 |
| rust-ort fp32 (no pair scorer) | 38/42/7 | .47 / .84 / .61 | .564 | .172 | .620 | 27/37 | 226 |

CoreML fp32 and torch are identical (parity). fp16-w8 differs by 1 pair in either direction: fp16-w8 is the production precision
and is not worse here.
JointIE is one decoder call per symmetric/asymmetric schema group (2 calls here); torch and JointIE latencies are CPU torch, not comparable to CoreML.

### What post-processing recovers per engine (overall untyped pairs, test)

A raw @0.5; B + type gate and max collapse @0.5; C = full pipeline (F1 point). Cell: tp/fp/fn, P R F1 | AUPRC
(`eval/experiments/recovery.py --split test`).

| engine | A raw | B type gate + collapse | C final pipeline |
|---|---|---|---|
| coreml-fp32 | 41/47/4, .47 .91 .62 \| .614 | 37/29/8, .56 .82 .67 \| .596 | 36/22/9, .62 .80 .70 \| .692 |
| coreml-fp16-w8 | 41/45/4, .48 .91 .63 \| .624 | 38/29/7, .57 .84 .68 \| .625 | 37/22/8, .63 .82 .71 \| .715 |
| torch JointIE | 42/64/3, .40 .93 .56 \| .574 | 42/63/3, .40 .93 .56 \| .583 | 42/61/3, .41 .93 .57 \| .661 |
| rust-ort | 42/92/3, .31 .93 .47 \| .653 | 38/51/7, .43 .84 .57 \| .626 | 38/42/7, .47 .84 .61 \| .564 |

**Rust recovery:** its raw output is `min(head, tail)` for every head x tail combination (precision .31). The type gate and collapse
remove 41 of 92 false positives at a cost of 4 true positives (F1 .47 to .57); NMS plus per-engine calibration and the 0.5 threshold add a
little (.61). It still ranks worse than the CoreML pair scorer (AUPRC .564 vs .692; per_org threshold at P>=0.8 keeps only 2 pairs):
**the learned pair scorer is what is missing, post-processing cannot replace it.** Rust's AUPRC *falls* from A to C (.653 to .564): its per-category temperatures are fitted on only 18 to 33 dev pairs, and its raw score is not a pair score, so calibration cannot fix the ranking.
(Type gate alone lost 4 of the true pairs on the CoreML engines because an endpoint such as "Amsterdam" or "Leiden" had no
entity-pass span above the 0.5 entity threshold; a lower-threshold snapping pass is an easy recall lever, not tried.)

## Cross-window loss and the unsplit comparison

- **4 of 103 gold pairs (3.9%) are not inside any single window** (dev 2 of 58, test 2 of 45): 2 per_org, 2 per_per, all in
  passages (e.g. "Nina Brunner ... Rhätia Datentechnik AG" separated by a long clause). Windows overlap by 25%.
  This is the structural recall ceiling of the 64-word bucket on gold; on production text with longer-range coreference it is likely larger.
- **Torch on unsplit passages** (no bucket) vs torch on 64-word windows, final pipeline, all splits, passages only (40 passages): windowed F1 .620 (AUPRC .545) vs unsplit F1 .598 (AUPRC .524); JointIE windowed .549 vs unsplit .586.
  Sentences are identical (they fit one window). So **the window split does not cost accuracy on this gold**
  (windowed is slightly better for the independent decoder because the pair generator sees fewer candidates). The cost lever "larger-window
  re-export" would buy the 4 cross-window pairs at most, i.e. at most about +4 points of recall, and may lose precision on long inputs.

## JointIE: gliner2 2.0.0 vs 2.1.0 (torch only, dev, unsplit text, relation threshold 0.05)

2.1.0 was run in an isolated environment (`uv run --no-project --isolated --with "gliner2[local]==2.1.0"`, version verified at runtime; `pyproject.toml`
and `uv.lock` untouched). Runner: `eval/experiments/run_jointie_rel.py` (adds `--relation-threshold` to `scripts/run_torch_jointie.py`).

| run | P / R / F1 at 0.5 | AUPRC | median score TP / FP | notes |
|---|---|---|---|---|
| 2.0.0, `--symmetric-mode undirected` | .548 / .793 / .648 | .586 | .82 / .33 | per_per pairs work |
| 2.1.0, `--symmetric-mode undirected` | .561 / .793 / .657 | .589 | .82 / .33 | 1 fp fewer |
| 2.1.0, `symmetric=True` relations | .528 / .655 / .585 | .426 | .85 / .40 | per_per returns 0 pairs; 4 of 124 items infeasible (bug not fixed in 2.1.0) |
| torch independent, v2.1 grouping (comparison) | .722 / .672 / .696 | .574 | .76 / .27 | |

**Verdict:** 2.1.0 changes nothing material (the symmetric bug persists; `undirected` is still required). JointIE's ranking is not better than
independent decoding with the same type gate (AUPRC .586 vs .574 on dev, .661 vs .692 on test), and score separation is the same
(TP/FP medians .82/.33 vs .76/.27, a gap of 0.49 in both). It does give higher recall (place 10 of 16 tp vs 5 to 6) at much lower precision
(place fp 19 to 20 vs 3), and a clearly better **qualifier** accuracy (test 38/42 = .90 vs 26/36 = .72), because its typed relation signatures
choose the relation name jointly with the endpoint types. **Not worth porting a JointIE-style decoder to Rust for the pair question.** If
the qualifier matters, a cheap alternative is to port only the typed qualifier choice (restrict the argmax to names allowed by the endpoint types, which
`pair_categories[].qualifiers` already does).

## Corpus run (localdb technical docs)

CoreML fp16-w8, CPU, 3450 windows, final schema, `--relation-threshold 0.0`: 4.6 min total model time (mean 80 ms per window for 3 passes),
run as four locked chunks (`eval/experiments/corpus_run.sh`). 307 document-level pairs without a threshold, **20 at calibrated 0.5**
(13 org_org, 3 per_per, 2 per_org, 2 place). A 40-pair stratified sample was reviewed by eye
(`eval/experiments/corpus-sample.md`): **correct 6, unsure 7, wrong 27**; strict precision 6/40 = 0.15, lenient 13/40 = 0.33.
At the production threshold (13 sampled pairs at >= 0.5): 2 correct, 2 unsure, 9 wrong. The five good per_per pairs are the
genuine people (Alice, Bob, Carol); everything else is code identifiers, tools and product names the entity layer tags as
ORG/PERSON/GPE. The bottleneck on technical text is the entity layer, not the pair scorer. Recommend restricting localdb's
pairs to entities with entity score >= 0.8 or to PERSON/ORG/GPE spans that are not code-like (backticks, `::`, snake_case) before trusting
this layer on technical docs.

## Recommendations for localdb (Rust port)

Port, in this order (all are pure arithmetic; reference implementation `src/gliner_test/relations.py`, tests in `tests/test_relations.py`):

1. **Type gate (`snap_and_type` + `categorize`).** Snap each relation endpoint to the largest-overlap entity span (ties: higher score, then
   earlier); categorise by the schema's `pair_categories` head/tail types; drop uncategorisable pairs. This is the biggest single gain on every
   engine (Rust F1 .47 to .57, fp precision .31 to .43).
2. **Collapse per (category, endpoints)** with max over the category's *own qualifier names* (undirected categories group both orientations;
   place is directed). The qualifier list (name, score) is the secondary output.
3. **Overlap NMS** (`suppress_overlapping`): by descending score, drop a pair whose head spans and tail spans both overlap a kept pair of
   the same category.
4. **Temperature calibration**, per category: `score' = sigmoid(logit(clamp(score, 1e-6, 1-1e-6)) / T)` with the T table above
   (store as `eval/calibration/<engine>.json`; refit on any new engine/precision with
   `python -m gliner_test.relations --fit-calibration --calibration-method temperature --gold eval/gold.jsonl --split dev ...`).
5. **Threshold 0.5** on the calibrated score (global). Expose it as a user knob; it traces the P/R trade-off (test: 0.35 gives R .82 at P .53; 0.65 gives P .64 at R .62).
6. **Document aggregation** (`aggregate_document`) with max: lift pairs to chunk offsets, merge duplicates, group by normalised endpoint text.

Do not port: noisy-or, per-category thresholds, descriptions, coarse prompts (unless latency matters), the JointIE decoder.
CoreML settings: 3 calls per window, `--relation-threshold 0.0`, arg threshold 0.2, pair cap 64.

## Open levers

1. **Larger-window CoreML re-export.** The model was trained on up to 4,096 words; we use a 64-word bucket. Cross-window loss on gold is only 4 of 103 pairs
   and torch-unsplit is not better, so the expected gain here is small; it matters for long-range pairs in prose that the gold does not cover, and it would cut the number
   of passes over overlapping text (25% overlap) and the 3 calls per window.
2. **ONNX relation-scorer fragment for gliner25-rs.** Rust's pair scores are `min(head, tail)`; with the full post-processing it reaches F1 .61 vs CoreML .70
   and AUPRC .564 vs .692. The fragment is the only way to close that gap; precision at a fixed recall is the main loss (fp 42 vs 22).
3. **Fine-tuning** on place and per_per: `place` is the weakest category in every run (dev F1 .48, test .52; the highest-scoring false positives are
   0.88 to 0.97 for plausible but unlabelled place pairs), and the model has a lot of unlabelled-pair ambiguity that guideline-conformant training data would fix.
4. **Lower-threshold entity snapping** (recover endpoints with no entity above 0.5; 4 of 45 test pairs were lost to the type gate for this reason), and a code-identifier filter for technical text.
5. **Prompt order and wording** are not stable knobs (order changed 56 of 124 item outputs; coarse wording changed F1 by .11). Any prompt change needs a re-calibration.

## Suspected gold issues (not edited)

- `gold-0041` (test): "Jeff Bezos founded Amazon in Seattle ... headquartered near the Space Needle." Gold has only per_org Bezos-Amazon; **Amazon -> Seattle (place)** is
  a clear omission (model score 0.83).
- `gold-0015` (dev): "Mark Rutte, de voormalige premier van Nederland, sprak ... met de NATO in Brussel." No gold pair; NATO -> Brussel (0.88) and Rutte -> Nederland
  (former PM) are arguable place/affiliation pairs under the guidelines (the schema's `employee_of` tail allows GPE, but per_org has only ORG tails, so a
  person-government pair has no category).
- `gold-p-fr-04` (test, "Hé Camille, tu as vu Nadia ? Elle est ma meilleure amie"): the model pairs Camille-Nadia (0.97); gold correctly omits it, a case of
  unlabelled-but-plausible ambiguity, not an error.
