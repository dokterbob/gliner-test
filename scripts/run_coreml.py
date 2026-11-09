"""Run entity + relation extraction over data/windows.jsonl with the CoreML export.

Writes one shared-format prediction line per window to
``data/preds/coreml-<precision>-<units>.jsonl``:

    {"window_id": ..., "entities": [{"label", "text", "start", "end", "score"}],
     "relations": [{"name", "head": {"text","start","end"}, "tail": {...}, "score"}],
     "latency_ms": float}

Entity/relation char offsets are within the *window* text (not the original
chunk), matching windows.jsonl's own ``text`` field.

schema/ontonotes.json (v2) has 8 entity types and 8 relation types split
into ``relation_passes`` of <= 4 each (the Core ML relation head is R4 and
the bucket allows only 8 extraction queries total; each relation type costs
2 queries, head role + tail role). Every window therefore runs through
``1 + len(relation_passes)`` ``extract()`` calls:
  - an entities pass:       Schema().entities([8 OntoNotes labels])
  - one pass per relation group (currently "affiliation", "social_place"):
       Schema().relations([<= 4 relation *prompts* in that group])
Each relation's ``prompt`` (not its canonical ``name``) is what's sent to
the model; predictions are emitted under ``name``. ``latency_ms`` is the
sum of all passes -- the real cost of getting full entities+relations for
that window. (v1 schemas with a flat "relations" list and no
"relation_passes"/"prompt" fields still work: they fall back to one
"default" pass using ``name`` as the prompt.)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import coremltools as ct

from gliner_test.compiled_models import CACHE_ROOT, patch_loader

UNITS = {
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}

FP16_W8_FEATURE_PACKAGE = "gliner2_multi_extraction_features_w8_embedding_linear_fp16_L128_W64_Q8.mlpackage"


def build_runtime(model_dir: Path, precision: str, compute_units: str):
    # Loads already-compiled .mlmodelc artifacts (see scripts/compile_models.py)
    # instead of letting ct.models.MLModel recompile the .mlpackage into the
    # boot-volume temp dir on every process start.
    patch_loader(model_dir)
    sys.path.insert(0, str(model_dir))
    from extraction_runtime import CoreMLAdaptiveBoundaryExtractor, CoreMLBoundaryExtractor

    cu = UNITS[compute_units]
    if precision == "fp32":
        return CoreMLBoundaryExtractor(str(model_dir), precision="fp32", compute_units=cu)
    if precision == "fp16-w8":
        return CoreMLBoundaryExtractor(
            str(model_dir), precision="fp16", compute_units=cu, feature_package=FP16_W8_FEATURE_PACKAGE
        )
    if precision == "adaptive":
        # Our schema never declares latent records, so CoreMLAdaptiveBoundaryExtractor's
        # own selected_precision() always picks the fp16/W8 path for this workload; see
        # docs/coreml-findings.md. We still point both internal precisions at the
        # requested compute units for a fair, host-ready comparison.
        return CoreMLAdaptiveBoundaryExtractor(str(model_dir), fp16_compute_units=cu, fp32_compute_units=cu)
    raise ValueError(f"unknown precision {precision!r}")


def load_schemas(schema_path: Path, Schema):
    """Build the entity schema plus one relation schema per ``relation_passes`` group.

    Returns ``(ent_schema, relation_pass_schemas, label_to_tag, threshold)`` where
    ``relation_pass_schemas`` is a list of ``(pass_name, schema, prompt_to_name)``.
    """
    cfg = json.loads(schema_path.read_text())
    ent_labels = [e["label"] for e in cfg["entities"]]
    label_to_tag = {e["label"]: e["ontonotes"] for e in cfg["entities"]}
    threshold = cfg.get("threshold", 0.5)
    ent_schema = Schema().entities(ent_labels)

    pass_names = cfg.get("relation_passes") or ["default"]
    relation_pass_schemas = []
    for pass_name in pass_names:
        rels_in_pass = [r for r in cfg["relations"] if r.get("pass", "default") == pass_name]
        if not rels_in_pass:
            continue
        prompt_to_name = {r.get("prompt", r["name"]): r["name"] for r in rels_in_pass}
        schema = Schema().relations(list(prompt_to_name.keys()))
        relation_pass_schemas.append((pass_name, schema, prompt_to_name))
    return ent_schema, relation_pass_schemas, label_to_tag, threshold


def extract_window(runtime, ent_schema, relation_pass_schemas, label_to_tag, threshold, text, rel_threshold=None):
    rel_threshold = threshold if rel_threshold is None else rel_threshold
    t0 = time.perf_counter()
    ent_out = runtime.extract(text, ent_schema, threshold=threshold, include_confidence=True, include_spans=True)

    entities = []
    for label, mentions in ent_out.get("entities", {}).items():
        tag = label_to_tag.get(label, label)
        for m in mentions:
            entities.append(
                {"label": tag, "text": m["text"], "start": m["start"], "end": m["end"], "score": float(m["confidence"])}
            )

    relations = []
    for _pass_name, rel_schema, prompt_to_name in relation_pass_schemas:
        rel_out = runtime.extract(text, rel_schema, threshold=rel_threshold, include_confidence=True, include_spans=True)
        for prompt, pairs in rel_out.get("relation_extraction", {}).items():
            name = prompt_to_name.get(prompt, prompt)
            for p in pairs:
                head, tail = p["head"], p["tail"]
                score = float(head.get("confidence", tail.get("confidence", 0.0)))
                relations.append(
                    {
                        "name": name,
                        "head": {"text": head["text"], "start": head["start"], "end": head["end"]},
                        "tail": {"text": tail["text"], "start": tail["start"], "end": tail["end"]},
                        "score": score,
                    }
                )
    latency_ms = (time.perf_counter() - t0) * 1000
    return entities, relations, latency_ms


def apply_pair_generator_overrides(owner, arg_threshold, pair_cap, max_pair_cap=None):
    """dataclasses.replace the frozen RelationProposalSettings on owner.relation_pair_generator
    (re-read by the generator on every call)."""
    import dataclasses

    if arg_threshold is None and pair_cap is None:
        return
    if pair_cap is not None and max_pair_cap is not None and pair_cap > max_pair_cap:
        raise SystemExit(f"--pair-cap {pair_cap} exceeds the maximum {max_pair_cap} supported by this engine")
    gen = owner.relation_pair_generator
    kw = {}
    if arg_threshold is not None:
        kw["argument_threshold"] = arg_threshold
    if pair_cap is not None:
        kw["pair_cap"] = pair_cap
    gen.settings = dataclasses.replace(gen.settings, **kw)
    print(f"[pair-generator] settings -> {gen.settings}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=None, help="defaults to .cache/coreml/<revision>")
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--windows", type=Path, default=Path("data/windows.jsonl"))
    ap.add_argument("--precision", choices=["fp32", "fp16-w8", "adaptive"], default="fp32")
    ap.add_argument("--compute-units", choices=list(UNITS), default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--relation-threshold", type=float, default=None,
                    help="threshold for the relation passes only (default: schema `threshold`; entities always use the "
                         "schema threshold). 0.0 keeps every scored pair. Note: gliner2 gates per pair at this threshold "
                         "and runs dedup AFTER it, so lowering it can also shift which of two overlapping pairs "
                         "survives dedup, not just add pairs.")
    ap.add_argument("--arg-threshold", type=float, default=None,
                    help="relation_pair_generator argument_threshold (default 0.2 from the model config): minimum "
                         "head/tail entity-candidate score for a span to be proposed as a pair endpoint")
    ap.add_argument("--pair-cap", type=int, default=None,
                    help="relation_pair_generator pair_cap (default 64) ; CoreML relation package is P256=R4x64 so must be <= 64")

    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    if args.model_dir is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from download_models import REVISION

        args.model_dir = CACHE_ROOT / REVISION

    out = args.out or Path(f"data/preds/coreml-{args.precision}-{args.compute_units}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    load_t0 = time.perf_counter()
    runtime = build_runtime(args.model_dir, args.precision, args.compute_units)
    from gliner2 import Schema  # noqa: E402  (import after sys.path insert in build_runtime)

    ent_schema, relation_pass_schemas, label_to_tag, threshold = load_schemas(args.schema, Schema)
    apply_pair_generator_overrides(runtime, args.arg_threshold, args.pair_cap, max_pair_cap=64)
    rel_threshold = threshold if args.relation_threshold is None else args.relation_threshold
    print(f"[run_coreml] entity threshold={threshold} relation threshold={rel_threshold}", file=sys.stderr)
    load_ms = (time.perf_counter() - load_t0) * 1000
    print(f"[run_coreml] loaded {args.precision}/{args.compute_units} in {load_ms:.0f} ms", file=sys.stderr)

    windows = []
    with args.windows.open() as f:
        for line in f:
            windows.append(json.loads(line))
    windows = windows[args.offset :]
    if args.limit is not None:
        windows = windows[: args.limit]

    n = 0
    with out.open("w") as fout:
        for w in windows:
            entities, relations, latency_ms = extract_window(
                runtime, ent_schema, relation_pass_schemas, label_to_tag, threshold, w["text"], rel_threshold
            )
            fout.write(
                json.dumps(
                    {
                        "window_id": w["window_id"],
                        "entities": entities,
                        "relations": relations,
                        "latency_ms": latency_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
            if n % args.progress_every == 0:
                print(f"[run_coreml] {n}/{len(windows)}", file=sys.stderr)

    del runtime
    gc.collect()
    print(f"[run_coreml] wrote {n} predictions to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
