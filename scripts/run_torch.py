"""Native PyTorch reference run: fastino/gliner2.5-multi-v1 via the `gliner2`
package (2.0.0, the version the CoreML export pins), over the same windows
JSONL used by the Rust (gliner25-rs/ONNX Runtime) and CoreML runners.

Writes predictions in the shared format:
    {"window_id", "entities": [{"label","text","start","end","score"}],
     "relations": [{"name","head":{...},"tail":{...},"score"}], "latency_ms"}

Labels and the shared threshold come from schema/ontonotes.json -- never
hard-coded -- so all three engines read the same prompt strings.

Usage:
    uv run python scripts/run_torch.py \
        --schema schema/ontonotes.json \
        --windows data/windows.jsonl \
        --out data/preds/torch-fp32.jsonl \
        --device cpu

## One-pass vs. multi-pass

`--mode multi-pass` (the default) runs one `model.extract()` call for
entities, then one more per `schema.relation_passes` group (schema v2:
"affiliation" then "social_place", 4 relations each), mirroring
scripts/run_coreml.py exactly -- that pipeline is limited to 8 queries per
call, so it *must* split entities (8 labels) and each relation pass (4
names x 2 roles = 8) into separate calls. GLiNER2 conditions every task's
attention on the whole prompt, so a one-pass call with everything together
is a different encoder input, not just a latency optimization --
`--mode one-pass` keeps the original combined-schema behavior for the cost
comparison in docs/cross-validation.md.

Schema v2 separates the relation's canonical output id (`name`, e.g.
"employee_of") from the text sent to the model (`prompt`, e.g. "works for").
`model.create_schema().relations([...])` takes prompts; the result dict is
keyed by whatever string was passed in, so we map prompt -> name on the way
out.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path


def load_schema(path: Path):
    schema = json.loads(path.read_text())
    label_to_tag = {e["label"]: e["ontonotes"] for e in schema["entities"]}
    entity_labels = [e["label"] for e in schema["entities"]]
    prompt_to_name = {r["prompt"]: r["name"] for r in schema["relations"]}
    passes = schema.get("relation_passes", [])
    prompts_by_pass = {
        pass_name: [r["prompt"] for r in schema["relations"] if r["pass"] == pass_name]
        for pass_name in passes
    }
    all_prompts = [r["prompt"] for r in schema["relations"]]
    threshold = schema["threshold"]
    return entity_labels, prompts_by_pass, all_prompts, prompt_to_name, label_to_tag, threshold


def peak_rss_mb() -> float:
    # macOS getrusage reports ru_maxrss in bytes; Linux in KB. This project
    # targets macOS, but guard anyway.
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


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
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="fastino/gliner2.5-multi-v1")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all windows")
    ap.add_argument("--mode", default="multi-pass", choices=["one-pass", "multi-pass"],
                     help="multi-pass (default) mirrors scripts/run_coreml.py's entities-then-per-relation-pass split")
    ap.add_argument("--relation-threshold", type=float, default=None,
                    help="threshold for the relation passes only (default: schema `threshold`; entities always use the "
                         "schema threshold). 0.0 keeps every scored pair. Note: gliner2 gates per pair at this threshold "
                         "and runs dedup AFTER it, so lowering it can also shift which of two overlapping pairs "
                         "survives dedup, not just add pairs.")
    ap.add_argument("--arg-threshold", type=float, default=None,
                    help="relation_pair_generator argument_threshold (default 0.2 from the model config): minimum "
                         "head/tail entity-candidate score for a span to be proposed as a pair endpoint")
    ap.add_argument("--pair-cap", type=int, default=None,
                    help="relation_pair_generator pair_cap (default 64)")
    args = ap.parse_args()

    entity_labels, prompts_by_pass, all_prompts, prompt_to_name, label_to_tag, threshold = load_schema(args.schema)

    import torch
    from gliner2 import AutoExtractor

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[run_torch] mps requested but not available, falling back to cpu")
        device = "cpu"

    print(f"[run_torch] loading {args.repo} on device={device} ...")
    t0 = time.perf_counter()
    try:
        model = AutoExtractor.from_pretrained(args.repo, map_location=device)
    except Exception as e:  # noqa: BLE001 -- report mps misbehavior rather than crash silently
        if device == "mps":
            print(f"[run_torch] mps load failed ({e!r}); retrying on cpu")
            device = "cpu"
            model = AutoExtractor.from_pretrained(args.repo, map_location=device)
        else:
            raise
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[run_torch] loaded in {load_ms:.1f} ms")

    apply_pair_generator_overrides(model, args.arg_threshold, args.pair_cap)
    rel_threshold = threshold if args.relation_threshold is None else args.relation_threshold
    print(f"[run_torch] entity threshold={threshold} relation threshold={rel_threshold}")
    multi_pass = args.mode == "multi-pass"
    if multi_pass:
        ent_schema = model.create_schema().entities(entity_labels)
        rel_schemas = [
            (pass_name, model.create_schema().relations(prompts))
            for pass_name, prompts in prompts_by_pass.items()
        ]
    else:
        combined_schema = model.create_schema().entities(entity_labels).relations(all_prompts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latencies_ms = []
    total_words = 0
    n = 0
    mps_error = None

    with args.windows.open() as fin, args.out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]

            try:
                if multi_pass:
                    t0 = time.perf_counter()
                    ent_result = model.extract(
                        text, ent_schema, threshold=threshold,
                        include_confidence=True, include_spans=True,
                    )
                    relations_src = {}
                    for _pass_name, rel_schema in rel_schemas:
                        rel_result = model.extract(
                            text, rel_schema, threshold=rel_threshold,
                            include_confidence=True, include_spans=True,
                        )
                        relations_src.update(rel_result.get("relation_extraction", {}))
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    entities_src = ent_result.get("entities", {})
                else:
                    t0 = time.perf_counter()
                    result = model.extract(
                        text, combined_schema, threshold=threshold,
                        include_confidence=True, include_spans=True,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    entities_src = result.get("entities", {})
                    relations_src = result.get("relation_extraction", {})
            except Exception as e:  # noqa: BLE001
                if device == "mps" and mps_error is None:
                    mps_error = repr(e)
                    print(f"[run_torch] mps extract failed ({e!r}); this window skipped")
                    continue
                raise
            latencies_ms.append(elapsed_ms)
            total_words += len(text.split())

            entities = []
            for label, items in entities_src.items():
                tag = label_to_tag.get(label, label)
                for it in items:
                    entities.append({
                        "label": tag,
                        "text": it["text"],
                        "start": it["start"],
                        "end": it["end"],
                        "score": it.get("confidence", 1.0),
                    })

            relations = []
            for prompt, pairs in relations_src.items():
                name = prompt_to_name.get(prompt, prompt)
                for pair in pairs:
                    head, tail = pair["head"], pair["tail"]
                    score = min(head.get("confidence", 1.0), tail.get("confidence", 1.0))
                    relations.append({
                        "name": name,
                        "head": {"text": head["text"], "start": head["start"], "end": head["end"]},
                        "tail": {"text": tail["text"], "start": tail["start"], "end": tail["end"]},
                        "score": score,
                    })

            fout.write(json.dumps({
                "window_id": rec["window_id"],
                "entities": entities,
                "relations": relations,
                "latency_ms": elapsed_ms,
            }, ensure_ascii=False) + "\n")

            n += 1
            if args.limit and n >= args.limit:
                break

    latencies_ms.sort()

    def pct(p):
        if not latencies_ms:
            return 0.0
        idx = min(len(latencies_ms) - 1, round((len(latencies_ms) - 1) * p))
        return latencies_ms[idx]

    total_ms = sum(latencies_ms)
    words_per_sec = total_words / (total_ms / 1000.0) if total_ms else 0.0
    timing = {
        "engine": "torch",
        "device": device,
        "mode": args.mode,
        "repo": args.repo,
        "windows": n,
        "load_ms": load_ms,
        "p50_latency_ms": pct(0.5),
        "p90_latency_ms": pct(0.9),
        "total_words": total_words,
        "words_per_sec": words_per_sec,
        "peak_rss_mb": peak_rss_mb(),
        "mps_error": mps_error,
    }
    timing_path = args.out.with_suffix("").with_suffix(".timing.json")
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[run_torch] windows={n} load_ms={load_ms:.1f} p50_ms={timing['p50_latency_ms']:.2f} "
          f"words_per_sec={words_per_sec:.1f} peak_rss_mb={timing['peak_rss_mb']:.1f}")
    print(f"[run_torch] wrote {args.out} and {timing_path}")


if __name__ == "__main__":
    main()
