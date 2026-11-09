"""GLiNER2.5 JointIE reference run, for the relation-decoding comparison in
docs/cross-validation.md.

`gliner2.joint_ie.JointIE` (same fastino/gliner2.5-multi-v1 checkpoint as
scripts/run_torch.py) scores typed relation *signatures*
(`.relation(name, head_type, tail_type, ...)`) and searches a globally
consistent graph (unique-head caps, no-self-loop, etc. are opt-in; this
script uses the library defaults -- see docs/references/gliner2-tutorial-15-joint-ie.md).
This is a genuinely different decoding path from independent
`extract_entities` + `extract_relations` (scripts/run_torch.py): JointIE adds
typed endpoints and a graph search on top of the same per-pair relation
scorer (gliner2/models/boundary/relations.py).

Entities and relations are requested together in ONE call per window --
JointIE's schema takes a relation's full head/tail type LIST directly
(`.relation(prompt, head=[...], tail=[...], symmetric=...)`; confirmed in
gliner2/joint_ie/schema.py:78, `_types()` accepts a string or an iterable),
so there is no need for gliner25-rs/run_coreml.py's per-pass splitting here.

Writes the shared prediction format, like scripts/run_torch.py.

Usage:
    uv run python scripts/run_torch_jointie.py \
        --schema schema/ontonotes.json \
        --windows eval/gold.jsonl \
        --out data/preds/gold/torch-jointie.jsonl
"""
import argparse
import json
import resource
import time
from pathlib import Path


def load_schema(path: Path):
    schema = json.loads(path.read_text())
    label_to_tag = {e["label"]: e["ontonotes"] for e in schema["entities"]}
    tag_to_label = {e["ontonotes"]: e["label"] for e in schema["entities"]}
    entity_labels = [e["label"] for e in schema["entities"]]
    # JointSchema.relation(head=..., tail=...) takes entity TYPE NAMES as
    # declared in .entities(...) -- i.e. the gliner prompt `label` strings
    # ("person"), not the OntoNotes tags ("PERSON") schema/ontonotes.json
    # stores in relations[].head/.tail. Convert once here.
    relations = []
    for r in schema["relations"]:
        relations.append({
            **r,
            "head_label_map": [tag_to_label[t] for t in r["head"]],
            "tail_label_map": [tag_to_label[t] for t in r["tail"]],
        })
    prompt_to_name = {r["prompt"]: r["name"] for r in relations}
    threshold = schema["threshold"]
    return entity_labels, relations, prompt_to_name, label_to_tag, threshold


def peak_rss_mb() -> float:
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="fastino/gliner2.5-multi-v1")
    ap.add_argument("--limit", type=int, default=0, help="0 = all windows")
    ap.add_argument("--optimizer", default="beam", choices=["beam", "greedy", "auto"])
    ap.add_argument("--include-symmetric", action="store_true",
                     help="also try the 3 PERSON<->PERSON symmetric relations, known to always come back empty (see comments)")
    ap.add_argument("--symmetric-mode", choices=["skip", "undirected"], default="skip",
                    help="skip (default): PERSON<->PERSON symmetric relations are not run (unless --include-symmetric). "
                         "undirected: run them as `.relation(name, h, t, directed=False)` WITHOUT symmetric=True, which "
                         "avoids the conflicting constraints that make symmetric=True always return empty in gliner2 2.0.0 "
                         "(joint_ie/compiler.py:36,40; optimizers/base.py:139-164; constraints.py:136-139). "
                         "All symmetric relations go into one extra schema/call.")
    args = ap.parse_args()

    entity_labels, relations, prompt_to_name, label_to_tag, threshold = load_schema(args.schema)

    from gliner2.joint_ie import JointIE, JointIEConfig

    print(f"[run_torch_jointie] loading {args.repo} (cpu) ...")
    t0 = time.perf_counter()
    joint = JointIE.from_pretrained(args.repo)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[run_torch_jointie] loaded in {load_ms:.1f} ms")

    # entity threshold mirrors the shared schema threshold; relation typing
    # and symmetry come straight from schema/ontonotes.json's head/tail
    # OntoNotes-type lists -- JointIE is the ONE engine in this repo that
    # actually consumes that typing, rather than it being a post-filter.
    #
    # KNOWN LIBRARY LIMITATION, not ours to fix: a symmetric relation whose
    # head and tail are the SAME single entity type (family, business_associate,
    # personal_relation -- all PERSON<->PERSON here) makes JointIE's decoder
    # return an EMPTY solution every time, confirmed reproducible:
    #  - declared together: fails on any sentence where only those 3 types
    #    have valid candidates ("greedy joint-IE decoding produced a
    #    constraint-violating assignment; returning empty solution").
    #  - declared ALONE, one relation per schema (tried as a workaround):
    #    still empty, every time, on all three types, on a minimal
    #    single-sentence repro ("Barack Obama and Joe Biden remained close
    #    friends after leaving office." -> 0 relations under optimizer=beam,
    #    0 entities AND 0 relations, i.e. fully infeasible, under
    #    optimizer=greedy). So this is not a co-declaration conflict, it is
    #    PERSON<->PERSON symmetric relations specifically. See
    #    docs/cross-validation.md for the full writeup; not investigated
    #    further inside gliner2.joint_ie internals (out of scope here).
    # Default: only run the 5 non-symmetric relations (one schema/call per
    # window); --include-symmetric re-adds the 3 PERSON<->PERSON ones (one
    # extra schema/call each) to re-confirm they stay empty, at ~4x the cost.
    asymmetric = [r for r in relations if not r.get("symmetric", False)]
    symmetric = [r for r in relations if r.get("symmetric", False)] if args.include_symmetric else []

    def build_schema(rels, undirected=False):
        s = joint.create_schema().entities({label: {"threshold": threshold} for label in entity_labels})
        for r in rels:
            if undirected:
                s = s.relation(
                    r["prompt"], head=r["head_label_map"], tail=r["tail_label_map"],
                    directed=False, threshold=threshold,
                )
            else:
                s = s.relation(
                    r["prompt"], head=r["head_label_map"], tail=r["tail_label_map"],
                    symmetric=r.get("symmetric", False), threshold=threshold,
                )
        return s

    schema_groups = [("asymmetric", build_schema(asymmetric))]
    schema_groups += [(r["name"], build_schema([r])) for r in symmetric]
    if args.symmetric_mode == "undirected":
        sym_all = [r for r in relations if r.get("symmetric", False)]
        schema_groups.append(("symmetric-undirected", build_schema(sym_all, undirected=True)))

    config = JointIEConfig(optimizer=args.optimizer)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latencies_ms = []
    total_words = 0
    n = 0
    infeasible = 0

    with args.windows.open() as fin, args.out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]

            t0 = time.perf_counter()
            entities_by_span = {}  # dedupe across the split calls: same text, same entities
            relations_out = []
            any_infeasible = False
            for _group_name, schema_obj in schema_groups:
                result = joint.extract(text, schema_obj, config=config)
                if not result.feasible:
                    any_infeasible = True
                for e in result.entities:
                    tag = label_to_tag.get(e.type, e.type)
                    entities_by_span[(e.start, e.end, tag)] = {
                        "label": tag, "text": e.text, "start": e.start, "end": e.end,
                        "score": e.confidence if e.confidence is not None else 1.0,
                    }
                for rel in result.relations:
                    head = result.entity(rel.head)
                    tail = result.entity(rel.tail)
                    name = prompt_to_name.get(rel.type, rel.type)
                    relations_out.append({
                        "name": name,
                        "head": {"text": head.text, "start": head.start, "end": head.end},
                        "tail": {"text": tail.text, "start": tail.start, "end": tail.end},
                        "score": rel.confidence if rel.confidence is not None else 1.0,
                    })
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            total_words += len(text.split())
            if any_infeasible:
                infeasible += 1
            entities = list(entities_by_span.values())

            fout.write(json.dumps({
                "window_id": rec["window_id"],
                "entities": entities,
                "relations": relations_out,
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
        "engine": "torch-jointie",
        "optimizer": args.optimizer,
        "repo": args.repo,
        "windows": n,
        "infeasible": infeasible,
        "calls_per_window": len(schema_groups),
        "latency_note": (
            f"sum of {len(schema_groups)} extract() calls per window (1 asymmetric-relations "
            "schema + 1 per symmetric relation, see the WORKAROUND comment in this file) -- "
            "not a native single-call JointIE latency"
        ),
        "load_ms": load_ms,
        "p50_latency_ms": pct(0.5),
        "p90_latency_ms": pct(0.9),
        "total_words": total_words,
        "words_per_sec": words_per_sec,
        "peak_rss_mb": peak_rss_mb(),
    }
    timing_path = args.out.with_suffix("").with_suffix(".timing.json")
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[run_torch_jointie] windows={n} infeasible={infeasible} load_ms={load_ms:.1f} "
          f"p50_ms={timing['p50_latency_ms']:.2f} words_per_sec={words_per_sec:.1f} "
          f"peak_rss_mb={timing['peak_rss_mb']:.1f}")
    print(f"[run_torch_jointie] wrote {args.out} and {timing_path}")


if __name__ == "__main__":
    main()
