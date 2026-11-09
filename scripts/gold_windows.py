"""Gold windowing for the bucket-limited CoreML engines.

    split  eval/gold.jsonl -> data/gold-windows.jsonl   (same windowing as the corpus:
           gliner_test.windowing.window_chunk, budgets as in its main(), against a schema)
    merge  window-level predictions -> parent-level predictions (parent-relative offsets)
    stats  cross-window gold pair count

Window ids are "<parent>#<k>". Predictions on windows are merged back with
parent-relative char offsets; entities are deduped on (start, end, label) and
relations on (head span, tail span, name), keeping the max score; latency_ms is
summed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _jl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_split(args):
    from download_models import REVISION

    from gliner_test import windowing as W
    from gliner_test.compiled_models import CACHE_ROOT

    model_dir = args.model_dir or (CACHE_ROOT / REVISION)
    processor, collator, Schema = W.load_runtime_pieces(model_dir)
    cfg = json.loads(args.schema.read_text())
    ent_schema = Schema().entities([e["label"] for e in cfg["entities"]]).build()
    # budgets exactly as windowing.main(): tighter of the two schema prompts + safety margin.
    # NB: build_schemas() flattens all relations into one schema (valid only for <=4 relations), so
    # with a multi-pass schema we use the worst single pass instead and re-validate every pass.
    ent_tokens = collator([("", ent_schema)]).input_ids.shape[1]
    pass_names = cfg.get("relation_passes") or ["default"]
    rel_tokens = 0
    pass_schemas = []
    for pn in pass_names:
        rels = [r for r in cfg["relations"] if r.get("pass", "default") == pn]
        if not rels:
            continue
        if any(r.get("description_short") for r in rels):  # experimental: prompt -> short description
            s = Schema().relations({r.get("prompt", r["name"]): (r.get("description_short") or "") for r in rels}).build()
        else:
            s = Schema().relations([r.get("prompt", r["name"]) for r in rels]).build()
        pass_schemas.append(s)
        rel_tokens = max(rel_tokens, collator([("", s)]).input_ids.shape[1])
    budget = W.TOTAL_BUDGET - max(int(ent_tokens), int(rel_tokens)) - W.SCHEMA_SAFETY_MARGIN
    print(f"entity prompt={ent_tokens} worst relation pass={rel_tokens} text token budget={budget}", file=sys.stderr)

    worst = max(pass_schemas, key=lambda s: collator([("", s)]).input_ids.shape[1])

    def windows_for(text):
        b = budget
        while True:
            wins = W.window_chunk(text, processor, collator, ent_schema, worst, b)
            if all(W.fits_bucket(collator, ps, t) for ps in pass_schemas for _, _, t in wins):
                return wins
            b -= 1

    n_items = n_win = n_multi = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for g in _jl(args.gold):
            n_items += 1
            wins = windows_for(g["text"])
            if len(wins) > 1:
                n_multi += 1
            for k, (s, e, t) in enumerate(wins):
                n_win += 1
                fout.write(json.dumps({"window_id": f"{g['window_id']}#{k}", "parent_id": g["window_id"],
                                       "char_start": s, "char_end": e, "text": t}, ensure_ascii=False) + "\n")
    print(f"items={n_items} windows={n_win} multi-window items={n_multi} -> {args.out}", file=sys.stderr)


def merge_records(preds, windows):
    """preds: window-level records; windows: {window_id: {parent_id, char_start}} -> {parent: rec}."""
    by_parent = {}
    for r in preds:
        w = windows[r["window_id"]]
        base = w["char_start"]
        P = by_parent.setdefault(w["parent_id"], {"window_id": w["parent_id"], "entities": {}, "relations": {},
                                                  "latency_ms": 0.0, "n_windows": 0})
        P["latency_ms"] += r.get("latency_ms", 0.0)
        P["n_windows"] += 1
        for e in r.get("entities", []):
            k = (e["start"] + base, e["end"] + base, e["label"])
            if k not in P["entities"] or e["score"] > P["entities"][k]["score"]:
                P["entities"][k] = {**e, "start": e["start"] + base, "end": e["end"] + base}
        for x in r.get("relations", []):
            h = {**x["head"], "start": x["head"]["start"] + base, "end": x["head"]["end"] + base}
            t = {**x["tail"], "start": x["tail"]["start"] + base, "end": x["tail"]["end"] + base}
            k = (h["start"], h["end"], t["start"], t["end"], x["name"])
            if k not in P["relations"] or x["score"] > P["relations"][k]["score"]:
                P["relations"][k] = {**x, "head": h, "tail": t}
    out = {}
    for pid, P in by_parent.items():
        out[pid] = {"window_id": pid, "entities": list(P["entities"].values()),
                    "relations": list(P["relations"].values()), "latency_ms": P["latency_ms"]}
    return out


def cmd_merge(args):
    windows = {w["window_id"]: w for w in _jl(args.windows)}
    merged = merge_records(_jl(args.preds), windows)
    gold_order = [g["window_id"] for g in _jl(args.gold)] if args.gold.exists() else list(merged)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pid in gold_order:
            if pid in merged:
                f.write(json.dumps(merged[pid], ensure_ascii=False) + "\n")
    print(f"merged {len(merged)} parents -> {args.out}", file=sys.stderr)


def cmd_stats(args):
    wins = {}
    for w in _jl(args.windows):
        wins.setdefault(w["parent_id"], []).append(w)
    tot = cross = 0
    by_cat = {}
    for g in _jl(args.gold):
        for r in g.get("relations", []):
            lo = min(r["head"]["start"], r["tail"]["start"])
            hi = max(r["head"]["end"], r["tail"]["end"])
            ok = any(w["char_start"] <= lo and hi <= w["char_end"] for w in wins.get(g["window_id"], []))
            c = by_cat.setdefault(r.get("category"), [0, 0])
            c[0] += 1
            tot += 1
            if not ok:
                cross += 1
                c[1] += 1
    print(f"gold relations (pair-level approx, relation rows) {tot}; no single window holds both endpoints: {cross}")
    for k, (a, b) in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {b}/{a}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    s.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    s.add_argument("--model-dir", type=Path, default=None)
    s.add_argument("--out", type=Path, default=Path("data/gold-windows.jsonl"))
    s.set_defaults(fn=cmd_split)
    m = sub.add_parser("merge")
    m.add_argument("--preds", type=Path, required=True)
    m.add_argument("--windows", type=Path, default=Path("data/gold-windows.jsonl"))
    m.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(fn=cmd_merge)
    t = sub.add_parser("stats")
    t.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    t.add_argument("--windows", type=Path, default=Path("data/gold-windows.jsonl"))
    t.set_defaults(fn=cmd_stats)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
