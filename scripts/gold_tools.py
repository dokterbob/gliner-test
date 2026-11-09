#!/usr/bin/env python3
"""Tools for building and validating gold v3 (`eval/gold.jsonl`). Stdlib only.

Subcommands (run with ``uv run python scripts/gold_tools.py <cmd> ...``):

  items      build annotation items + batches from gold-v2.1 texts and the passages
  parse      parse one raw annotator output, align strings to char offsets, flag problems
  agreement  inter-annotator agreement between two parsed outputs (A1 vs A2)
  merge      merge A1/A2 aligned outputs: agreed items + keyed disagreements
  show       print disagreements
  adjudicate apply a decisions file to the merged file
  pool-cands extract pooling candidates from a prediction file against adjudicated gold
  build      turn the adjudicated file into eval/gold.jsonl (adds kind, split, categories)
  validate   check a gold file

Annotator output schema (strict JSONL, one object per text; see docs/annotation-guidelines.md s7):

  {"id": ..., "entities": [{"text", "occurrence", "type"}],
   "pairs": [{"category", "head": {"text","occurrence"}, "tail": {...}, "qualifiers": [...]}]}

Alignment: ``occurrence`` is the 0-based index of ``text`` among all its occurrences in the passage,
counting overlapping matches (search restarts at idx+1). Anything that cannot be aligned is
reported in ``issues`` and dropped from the aligned record.

Split method (``assign_splits``): items are grouped by (lang, sorted set of pair categories present,
or "none"); within each group items are sorted by window_id, shuffled with ``random.Random(SEED)``
(one RNG, groups visited in sorted key order) and assigned dev, test, dev, test, ...; the starting
side alternates per group so odd-sized groups balance out. SEED is fixed below.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = ROOT / "schema" / "ontonotes.json"
SEED = 20261109
MAX_PASSAGE_WORDS = 64
ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "DATE", "TIME"]
# canonical qualifier order per category (primary qualifier = first that holds)
QUAL_ORDER = [
    "employee_of", "founder_of", "member_of", "subsidiary_of",
    "located_in", "resident_of",
    "family", "business_associate", "personal_relation",
]
UNDIRECTED = {"per_per", "org_org", "per_org"}


def load_categories() -> dict:
    schema = json.loads(SCHEMA.read_text())
    return {c["id"]: c for c in schema["pair_categories"]}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- alignment

def find_occurrence(text: str, needle: str, occurrence: int) -> int:
    """Char index of the occurrence-th (0-based, overlaps counted) match, or -1."""
    if not needle:
        return -1
    idx, n = -1, -1
    while True:
        idx = text.find(needle, idx + 1)
        if idx < 0:
            return -1
        n += 1
        if n == occurrence:
            return idx


def extract_json_lines(raw: str) -> list[dict]:
    """Pull JSON objects out of an annotator message (tolerates fences / prose)."""
    out = []
    for line in raw.splitlines():
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"_unparsed": line})
    return out


def align_record(rec: dict, text: str, categories: dict) -> tuple[dict, list[str]]:
    """Align one annotator record to offsets. Returns (aligned, issues)."""
    issues: list[str] = []
    ents: dict[tuple[str, int], dict] = {}
    for e in rec.get("entities", []):
        s, occ, typ = e.get("text"), int(e.get("occurrence", 0) or 0), e.get("type")
        if typ not in ENTITY_TYPES:
            issues.append(f"bad entity type {typ!r} for {s!r}")
            continue
        i = find_occurrence(text, s or "", occ)
        if i < 0:
            issues.append(f"entity not found: {s!r} occ={occ}")
            continue
        key = (s, occ)
        if key in ents:
            issues.append(f"duplicate entity {s!r} occ={occ}")
            continue
        ents[key] = {"label": typ, "text": s, "start": i, "end": i + len(s)}
    elist = sorted(ents.values(), key=lambda e: (e["start"], e["end"]))
    for a in range(len(elist)):
        for b in range(a + 1, len(elist)):
            x, y = elist[a], elist[b]
            if y["start"] < x["end"] and x["start"] < y["end"]:
                issues.append(f"overlapping entities {x['text']!r} / {y['text']!r}")
    pairs = []
    for p in rec.get("pairs", []):
        cat = p.get("category")
        if cat not in categories:
            issues.append(f"bad category {cat!r}")
            continue
        ends = []
        for side in ("head", "tail"):
            ref = p.get(side) or {}
            key = (ref.get("text"), int(ref.get("occurrence", 0) or 0))
            if key not in ents:
                issues.append(f"{cat} {side} not an annotated entity: {key}")
                break
            ends.append(ents[key])
        else:
            quals = [q for q in p.get("qualifiers", []) if q in categories[cat]["qualifiers"]]
            bad = [q for q in p.get("qualifiers", []) if q not in categories[cat]["qualifiers"]]
            if bad:
                issues.append(f"qualifiers {bad} not in category {cat}")
            if not quals:
                issues.append(f"{cat} pair with no valid qualifier: {ends[0]['text']!r}/{ends[1]['text']!r}")
                continue
            if ends[0] is ends[1]:
                issues.append(f"self pair {ends[0]['text']!r}")
                continue
            pairs.append({"category": cat, "head": ends[0], "tail": ends[1], "qualifiers": quals})
    return {"id": rec["id"], "entities": elist, "pairs": pairs}, issues


def cmd_parse(args) -> None:
    categories = load_categories()
    items = {r["id"]: r for r in read_jsonl(args.items)}
    recs = extract_json_lines(Path(args.raw).read_text())
    out, all_issues, seen = [], {}, set()
    for rec in recs:
        if "_unparsed" in rec or "id" not in rec or rec["id"] not in items:
            all_issues.setdefault("_global", []).append(f"unusable line: {str(rec)[:120]}")
            continue
        if rec["id"] in seen:
            all_issues.setdefault(rec["id"], []).append("duplicate id in output")
            continue
        seen.add(rec["id"])
        al, iss = align_record(rec, items[rec["id"]]["text"], categories)
        out.append(al)
        if iss:
            all_issues[rec["id"]] = iss
    if args.batch:
        want = {r["id"] for r in read_jsonl(args.batch)}
        for miss in sorted(want - seen):
            all_issues.setdefault(miss, []).append("missing from output")
    write_jsonl(args.out, out)
    Path(str(args.out) + ".issues.json").write_text(json.dumps(all_issues, indent=1, ensure_ascii=False))
    print(f"{args.raw}: {len(out)} records, {sum(len(v) for v in all_issues.values())} issues in {len(all_issues)} items")


# ---------------------------------------------------------------- agreement

def span(e) -> tuple[int, int]:
    return (e["start"], e["end"])


def pair_key(p: dict, with_quals: bool = False):
    h, t = span(p["head"]), span(p["tail"])
    k = (p["category"], (h, t) if p["category"] not in UNDIRECTED else frozenset((h, t)))
    return k + (tuple(sorted(p["qualifiers"])),) if with_quals else k


def prf(tp: int, np_: int, ng: int) -> dict:
    p = tp / np_ if np_ else (1.0 if ng == 0 else 0.0)
    r = tp / ng if ng else (1.0 if np_ == 0 else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"f1": round(f, 4), "precision": round(p, 4), "recall": round(r, 4), "n_a": ng, "n_b": np_, "tp": tp}


def agreement(a_recs: list[dict], b_recs: list[dict]) -> dict:
    A = {r["id"]: r for r in a_recs}
    B = {r["id"]: r for r in b_recs}
    ids = sorted(set(A) & set(B))
    ent_tp = ent_a = ent_b = 0
    sp_tp = 0
    pair_c = defaultdict(lambda: [0, 0, 0])  # tp, nb, na
    pairq_c = defaultdict(lambda: [0, 0, 0])
    for i in ids:
        ea = {(span(e), e["label"]) for e in A[i]["entities"]}
        eb = {(span(e), e["label"]) for e in B[i]["entities"]}
        ent_tp += len(ea & eb); ent_a += len(ea); ent_b += len(eb)
        sp_tp += len({s for s, _ in ea} & {s for s, _ in eb})
        for with_q, store in ((False, pair_c), (True, pairq_c)):
            ka = {pair_key(p, with_q) for p in A[i]["pairs"]}
            kb = {pair_key(p, with_q) for p in B[i]["pairs"]}
            cats = {k[0] for k in ka | kb}
            for c in cats | {"all"}:
                sa = {k for k in ka if c == "all" or k[0] == c}
                sb = {k for k in kb if c == "all" or k[0] == c}
                store[c][0] += len(sa & sb); store[c][1] += len(sb); store[c][2] += len(sa)
    return {
        "n_items": len(ids),
        "entity_strict": prf(ent_tp, ent_b, ent_a),
        "entity_span_only": prf(sp_tp, sum(len(B[i]["entities"]) for i in ids), sum(len(A[i]["entities"]) for i in ids)),
        "pair": {c: prf(*v) for c, v in sorted(pair_c.items())},
        "pair_with_qualifiers": {c: prf(*v) for c, v in sorted(pairq_c.items())},
    }


def cmd_agreement(args) -> None:
    res = agreement(read_jsonl(args.a), read_jsonl(args.b))
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- items / splits

def make_items(old_gold: Path, passages: Path) -> list[dict]:
    items = []
    for r in read_jsonl(old_gold):
        items.append({"id": r["window_id"], "lang": r["lang"], "kind": "sentence", "text": r["text"]})
    for r in read_jsonl(passages):
        items.append({"id": "gold-p-" + r["id"], "lang": r["id"].split("-")[0], "kind": "passage",
                      "ptype": r.get("kind"), "text": r["text"]})
    return items


def cmd_items(args) -> None:
    items = make_items(Path(args.old_gold), Path(args.passages))
    write_jsonl(args.out, items)
    rng = random.Random(SEED)
    order = items[:]
    rng.shuffle(order)  # batches mix languages / kinds
    n = args.batches
    outdir = Path(args.batch_dir)
    for b in range(n):
        chunk = order[b::n]
        write_jsonl(outdir / f"batch-{b + 1}.jsonl", [{"id": r["id"], "text": r["text"]} for r in chunk])
        print(f"batch-{b + 1}: {len(chunk)} items")
    print(f"{len(items)} items total")


def assign_splits(rows: list[dict], cats_of) -> dict[str, str]:
    groups = defaultdict(list)
    for r in rows:
        key = (r["lang"], tuple(sorted(cats_of(r))) or ("none",))
        groups[key].append(r["window_id"])
    rng = random.Random(SEED)
    out, flip = {}, 0
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        for j, wid in enumerate(ids):
            out[wid] = "dev" if (j + flip) % 2 == 0 else "test"
        if len(ids) % 2:
            flip += 1
    return out


# ---------------------------------------------------------------- build

def build_gold(adjudicated: list[dict], categories: dict) -> list[dict]:
    rows = []
    for a in adjudicated:
        ents = [{"label": e["label"], "text": e["text"], "start": e["start"], "end": e["end"]}
                for e in sorted(a["entities"], key=lambda e: (e["start"], e["end"]))]
        typ = {(e["start"], e["end"]): e["label"] for e in ents}
        rels = []
        for p in a["pairs"]:
            h, t = p["head"], p["tail"]
            cat = p["category"]
            if cat == "per_org" and typ[(h["start"], h["end"])] != "PERSON":
                h, t = t, h
            elif cat in UNDIRECTED and cat != "per_org" and (t["start"], t["end"]) < (h["start"], h["end"]):
                h, t = t, h
            quals = sorted(set(p["qualifiers"]), key=QUAL_ORDER.index)
            ep = lambda x: {"text": x["text"], "start": x["start"], "end": x["end"], "type": typ[(x["start"], x["end"])]}
            rels.append({"name": quals[0], "category": cat, "head": ep(h), "tail": ep(t),
                         "qualifiers": quals, "source": p.get("source", "annotator")})
        rels.sort(key=lambda r: (r["head"]["start"], r["tail"]["start"], r["name"]))
        rows.append({"window_id": a["id"], "chunk_id": None, "lang": a["lang"], "kind": a["kind"],
                     "text": a["text"], "entities": ents, "relations": rels})
    splits = assign_splits(rows, lambda r: {x["category"] for x in r["relations"]})
    for r in rows:
        r["split"] = splits[r["window_id"]]
    rows.sort(key=lambda r: (r["kind"] != "sentence", r["window_id"]))
    return rows


def cmd_build(args) -> None:
    rows = build_gold(read_jsonl(args.adjudicated), load_categories())
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} items to {args.out}")


# ---------------------------------------------------------------- validate

def validate(rows: list[dict]) -> list[str]:
    cats = load_categories()
    errs: list[str] = []
    seen = set()
    for r in rows:
        w = r.get("window_id")
        if w in seen:
            errs.append(f"{w}: duplicate window_id")
        seen.add(w)
        text = r["text"]
        for k in ("kind", "split", "lang"):
            if k not in r:
                errs.append(f"{w}: missing {k}")
        if r.get("kind") not in ("sentence", "passage"):
            errs.append(f"{w}: bad kind {r.get('kind')!r}")
        if r.get("split") not in ("dev", "test"):
            errs.append(f"{w}: bad split {r.get('split')!r}")
        if r.get("kind") == "passage" and len(text.split()) > MAX_PASSAGE_WORDS:
            errs.append(f"{w}: passage has {len(text.split())} words > {MAX_PASSAGE_WORDS}")
        spans = {}
        for e in r["entities"]:
            if text[e["start"]:e["end"]] != e["text"]:
                errs.append(f"{w}: entity offset mismatch {e}")
            if e["label"] not in ENTITY_TYPES:
                errs.append(f"{w}: bad entity label {e['label']}")
            spans[(e["start"], e["end"])] = e["label"]
        es = sorted(r["entities"], key=lambda e: e["start"])
        for a, b in zip(es, es[1:]):
            if b["start"] < a["end"]:
                errs.append(f"{w}: overlapping entities {a['text']!r} / {b['text']!r}")
        pairs_seen = set()
        for x in r["relations"]:
            for side in ("head", "tail"):
                ep = x[side]
                if text[ep["start"]:ep["end"]] != ep["text"]:
                    errs.append(f"{w}: relation {side} offset mismatch {ep}")
                lab = spans.get((ep["start"], ep["end"]))
                if lab is None:
                    errs.append(f"{w}: relation {side} {ep['text']!r} is not a gold entity span")
                elif ep.get("type") != lab:
                    errs.append(f"{w}: relation {side} type {ep.get('type')} != entity label {lab}")
            c = cats.get(x.get("category"))
            if c is None:
                errs.append(f"{w}: unknown category {x.get('category')!r}")
                continue
            ht, tt = x["head"].get("type"), x["tail"].get("type")
            if ht not in c["head"] or tt not in c["tail"]:
                errs.append(f"{w}: {x['category']} endpoint types {ht}->{tt} not allowed")
            if not x.get("qualifiers") or any(q not in c["qualifiers"] for q in x["qualifiers"]):
                errs.append(f"{w}: qualifiers {x.get('qualifiers')} not in category {x['category']}")
            elif x.get("name") != x["qualifiers"][0]:
                errs.append(f"{w}: name {x.get('name')!r} != first qualifier {x['qualifiers'][0]!r}")
            if "resident_of" in x.get("qualifiers", []) and ht != "PERSON":
                errs.append(f"{w}: resident_of needs a PERSON head")
            if "located_in" in x.get("qualifiers", []) and ht == "PERSON":
                errs.append(f"{w}: located_in with PERSON head (use resident_of)")
            if x.get("source") not in ("annotator", "pool"):
                errs.append(f"{w}: bad source {x.get('source')!r}")
            hs, ts = (x["head"]["start"], x["head"]["end"]), (x["tail"]["start"], x["tail"]["end"])
            if hs == ts:
                errs.append(f"{w}: self pair")
            k = (x["category"], (hs, ts) if x["category"] not in UNDIRECTED else frozenset((hs, ts)))
            if k in pairs_seen:
                errs.append(f"{w}: duplicate pair {x['head']['text']!r}/{x['tail']['text']!r}")
            pairs_seen.add(k)
            if x["category"] == "per_org" and ht != "PERSON":
                errs.append(f"{w}: per_org head must be PERSON")
    return errs


def cmd_validate(args) -> None:
    rows = read_jsonl(args.gold)
    errs = validate(rows)
    n_rel = sum(len(r["relations"]) for r in rows)
    print(f"{args.gold}: {len(rows)} items, {n_rel} relations, {len(errs)} errors")
    for e in errs[:200]:
        print("  ERROR", e)
    sys.exit(1 if errs else 0)


# ---------------------------------------------------------------- merge / adjudicate

def ekey(e) -> str:
    return f"E:{e['start']}-{e['end']}:{e['label']}"


def pkey(p) -> str:
    h, t = span(p["head"]), span(p["tail"])
    if p["category"] in UNDIRECTED and t < h:
        h, t = t, h
    return f"P:{p['category']}:{h[0]}-{h[1]}:{t[0]}-{t[1]}"


def merge_pair_sets(A: dict, B: dict, items: dict) -> dict:
    """Merge two aligned annotator records for one item.

    Agreed entities / pairs are accepted automatically. Everything else is a *disagreement* with a
    stable key (``ekey``/``pkey``; qualifier-only differences get ``Q:<pkey>:<qualifier>``), which the
    adjudication decisions file refers to.
    """
    ea = {(e["start"], e["end"], e["label"]): e for e in A["entities"]}
    eb = {(e["start"], e["end"], e["label"]): e for e in B["entities"]}
    agreed_e = [ea[k] for k in sorted(ea) if k in eb]
    dis = {}
    for k in sorted(set(ea) ^ set(eb)):
        e = ea.get(k) or eb[k]
        dis[ekey(e)] = {"kind": "entity", "by": ["A1" if k in ea else "A2"], "entity": e}
    pa = {pkey(p): p for p in A["pairs"]}
    pb = {pkey(p): p for p in B["pairs"]}
    agreed_p = []
    for k in sorted(set(pa) | set(pb)):
        if k in pa and k in pb:
            qa, qb = set(pa[k]["qualifiers"]), set(pb[k]["qualifiers"])
            # same endpoints; directed pairs must also agree on direction (pkey is order-sensitive for place)
            base = pa[k]
            agreed_p.append({**base, "qualifiers": sorted(qa & qb, key=QUAL_ORDER.index)})
            for q in sorted(qa ^ qb):
                dis[f"Q:{k}:{q}"] = {"kind": "qualifier", "by": ["A1" if q in qa else "A2"], "pair": base, "qualifier": q}
        else:
            p = pa.get(k) or pb[k]
            dis[k] = {"kind": "pair", "by": ["A1" if k in pa else "A2"], "pair": p}
    return {"id": A["id"], "entities": agreed_e, "pairs": agreed_p, "disagreements": dis}


def cmd_merge(args) -> None:
    items = {r["id"]: r for r in read_jsonl(args.items)}
    A = {}
    B = {}
    for f in args.a:
        A.update({r["id"]: r for r in read_jsonl(f)})
    for f in args.b:
        B.update({r["id"]: r for r in read_jsonl(f)})
    out = []
    for i in items:
        m = merge_pair_sets(A[i], B[i], items)
        m.update({"lang": items[i]["lang"], "kind": items[i]["kind"], "text": items[i]["text"]})
        out.append(m)
    write_jsonl(args.out, out)
    n = sum(len(m["disagreements"]) for m in out)
    print(f"{len(out)} items merged, {n} disagreements in {sum(1 for m in out if m['disagreements'])} items")


def apply_decisions(merged: list[dict], decisions: dict, extra_pairs: dict | None = None) -> list[dict]:
    """Build adjudicated records. ``decisions``: {id: {"accept": [keys], "drop_agreed": [keys], "add": {...}}}.

    accept       disagreement keys to accept (entities, pairs, qualifier additions)
    drop_agreed  agreed entity/pair keys to remove (adjudicator overrides both annotators)
    add_pairs    list of {category, head:{start,end}, tail:{start,end}, qualifiers, source?} to add directly
    """
    out = []
    for m in merged:
        d = decisions.get(m["id"], {})
        acc = set(d.get("accept", []))
        drop = set(d.get("drop_agreed", []))
        ents = {(e["start"], e["end"]): e for e in m["entities"] if ekey(e) not in drop}
        pairs = {pkey(p): {**p, "source": "annotator"} for p in m["pairs"] if pkey(p) not in drop}
        for k, v in m["disagreements"].items():
            if k not in acc:
                continue
            if v["kind"] == "entity":
                ents[(v["entity"]["start"], v["entity"]["end"])] = v["entity"]
            elif v["kind"] == "pair":
                pairs[k] = {**v["pair"], "source": "annotator"}
            else:
                pk = k[2:].rsplit(":", 1)[0]
                if pk in pairs:
                    pairs[pk]["qualifiers"] = sorted(set(pairs[pk]["qualifiers"]) | {v["qualifier"]}, key=QUAL_ORDER.index)
                else:
                    pairs[pk] = {**v["pair"], "qualifiers": [v["qualifier"]], "source": "annotator"}
        for p in d.get("add_pairs", []) + (extra_pairs or {}).get(m["id"], []):
            for side in ("head", "tail"):
                for e in m["entities"] + [x["entity"] for x in m["disagreements"].values() if x["kind"] == "entity"]:
                    if (e["start"], e["end"]) == (p[side]["start"], p[side]["end"]):
                        ents.setdefault((e["start"], e["end"]), e)
                        p[side] = e
            pairs[pkey(p)] = {**p, "source": p.get("source", "annotator")}
        # pairs need their endpoint entities (take them from either annotator's disagreement list)
        pool = {(x["entity"]["start"], x["entity"]["end"]): x["entity"] for x in m["disagreements"].values() if x["kind"] == "entity"}
        final_pairs = []
        for p in pairs.values():
            if not p["qualifiers"]:
                continue
            for side in ("head", "tail"):
                sp = (p[side]["start"], p[side]["end"])
                if sp not in ents and sp in pool:
                    ents[sp] = pool[sp]
            final_pairs.append(p)
        out.append({"id": m["id"], "lang": m["lang"], "kind": m["kind"], "text": m["text"],
                    "entities": sorted(ents.values(), key=lambda e: (e["start"], e["end"])), "pairs": final_pairs})
    return out


def cmd_adjudicate(args) -> None:
    merged = read_jsonl(args.merged)
    decisions = json.loads(Path(args.decisions).read_text()) if Path(args.decisions).exists() else {}
    extra = json.loads(Path(args.extra).read_text()) if args.extra and Path(args.extra).exists() else None
    out = apply_decisions(merged, decisions, extra)
    write_jsonl(args.out, out)
    print(f"{len(out)} items, {sum(len(r['pairs']) for r in out)} pairs -> {args.out}")


def cmd_show(args) -> None:
    """Print disagreements readably for adjudication."""
    for m in read_jsonl(args.merged):
        if not m["disagreements"]:
            continue
        print(f"### {m['id']} [{m['lang']}] {m['text']}")
        for k, v in m["disagreements"].items():
            if v["kind"] == "entity":
                e = v["entity"]
                print(f"  {','.join(v['by'])} only  {k}  {e['text']!r}")
            else:
                p = v["pair"]
                print(f"  {','.join(v['by'])} only  {k}  {p['head']['text']!r} -> {p['tail']['text']!r} {p['qualifiers'] if v['kind']=='pair' else v['qualifier']}")


# ---------------------------------------------------------------- pool candidates

def cmd_pool_cands(args) -> None:
    """Pool candidates: predicted pairs (score>=cutoff) not in adjudicated gold, typed by span overlap."""
    cats = load_categories()
    adj = {r["id"]: r for r in read_jsonl(args.adjudicated)}
    cands = []
    for pr in read_jsonl(args.preds):
        wid = pr["window_id"]
        if wid not in adj:
            continue
        g = adj[wid]
        gold_keys = {pair_key(p) for p in g["pairs"]}
        ents = g["entities"]

        def etype(sp):
            """Map a predicted span to a gold entity by max overlap."""
            best, bo = None, 0
            for e in ents:
                o = min(sp["end"], e["end"]) - max(sp["start"], e["start"])
                if o > bo:
                    best, bo = e, o
            return best

        agg = {}
        for r in pr["relations"]:
            if r["score"] < args.cutoff:
                continue
            h, t = etype(r["head"]), etype(r["tail"])
            if h is None or t is None or h is t:
                continue
            for cid, c in cats.items():
                ok = (h["label"] in c["head"] and t["label"] in c["tail"])
                swapped = False
                if not ok and cid in UNDIRECTED:
                    ok = (t["label"] in c["head"] and h["label"] in c["tail"])
                    swapped = ok
                if not ok:
                    continue
                if r["name"] not in c["qualifiers"]:
                    continue
                hh, tt = (t, h) if swapped else (h, t)
                k = pair_key({"category": cid, "head": hh, "tail": tt, "qualifiers": []})
                if k in gold_keys:
                    continue
                a = agg.setdefault(k, {"id": wid, "category": cid, "head": hh, "tail": tt,
                                       "preds": [], "max_score": 0.0})
                a["preds"].append({"name": r["name"], "score": round(r["score"], 3)})
                a["max_score"] = max(a["max_score"], r["score"])
        cands.extend(agg.values())
    for i, c in enumerate(cands):
        c["cand_id"] = f"c{i:03d}"
        c["text"] = adj[c["id"]]["text"]
    write_jsonl(args.out, cands)
    print(f"{len(cands)} candidates")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("items")
    p.add_argument("--old-gold", default=ROOT / "eval/annotation/gold-v2.1.jsonl")
    p.add_argument("--passages", default=ROOT / "eval/annotation/passages.jsonl")
    p.add_argument("--out", default=ROOT / "eval/annotation/items.jsonl")
    p.add_argument("--batch-dir", default=ROOT / "eval/annotation/batches")
    p.add_argument("--batches", type=int, default=5)
    p.set_defaults(fn=cmd_items)
    p = sub.add_parser("parse")
    p.add_argument("raw"); p.add_argument("--items", default=ROOT / "eval/annotation/items.jsonl")
    p.add_argument("--batch"); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_parse)
    p = sub.add_parser("agreement")
    p.add_argument("a"); p.add_argument("b"); p.add_argument("--out")
    p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser("pool-cands")
    p.add_argument("--preds", required=True); p.add_argument("--adjudicated", required=True)
    p.add_argument("--cutoff", type=float, default=0.3); p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_pool_cands)
    p = sub.add_parser("merge")
    p.add_argument("--items", default=ROOT / "eval/annotation/items.jsonl")
    p.add_argument("--a", nargs="+", required=True); p.add_argument("--b", nargs="+", required=True)
    p.add_argument("--out", default=ROOT / "eval/annotation/merged.jsonl")
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("show")
    p.add_argument("--merged", default=ROOT / "eval/annotation/merged.jsonl")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("adjudicate")
    p.add_argument("--merged", default=ROOT / "eval/annotation/merged.jsonl")
    p.add_argument("--decisions", default=ROOT / "eval/annotation/decisions.json")
    p.add_argument("--extra"); p.add_argument("--out", default=ROOT / "eval/annotation/adjudicated.jsonl")
    p.set_defaults(fn=cmd_adjudicate)
    p = sub.add_parser("build")
    p.add_argument("--adjudicated", default=ROOT / "eval/annotation/adjudicated.jsonl")
    p.add_argument("--out", default=ROOT / "eval/gold.jsonl")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("validate")
    p.add_argument("gold", nargs="?", default=ROOT / "eval/gold.jsonl")
    p.set_defaults(fn=cmd_validate)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
