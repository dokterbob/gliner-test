"""Pair-centric relation post-processing (schema v2.2 `pair_categories`).

Turns the shared prediction format (entities + named relations with a
per-relation score) into *pairs*: which two entities are related, in which
pair category, with one confidence and the relation names as qualifiers.

    relations --snap_and_type--> typed --categorize--> category + canonical order
              --collapse_pairs--> one pair per (window, endpoints) with qualifiers
              --calibrate (opt)--> per-category isotonic map
              --apply_thresholds--> per-category thresholds
              --aggregate_document (opt)--> merge across overlapping windows

Design constraints: every core function is pure and deterministic and uses
plain dicts/lists/floats only (no numpy, no sets with unstable iteration in
the output path), so it can be ported 1:1 to Rust. Offsets are char offsets
relative to the window text and are never altered, only copied.

Data shapes
    entity   {"label", "text", "start", "end", "score"}
    endpoint {"text", "start", "end"}                  (+ "type" once typed)
    relation {"name", "head": endpoint, "tail": endpoint, "score"}
    pair     {"window_id", "category", "directed", "head": endpoint+type,
              "tail": endpoint+type, "score", "qualifiers": [{"name","score"}]}

CLI
    uv run python -m gliner_test.relations --preds data/preds/gold/coreml-fp32-cpu.jsonl \
        --schema schema/ontonotes.json --out data/pairs/coreml-fp32-cpu.jsonl \
        [--collapse max|noisy-or] [--score-names all|qualifiers] \
        [--global-threshold 0.5] [--cat-threshold place=0.6 ...] \
        [--calibration eval/calibration/coreml-fp32-cpu.json] \
        [--aggregate data/pairs-doc/x.jsonl --windows data/windows.jsonl --aggregate-mode max]

    # fit a per-category isotonic map on gold dev and store the lookup table
    uv run python -m gliner_test.relations --preds ... --schema ... --fit-calibration \
        --gold eval/gold.jsonl --split dev --calibration-out eval/calibration/<engine>.json

Order of operations in the CLI: collapse -> calibrate -> threshold. When a
calibration table is given, thresholds therefore live on the *calibrated*
scale. Pass `--global-threshold 0` to keep everything (e.g. when feeding
compare.py's threshold sweep).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def load_schema(path):
    """-> {"threshold": float, "categories": [{"id","directed","head":[..],"tail":[..],
    "qualifiers":[..],"threshold": float|None}]} in schema order."""
    with open(path) as f:
        raw = json.load(f)
    cats = []
    for c in raw.get("pair_categories", []):
        cats.append(
            {
                "id": c["id"],
                "directed": bool(c["directed"]),
                "head": list(c["head"]),
                "tail": list(c["tail"]),
                "qualifiers": list(c.get("qualifiers", [])),
                "threshold": c.get("threshold"),
            }
        )
    return {"threshold": float(raw.get("threshold", 0.5)), "categories": cats}


# --------------------------------------------------------------------------
# 1. snap_and_type
# --------------------------------------------------------------------------


def _overlap(a_start, a_end, b_start, b_end):
    lo = a_start if a_start > b_start else b_start
    hi = a_end if a_end < b_end else b_end
    return hi - lo if hi > lo else 0


def snap_endpoint(endpoint, entities):
    """OntoNotes type of the entity with the largest char overlap with
    `endpoint` (ties: higher score, then earlier in `entities`); None when no
    entity overlaps. The endpoint's own text/offsets are preserved."""
    best_type = None
    best_key = None
    for i, e in enumerate(entities):
        ov = _overlap(endpoint["start"], endpoint["end"], e["start"], e["end"])
        if ov <= 0:
            continue
        key = (ov, e.get("score", 1.0), -i)
        if best_key is None or key > best_key:
            best_key = key
            best_type = e["label"]
    return best_type


def snap_and_type(relations, entities):
    """Copy each relation with `type` added to head and tail (None when the
    endpoint overlaps no entity-pass span)."""
    out = []
    for r in relations:
        head = dict(r["head"])
        tail = dict(r["tail"])
        head["type"] = snap_endpoint(head, entities)
        tail["type"] = snap_endpoint(tail, entities)
        out.append({**r, "head": head, "tail": tail})
    return out


# --------------------------------------------------------------------------
# 2. categorize
# --------------------------------------------------------------------------


def _fits(head_type, tail_type, cat):
    return head_type in cat["head"] and tail_type in cat["tail"]


def _span_order(a, b):
    """(first, second) by start offset, then end offset; stable for equal spans."""
    if (b["start"], b["end"]) < (a["start"], a["end"]):
        return b, a
    return a, b


def categorize(relation, categories):
    """Assign a pair category from the endpoint types.

    Returns None if no category fits (or an endpoint is untyped). Otherwise
    {"category", "directed", "head", "tail"} where, for directed categories,
    head/tail keep the model's orientation, and for undirected categories
    they are put in a canonical order: PERSON first for per_org, otherwise
    by (start, end). The first fitting category in schema order wins."""
    h, t = relation["head"], relation["tail"]
    ht, tt = h.get("type"), t.get("type")
    if ht is None or tt is None:
        return None
    for cat in categories:
        fwd = _fits(ht, tt, cat)
        rev = _fits(tt, ht, cat)
        if cat["directed"]:
            if fwd:
                return {"category": cat["id"], "directed": True, "head": h, "tail": t}
            continue
        if not (fwd or rev):
            continue
        if cat["id"] == "per_org":
            first, second = (h, t) if ht == "PERSON" else (t, h)
        else:
            first, second = _span_order(h, t)
        return {"category": cat["id"], "directed": False, "head": first, "tail": second}
    return None


# --------------------------------------------------------------------------
# 3. collapse_pairs
# --------------------------------------------------------------------------


def _pair_key(category, directed, head, tail):
    a = (head["start"], head["end"])
    b = (tail["start"], tail["end"])
    if not directed and b < a:
        a, b = b, a
    return (category, a, b)


def combine_scores(scores, mode):
    """`max` or `noisy-or` (1 - prod(1 - s)) over an iterable of scores."""
    if mode == "max":
        best = 0.0
        for s in scores:
            if s > best:
                best = s
        return best
    if mode == "noisy-or":
        prod = 1.0
        for s in scores:
            prod *= 1.0 - s
        return 1.0 - prod
    raise ValueError(f"unknown collapse mode {mode!r}")


def collapse_pairs(relations, categories, window_id=None, mode="max", score_names="all"):
    """Group typed relations into one pair per (category, endpoints).

    relations: output of snap_and_type. Pairs fitting no category are dropped.
    Undirected categories group both orientations together. Per relation name
    the best score is kept; the pair `score` is `mode` (max | noisy-or) over
    those per-name scores. `score_names="all"` uses every name seen for the
    pair; `"qualifiers"` only names in the category's qualifier list (pairs
    with none of them are dropped). `qualifiers` is [{name, score}], limited
    to the category's qualifiers, sorted by score descending then name."""
    cat_by_id = {c["id"]: c for c in categories}
    groups = {}
    order = []
    for r in relations:
        cz = categorize(r, categories)
        if cz is None:
            continue
        key = _pair_key(cz["category"], cz["directed"], cz["head"], cz["tail"])
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"cz": cz, "names": {}}
            order.append(key)
        sc = r.get("score", 1.0)
        if sc > g["names"].get(r["name"], -1.0):
            g["names"][r["name"]] = sc
    pairs = []
    for key in order:
        g = groups[key]
        cz = g["cz"]
        cat = cat_by_id[cz["category"]]
        quals = [(n, s) for n, s in g["names"].items() if n in cat["qualifiers"]]
        quals.sort(key=lambda ns: (-ns[1], ns[0]))
        if score_names == "qualifiers":
            used = [s for _, s in quals]
        else:
            used = [g["names"][n] for n in sorted(g["names"])]
        if not used:
            continue
        pair = {
            "window_id": window_id,
            "category": cz["category"],
            "directed": cz["directed"],
            "head": dict(cz["head"]),
            "tail": dict(cz["tail"]),
            "score": combine_scores(used, mode),
            "qualifiers": [{"name": n, "score": s} for n, s in quals],
        }
        pairs.append(pair)
    pairs.sort(key=lambda p: (p["head"]["start"], p["head"]["end"], p["tail"]["start"], p["tail"]["end"], p["category"]))
    return pairs


# --------------------------------------------------------------------------
# 4. apply_thresholds
# --------------------------------------------------------------------------


def category_threshold(category_id, categories, global_threshold, overrides=None):
    """override (CLI) > schema per-category threshold > global threshold."""
    if overrides and overrides.get(category_id) is not None:
        return overrides[category_id]
    for c in categories:
        if c["id"] == category_id:
            if c["threshold"] is not None:
                return c["threshold"]
            break
    return global_threshold


def apply_thresholds(pairs, categories, global_threshold, overrides=None):
    """Keep pairs with score >= their category's threshold."""
    return [
        p for p in pairs
        if p["score"] >= category_threshold(p["category"], categories, global_threshold, overrides)
    ]


# --------------------------------------------------------------------------
# 5. calibrate (per-category isotonic regression, PAV)
# --------------------------------------------------------------------------


def fit_isotonic(scores, labels):
    """Pool-adjacent-violators fit of P(label=1 | score), non-decreasing.

    Returns a lookup table {"x": [...], "y": [...]} of knots (strictly
    increasing x, non-decreasing y) to be applied by linear interpolation
    with clamping at both ends. Each pooled block contributes one knot at
    its mean score. Empty input gives the identity-free empty table."""
    pts = sorted(zip(scores, labels), key=lambda t: (t[0], t[1]))
    # blocks: [sum_y, weight, sum_x]
    blocks = []
    for x, y in pts:
        blocks.append([float(y), 1.0, float(x)])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    xs, ys = [], []
    for sy, w, sx in blocks:
        x = sx / w
        y = sy / w
        if xs and x <= xs[-1]:
            continue  # numerical duplicate; y is already >= previous by PAV
        xs.append(x)
        ys.append(y)
    return {"x": xs, "y": ys}


def apply_isotonic(table, score):
    """Piecewise-linear lookup with clamping; identity if the table is empty."""
    xs, ys = table["x"], table["y"]
    n = len(xs)
    if n == 0:
        return score
    if score <= xs[0]:
        return ys[0]
    if score >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= score:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * ((score - xs[lo]) / span)


def fit(samples_by_category, engine=None, min_samples=1):
    """samples_by_category: {category: [(score, label 0/1), ...]} -> calibration
    JSON object {"method","engine","categories":{cat:{"n","x","y"}}}. Categories
    with fewer than `min_samples` samples are omitted (left uncalibrated)."""
    cats = {}
    for cid in sorted(samples_by_category):
        samples = samples_by_category[cid]
        if len(samples) < min_samples:
            continue
        table = fit_isotonic([s for s, _ in samples], [l for _, l in samples])
        cats[cid] = {"n": len(samples), "x": table["x"], "y": table["y"]}
    return {"method": "isotonic", "engine": engine, "categories": cats}


def apply(pairs, calibration):
    """Replace each pair's score by its calibrated value, keeping the raw
    score as `raw_score`. Qualifier scores are left raw. Categories absent
    from the table pass through unchanged."""
    out = []
    for p in pairs:
        table = calibration["categories"].get(p["category"])
        q = dict(p)
        if table is not None:
            q["raw_score"] = p["score"]
            q["score"] = apply_isotonic(table, p["score"])
        out.append(q)
    return out


# --------------------------------------------------------------------------
# 6. aggregate_document
# --------------------------------------------------------------------------

_PUNCT = " \t\r\n\"'`.,;:!?()[]{}<>"


def normalize_text(text):
    """Casefold, collapse whitespace, strip surrounding punctuation."""
    return re.sub(r"\s+", " ", text.casefold()).strip(_PUNCT)


def aggregate_document(pairs, windows_by_id, mode="max"):
    """Merge the same pair across overlapping windows of one chunk.

    pairs: window-level pairs (with window_id). windows_by_id:
    {window_id: {"chunk_id","char_start",...}}. Pairs are first lifted to
    chunk offsets (char_start + offset) and exact duplicates (same chunk
    offsets and category, i.e. the same mention pair seen from overlapping
    windows) are merged by max. The remaining distinct mention pairs are
    grouped by (chunk_id, category, normalised endpoint texts; unordered for
    undirected categories) and combined with `mode` (max | noisy-or).
    `evidence` = number of distinct mention pairs merged, `n_windows` =
    number of windows contributing. Output offsets are those of the
    best-scoring mention pair, in chunk coordinates."""
    # step 1: dedupe identical mention pairs seen in several windows
    mentions = {}
    mention_order = []
    for p in pairs:
        w = windows_by_id.get(p["window_id"])
        if w is None:
            continue
        base = w["char_start"]
        head = {**p["head"], "start": p["head"]["start"] + base, "end": p["head"]["end"] + base}
        tail = {**p["tail"], "start": p["tail"]["start"] + base, "end": p["tail"]["end"] + base}
        key = (w["chunk_id"],) + _pair_key(p["category"], p["directed"], head, tail)
        m = mentions.get(key)
        if m is None:
            mentions[key] = {
                "chunk_id": w["chunk_id"], "category": p["category"], "directed": p["directed"],
                "head": head, "tail": tail, "score": p["score"], "windows": [p["window_id"]],
                "quals": {q["name"]: q["score"] for q in p.get("qualifiers", [])},
            }
            mention_order.append(key)
        else:
            m["windows"].append(p["window_id"])
            if p["score"] > m["score"]:
                m["score"] = p["score"]
                m["head"], m["tail"] = head, tail
            for q in p.get("qualifiers", []):
                if q["score"] > m["quals"].get(q["name"], -1.0):
                    m["quals"][q["name"]] = q["score"]
    # step 2: group by normalised text
    groups = {}
    group_order = []
    for key in mention_order:
        m = mentions[key]
        a = (normalize_text(m["head"]["text"]), m["head"].get("type"))
        b = (normalize_text(m["tail"]["text"]), m["tail"].get("type"))
        if not m["directed"] and b < a:
            a, b = b, a
        gk = (m["chunk_id"], m["category"], a, b)
        g = groups.get(gk)
        if g is None:
            groups[gk] = [m]
            group_order.append(gk)
        else:
            g.append(m)
    out = []
    for gk in group_order:
        ms = groups[gk]
        best = ms[0]
        for m in ms:
            if m["score"] > best["score"]:
                best = m
        quals = {}
        wins = []
        for m in ms:
            for n, s in m["quals"].items():
                if s > quals.get(n, -1.0):
                    quals[n] = s
            for wid in m["windows"]:
                if wid not in wins:
                    wins.append(wid)
        out.append(
            {
                "chunk_id": best["chunk_id"],
                "category": best["category"],
                "directed": best["directed"],
                "head": best["head"],
                "tail": best["tail"],
                "score": combine_scores([m["score"] for m in ms], mode),
                "evidence": len(ms),
                "n_windows": len(wins),
                "qualifiers": [{"name": n, "score": s} for n, s in sorted(quals.items(), key=lambda ns: (-ns[1], ns[0]))],
            }
        )
    out.sort(key=lambda p: (p["chunk_id"], p["head"]["start"], p["tail"]["start"], p["category"]))
    return out


# --------------------------------------------------------------------------
# glue (I/O, gold labelling) -- not part of the Rust-port core
# --------------------------------------------------------------------------


def predictions_to_pairs(rec, categories, mode="max", score_names="all"):
    typed = snap_and_type(rec.get("relations", []), rec.get("entities", []))
    return collapse_pairs(typed, categories, rec["window_id"], mode, score_names)


def _read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _ep_match(a, b):
    return a["text"].casefold() == b["text"].casefold() or _overlap(a["start"], a["end"], b["start"], b["end"]) > 0


def label_pairs(pairs, gold_rec):
    """0/1 label per pair: 1 if it matches a not-yet-used gold relation
    (endpoint text equal or char-overlapping; unordered unless the pair is
    directed). Gold `category`, when present, must equal the pair's. Pairs are
    visited by descending score so the best-scoring duplicate claims the gold."""
    gold = gold_rec.get("relations", [])
    used = [False] * len(gold)
    labels = [0] * len(pairs)
    for i in sorted(range(len(pairs)), key=lambda i: (-pairs[i]["score"], i)):
        p = pairs[i]
        for j, g in enumerate(gold):
            if used[j]:
                continue
            if g.get("category") not in (None, p["category"]):
                continue
            fwd = _ep_match(p["head"], g["head"]) and _ep_match(p["tail"], g["tail"])
            rev = (not p["directed"]) and _ep_match(p["head"], g["tail"]) and _ep_match(p["tail"], g["head"])
            if fwd or rev:
                used[j] = True
                labels[i] = 1
                break
    return labels


def _parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preds", type=Path, required=True, help="shared-format predictions JSONL")
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--out", type=Path, default=None, help="pairs JSONL (one pair per line)")
    ap.add_argument("--collapse", choices=["max", "noisy-or"], default="max")
    ap.add_argument("--score-names", choices=["all", "qualifiers"], default="all",
                    help="names contributing to the pair score (default all; qualifiers = category's own names)")
    ap.add_argument("--global-threshold", type=float, default=None,
                    help="fallback threshold for categories whose schema threshold is null (default: schema threshold)")
    ap.add_argument("--cat-threshold", action="append", metavar="CAT=VALUE",
                    help="override a category threshold (repeatable)")
    ap.add_argument("--calibration", type=Path, default=None, help="calibration JSON to apply (collapse -> calibrate -> threshold)")
    ap.add_argument("--fit-calibration", action="store_true", help="fit isotonic maps on --gold and write --calibration-out")
    ap.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    ap.add_argument("--split", default="dev", help="gold split used for fitting (gold without `split` counts as all)")
    ap.add_argument("--calibration-out", type=Path, default=None)
    ap.add_argument("--aggregate", type=Path, default=None, help="also write document-level aggregated pairs here")
    ap.add_argument("--windows", type=Path, default=Path("data/windows.jsonl"))
    ap.add_argument("--aggregate-mode", choices=["max", "noisy-or"], default="max")
    args = ap.parse_args(argv)

    schema = load_schema(args.schema)
    cats = schema["categories"]
    gthr = schema["threshold"] if args.global_threshold is None else args.global_threshold
    overrides = _parse_overrides(args.cat_threshold)
    recs = _read_jsonl(args.preds)
    by_window = [(r["window_id"], predictions_to_pairs(r, cats, args.collapse, args.score_names)) for r in recs]

    if args.fit_calibration:
        if args.calibration_out is None:
            ap.error("--fit-calibration needs --calibration-out")
        gold = {g["window_id"]: g for g in _read_jsonl(args.gold)}
        samples = {c["id"]: [] for c in cats}
        for wid, pairs in by_window:
            g = gold.get(wid)
            if g is None or (args.split != "all" and g.get("split", args.split) != args.split):
                continue
            for p, lab in zip(pairs, label_pairs(pairs, g)):
                samples[p["category"]].append((p["score"], lab))
        table = fit(samples, engine=args.preds.stem)
        args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_out.write_text(json.dumps(table, indent=1))
        print(f"[relations] fitted {args.calibration_out}: "
              + ", ".join(f"{k}:n={v['n']}" for k, v in table["categories"].items()), file=sys.stderr)
        return

    calibration = json.loads(args.calibration.read_text()) if args.calibration else None
    all_pairs = []
    for _wid, pairs in by_window:
        if calibration is not None:
            pairs = apply(pairs, calibration)
        all_pairs.extend(apply_thresholds(pairs, cats, gthr, overrides))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[relations] wrote {len(all_pairs)} pairs to {args.out}", file=sys.stderr)

    if args.aggregate is not None:
        windows = {w["window_id"]: w for w in _read_jsonl(args.windows)}
        agg = aggregate_document(all_pairs, windows, args.aggregate_mode)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        with args.aggregate.open("w") as f:
            for p in agg:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[relations] wrote {len(agg)} document-level pairs to {args.aggregate}", file=sys.stderr)


if __name__ == "__main__":
    main()
