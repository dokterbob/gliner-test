"""Cross-engine and gold-set comparison for the GLiNER2.5 test drive.

Three things, all driven by the shared prediction format
({"window_id","entities":[{"label","text","start","end","score"}],
  "relations":[...],"latency_ms"}):

  1. Pairwise engine agreement over data/windows.jsonl: span-level exact-match
     and overlap P/R/F1 per label against a reference engine (default: the
     torch run), plus score-diff stats on exactly-matched spans.
  2. Gold-set P/R/F1 per label, per engine, against eval/gold.jsonl.
  3. The biggest disagreements between each engine and the reference, with
     the window text, for manual inspection.

Engines are discovered by filename: every `*.jsonl` directly under
--preds-dir (not matching *.timing.json) is one engine's corpus run, named
after its stem (e.g. "rust-ort-fp32", "torch-fp32", "coreml-fp16-ane").
Gold-set runs are the same, under --preds-dir/gold/.

Usage:
    uv run python scripts/compare.py \
        --windows data/windows.jsonl \
        --preds-dir data/preds \
        --gold eval/gold.jsonl \
        --reference torch-fp32 \
        --report docs/cross-validation-report.md
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["window_id"]] = rec
    return out


def discover_engines(preds_dir: Path) -> dict:
    engines = {}
    if not preds_dir.exists():
        return engines
    for p in sorted(preds_dir.glob("*.jsonl")):
        if p.name.endswith(".timing.json"):
            continue
        engines[p.stem] = load_jsonl(p)
    return engines


def discover_gold_engines(preds_dir: Path) -> dict:
    gold_dir = preds_dir / "gold"
    engines = {}
    if not gold_dir.exists():
        return engines
    for p in sorted(gold_dir.glob("*.jsonl")):
        engines[p.stem] = load_jsonl(p)
    return engines


def spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and b_start < a_end


def match_spans(pred_ents, ref_ents, mode: str):
    """Greedy one-to-one matching per label. mode: 'exact' or 'overlap'.

    Returns (matched_pairs, unmatched_pred, unmatched_ref) where matched_pairs
    is a list of (pred_entity, ref_entity).
    """
    by_label_pred = defaultdict(list)
    by_label_ref = defaultdict(list)
    for e in pred_ents:
        by_label_pred[e["label"]].append(e)
    for e in ref_ents:
        by_label_ref[e["label"]].append(e)

    matched = []
    unmatched_pred = []
    unmatched_ref = []

    for label in set(by_label_pred) | set(by_label_ref):
        preds = list(by_label_pred.get(label, []))
        refs = list(by_label_ref.get(label, []))
        ref_used = [False] * len(refs)

        if mode == "exact":
            for p in preds:
                found = None
                for i, r in enumerate(refs):
                    if ref_used[i]:
                        continue
                    if p["start"] == r["start"] and p["end"] == r["end"]:
                        found = i
                        break
                if found is not None:
                    ref_used[found] = True
                    matched.append((p, refs[found]))
                else:
                    unmatched_pred.append(p)
        else:  # overlap: greedy by descending overlap length
            candidates = []
            for pi, p in enumerate(preds):
                for ri, r in enumerate(refs):
                    if spans_overlap(p["start"], p["end"], r["start"], r["end"]):
                        ov = min(p["end"], r["end"]) - max(p["start"], r["start"])
                        candidates.append((ov, pi, ri))
            candidates.sort(key=lambda t: -t[0])
            pred_used = [False] * len(preds)
            for ov, pi, ri in candidates:
                if pred_used[pi] or ref_used[ri]:
                    continue
                pred_used[pi] = True
                ref_used[ri] = True
                matched.append((preds[pi], refs[ri]))
            for pi, used in enumerate(pred_used):
                if not used:
                    unmatched_pred.append(preds[pi])

        for ri, used in enumerate(ref_used):
            if not used:
                unmatched_ref.append(refs[ri])

    return matched, unmatched_pred, unmatched_ref


def prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def entity_metrics(pred_by_window: dict, ref_by_window: dict, mode: str):
    """Per-label TP/FP/FN across all shared windows, plus score diffs on
    exact matches (only meaningful for mode='exact', but harmless either way)."""
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    score_diffs = []
    shared_ids = set(pred_by_window) & set(ref_by_window)
    for wid in shared_ids:
        pred_ents = pred_by_window[wid].get("entities", [])
        ref_ents = ref_by_window[wid].get("entities", [])
        matched, unmatched_pred, unmatched_ref = match_spans(pred_ents, ref_ents, mode)
        for p, r in matched:
            per_label[p["label"]]["tp"] += 1
            score_diffs.append(p.get("score", 1.0) - r.get("score", 1.0))
        for p in unmatched_pred:
            per_label[p["label"]]["fp"] += 1
        for r in unmatched_ref:
            per_label[r["label"]]["fn"] += 1
    return per_label, score_diffs, len(shared_ids)


def load_relation_schema(schema_path: Path) -> dict:
    """name -> {"symmetric": bool, "head": [...], "tail": [...]} from schema/ontonotes.json."""
    if not schema_path.exists():
        return {}
    schema = json.loads(schema_path.read_text())
    return {
        r["name"]: {
            "symmetric": bool(r.get("symmetric", False)),
            "head": r.get("head", []),
            "tail": r.get("tail", []),
        }
        for r in schema.get("relations", [])
    }


def relation_key(rel, symmetric_names: set):
    """(name, head_text, tail_text), order-insensitive for symmetric relations
    (schema v2: family, business_associate, personal_relation) so that e.g.
    family(A, B) and family(B, A) count as the same instance."""
    head = rel["head"]["text"].lower()
    tail = rel["tail"]["text"].lower()
    if rel["name"] in symmetric_names:
        head, tail = sorted((head, tail))
    return (rel["name"], head, tail)


def relation_metrics(pred_by_window: dict, ref_by_window: dict, symmetric_names: set = frozenset()):
    tp = fp = fn = 0
    shared_ids = set(pred_by_window) & set(ref_by_window)
    for wid in shared_ids:
        pred_rels = {relation_key(r, symmetric_names) for r in pred_by_window[wid].get("relations", [])}
        ref_rels = {relation_key(r, symmetric_names) for r in ref_by_window[wid].get("relations", [])}
        tp += len(pred_rels & ref_rels)
        fp += len(pred_rels - ref_rels)
        fn += len(ref_rels - pred_rels)
    return prf1(tp, fp, fn), (tp, fp, fn)


def build_type_index(entities):
    return [(e["start"], e["end"], e["label"]) for e in entities]


def span_has_type(start, end, type_index, allowed_labels: set) -> bool:
    return any(l in allowed_labels and spans_overlap(start, end, s, e) for s, e, l in type_index)


def type_filter_relations(pred_by_window: dict, relation_schema: dict) -> dict:
    """Optional post-filter variant: drop a relation pair unless its head/tail
    span overlaps one of THIS ENGINE's OWN entity predictions whose label is
    among the relation's declared OntoNotes head/tail types in
    schema/ontonotes.json. Neither gliner25-rs's pair_relations nor gliner2's
    independent extract_relations apply this typing themselves (see
    docs/cross-validation.md); this is an evaluation-side variant, not a
    claim about what either engine does natively."""
    filtered = {}
    for wid, rec in pred_by_window.items():
        idx = build_type_index(rec.get("entities", []))
        kept = []
        for rel in rec.get("relations", []):
            spec = relation_schema.get(rel["name"])
            if spec is None:
                kept.append(rel)
                continue
            h = rel["head"]
            t = rel["tail"]
            if span_has_type(h["start"], h["end"], idx, set(spec["head"])) and \
               span_has_type(t["start"], t["end"], idx, set(spec["tail"])):
                kept.append(rel)
        filtered[wid] = {**rec, "relations": kept}
    return filtered


# ---------------------------------------------------------------------------
# Pair-centric relation metrics (schema v2.2 pair_categories)
#
# A "pair" is two entity endpoints regardless of relation name. Predictions are
# collapsed to the best score per pair; each gold pair is matched at most once
# (greedy, highest score first). Metrics are computed from one ranked list per
# view (overall / per category), so the threshold sweep, operating point and
# AUPRC are all consistent with each other.
# ---------------------------------------------------------------------------
SWEEP = [round(0.10 + 0.05 * i, 2) for i in range(18)]  # 0.10 .. 0.95


def load_pair_schema(schema_path: Path):
    """-> (categories, global_threshold). categories: list of dicts
    {id, directed, head:set, tail:set, qualifiers:list, threshold}."""
    if not schema_path.exists():
        return [], 0.5
    schema = json.loads(schema_path.read_text())
    cats = [
        {
            "id": c["id"],
            "directed": bool(c["directed"]),
            "head": set(c["head"]),
            "tail": set(c["tail"]),
            "qualifiers": list(c.get("qualifiers", [])),
            "threshold": c.get("threshold"),
        }
        for c in schema.get("pair_categories", [])
    ]
    return cats, float(schema.get("threshold", 0.5))


def snap_type(span, entities, exact_first=False):
    """OntoNotes label of the entity overlapping `span` most (ties: higher
    score, then list order). exact_first prefers an identical span."""
    best = None
    best_key = None
    for i, e in enumerate(entities):
        if not spans_overlap(span["start"], span["end"], e["start"], e["end"]):
            continue
        ov = min(span["end"], e["end"]) - max(span["start"], e["start"])
        exact = int(exact_first and span["start"] == e["start"] and span["end"] == e["end"])
        key = (exact, ov, e.get("score", 1.0), -i)
        if best_key is None or key > best_key:
            best, best_key = e["label"], key
    return best


def categories_fitting(htype, ttype, cats):
    out = []
    for c in cats:
        fwd = htype in c["head"] and ttype in c["tail"]
        rev = ttype in c["head"] and htype in c["tail"]
        if fwd or (not c["directed"] and rev):
            out.append(c["id"])
    return out


def infer_gold_category(name, htype, ttype, cats):
    """Category for an old-format gold relation: types fit AND the relation
    name is one of the category's qualifiers; falls back to name alone when
    unique (e.g. gold-0002, whose tail is a substring of an ORG span)."""
    by_type = categories_fitting(htype, ttype, cats) if htype and ttype else []
    by_name = [c["id"] for c in cats if name in c["qualifiers"]]
    both = [c for c in by_type if c in by_name]
    if both:
        return both[0]
    if len(by_name) == 1:
        return by_name[0]
    return by_type[0] if by_type else None


def endpoints_match(a, b, rule: str) -> bool:
    if a["text"].casefold() == b["text"].casefold():
        return True
    return rule == "overlap" and spans_overlap(a["start"], a["end"], b["start"], b["end"])


def _pair_key(h, t, directed: bool):
    a, b = (h["start"], h["end"]), (t["start"], t["end"])
    return (a, b) if directed else tuple(sorted((a, b)))


def gold_pairs(rec, cats, include_pool: bool):
    """Collapse a gold record to pairs: {head, tail, category, qualifiers:set}."""
    ents = rec.get("entities", [])
    pairs = {}
    for r in rec.get("relations", []):
        if not include_pool and r.get("source") == "pool":
            continue
        h, t = r["head"], r["tail"]
        htype = h.get("type") or snap_type(h, ents, exact_first=True)
        ttype = t.get("type") or snap_type(t, ents, exact_first=True)
        cat = r.get("category") or infer_gold_category(r["name"], htype, ttype, cats)
        directed = next((c["directed"] for c in cats if c["id"] == cat), False)
        quals = set(r.get("qualifiers") or [r["name"]])
        key = _pair_key(h, t, directed)
        if key in pairs:
            pairs[key]["qualifiers"] |= quals
        else:
            pairs[key] = {"head": h, "tail": t, "category": cat, "directed": directed, "qualifiers": quals}
    return list(pairs.values())


def pred_candidates(rec, cats, is_pairs_file: bool):
    """Collapse a prediction record to ordered-pair candidates:
    {head, tail, score, names:{name:score}, category}. Category comes from the
    predicted entity types (relations files) or the file itself (pairs files)."""
    ents = rec.get("entities", [])
    out = {}
    if is_pairs_file:
        for p in rec.get("pairs", []):
            names = {q["name"]: q["score"] for q in p.get("qualifiers", [])}
            out[_pair_key(p["head"], p["tail"], True)] = {
                "head": p["head"], "tail": p["tail"], "score": p["score"], "names": names,
                "category": p.get("category"),
            }
        return list(out.values())
    for r in rec.get("relations", []):
        key = _pair_key(r["head"], r["tail"], True)
        c = out.get(key)
        if c is None:
            ht = snap_type(r["head"], ents)
            tt = snap_type(r["tail"], ents)
            fit = categories_fitting(ht, tt, cats) if ht and tt else []
            # directed categories: keep only the orientation the types allow
            c = out[key] = {"head": r["head"], "tail": r["tail"], "score": 0.0, "names": {},
                            "category": fit[0] if fit else None}
        c["score"] = max(c["score"], r.get("score", 1.0))
        c["names"][r["name"]] = max(c["names"].get(r["name"], 0.0), r.get("score", 1.0))
    return list(out.values())


def _merge_unordered(cands):
    merged = {}
    for c in cands:
        k = _pair_key(c["head"], c["tail"], False)
        m = merged.get(k)
        if m is None:
            merged[k] = {**c, "names": dict(c["names"])}
        else:
            if c["score"] > m["score"]:
                m["head"], m["tail"], m["score"], m["category"] = c["head"], c["tail"], c["score"], c["category"]
            for n, sc in c["names"].items():
                m["names"][n] = max(m["names"].get(n, 0.0), sc)
    return list(merged.values())


def match_window(cands, golds, directed_view: bool, rule: str):
    """Greedy highest-score-first one-to-one matching. Returns
    (ranked [(score, matched_gold|None, cand)], n_unmatched_gold)."""
    used = [False] * len(golds)
    ranked = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        hit = None
        for i, g in enumerate(golds):
            if used[i]:
                continue
            fwd = endpoints_match(c["head"], g["head"], rule) and endpoints_match(c["tail"], g["tail"], rule)
            rev = (not directed_view) and endpoints_match(c["head"], g["tail"], rule) \
                and endpoints_match(c["tail"], g["head"], rule)
            if fwd or rev:
                hit = i
                break
        if hit is not None:
            used[hit] = True
        ranked.append((c["score"], golds[hit] if hit is not None else None, c))
    return ranked, used.count(False)


def average_precision(ranked, n_gold):
    """Non-interpolated AP over the score-sorted list; unrecalled gold adds 0."""
    if n_gold == 0:
        return float("nan")
    tp = 0
    total = 0.0
    for i, (_, g, _) in enumerate(sorted(ranked, key=lambda t: -t[0]), 1):
        if g is not None:
            tp += 1
            total += tp / i
    return total / n_gold


def ece(scores, hits, bins=10):
    """Equal-width ECE; same definition as gliner2.joint_ie.calibration.
    expected_calibration_error (re-implemented in pure python so compare.py
    needs no torch; last bin closed on the right)."""
    n = len(scores)
    if n == 0:
        return float("nan")
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores) if s >= lo and (s <= hi if b == bins - 1 else s < hi)]
        if idx:
            conf = sum(scores[i] for i in idx) / len(idx)
            acc = sum(hits[i] for i in idx) / len(idx)
            err += len(idx) / n * abs(conf - acc)
    return err


def load_pairs_dir(pairs_dir: Path) -> dict:
    """data/pairs/*.jsonl (one pair per line) -> {engine: {window_id: {"pairs": [...]}}}."""
    out = {}
    if not pairs_dir.exists():
        return out
    for p in sorted(pairs_dir.glob("*.jsonl")):
        recs = defaultdict(lambda: {"entities": [], "pairs": []})
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[r["window_id"]]["pairs"].append(r)
        out[p.stem] = dict(recs)
    return out


def load_gold(path: Path, split: str) -> dict:
    gold = load_jsonl(path)
    if split != "all":
        gold = {w: r for w, r in gold.items() if r.get("split", split) == split}
    return gold


class PairEval:
    """All ranked lists for one engine against one gold set."""

    def __init__(self, preds, gold, cats, rule, include_pool, is_pairs_file):
        self.cats = cats
        self.views = ["overall"] + [c["id"] for c in cats]
        self.ranked = {v: [] for v in self.views}
        self.n_gold = {v: 0 for v in self.views}
        self.uncat = []  # scores of predicted pairs fitting no category
        self.qual = []   # (gold_pair, cand) matched in the overall view
        directed = {c["id"]: c["directed"] for c in cats}
        for wid, g in gold.items():
            gp = gold_pairs(g, cats, include_pool)
            cands = pred_candidates(preds.get(wid, {}), cats, is_pairs_file)
            # overall: untyped, unordered, name ignored
            ranked, _ = match_window(_merge_unordered(cands), gp, False, rule)
            self.ranked["overall"] += ranked
            self.n_gold["overall"] += len(gp)
            for c in cats:
                cid = c["id"]
                gc = [x for x in gp if x["category"] == cid]
                cc = [x for x in cands if x["category"] == cid]
                if not c["directed"]:
                    cc = _merge_unordered(cc)
                ranked, _ = match_window(cc, gc, directed[cid], rule)
                self.ranked[cid] += ranked
                self.n_gold[cid] += len(gc)
            self.uncat += [x["score"] for x in cands if x["category"] is None]
        for sc, gm, cand in self.ranked["overall"]:
            if gm is not None:
                self.qual.append((gm, cand, sc))

    def counts(self, view, thr):
        r = [t for t in self.ranked[view] if t[0] >= thr]
        tp = sum(1 for t in r if t[1] is not None)
        return tp, len(r) - tp, self.n_gold[view] - tp

    def typed_counts(self, thr):
        tp = fp = 0
        for c in self.cats:
            a, b, _ = self.counts(c["id"], thr)
            tp += a
            fp += b
        fp += sum(1 for s in self.uncat if s >= thr)
        return tp, fp, self.n_gold["overall"] - tp

    def qualifier_accuracy(self, thr, cats):
        qs = {c["id"]: set(c["qualifiers"]) for c in cats}
        ok = n = 0
        for g, cand, sc in self.qual:
            if sc < thr or not cand["names"]:
                continue
            allowed = qs.get(g["category"]) or qs.get(cand["category"])
            names = {k: v for k, v in cand["names"].items() if not allowed or k in allowed}
            if not names:
                continue
            n += 1
            ok += max(names, key=names.get) in g["qualifiers"]
        return ok, n


def pair_report(name, ev, thr, cats):
    L = []
    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    all_scores = [t[0] for t in ev.ranked["overall"]]
    lo = min(all_scores) if all_scores else None
    L.append(f"\n#### {name}  (operating threshold {thr:g}; min predicted pair score "
             f"{'n/a' if lo is None else f'{lo:.3f}'})\n")
    if lo is not None and lo > SWEEP[0] + 1e-9:
        L.append(f"_Note: scores below {lo:.2f} were gated out when this file was produced, so sweep rows "
                 f"below that are identical (recall ceiling). Re-run with --relation-threshold 0.0 for the full curve._\n")
    rows = []
    for v in ev.views:
        tp, fp, fn = ev.counts(v, thr)
        p, r, f1 = prf1(tp, fp, fn)
        ranked = ev.ranked[v]
        rows.append((v, ev.n_gold[v], tp + fp, tp, fp, fn, f(p), f(r), f(f1),
                     f(average_precision(ranked, ev.n_gold[v])),
                     f(ece([t[0] for t in ranked], [t[1] is not None for t in ranked]))))
    tp, fp, fn = ev.typed_counts(thr)
    p, r, f1 = prf1(tp, fp, fn)
    rows.append(("sum of categories (+uncat. as FP)", ev.n_gold["overall"], tp + fp, tp, fp, fn, f(p), f(r), f(f1), "", ""))
    n_unc = sum(1 for s in ev.uncat if s >= thr)
    L.append(md_table(["view", "gold", "pred", "tp", "fp", "fn", "P", "R", "F1", "AUPRC", "ECE(10)"], rows))
    L.append(f"\nuncategorised predicted pairs (fit no category by predicted entity types): "
             f"{n_unc} at >= {thr:g} ({len(ev.uncat)} total); excluded from per-category rows, included in overall.\n")
    ok, n = ev.qualifier_accuracy(thr, cats)
    L.append(f"qualifier top-1 accuracy on matched pairs: {ok}/{n}" + (f" = {ok / n:.3f}" if n else "") + "\n")
    srows = []
    for t in SWEEP:
        row = [f"{t:.2f}"]
        for v in ev.views:
            tp, fp, fn = ev.counts(v, t)
            row.append(f"{prf1(tp, fp, fn)[2]:.3f}")
        tp, fp, fn = ev.counts("overall", t)
        p, r, _ = prf1(tp, fp, fn)
        row += [f"{p:.3f}", f"{r:.3f}"]
        srows.append(row)
    L.append("Threshold sweep (F1 per view; overall P/R last):\n")
    L.append(md_table(["thr"] + [f"F1 {v}" for v in ev.views] + ["P overall", "R overall"], srows))
    return "\n".join(L)


def top_disagreements(pred_by_window, ref_by_window, windows_text, k=15):
    rows = []
    shared_ids = set(pred_by_window) & set(ref_by_window)
    for wid in shared_ids:
        pred_ents = pred_by_window[wid].get("entities", [])
        ref_ents = ref_by_window[wid].get("entities", [])
        _, unmatched_pred, unmatched_ref = match_spans(pred_ents, ref_ents, "overlap")
        n_diff = len(unmatched_pred) + len(unmatched_ref)
        if n_diff:
            rows.append((n_diff, wid, unmatched_pred, unmatched_ref))
    rows.sort(key=lambda t: -t[0])
    out = []
    for n_diff, wid, up, ur in rows[:k]:
        out.append({
            "window_id": wid,
            "text": windows_text.get(wid, ""),
            "pred_only": [f'{e["label"]}:"{e["text"]}"@{e["score"]:.2f}' for e in up],
            "ref_only": [f'{e["label"]}:"{e["text"]}"@{e["score"]:.2f}' for e in ur],
        })
    return out


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=Path, default=Path("data/windows.jsonl"))
    ap.add_argument("--preds-dir", type=Path, default=Path("data/preds"))
    ap.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    ap.add_argument("--reference", default="torch-fp32",
                     help="engine (filename stem under preds-dir) to use as the corpus-agreement reference")
    ap.add_argument("--report", type=Path, default=None, help="write markdown report here; also printed to stdout")
    ap.add_argument("--top-k-disagreements", type=int, default=15)
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--pairs-dir", type=Path, default=Path("data/pairs"),
                    help="dir of pair-level prediction files (output of gliner_test.relations); evaluated like engines")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="gold split to score (gold without a `split` field is treated as all)")
    pool = ap.add_mutually_exclusive_group()
    pool.add_argument("--include-pool", dest="include_pool", action="store_true", default=True,
                      help="score gold relations whose source is `pool` (default)")
    pool.add_argument("--exclude-pool", dest="include_pool", action="store_false",
                      help="drop gold relations whose source is `pool`")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating point for pair metrics (default: schema global threshold)")
    ap.add_argument("--endpoint-match", choices=["text", "overlap"], default="overlap",
                    help="pair endpoint rule: `text` = case-folded text equality; `overlap` = text equality OR char overlap")
    ap.add_argument("--pairs-only", action="store_true",
                    help="print only the pair-level relation report (fast; skips corpus agreement and legacy gold tables)")
    ap.add_argument("--engine", action="append", default=None,
                    help="restrict pair report to these engine names (repeatable)")
    args = ap.parse_args()

    relation_schema = load_relation_schema(args.schema)
    symmetric_names = {name for name, spec in relation_schema.items() if spec["symmetric"]}

    windows_text = {}
    if args.windows.exists():
        with args.windows.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                windows_text[r["window_id"]] = r["text"]

    gold = load_gold(args.gold, args.split)
    if not args.include_pool:
        for rec in gold.values():
            rec["relations"] = [r for r in rec.get("relations", []) if r.get("source") != "pool"]
    gold_engines = discover_gold_engines(args.preds_dir)
    pairs_engines = load_pairs_dir(args.pairs_dir)
    corpus_engines = {} if args.pairs_only else discover_engines(args.preds_dir)
    pair_cats, global_thr = load_pair_schema(args.schema)
    op_thr = global_thr if args.threshold is None else args.threshold

    out_lines = []
    out_lines.append("# GLiNER2.5 cross-validation report\n")
    if not args.pairs_only:
        out_lines.append(f"Corpus engines found: {sorted(corpus_engines)}\n")
    out_lines.append(f"Gold engines found: {sorted(gold_engines)}; pair files: {sorted(pairs_engines)}\n")

    if not args.pairs_only:
        # -------------------------------------------------- (i) corpus agreement
        out_lines.append("\n## 1. Pairwise agreement vs. reference on data/windows.jsonl\n")
        ref = corpus_engines.get(args.reference)
        if ref is None:
            out_lines.append(f"(no reference engine `{args.reference}` found under {args.preds_dir} -- skipped)\n")
        else:
            for name, preds in corpus_engines.items():
                if name == args.reference:
                    continue
                out_lines.append(f"\n### {name} vs {args.reference}\n")
                for mode in ("exact", "overlap"):
                    per_label, score_diffs, n_shared = entity_metrics(preds, ref, mode)
                    rows = []
                    tp_all = fp_all = fn_all = 0
                    for label in sorted(per_label):
                        c = per_label[label]
                        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
                        rows.append((label, c["tp"], c["fp"], c["fn"], f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                        tp_all += c["tp"]; fp_all += c["fp"]; fn_all += c["fn"]
                    p, r, f1 = prf1(tp_all, fp_all, fn_all)
                    rows.append(("**micro-all**", tp_all, fp_all, fn_all, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                    out_lines.append(f"\n**{mode} match** ({n_shared} shared windows)\n")
                    out_lines.append(md_table(["label", "tp", "fp", "fn", "P", "R", "F1"], rows))
                    if mode == "exact" and score_diffs:
                        out_lines.append(
                            f"\nScore diff on exact matches ({name} - {args.reference}): "
                            f"mean={statistics.mean(score_diffs):+.4f}, "
                            f"mean_abs={statistics.mean(abs(x) for x in score_diffs):.4f}, "
                            f"stdev={statistics.pstdev(score_diffs):.4f}, n={len(score_diffs)}\n"
                        )
                (p, r, f1), (tp, fp, fn) = relation_metrics(preds, ref, symmetric_names)
                out_lines.append(
                    f"\n**relations** (exact name+head_text+tail_text match, symmetric relations order-insensitive): "
                    f"tp={tp} fp={fp} fn={fn} P={p:.3f} R={r:.3f} F1={f1:.3f}\n"
                )

        # ------------------------------------------------------- (ii) gold P/R/F1
        out_lines.append("\n## 2. Gold-set P/R/F1 per engine and label (eval/gold.jsonl)\n")
        if not gold:
            out_lines.append(f"(no gold set found at {args.gold})\n")
        elif not gold_engines:
            out_lines.append(f"(no gold predictions found under {args.preds_dir}/gold/)\n")
        else:
            for name, preds in gold_engines.items():
                out_lines.append(f"\n### {name} vs gold\n")
                per_label, score_diffs, n_shared = entity_metrics(preds, gold, "exact")
                rows = []
                tp_all = fp_all = fn_all = 0
                for label in sorted(per_label):
                    c = per_label[label]
                    p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
                    rows.append((label, c["tp"], c["fp"], c["fn"], f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                    tp_all += c["tp"]; fp_all += c["fp"]; fn_all += c["fn"]
                p, r, f1 = prf1(tp_all, fp_all, fn_all)
                rows.append(("**micro-all**", tp_all, fp_all, fn_all, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                out_lines.append(f"\nexact match, {n_shared} shared gold sentences\n")
                out_lines.append(md_table(["label", "tp", "fp", "fn", "P", "R", "F1"], rows))

                per_label_ov, _, _ = entity_metrics(preds, gold, "overlap")
                rows_ov = []
                tp_o = fp_o = fn_o = 0
                for label in sorted(per_label_ov):
                    c = per_label_ov[label]
                    p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
                    rows_ov.append((label, c["tp"], c["fp"], c["fn"], f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                    tp_o += c["tp"]; fp_o += c["fp"]; fn_o += c["fn"]
                p, r, f1 = prf1(tp_o, fp_o, fn_o)
                rows_ov.append(("**micro-all**", tp_o, fp_o, fn_o, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"))
                out_lines.append("\noverlap match\n")
                out_lines.append(md_table(["label", "tp", "fp", "fn", "P", "R", "F1"], rows_ov))

                (p, r, f1), (tp, fp, fn) = relation_metrics(preds, gold, symmetric_names)
                out_lines.append(
                    f"\n**relations, independent** (exact name+head_text+tail_text match, symmetric order-insensitive): "
                    f"tp={tp} fp={fp} fn={fn} P={p:.3f} R={r:.3f} F1={f1:.3f}\n"
                )

                if relation_schema:
                    filtered_preds = type_filter_relations(preds, relation_schema)
                    (p, r, f1), (tp, fp, fn) = relation_metrics(filtered_preds, gold, symmetric_names)
                    out_lines.append(
                        f"\n**relations, independent + type post-filter** (head/tail span must overlap this "
                        f"engine's own entity prediction of a type the schema allows for that relation): "
                        f"tp={tp} fp={fp} fn={fn} P={p:.3f} R={r:.3f} F1={f1:.3f}\n"
                    )

        # -------------------------------------------------- (iii) disagreements
        out_lines.append("\n## 3. Top disagreements vs reference (corpus windows)\n")
        if ref is not None:
            for name, preds in corpus_engines.items():
                if name == args.reference:
                    continue
                dis = top_disagreements(preds, ref, windows_text, k=args.top_k_disagreements)
                out_lines.append(f"\n### {name} vs {args.reference}\n")
                for d in dis:
                    out_lines.append(f"\n- `{d['window_id']}`: {d['text']!r}")
                    if d["pred_only"]:
                        out_lines.append(f"\n  - only in {name}: {', '.join(d['pred_only'])}")
                    if d["ref_only"]:
                        out_lines.append(f"\n  - only in {args.reference}: {', '.join(d['ref_only'])}")
        else:
            out_lines.append("(no reference engine available)\n")


    # ------------------------------------------- (iv) pair-level relation metrics
    out_lines.append("\n## 4. Pair-level relation metrics (schema pair_categories; relation name ignored)\n")
    out_lines.append(
        f"gold: {args.gold}, split={args.split}, pool {'included' if args.include_pool else 'excluded'}, "
        f"{len(gold)} windows; endpoint match = {args.endpoint_match}; best score per pair; each gold pair matched at most once; "
        f"category of predicted pairs from predicted entity types; place is directed, other categories unordered "
        f"(overall row: all pairs, unordered).\n")
    if not pair_cats:
        out_lines.append("(schema has no pair_categories -- only the overall row is meaningful)\n")
    sel = lambda n: args.engine is None or n in args.engine
    for name, preds in gold_engines.items():
        if sel(name):
            out_lines.append(pair_report(name, PairEval(preds, gold, pair_cats, args.endpoint_match,
                                                        args.include_pool, False), op_thr, pair_cats))
    for name, preds in pairs_engines.items():
        if sel(name):
            out_lines.append(pair_report("pairs:" + name, PairEval(preds, gold, pair_cats, args.endpoint_match,
                                                                  args.include_pool, True), op_thr, pair_cats))

    report = "\n".join(out_lines) + "\n"
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"\n[compare] wrote {args.report}")


if __name__ == "__main__":
    main()
