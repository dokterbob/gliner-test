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

    gold = load_jsonl(args.gold)
    corpus_engines = discover_engines(args.preds_dir)
    gold_engines = discover_gold_engines(args.preds_dir)

    out_lines = []
    out_lines.append("# GLiNER2.5 cross-validation report\n")
    out_lines.append(f"Corpus engines found: {sorted(corpus_engines)}\n")
    out_lines.append(f"Gold engines found: {sorted(gold_engines)}\n")

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

    report = "\n".join(out_lines) + "\n"
    print(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report)
        print(f"\n[compare] wrote {args.report}")


if __name__ == "__main__":
    main()
