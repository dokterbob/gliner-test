"""How much does post-processing recover per engine? (rows: overall untyped pair P/R/F1 + AUPRC)
    uv run python eval/experiments/recovery.py --split test
A. raw relations as emitted, compare.py raw path, global thr 0.5 (no collapse/gate beyond compare's own pair merge; incl. uncategorised)
B. + type gate and max collapse (relations.py, all names, thr 0.5)
C. + qualifier-name scoring, NMS, temperature calibration, per-category thresholds (final pipeline, F1 point)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp  # noqa: E402

C = exp.C
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--engines", nargs="*", default=["coreml-fp32-cpu", "coreml-fp16-w8-cpu", "torch-fp32", "torch-jointie", "rust-ort-fp32"])
a = ap.parse_args()
pcats, _ = C.load_pair_schema(exp.ROOT / "schema/ontonotes.json")
gold = C.load_gold(exp.ROOT / "eval/gold.jsonl", a.split)
rows = []
for e in a.engines:
    raw = {r["window_id"]: r for r in exp.jl(exp.ROOT / f"data/preds/gold/{e}.jsonl")}
    ev = C.PairEval(raw, gold, pcats, "overlap", True, False)
    out = {}
    for tag, thr in (("A", 0.5),):
        tp, fp, fn = ev.counts("overall", thr)
        p, r, f1 = C.prf1(tp, fp, fn)
        out[tag] = (tp, fp, fn, p, r, f1, C.average_precision(ev.ranked["overall"], ev.n_gold["overall"]))
    base = exp.load_cfg(exp.ROOT / f"eval/experiments/final-{e}-f1.json")
    b = json.loads(json.dumps(base))
    b["post"] = {"collapse": "max", "score_names": "all", "threshold": 0.5}
    rb = exp.evaluate(b, a.split, True)
    rc = exp.evaluate(base, a.split, True)
    for tag, r in (("B", rb), ("C", rc)):
        o = r["overall"]
        out[tag] = (o["tp"], o["fp"], o["fn"], o["P"], o["R"], o["F1"], o["AUPRC"])
    rows.append((e, out))
print(f"split={a.split}, overall (untyped, unordered) pairs; cells tp/fp/fn P R F1 | AUPRC\n")
print("| engine | A raw @0.5 | B + type gate + max collapse @0.5 | C final pipeline (F1 point) |")
print("|---|---|---|---|")
for e, o in rows:
    f = lambda t: f"{t[0]}/{t[1]}/{t[2]} {t[3]:.2f} {t[4]:.2f} {t[5]:.2f} \\| {t[6]:.3f}"
    print(f"| {e} | {f(o['A'])} | {f(o['B'])} | {f(o['C'])} |")
