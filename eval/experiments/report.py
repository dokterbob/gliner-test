"""Markdown tables for the final configs.
    uv run python eval/experiments/report.py --split test [--engines coreml-fp32-cpu ...] [--exclude-pool]
Per category and engine: gold pairs, counts tp/fp/fn and P/R/F1 at both operating points (F1-max, P>=0.8), AUPRC and
ECE(10) over ALL scored pairs (unthresholded, calibrated), and qualifier top-1 accuracy of the F1 point."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--engines", nargs="*", default=["coreml-fp32-cpu"])
ap.add_argument("--exclude-pool", action="store_true")
a = ap.parse_args()
cats = ["overall", "place", "per_per", "org_org", "per_org"]


def cell(r):
    if r["tp"] + r["fp"] == 0:
        return "no pairs kept"
    return f"{r['tp']}/{r['fp']}/{r['fn']} P{r['P']:.2f} R{r['R']:.2f} F{r['F1']:.2f}"


for e in a.engines:
    f1 = exp.evaluate(exp.load_cfg(exp.ROOT / f"eval/experiments/final-{e}-f1.json"), a.split, not a.exclude_pool)
    p8 = exp.evaluate(exp.load_cfg(exp.ROOT / f"eval/experiments/final-{e}-p80.json"), a.split, not a.exclude_pool)
    print(f"\n**{e}**, split={a.split}, pool {'excluded' if a.exclude_pool else 'included'}; "
          f"macro-F1 over categories: F1 point {f1['macro_F1']}, P>=0.8 point {p8['macro_F1']}; "
          f"qualifier top-1: {f1['qual_acc'][0]}/{f1['qual_acc'][1]}\n")
    print("| category | gold | F1 point (tp/fp/fn, P, R, F1) | P>=0.8 point | AUPRC | ECE |")
    print("|---|---|---|---|---|---|")
    for c in cats:
        print(f"| {c} | {f1[c]['gold']} | {cell(f1[c])} | {cell(p8[c])} | {f1[c]['AUPRC']} | {f1[c]['ECE']} |")
