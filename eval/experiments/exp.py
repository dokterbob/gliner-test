"""Experiment driver: run a model config once (threshold 0.0), post-process offline, score on gold.

    uv run python eval/experiments/exp.py run  eval/experiments/01a-*.json     # model run (under lockf) + merge to parent items
    uv run python eval/experiments/exp.py eval eval/experiments/01a-*.json --split dev [--no-record]

Config (JSON):
  id, desc
  schema        schema file sent to the model (prompts / passes)
  post_schema   schema whose pair_categories drive post-processing (default: schema, else schema/ontonotes.json)
  engine        coreml-fp32 (default) | coreml-fp16-w8 | torch
  raw           optional experiment id whose raw predictions to reuse (no model run)
  flags         {"arg_threshold": x, "pair_cap": n}
  post          {"collapse": max|noisy-or, "score_names": all|qualifiers|detectors, "threshold": 0.5,
                 "cat_thresholds": {cat: x}, "calibration": null|"isotonic"|"temperature"}
Raw outputs live in data/exp/<id>/ (not committed); results go to eval/experiments/results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
LOCK = str(ROOT / ".cache" / "model.lock")

import compare as C  # noqa: E402
from gliner_test import relations as R  # noqa: E402


def jl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def load_cfg(path):
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("engine", "coreml-fp32")
    cfg.setdefault("flags", {})
    cfg.setdefault("post", {})
    return cfg


def exp_dir(eid):
    return ROOT / "data" / "exp" / eid


def windows_for_schema(schema):
    """Windows are sized against the schema's own prompts (a longer relation prompt = smaller windows)."""
    stem = Path(schema).stem
    path = ROOT / "data" / ("gold-windows.jsonl" if schema == "schema/ontonotes.json" else f"gold-windows--{stem}.jsonl")
    if not path.exists():
        subprocess.run([sys.executable, "scripts/gold_windows.py", "split", "--schema", schema, "--out", str(path)],
                       cwd=ROOT, check=True)
    return path


def run_model(cfg, windows=None, out_name="win.jsonl"):
    eid = cfg["id"]
    d = exp_dir(eid)
    d.mkdir(parents=True, exist_ok=True)
    windows = windows or windows_for_schema(cfg["schema"])
    out = d / out_name
    eng = cfg["engine"]
    f = cfg["flags"]
    if eng.startswith("coreml"):
        prec = "fp32" if eng == "coreml-fp32" else "fp16-w8"
        cmd = ["uv", "run", "python", "scripts/run_coreml.py", "--windows", str(windows), "--schema", cfg["schema"],
               "--precision", prec, "--compute-units", "cpu", "--relation-threshold", "0.0", "--out", str(out),
               "--progress-every", "50"]
    elif eng == "torch":
        cmd = ["uv", "run", "python", "scripts/run_torch.py", "--windows", str(windows), "--schema", cfg["schema"],
               "--relation-threshold", "0.0", "--out", str(out)]
    else:
        raise SystemExit(f"engine {eng}")
    if f.get("arg_threshold") is not None:
        cmd += ["--arg-threshold", str(f["arg_threshold"])]
    if f.get("pair_cap") is not None:
        cmd += ["--pair-cap", str(f["pair_cap"])]
    full = ["/usr/bin/lockf", "-k", LOCK] + cmd
    print("+", " ".join(full), file=sys.stderr)
    subprocess.run(full, cwd=ROOT, check=True)
    return out


def cmd_run(args):
    for p in args.configs:
        cfg = load_cfg(p)
        if cfg.get("raw"):
            print(f"{cfg['id']}: reuses raw of {cfg['raw']}", file=sys.stderr)
            continue
        out = run_model(cfg)
        subprocess.run([sys.executable, "scripts/gold_windows.py", "merge", "--preds", str(out), "--windows",
                        str(windows_for_schema(cfg["schema"])), "--out",
                        str(exp_dir(cfg["id"]) / "raw.jsonl")], cwd=ROOT, check=True)


def pairs_for(cfg, raw_path, categories, apply_threshold=False):
    post = cfg["post"]
    mode = post.get("collapse", "max")
    sn = post.get("score_names", "all")
    out = {}
    cal = None
    if post.get("calibration"):
        cal = json.loads((ROOT / post["calibration"]).read_text())
    for rec in jl(raw_path):
        pairs = R.predictions_to_pairs(rec, categories, mode, sn)
        if cal is not None:
            pairs = apply_cal(pairs, cal)
        out[rec["window_id"]] = pairs
    return out


def apply_cal(pairs, cal):
    if cal["method"] == "isotonic":
        return R.apply(pairs, cal)
    if cal["method"] == "temperature":
        return R.apply_temperature(pairs, cal)
    raise ValueError(cal["method"])


def prf(tp, fp, fn):
    return C.prf1(tp, fp, fn)


def evaluate(cfg, split="dev", pool=True, thr_override=None):
    eid = cfg["raw"] or cfg["id"] if cfg.get("raw") else cfg["id"]
    raw = exp_dir(eid) / "raw.jsonl"
    schema_path = ROOT / (cfg.get("post_schema") or cfg["schema"])
    sch = R.load_schema(schema_path)
    if not sch["categories"]:
        sch = R.load_schema(ROOT / "schema/ontonotes.json")
    cats_r = sch["categories"]
    gthr = cfg["post"].get("threshold", sch["threshold"])
    over = cfg["post"].get("cat_thresholds", {})
    pairs = pairs_for(cfg, raw, cats_r)
    gold = C.load_gold(ROOT / "eval/gold.jsonl", split)
    if not pool:
        for rec in gold.values():
            rec["relations"] = [r for r in rec.get("relations", []) if r.get("source") != "pool"]
    pcats, _ = C.load_pair_schema(ROOT / "schema/ontonotes.json")

    def rec_of(pdict, thr=False):
        out = {}
        for wid, ps in pdict.items():
            if thr:
                ps = R.apply_thresholds(ps, cats_r, gthr, over)
            out[wid] = {"entities": [], "pairs": ps}
        return out

    ev_all = C.PairEval(rec_of(pairs), gold, pcats, "overlap", pool, True)
    ev_thr = C.PairEval(rec_of(pairs, True), gold, pcats, "overlap", pool, True)
    row = {"id": cfg["id"], "desc": cfg.get("desc", ""), "split": split, "pool": pool, "engine": cfg["engine"],
           "threshold": gthr, "cat_thresholds": over}
    cat_f1 = []
    for v in ev_thr.views:
        tp, fp, fn = ev_thr.counts(v, 0.0)
        p, r, f1 = prf(tp, fp, fn)
        ra = ev_all.ranked[v]
        row[v] = {"gold": ev_thr.n_gold[v], "tp": tp, "fp": fp, "fn": fn, "P": round(p, 3), "R": round(r, 3),
                  "F1": round(f1, 3), "AUPRC": _r(C.average_precision(ra, ev_all.n_gold[v])),
                  "ECE": _r(C.ece([t[0] for t in ra], [t[1] is not None for t in ra]))}
        if v != "overall" and ev_thr.n_gold[v]:
            cat_f1.append(f1)
    row["macro_F1"] = round(sum(cat_f1) / len(cat_f1), 3)
    ok, n = ev_thr.qualifier_accuracy(0.0, pcats)
    row["qual_acc"] = [ok, n]
    lat = None
    wp = exp_dir(eid) / "win.jsonl"
    if wp.exists():
        ls = [r["latency_ms"] for r in jl(wp)]
        lat = round(sum(ls) / len(ls), 1)
    row["lat_ms_per_window"] = lat
    # score separation (TP vs FP) at no threshold, overall view
    tps = sorted(t[0] for t in ev_all.ranked["overall"] if t[1] is not None)
    fps = sorted(t[0] for t in ev_all.ranked["overall"] if t[1] is None)
    row["median_tp"] = _r(_med(tps))
    row["median_fp"] = _r(_med(fps))
    return row


def _med(xs):
    return xs[len(xs) // 2] if xs else float("nan")


def _r(x):
    return None if x != x else round(x, 3)


def fmt_row(r):
    cats = ["place", "per_per", "org_org", "per_org"]
    s = f"{r['id']:<22} {r['split']:<4} thr={r['threshold']:<4} overall P/R/F1 " \
        f"{r['overall']['P']:.3f}/{r['overall']['R']:.3f}/{r['overall']['F1']:.3f} macroF1 {r['macro_F1']:.3f} " \
        f"AUPRC {r['overall']['AUPRC']} | " + " ".join(f"{c}:{r[c]['F1']:.2f}({r[c]['tp']}/{r[c]['fp']}/{r[c]['fn']})" for c in cats)
    return s + f" | lat {r['lat_ms_per_window']} ms/win"


def write_md():
    rows = [json.loads(x) for x in open(ROOT / "eval/experiments/results.jsonl") if x.strip()]
    L = ["# Experiment results (generated by exp.py; do not edit)\n",
         "Cell format `tp/fp/fn`. All dev unless noted. overall = untyped, unordered pairs; macro = mean F1 over the 4 categories.\n",
         "| id | split | pool | thr | P | R | F1 | macro F1 | AUPRC | ECE | place F1 | per_per F1 | org_org F1 | per_org F1 | tp/fp/fn | qual acc | lat ms/win | desc |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r["overall"]
        qa = f"{r['qual_acc'][0]}/{r['qual_acc'][1]}"
        L.append(f"| {r['id']} | {r['split']} | {'y' if r['pool'] else 'n'} | {r['threshold']} | {o['P']} | {o['R']} | {o['F1']} | {r['macro_F1']} | "
                 f"{o['AUPRC']} | {o['ECE']} | " + " | ".join(f"{r[c]['F1']}" for c in ["place", "per_per", "org_org", "per_org"]) +
                 f" | {o['tp']}/{o['fp']}/{o['fn']} | {qa} | {r['lat_ms_per_window']} | {r['desc']} |")
    (ROOT / "eval/experiments/results.md").write_text("\n".join(L) + "\n")


def cmd_eval(args):
    for p in args.configs:
        cfg = load_cfg(p)
        row = evaluate(cfg, args.split, not args.exclude_pool)
        print(fmt_row(row))
        if not args.no_record:
            with open(ROOT / "eval/experiments/results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            write_md()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("configs", nargs="+")
    r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval")
    e.add_argument("configs", nargs="+")
    e.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    e.add_argument("--exclude-pool", action="store_true")
    e.add_argument("--no-record", action="store_true")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
