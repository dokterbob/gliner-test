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
        cmd = ["uv", "run", "python", "eval/experiments/run_desc.py" if f.get("descriptions") else "scripts/run_coreml.py", "--windows", str(windows), "--schema", cfg["schema"],
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


def raw_file(cfg):
    if cfg.get("raw_file"):
        return ROOT / cfg["raw_file"]
    return exp_dir(cfg.get("raw") or cfg["id"]) / "raw.jsonl"


def pairs_for(cfg, raw_path, categories, apply_threshold=False):
    post = cfg["post"]
    mode = post.get("collapse", "max")
    sn = post.get("score_names", "all")
    out = {}
    cal = None
    if post.get("calibration"):
        cal = json.loads((ROOT / post["calibration"]).read_text())
    for rec in jl(raw_path):
        pairs = R.predictions_to_pairs(rec, categories, mode, sn, bool(post.get("nms")))
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


def evaluate(cfg, split="dev", pool=True, thr_override=None, kind=None):
    eid = cfg.get("raw") or cfg["id"]
    raw = raw_file(cfg)
    schema_path = ROOT / (cfg.get("post_schema") or cfg["schema"])
    sch = R.load_schema(schema_path)
    if not sch["categories"]:
        sch = R.load_schema(ROOT / "schema/ontonotes.json")
    cats_r = sch["categories"]
    gthr = cfg["post"].get("threshold", sch["threshold"])
    over = cfg["post"].get("cat_thresholds", {})
    if not over and not cfg["post"].get("schema_thresholds"):
        over = {c["id"]: gthr for c in cats_r}  # plain global threshold, ignoring the schema's per-category ones
    pairs = pairs_for(cfg, raw, cats_r)
    gold = C.load_gold(ROOT / "eval/gold.jsonl", split)
    if not pool:
        for rec in gold.values():
            rec["relations"] = [r for r in rec.get("relations", []) if r.get("source") != "pool"]
    if kind:
        gold = {w: r for w, r in gold.items() if r.get("kind") == kind}
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
    row = {"id": cfg["id"], "desc": cfg.get("desc", ""), "split": split + (f"/{kind}" if kind else ""), "pool": pool, "engine": cfg["engine"],
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
    # oracle single global threshold on dev (0.05 steps): what the score ranking supports without per-category tuning
    best = (-1.0, None, None)
    for i in range(1, 20):
        t = i / 20
        tp, fp, fn = ev_all.counts("overall", t)
        f1 = prf(tp, fp, fn)[2]
        mf = []
        for c in ev_all.views[1:]:
            a, b, d = ev_all.counts(c, t)
            if ev_all.n_gold[c]:
                mf.append(prf(a, b, d)[2])
        m = sum(mf) / len(mf)
        if f1 > best[0]:
            best = (f1, t, m)
    row["best_global"] = {"thr": best[1], "F1": round(best[0], 3), "macro_F1": round(best[2], 3)}
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
        f"AUPRC {r['overall']['AUPRC']} bestglobal {r['best_global']['F1']}@{r['best_global']['thr']} | " + " ".join(f"{c}:{r[c]['F1']:.2f}({r[c]['tp']}/{r[c]['fp']}/{r[c]['fn']})" for c in cats)
    return s + f" | lat {r['lat_ms_per_window']} ms/win"


def write_md():
    rows = [json.loads(x) for x in open(ROOT / "eval/experiments/results.jsonl") if x.strip()]
    L = ["# Experiment results (generated by exp.py; do not edit)\n",
         "Cell format `tp/fp/fn`. All dev unless noted. overall = untyped, unordered pairs; macro = mean F1 over the 4 categories.\n",
         "| id | split | pool | thr | P | R | F1 | macro F1 | AUPRC | ECE | best global thr: F1 | place F1 | per_per F1 | org_org F1 | per_org F1 | tp/fp/fn | qual acc | lat ms/win | desc |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        o = r["overall"]
        qa = f"{r['qual_acc'][0]}/{r['qual_acc'][1]}"
        L.append(f"| {r['id']} | {r['split']} | {'y' if r['pool'] else 'n'} | {r['threshold']} | {o['P']} | {o['R']} | {o['F1']} | {r['macro_F1']} | "
                 f"{o['AUPRC']} | {o['ECE']} | {r['best_global']['thr']}: {r['best_global']['F1']} | " + " | ".join(f"{r[c]['F1']}" for c in ["place", "per_per", "org_org", "per_org"]) +
                 f" | {o['tp']}/{o['fp']}/{o['fn']} | {qa} | {r['lat_ms_per_window']} | {r['desc']} |")
    (ROOT / "eval/experiments/results.md").write_text("\n".join(L) + "\n")


def cmd_eval(args):
    for p in args.configs:
        cfg = load_cfg(p)
        row = evaluate(cfg, args.split, not args.exclude_pool, kind=args.kind)
        print(fmt_row(row))
        if not args.no_record:
            with open(ROOT / "eval/experiments/results.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            write_md()


def tune_thresholds(cfg, split="dev", precision_target=0.8):
    """Per-category thresholds on `split` (dev!): (a) maximise F1 -- the middle of the plateau of equally good
    0.05-grid thresholds; (b) lowest threshold with precision >= target (else the highest grid point)."""
    eid = cfg.get("raw") or cfg["id"]
    sch = R.load_schema(ROOT / (cfg.get("post_schema") or cfg["schema"]))
    if not sch["categories"]:
        sch = R.load_schema(ROOT / "schema/ontonotes.json")
    pairs = pairs_for(cfg, raw_file(cfg), sch["categories"])
    gold = C.load_gold(ROOT / "eval/gold.jsonl", split)
    pcats, _ = C.load_pair_schema(ROOT / "schema/ontonotes.json")
    ev = C.PairEval({w: {"entities": [], "pairs": ps} for w, ps in pairs.items()}, gold, pcats, "overlap", True, True)
    grid = [i / 20 for i in range(1, 20)]
    f1_thr, p_thr = {}, {}
    for c in ev.views[1:]:
        scored = []
        for t in grid:
            tp, fp, fn = ev.counts(c, t)
            p, r, f1 = prf(tp, fp, fn)
            scored.append((t, f1, p, tp + fp))
        best = max(x[1] for x in scored)
        plateau = [x[0] for x in scored if x[1] >= best - 1e-9]
        f1_thr[c] = round(plateau[len(plateau) // 2], 2)
        ok = [x[0] for x in scored if x[2] >= precision_target and x[3] > 0]
        p_thr[c] = round(ok[0], 2) if ok else 0.95
    return f1_thr, p_thr


def calib_samples(cfg, split="dev"):
    """{category: [(score, 0/1)]} over every window-level pair of `split`, labelled against gold; plus per-window lists."""
    eid = cfg.get("raw") or cfg["id"]
    sch = R.load_schema(ROOT / (cfg.get("post_schema") or cfg["schema"]))
    if not sch["categories"]:
        sch = R.load_schema(ROOT / "schema/ontonotes.json")
    c2 = json.loads(json.dumps(cfg))
    c2["post"] = {k: v for k, v in cfg["post"].items() if k != "calibration"}
    pairs = pairs_for(c2, raw_file(c2), sch["categories"])
    gold = {g["window_id"]: g for g in jl(ROOT / "eval/gold.jsonl") if split == "all" or g.get("split") == split}
    per_win = []
    for wid, ps in pairs.items():
        g = gold.get(wid)
        if g is None:
            continue
        per_win.append([(p["category"], p["score"], lab) for p, lab in zip(ps, R.label_pairs(ps, g))])
    return per_win


def _bucket(per_win):
    out = {}
    for w in per_win:
        for c, s, l in w:
            out.setdefault(c, []).append((s, l))
    return out


def cmd_calibrate(args):
    cfg = load_cfg(args.config)
    per_win = calib_samples(cfg, "dev")
    name = Path(args.config).stem
    outd = ROOT / "eval/calibration"
    outd.mkdir(parents=True, exist_ok=True)
    iso = R.fit(_bucket(per_win), engine=cfg["engine"], min_samples=args.min_samples)
    tmp = R.fit_temperature_table(_bucket(per_win), engine=cfg["engine"], min_samples=args.min_samples)
    (outd / f"{args.tag}-isotonic.json").write_text(json.dumps(iso, indent=1))
    (outd / f"{args.tag}-temperature.json").write_text(json.dumps(tmp, indent=1))
    print("isotonic n:", {k: v["n"] for k, v in iso["categories"].items()})
    print("temperature:", {"global": round(tmp["global"]["T"], 3), **{k: round(v["T"], 3) for k, v in tmp["categories"].items()}})
    # 5-fold (by window) cross-fitted ECE on dev: honest comparison of the two methods without touching test
    folds = [per_win[i::5] for i in range(5)]
    res = {"raw": ([], []), "isotonic": ([], []), "temperature": ([], [])}
    for k in range(5):
        tr = _bucket([w for j, f in enumerate(folds) if j != k for w in f])
        iso_k = R.fit(tr, min_samples=args.min_samples)
        tmp_k = R.fit_temperature_table(tr, min_samples=args.min_samples)
        for w in folds[k]:
            for c, sc, l in w:
                pr = {"category": c, "score": sc}
                res["raw"][0].append(sc)
                res["raw"][1].append(l)
                res["isotonic"][0].append(R.apply([pr], iso_k)[0]["score"])
                res["isotonic"][1].append(l)
                res["temperature"][0].append(R.apply_temperature([pr], tmp_k)[0]["score"])
                res["temperature"][1].append(l)
    for m, (sc, lab) in res.items():
        nll = -sum(math.log(max(min(s, 1 - 1e-6), 1e-6)) if l else math.log(max(1 - min(s, 1 - 1e-6), 1e-6)) for s, l in zip(sc, lab)) / len(sc)
        print(f"5-fold dev  {m:<12} ECE(10)={C.ece(sc, lab):.3f}  logloss={nll:.3f}  n={len(sc)}")


def cmd_cvthr(args):
    """5-fold (by window) cross-validation on dev: per-category thresholds vs one global threshold, both tuned on 4 folds
    (max F1 on the 0.05 grid, plateau middle) and scored pooled on the held-out folds. Uses calibrated scores of the config."""
    cfg = load_cfg(args.config)
    sch = R.load_schema(ROOT / (cfg.get("post_schema") or cfg["schema"]))
    pairs = pairs_for(cfg, raw_file(cfg), sch["categories"])
    gold = C.load_gold(ROOT / "eval/gold.jsonl", "dev")
    pcats, _ = C.load_pair_schema(ROOT / "schema/ontonotes.json")
    wids = sorted(gold)
    grid = [i / 20 for i in range(1, 20)]

    def ev_of(ws):
        return C.PairEval({w: {"entities": [], "pairs": pairs.get(w, [])} for w in ws}, {w: gold[w] for w in ws},
                          pcats, "overlap", True, True)

    def best(ev, view):
        sc = [(prf(*ev.counts(view, t))[2], t) for t in grid]
        top = max(f for f, _ in sc)
        pl = [t for f, t in sc if f >= top - 1e-9]
        return pl[len(pl) // 2]

    tot = {"global": [0, 0, 0], "percat": [0, 0, 0], "fixed0.5": [0, 0, 0]}
    for k in range(5):
        test = wids[k::5]
        train = [w for w in wids if w not in test]
        evtr, evte = ev_of(train), ev_of(test)
        tg = best(evtr, "overall")
        for name, thr in (("global", {c["id"]: tg for c in pcats}), ("percat", {c["id"]: best(evtr, c["id"]) for c in pcats}),
                          ("fixed0.5", {c["id"]: 0.5 for c in pcats})):
            for c in pcats:
                a, b, d = evte.counts(c["id"], thr[c["id"]])
                tot[name][0] += a
                tot[name][1] += b
                tot[name][2] += d
    for name, (a, b, d) in tot.items():
        p, r, f = prf(a, b, d)
        print(f"{name:<9} tp/fp/fn {a}/{b}/{d}  P {p:.3f} R {r:.3f} F1 {f:.3f}   (sum over categories)")


def cmd_tune(args):
    cfg = load_cfg(args.config)
    f1_thr, p_thr = tune_thresholds(cfg)
    print("max-F1 thresholds:", json.dumps(f1_thr))
    print("P>=0.8 thresholds:", json.dumps(p_thr))


def cmd_rebuild(args):
    (ROOT / "eval/experiments/results.jsonl").write_text("")
    for p in sorted(Path(ROOT / "eval/experiments").glob("[0-9][0-9]*.json")):
        row = evaluate(load_cfg(p), args.split, True)
        print(fmt_row(row))
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
    e.add_argument("--kind", choices=["sentence", "passage"], default=None)
    e.set_defaults(fn=cmd_eval)
    c = sub.add_parser("calibrate", help="fit isotonic + temperature on dev; print 5-fold dev ECE")
    c.add_argument("config")
    c.add_argument("--tag", required=True)
    c.add_argument("--min-samples", type=int, default=8)
    c.set_defaults(fn=cmd_calibrate)
    v = sub.add_parser("cvthr", help="5-fold dev CV: per-category vs global threshold on a config's calibrated scores")
    v.add_argument("config")
    v.set_defaults(fn=cmd_cvthr)
    t = sub.add_parser("tune", help="print dev-tuned per-category thresholds for a config")
    t.add_argument("config")
    t.set_defaults(fn=cmd_tune)
    b = sub.add_parser("rebuild", help="re-evaluate every NN-*.json config (dev) and rewrite results.jsonl/md")
    b.add_argument("--split", default="dev", choices=["dev"])
    b.set_defaults(fn=cmd_rebuild)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
