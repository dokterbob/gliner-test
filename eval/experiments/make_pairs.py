"""Builds data/pairs (F1 operating point), data/pairs-p80 and data/pairs-all (no threshold) with the
gliner_test.relations CLI from the final-*.json configs. Run mk_final.py first."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for cfgp in sorted((ROOT / "eval/experiments").glob("final-*-*.json")):
    cfg = json.loads(cfgp.read_text())
    e = cfg["raw_file"].split("/")[-1].removesuffix(".jsonl")
    kind = cfgp.stem.rsplit("-", 1)[1]  # f1 | p80
    p = cfg["post"]
    base = [sys.executable, "-m", "gliner_test.relations", "--preds", cfg["raw_file"], "--score-names", p["score_names"],
            "--collapse", p["collapse"], "--calibration", p["calibration"]]
    if p.get("nms"):
        base.append("--nms")
    thr = [x for k, v in p.get("cat_thresholds", {}).items() for x in ("--cat-threshold", f"{k}={v}")]
    outs = {"f1": "data/pairs", "p80": "data/pairs-p80", "percat": "data/pairs-percat"}[kind]
    subprocess.run(base + thr + ["--out", f"{outs}/{e}.jsonl"], cwd=ROOT, check=True)
    if kind == "f1":
        zero = [x for k in ("place", "per_per", "org_org", "per_org") for x in ("--cat-threshold", f"{k}=0")]
        subprocess.run(base + zero + ["--out", f"data/pairs-all/{e}.jsonl"], cwd=ROOT, check=True)
