"""Writes eval/experiments/final-<engine>-{f1,p80}.json and the CLI commands that produce data/pairs*/ files.
    uv run python eval/experiments/mk_final.py
f1 point: one global threshold 0.5 on the temperature-calibrated pair score (schema thresholds are null).
p80 point: per-engine lowest calibrated threshold with dev precision >= 0.8."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import exp  # noqa: E402

ENGINES = {"coreml-fp32-cpu": "coreml-fp32", "coreml-fp16-w8-cpu": "coreml-fp16-w8", "torch-fp32": "torch",
           "torch-jointie": "torch-jointie", "rust-ort-fp32": "rust-ort"}
OWN_F1 = {"torch-jointie", "rust-ort-fp32"}
out = {}
for e, kind in ENGINES.items():
    base = {"id": f"final-{e}-f1", "desc": f"FINAL {e}: max/qualifiers/NMS, temperature calibration, F1 operating point",
            "schema": "schema/ontonotes.json", "engine": kind, "raw_file": f"data/preds/gold/{e}.jsonl", "flags": {},
            "post": {"collapse": "max", "score_names": "qualifiers", "nms": True, "threshold": 0.5,
                     "calibration": f"eval/calibration/{e}.json", "cat_thresholds": {}}}
    f1_thr, p_thr = exp.tune_thresholds(base)  # tuned on dev only
    c1 = json.loads(json.dumps(base))
    # F1 point = one global threshold 0.5 on the calibrated score (schema thresholds are null)
    c2 = json.loads(json.dumps(base))
    c2["id"] = f"final-{e}-p80"
    c2["desc"] = f"FINAL {e}: same, precision>=0.8 operating point (dev-tuned per category)"
    c2["post"]["cat_thresholds"] = p_thr
    json.dump(c1, open(Path(__file__).parent / f"final-{e}-f1.json", "w"), indent=1)
    json.dump(c2, open(Path(__file__).parent / f"final-{e}-p80.json", "w"), indent=1)
    if e == "coreml-fp32-cpu":  # dev-tuned per-category thresholds: negative result, kept for the report
        c3 = json.loads(json.dumps(base))
        c3["id"] = f"final-{e}-percat"
        c3["desc"] = f"{e}: per-category thresholds maximising dev F1 (in-sample on dev; does not transfer)"
        c3["post"]["cat_thresholds"] = f1_thr
        json.dump(c3, open(Path(__file__).parent / f"final-{e}-percat.json", "w"), indent=1)
    print(e, "f1 thr", f1_thr, "p80", p_thr)
