"""Prompt-token cost of each pass of a schema file (bucket: 128 total, windows shrink if a pass > 59).

    uv run python eval/experiments/tokens.py schema/ontonotes.json eval/experiments/schemas/*.json
A relation entry may carry a `description`; it is sent as {prompt: {"description": ...}}.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from download_models import REVISION  # noqa: E402

from gliner_test import windowing as W  # noqa: E402
from gliner_test.compiled_models import CACHE_ROOT  # noqa: E402

processor, collator, Schema = W.load_runtime_pieces(CACHE_ROOT / REVISION)


def rel_arg(r):
    return {r.get("prompt", r["name"]): {"description": r["description_short"]}} if r.get("description_short") else r.get("prompt", r["name"])


def toks(s):
    return int(collator([("", s)]).input_ids.shape[1])


for path in sys.argv[1:]:
    cfg = json.loads(Path(path).read_text())
    ent = toks(Schema().entities([e["label"] for e in cfg["entities"]]).build())
    out = []
    for pn in cfg.get("relation_passes") or ["default"]:
        rels = [r for r in cfg["relations"] if r.get("pass", "default") == pn]
        prompts = {}
        for r in rels:
            if r.get("description_short"):
                prompts[r["prompt"]] = r["description_short"]
        if prompts:
            built = Schema().relations({r["prompt"]: (r.get("description_short") or "") for r in rels}).build()
        else:
            built = Schema().relations([r["prompt"] for r in rels]).build()
        out.append(f"{pn}={toks(built)}")
    print(f"{path}: entities={ent} " + " ".join(out))
