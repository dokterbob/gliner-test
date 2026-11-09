"""Generates the coarse-prompt schema variants (experiment 03) from schema/ontonotes.json + ontonotes-v2.1.json.
    uv run python eval/experiments/schemas/make_coarse.py
"""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
v21 = json.load(open(ROOT / "schema/ontonotes-v2.1.json"))
v22 = json.load(open(ROOT / "schema/ontonotes.json"))


def coarse(p):
    return [
        {"name": "coarse_place", "prompt": p[0], "pass": "coarse", "head": ["PERSON", "ORG", "FAC", "GPE", "LOC"], "tail": ["GPE", "LOC", "FAC"], "symmetric": False},
        {"name": "coarse_affil", "prompt": p[1], "pass": "coarse", "head": ["PERSON"], "tail": ["ORG"], "symmetric": False},
        {"name": "coarse_partner", "prompt": p[2], "pass": "coarse", "head": ["ORG"], "tail": ["ORG"], "symmetric": True},
        {"name": "coarse_related", "prompt": p[3], "pass": "coarse", "head": ["PERSON"], "tail": ["PERSON"], "symmetric": True},
    ]


def mk(fine, prompts, name, comment):
    s = copy.deepcopy(v22)
    s["$comment"] = comment
    det = {"place": "coarse_place", "per_org": "coarse_affil", "org_org": "coarse_partner", "per_per": "coarse_related"}
    for c in s["pair_categories"]:
        c["detectors"] = [det[c["id"]]]
        if not fine:
            c["qualifiers"] = []
    if fine:
        s["relations"] = coarse(prompts) + copy.deepcopy(v21["relations"])
        s["relation_passes"] = ["coarse"] + v21["relation_passes"]
    else:
        s["relations"] = coarse(prompts)
        s["relation_passes"] = ["coarse"]
    json.dump(s, open(ROOT / f"eval/experiments/schemas/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    P = ["located in", "affiliated with", "partner of", "related to"]
    mk(False, P, "coarse-only", "Exp 03a: one coarse detector pass (4 prompts) + entity pass = 2 calls per window")
    mk(True, P, "coarse-fine", "Exp 03b: coarse pass + v2.1 fine passes (4 calls); coarse names detect, fine names qualify")
    mk(False, ["based in", "works for", "affiliated with", "relationship"], "coarse-only-b",
       "Exp 03e: alternative coarse wording; 2 calls per window")
