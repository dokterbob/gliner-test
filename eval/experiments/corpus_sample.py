"""Stratified sample of document-level corpus pairs for manual review (writes the sample JSON; the verdicts live in
corpus-sample.md). Per category: up to 6 pairs with score>=0.5, 4 in [0.2,0.5), 4 below 0.2 (evenly spaced ranks)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
pairs = [json.loads(l) for l in open(ROOT / "data/pairs-corpus/coreml-fp16-w8-cpu.docs-all.jsonl")]
text = {}
want = {p["chunk_id"] for p in pairs}
for l in open(ROOT / "data/corpus.jsonl"):
    r = json.loads(l)
    if r["chunk_id"] in want:
        text[r["chunk_id"]] = r["text"]
out = []
for cat in ("place", "per_per", "org_org", "per_org"):
    ps = sorted([p for p in pairs if p["category"] == cat], key=lambda p: -p["score"])
    for lo, hi, k in ((0.5, 9, 6), (0.2, 0.5, 4), (-1, 0.2, 4)):
        band = [p for p in ps if lo <= p["score"] < hi]
        idx = sorted({int(i * len(band) / k) for i in range(k)}) if len(band) > k else range(len(band))
        out += [band[i] for i in idx]
out = out[:40]
for i, p in enumerate(out):
    t = text[p["chunk_id"]]
    a = min(p["head"]["start"], p["tail"]["start"])
    b = max(p["head"]["end"], p["tail"]["end"])
    ctx = t[max(0, a - 70): b + 70].replace("\n", " ")
    print(f"[{i + 1}] {p['category']} {p['score']:.2f} ev={p['evidence']} | {p['head']['text']!r} <-> {p['tail']['text']!r} | quals={[q['name'] for q in p['qualifiers']][:2]}\n    ...{ctx}...")
json.dump(out, open(ROOT / "data/pairs-corpus/sample.json", "w"))
