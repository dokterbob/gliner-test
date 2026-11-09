"""Sample windows with entities for a quick qualitative read, plus label counts.

Usage:
    uv run python scripts/quality_review.py --preds data/preds/coreml-fp32-gpu.jsonl --sample 40
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--windows", type=Path, default=Path("data/windows.jsonl"))
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    windows_by_id = {json.loads(line)["window_id"]: json.loads(line) for line in args.windows.open()}

    preds = [json.loads(line) for line in args.preds.open()]
    label_counts = Counter()
    relation_counts = Counter()
    n_with_entities = 0
    for p in preds:
        if p["entities"]:
            n_with_entities += 1
        for e in p["entities"]:
            label_counts[e["label"]] += 1
        for r in p["relations"]:
            relation_counts[r["name"]] += 1

    print(f"windows: {len(preds)}, with >=1 entity: {n_with_entities}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label:10s} {count}")
    print("\nRelation distribution:")
    for name, count in relation_counts.most_common():
        print(f"  {name:12s} {count}")

    with_entities = [p for p in preds if p["entities"]]
    random.seed(args.seed)
    sample = random.sample(with_entities, min(args.sample, len(with_entities)))
    print(f"\n=== sample of {len(sample)} windows with entities ===")
    for p in sample:
        w = windows_by_id.get(p["window_id"], {})
        print(f"\n--- {p['window_id']} ---")
        print(w.get("text", "<window text not found>"))
        for e in p["entities"]:
            print(f"  [{e['label']:5s} {e['score']:.2f}] {e['text']!r}")
        for r in p["relations"]:
            print(f"  ({r['name']} {r['score']:.2f}) {r['head']['text']!r} -> {r['tail']['text']!r}")


if __name__ == "__main__":
    main()
