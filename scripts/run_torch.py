"""Native PyTorch reference run: fastino/gliner2.5-multi-v1 via the `gliner2`
package (2.0.0, the version the CoreML export pins), over the same windows
JSONL used by the Rust (gliner25-rs/ONNX Runtime) and CoreML runners.

Writes predictions in the shared format:
    {"window_id", "entities": [{"label","text","start","end","score"}],
     "relations": [{"name","head":{...},"tail":{...},"score"}], "latency_ms"}

Labels and the shared threshold come from schema/ontonotes.json -- never
hard-coded -- so all three engines read the same prompt strings.

Usage:
    uv run python scripts/run_torch.py \
        --schema schema/ontonotes.json \
        --windows data/windows.jsonl \
        --out data/preds/torch-fp32.jsonl \
        --device cpu
"""
import argparse
import json
import resource
import time
from pathlib import Path


def load_schema(path: Path):
    schema = json.loads(path.read_text())
    label_to_tag = {e["label"]: e["ontonotes"] for e in schema["entities"]}
    entity_labels = [e["label"] for e in schema["entities"]]
    relation_names = [r["name"] for r in schema["relations"]]
    threshold = schema["threshold"]
    return entity_labels, relation_names, label_to_tag, threshold


def peak_rss_mb() -> float:
    # macOS getrusage reports ru_maxrss in bytes; Linux in KB. This project
    # targets macOS, but guard anyway.
    import sys

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default="fastino/gliner2.5-multi-v1")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all windows")
    args = ap.parse_args()

    entity_labels, relation_names, label_to_tag, threshold = load_schema(args.schema)

    import torch
    from gliner2 import AutoExtractor

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[run_torch] mps requested but not available, falling back to cpu")
        device = "cpu"

    print(f"[run_torch] loading {args.repo} on device={device} ...")
    t0 = time.perf_counter()
    try:
        model = AutoExtractor.from_pretrained(args.repo, map_location=device)
    except Exception as e:  # noqa: BLE001 -- report mps misbehavior rather than crash silently
        if device == "mps":
            print(f"[run_torch] mps load failed ({e!r}); retrying on cpu")
            device = "cpu"
            model = AutoExtractor.from_pretrained(args.repo, map_location=device)
        else:
            raise
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[run_torch] loaded in {load_ms:.1f} ms")

    schema_obj = model.create_schema().entities(entity_labels).relations(relation_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    latencies_ms = []
    total_words = 0
    n = 0
    mps_error = None

    with args.windows.open() as fin, args.out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["text"]

            t0 = time.perf_counter()
            try:
                result = model.extract(
                    text,
                    schema_obj,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
            except Exception as e:  # noqa: BLE001
                if device == "mps" and mps_error is None:
                    mps_error = repr(e)
                    print(f"[run_torch] mps extract failed ({e!r}); this window skipped")
                    continue
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            total_words += len(text.split())

            entities = []
            for label, items in result.get("entities", {}).items():
                tag = label_to_tag.get(label, label)
                for it in items:
                    entities.append({
                        "label": tag,
                        "text": it["text"],
                        "start": it["start"],
                        "end": it["end"],
                        "score": it.get("confidence", 1.0),
                    })

            relations = []
            for name, pairs in result.get("relation_extraction", {}).items():
                for pair in pairs:
                    head, tail = pair["head"], pair["tail"]
                    score = min(head.get("confidence", 1.0), tail.get("confidence", 1.0))
                    relations.append({
                        "name": name,
                        "head": {"text": head["text"], "start": head["start"], "end": head["end"]},
                        "tail": {"text": tail["text"], "start": tail["start"], "end": tail["end"]},
                        "score": score,
                    })

            fout.write(json.dumps({
                "window_id": rec["window_id"],
                "entities": entities,
                "relations": relations,
                "latency_ms": elapsed_ms,
            }, ensure_ascii=False) + "\n")

            n += 1
            if args.limit and n >= args.limit:
                break

    latencies_ms.sort()

    def pct(p):
        if not latencies_ms:
            return 0.0
        idx = min(len(latencies_ms) - 1, round((len(latencies_ms) - 1) * p))
        return latencies_ms[idx]

    total_ms = sum(latencies_ms)
    words_per_sec = total_words / (total_ms / 1000.0) if total_ms else 0.0
    timing = {
        "engine": "torch",
        "device": device,
        "repo": args.repo,
        "windows": n,
        "load_ms": load_ms,
        "p50_latency_ms": pct(0.5),
        "p90_latency_ms": pct(0.9),
        "total_words": total_words,
        "words_per_sec": words_per_sec,
        "peak_rss_mb": peak_rss_mb(),
        "mps_error": mps_error,
    }
    timing_path = args.out.with_suffix("").with_suffix(".timing.json")
    timing_path.write_text(json.dumps(timing, indent=2))
    print(f"[run_torch] windows={n} load_ms={load_ms:.1f} p50_ms={timing['p50_latency_ms']:.2f} "
          f"words_per_sec={words_per_sec:.1f} peak_rss_mb={timing['peak_rss_mb']:.1f}")
    print(f"[run_torch] wrote {args.out} and {timing_path}")


if __name__ == "__main__":
    main()
