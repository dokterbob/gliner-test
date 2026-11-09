"""Benchmark one (precision, compute-units) combination over a fixed window sample.

Run once per combination (short, under the shared-machine model lock), e.g.:

    /usr/bin/lockf -k <lockfile> uv run python scripts/bench.py \
        --precision fp32 --compute-units gpu --sample-size 300

Writes ``data/bench/<precision>-<units>.json`` with:
  - load_ms (model construction, from compiled .mlmodelc cache)
  - p50_ms / p95_ms / mean_ms (per-window latency, both passes combined)
  - windows_per_s, words_per_s
  - peak_rss_bytes
  - a latency breakdown (coreml feature-graph / scorer-head / relation-head
    predict() time vs "python" preprocessing+decode time), via instrumented
    predict() calls on the runtime's own MLModel/CompiledMLModel objects.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import coremltools as ct
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_models import REVISION  # noqa: E402
from run_coreml import FP16_W8_FEATURE_PACKAGE, UNITS, extract_window, load_schemas  # noqa: E402

from gliner_test.compiled_models import CACHE_ROOT, patch_loader  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round(fraction * (len(ordered) - 1)), len(ordered) - 1)]


def instrument(obj, name: str, timers: dict) -> None:
    if obj is None or not hasattr(obj, "predict"):
        return
    original = obj.predict

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        timers[name] = timers.get(name, 0.0) + (time.perf_counter() - t0)
        timers[name + "_n"] = timers.get(name + "_n", 0) + 1
        return result

    obj.predict = wrapped


def instrument_runtime(rt, timers: dict) -> None:
    instrument(getattr(rt, "features_model", None), "coreml_feature_graph", timers)
    instrument(getattr(rt, "scorer_model", None), "coreml_scorer_head", timers)
    relation_scorer = getattr(rt, "relation_scorer", None)
    if relation_scorer is not None:
        instrument(getattr(relation_scorer, "model", None), "coreml_relation_head", timers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--schema", type=Path, default=Path("schema/ontonotes.json"))
    ap.add_argument("--windows", type=Path, default=Path("data/windows.jsonl"))
    ap.add_argument("--precision", choices=["fp32", "fp16-w8", "adaptive"], required=True)
    ap.add_argument("--compute-units", choices=list(UNITS), required=True)
    ap.add_argument("--sample-size", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model_dir = args.model_dir or (CACHE_ROOT / REVISION)
    out = args.out or Path(f"data/bench/{args.precision}-{args.compute_units}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    windows = [json.loads(line) for line in args.windows.open()]
    step = max(1, len(windows) // args.sample_size)
    sample = windows[::step][: args.sample_size]
    total_words = sum(len(w["text"].split()) for w in sample)

    patch_loader(model_dir)
    sys.path.insert(0, str(model_dir))
    from extraction_runtime import CoreMLAdaptiveBoundaryExtractor, CoreMLBoundaryExtractor
    from gliner2 import Schema

    cu = UNITS[args.compute_units]
    load_t0 = time.perf_counter()
    if args.precision == "fp32":
        runtime = CoreMLBoundaryExtractor(str(model_dir), precision="fp32", compute_units=cu)
    elif args.precision == "fp16-w8":
        runtime = CoreMLBoundaryExtractor(
            str(model_dir), precision="fp16", compute_units=cu, feature_package=FP16_W8_FEATURE_PACKAGE
        )
    else:
        runtime = CoreMLAdaptiveBoundaryExtractor(str(model_dir), fp16_compute_units=cu, fp32_compute_units=cu)
    load_ms = (time.perf_counter() - load_t0) * 1000

    ent_schema, relation_pass_schemas, label_to_tag, threshold = load_schemas(args.schema, Schema)

    timers: dict = {}
    process = psutil.Process()
    peak_rss = process.memory_info().rss

    # Warm up (first calls pay one-time lazy init costs; also forces the
    # adaptive runtime to build its internal CoreMLBoundaryExtractor so we
    # can instrument its model objects).
    for w in sample[: args.warmup]:
        extract_window(runtime, ent_schema, relation_pass_schemas, label_to_tag, threshold, w["text"])
    peak_rss = max(peak_rss, process.memory_info().rss)

    target = runtime._runtime if isinstance(runtime, CoreMLAdaptiveBoundaryExtractor) else runtime
    instrument_runtime(target, timers)

    latencies = []
    t_start = time.perf_counter()
    for w in sample:
        _, _, latency_ms = extract_window(runtime, ent_schema, relation_pass_schemas, label_to_tag, threshold, w["text"])
        latencies.append(latency_ms)
        peak_rss = max(peak_rss, process.memory_info().rss)
    wall_s = time.perf_counter() - t_start

    coreml_ms = sum(v for k, v in timers.items() if k.startswith("coreml_") and not k.endswith("_n")) * 1000
    total_ms = sum(latencies)
    python_ms = max(total_ms - coreml_ms, 0.0)

    report = {
        "precision": args.precision,
        "compute_units": args.compute_units,
        "sample_size": len(sample),
        "load_ms": load_ms,
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": statistics.mean(latencies),
        "windows_per_s": len(sample) / wall_s,
        "words_per_s": total_words / wall_s,
        "peak_rss_bytes": peak_rss,
        "peak_rss_mb": peak_rss / 1e6,
        "latency_breakdown_ms": {
            "coreml_feature_graph": timers.get("coreml_feature_graph", 0.0) * 1000,
            "coreml_scorer_head": timers.get("coreml_scorer_head", 0.0) * 1000,
            "coreml_relation_head": timers.get("coreml_relation_head", 0.0) * 1000,
            "python_preprocess_and_decode": python_ms,
            "total": total_ms,
        },
        "coremltools_version": ct.__version__,
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    del runtime
    gc.collect()


if __name__ == "__main__":
    main()
