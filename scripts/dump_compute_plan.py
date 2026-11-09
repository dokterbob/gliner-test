"""Dump a per-operation device assignment (CPU/GPU/ANE) for one compiled package.

On this VM (no Neural Engine) every op lands on CPU or GPU; the same command
on bare-metal Apple Silicon will show real ANE coverage, which is the whole
point of running this on both. Usage:

    uv run python scripts/dump_compute_plan.py \
        --package gliner2_multi_extraction_features_w8_embedding_linear_fp16_L128_W64_Q8 \
        --compute-units all
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

from gliner_test.compiled_models import CACHE_ROOT, compiled_path_for

UNITS = {
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "ane": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


def device_name(usage) -> str:
    if usage is None:
        return "unsupported/unknown"
    names = []
    for device in usage.supported_compute_devices:
        names.append(type(device).__name__)
    preferred = type(usage.preferred_compute_device).__name__ if usage.preferred_compute_device else "none"
    return f"preferred={preferred} supported={sorted(set(names))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--package", required=True, help="package name without .mlmodelc/.mlpackage suffix")
    ap.add_argument("--compute-units", choices=list(UNITS), default="all")
    args = ap.parse_args()

    if args.model_dir is None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from download_models import REVISION

        args.model_dir = CACHE_ROOT / (args.revision or REVISION)

    compiled_path = compiled_path_for(args.model_dir, args.package)
    if not compiled_path.exists():
        raise SystemExit(f"{compiled_path} not found -- run scripts/compile_models.py first")

    plan = MLComputePlan.load_from_path(str(compiled_path), compute_units=UNITS[args.compute_units])
    program = plan.model_structure.program
    main_function = program.functions["main"]

    device_counts = Counter()
    preferred_counts = Counter()
    for op in main_function.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            device_counts["unknown"] += 1
            continue
        preferred = type(usage.preferred_compute_device).__name__.replace("MLComputeDevice", "")
        preferred_counts[preferred] += 1
        for device in usage.supported_compute_devices:
            device_counts[type(device).__name__.replace("MLComputeDevice", "")] += 1

    report = {
        "package": args.package,
        "compute_units": args.compute_units,
        "preferred_device_op_counts": dict(preferred_counts),
        "supported_device_op_counts": dict(device_counts),
        "total_ops": sum(preferred_counts.values()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
