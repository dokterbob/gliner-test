"""Warm the compiled-model cache: compile every needed .mlpackage -> .mlmodelc
exactly once, smallest package first, checking free disk before each compile.

Run this once (under the shared-machine model lock) before benchmarking:

    /usr/bin/lockf -k <lockfile> uv run python scripts/compile_models.py

Compiling is compute-units-independent (compute_units is chosen at *load*
time via ``ct.models.CompiledMLModel(path, compute_units=...)``), so this
only needs to happen once regardless of how many compute-unit / precision
combinations are benchmarked afterward. Reports cold compile time per
package so bench.py can contrast it with warm (compiled-cache) load time.
"""

import argparse
import time
from pathlib import Path

from gliner_test.compiled_models import CACHE_ROOT, ensure_compiled, package_size
from download_models import NEEDED_PACKAGES, REVISION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    dest_root = CACHE_ROOT / args.revision
    if not dest_root.exists():
        raise SystemExit(f"{dest_root} does not exist -- run scripts/download_models.py first")

    sized = sorted(NEEDED_PACKAGES, key=lambda pkg: package_size(dest_root, pkg))
    for pkg in sized:
        out_path = dest_root / f"{pkg}.mlmodelc"
        if out_path.exists():
            print(f"[compile_models] {pkg}: already compiled, skipping")
            continue
        size_mb = package_size(dest_root, pkg) / 1e6
        t0 = time.perf_counter()
        ensure_compiled(dest_root, pkg)
        dt = time.perf_counter() - t0
        print(f"[compile_models] {pkg}: {size_mb:.1f} MB package, cold compile {dt:.2f}s")


if __name__ == "__main__":
    main()
