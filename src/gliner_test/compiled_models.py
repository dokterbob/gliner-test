"""Materialize the GLiNER2 CoreML snapshot and compile .mlpackage -> .mlmodelc once.

Two problems this module fixes:

1. Loading an ``.mlpackage`` straight out of the Hugging Face snapshot fails:
   its files are symlinks into the HF blob cache, and the CoreML compiler
   rejects a symlinked weights file ("weight.bin doesn't exist"). We instead
   *materialize* (hard-link, falling back to a copy across filesystems) the
   snapshot's real blob files into ``.cache/coreml/<revision>/`` -- this costs
   no extra disk on the common case (hard link, not a copy) and gives CoreML
   real files to compile.

2. ``ct.models.MLModel(path)`` on an ``.mlpackage`` recompiles it into a
   per-process Darwin temp dir (``/var/folders/.../T/TemporaryItems/NSIRD_*``)
   on *every* load, regardless of ``$TMPDIR`` -- this is what filled the boot
   volume during the first run. We compile each ``.mlpackage`` to a
   ``.mlmodelc`` exactly once (into the same materialized dir) and load that
   compiled artifact directly with ``ct.models.CompiledMLModel``, which does
   not recompile. ``extraction_runtime.py`` (third-party, vendored unmodified)
   is unaware of any of this: we monkeypatch ``coremltools.models.MLModel`` so
   its unmodified ``ct.models.MLModel(path, compute_units=...)`` calls
   transparently resolve to our compiled cache when one exists.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import coremltools as ct

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "coreml"
MIN_FREE_BYTES_FOR_COMPILE = int(1.5 * 1024**3)

_REAL_ML_MODEL = ct.models.MLModel
_compiled_lookup: dict[str, Path] = {}
_patched = False


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    real_src = Path(os.path.realpath(src))
    try:
        os.link(real_src, dst)
    except OSError:
        # Cross-device (EXDEV) or other link failure: fall back to a copy.
        shutil.copy2(real_src, dst)


def materialize(snapshot_dir: Path, revision: str, relpaths: list[str]) -> Path:
    """Hard-link (or copy) the given snapshot paths into a real, writable dir.

    ``relpaths`` may name plain files or directories (e.g. an ``.mlpackage``
    bundle or ``tokenizer/``); directories are walked recursively.
    """
    dest_root = CACHE_ROOT / revision
    dest_root.mkdir(parents=True, exist_ok=True)
    for rel in relpaths:
        src = snapshot_dir / rel
        if src.is_dir():
            for file in src.rglob("*"):
                if file.is_file() or file.is_symlink():
                    _hardlink_or_copy(file, dest_root / file.relative_to(snapshot_dir))
        elif src.exists():
            _hardlink_or_copy(src, dest_root / rel)
    return dest_root


def package_size(dest_root: Path, package_name: str) -> int:
    pkg_path = dest_root / f"{package_name}.mlpackage"
    return sum(f.stat().st_size for f in pkg_path.rglob("*") if f.is_file())


def compiled_path_for(dest_root: Path, package_name: str) -> Path:
    return dest_root / f"{package_name}.mlmodelc"


def ensure_compiled(dest_root: Path, package_name: str, *, verbose: bool = True) -> Path:
    """Compile one ``.mlpackage`` to ``.mlmodelc`` if not already cached.

    Call this one package at a time, under the shared-machine model lock --
    never compile two packages concurrently. Refuses to run if ``/`` has
    less than ``MIN_FREE_BYTES_FOR_COMPILE`` free, since compilation stages a
    transient copy of the package's weights on the boot volume regardless of
    ``$TMPDIR``.
    """
    pkg_path = dest_root / f"{package_name}.mlpackage"
    out_path = compiled_path_for(dest_root, package_name)
    if out_path.exists():
        return out_path
    free = shutil.disk_usage("/").free
    if free < MIN_FREE_BYTES_FOR_COMPILE:
        raise RuntimeError(
            f"Refusing to compile {package_name}: only {free / 1e9:.2f} GB free on /, "
            f"need >= {MIN_FREE_BYTES_FOR_COMPILE / 1e9:.1f} GB (CoreML compile stages weights "
            "on the boot volume regardless of $TMPDIR)."
        )
    if verbose:
        print(f"[compiled_models] compiling {package_name} ({free / 1e9:.2f} GB free on /)...")
    compiled = ct.utils.compile_model(str(pkg_path), destination_path=str(out_path))
    return Path(compiled)


def patch_loader(dest_root: Path) -> None:
    """Monkeypatch ``coremltools.models.MLModel`` to load compiled caches.

    extraction_runtime.py always does ``ct.models.MLModel(path, compute_units=...)``
    on an ``.mlpackage`` path; after this patch, any such call for a package
    with a compiled ``.mlmodelc`` already in ``dest_root`` transparently loads
    that compiled artifact via ``ct.models.CompiledMLModel`` instead (no
    recompile, no boot-volume temp usage, ~0.1s instead of several seconds).
    Packages with no compiled cache yet fall back to the real ``MLModel``
    (which will recompile, as before) so this is always safe to apply.
    """
    global _patched
    for mlmodelc in dest_root.glob("*.mlmodelc"):
        package_name = mlmodelc.stem
        _compiled_lookup[f"{package_name}.mlpackage"] = mlmodelc
    if _patched:
        return

    def _loader(path, *args, compute_units=ct.ComputeUnit.ALL, **kwargs):
        name = Path(path).name
        compiled = _compiled_lookup.get(name)
        if compiled is not None:
            return ct.models.CompiledMLModel(str(compiled), compute_units=compute_units)
        return _REAL_ML_MODEL(path, *args, compute_units=compute_units, **kwargs)

    ct.models.MLModel = _loader
    _patched = True
