"""Download the GLiNER2.5 multi CoreML export into a materialized directory.

A materialized (non-symlinked) directory is required: the default Hugging Face
hub cache stores large files as symlinks into a shared blob store, and CoreML's
compiler refuses to open a symlinked .mlpackage. We therefore pass
`local_dir=...` to `snapshot_download`, which copies real files.

We only fetch what `extraction_runtime.py` can actually load for entity +
relation extraction with `Schema().entities(...).relations(...)`:
  - runtime source files (extraction_runtime.py, extraction_pool.py,
    preprocessing.py, convert_extraction_names.py)
  - config.json / encoder_config/config.json / tokenizer/*
  - the extraction_features + extraction_scorer + explicit + relation +
    record_assignment + record_anchorless mlpackages, for fp32 and fp16,
    plus the two W8-embedding feature variants used by the adaptive runtime.

We deliberately skip the classification_*, explicit... (wait, explicit IS
needed, see below) packages:
  - classification_*.mlpackage: unused (no classification task in our schema)
  - conversion/export/verify/*.py, tests/, reports/: conversion-time tooling,
    not needed to run the exported model

Note: `CoreMLBoundaryExtractor.__init__` unconditionally loads the explicit,
relation, record_assignment and record_anchorless packages even though our
schema only uses entities()+relations(), so those are NOT skippable.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "FluidInference/gliner2-5-multi-coreml"

NEEDED_PACKAGES = [
    "gliner2_multi_extraction_features_fp32_L128_W64_Q8",
    "gliner2_multi_extraction_features_w8_embedding_fp32_L128_W64_Q8",
    "gliner2_multi_extraction_features_w8_embedding_linear_fp16_L128_W64_Q8",
    "gliner2_multi_extraction_scorer_fp32_L128_W64_Q8",
    "gliner2_multi_extraction_scorer_fp16_L128_W64_Q8",
    "gliner2_multi_explicit_fp32_W64_Q8_S64",
    "gliner2_multi_explicit_fp16_W64_Q8_S64",
    "gliner2_multi_relation_fp32_W64_R4_P256",
    "gliner2_multi_relation_fp16_W64_R4_P256",
    "gliner2_multi_record_assignment_fp32_F8_C192_I1536",
    "gliner2_multi_record_assignment_fp16_F8_C192_I1536",
    "gliner2_multi_record_anchorless_fp32_F8_C192_I1536",
    "gliner2_multi_record_anchorless_fp16_F8_C192_I1536",
]

NEEDED_FILES = [
    "README.md",
    "LICENSE",
    "config.json",
    "encoder_config/config.json",
    "tokenizer/*",
    "extraction_runtime.py",
    "extraction_pool.py",
    "preprocessing.py",
    "convert_extraction_names.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", type=Path, default=Path("models/gliner2-multi-coreml"))
    args = ap.parse_args()

    allow_patterns = list(NEEDED_FILES) + [f"{pkg}.mlpackage/**" for pkg in NEEDED_PACKAGES]

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(args.local_dir),
        allow_patterns=allow_patterns,
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
