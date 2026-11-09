"""Fetch the GLiNER2.5 multi CoreML export via HF's normal cache, then
materialize (hard-link) the files we need into a real, writable, gitignored
directory that CoreML can compile from.

Two separate problems, two fixes:

1. ``snapshot_download`` with no ``local_dir`` uses the default HF cache
   (``~/.cache/huggingface``, on the big data volume, not duplicated across
   revisions) and stores large files as symlinks into a content-addressed
   blob store -- cheap, normal HF behavior.
2. CoreML's compiler refuses a symlinked weights file ("weight.bin doesn't
   exist"), so we hard-link (falling back to a copy only across filesystems)
   the real blob files into ``.cache/coreml/<revision>/`` via
   ``gliner_test.compiled_models.materialize``. This costs no extra disk in
   the common case (a hard link, not a copy) and gives CoreML real files.

We only fetch what ``extraction_runtime.py`` can actually load for entity +
relation extraction with ``Schema().entities(...)`` / ``Schema().relations(...)``:
  - runtime source files (extraction_runtime.py, extraction_pool.py,
    preprocessing.py, convert_extraction_names.py)
  - config.json / encoder_config/config.json / tokenizer/*
  - the extraction_features + extraction_scorer + explicit + relation +
    record_assignment + record_anchorless mlpackages, for fp32 and fp16,
    plus the two W8-embedding feature variants used by the adaptive runtime.

We skip classification_*.mlpackage (unused: no classification task in our
schema) and conversion/export/verify/test tooling (not needed to run the
exported model).

Note: ``CoreMLBoundaryExtractor.__init__`` unconditionally loads the
explicit, relation, record_assignment and record_anchorless packages even
though our schema only uses entities()+relations(), so those are NOT
skippable.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from gliner_test.compiled_models import materialize

REPO_ID = "FluidInference/gliner2-5-multi-coreml"
REVISION = "d35165dfcf7aac0df1d588c3eb9171249b9515cd"

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
    "tokenizer",  # directory, walked recursively by materialize()
    "extraction_runtime.py",
    "extraction_pool.py",
    "preprocessing.py",
    "convert_extraction_names.py",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    file_patterns = [f"{f}/**" if f == "tokenizer" else f for f in NEEDED_FILES]
    allow_patterns = file_patterns + [f"{pkg}.mlpackage/**" for pkg in NEEDED_PACKAGES]
    snapshot_dir = Path(
        snapshot_download(repo_id=REPO_ID, revision=args.revision, allow_patterns=allow_patterns)
    )
    print(f"HF snapshot (symlinked cache): {snapshot_dir}")

    relpaths = list(NEEDED_FILES) + [f"{pkg}.mlpackage" for pkg in NEEDED_PACKAGES]
    dest_root = materialize(snapshot_dir, args.revision, relpaths)
    print(f"Materialized (hard-linked) real files to: {dest_root}")


if __name__ == "__main__":
    main()
