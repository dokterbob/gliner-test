#!/bin/zsh
# CoreML fp16-w8 (CPU) over the localdb corpus windows with the final schema, relation threshold 0.0, in locked chunks.
set -e
cd "/Volumes/User Home/dev/dev/gliner-test"
L=(/usr/bin/lockf -k .cache/model.lock)
N=$(wc -l < data/windows.jsonl)
CH=1000
off=0
i=0
while [ $off -lt $N ]; do
  "${L[@]}" uv run python scripts/run_coreml.py --windows data/windows.jsonl --precision fp16-w8 --compute-units cpu \
     --relation-threshold 0.0 --offset $off --limit $CH --progress-every 250 --out data/corpus-run/chunk$i.jsonl
  off=$((off+CH)); i=$((i+1))
done
cat data/corpus-run/chunk*.jsonl > data/corpus-run/coreml-fp16-w8-cpu.jsonl
echo CORPUS_DONE
