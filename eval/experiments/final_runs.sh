#!/bin/zsh
# Final runs on gold v3 windows (schema/ontonotes.json, relation thr 0.0), merged back to parent items.
# Each model run is a separate lockf-held process (one model loaded at a time).
set -e
cd "/Volumes/User Home/dev/dev/gliner-test"
L=(/usr/bin/lockf -k .cache/model.lock)
W=data/gold-windows.jsonl
X=data/exp/final
uv run python scripts/gold_windows.py split --schema schema/ontonotes.json --out $W 2>/dev/null
"${L[@]}" uv run python scripts/run_coreml.py --windows $W --precision fp32 --compute-units cpu --relation-threshold 0.0 --out $X/coreml-fp32-cpu.win.jsonl
"${L[@]}" uv run python scripts/run_coreml.py --windows $W --precision fp16-w8 --compute-units cpu --relation-threshold 0.0 --out $X/coreml-fp16-w8-cpu.win.jsonl
"${L[@]}" uv run python scripts/run_torch.py --windows $W --relation-threshold 0.0 --out $X/torch-fp32.win.jsonl
"${L[@]}" uv run python scripts/run_torch.py --windows eval/gold.jsonl --relation-threshold 0.0 --out $X/torch-fp32-unsplit.jsonl
"${L[@]}" uv run python eval/experiments/run_jointie_rel.py --windows $W --symmetric-mode undirected --out $X/torch-jointie.win.jsonl
"${L[@]}" uv run python eval/experiments/run_jointie_rel.py --windows eval/gold.jsonl --symmetric-mode undirected --out $X/torch-jointie-unsplit.jsonl
ORT_DYLIB_PATH=/opt/homebrew/opt/onnxruntime/lib/libonnxruntime.dylib "${L[@]}" rust/target/release/gliner-rust-eval --schema schema/ontonotes.json --models-dir models/gliner2.5-multi-v1-onnx --windows $W --out $X/rust-ort-fp32.win.jsonl
for e in coreml-fp32-cpu coreml-fp16-w8-cpu torch-fp32 torch-jointie rust-ort-fp32; do
  uv run python scripts/gold_windows.py merge --preds $X/$e.win.jsonl --windows $W --out data/preds/gold/$e.jsonl
done
cp $X/torch-fp32-unsplit.jsonl data/exp/final/torch-fp32-unsplit.merged.jsonl
echo FINAL_RUNS_DONE
