#!/usr/bin/env bash
# Like sweep.sh but with longer / higher-effort QAT (for low-bit and QR-limit candidates).
#   tools/sweep2.sh h46_w4_L:46,46:4 ...
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for cfg in "$@"; do
  IFS=: read -r name hidden bits <<< "$cfg"
  if [ ! -f "results/models/$name.bin" ]; then
    ~/projects/bin/sim-run $PY tools/distill.py --name "$name" --hidden "$hidden" --bits "$bits" \
      --float-iters "${FLOAT_ITERS:-6}" --qat-iters "${QAT_ITERS:-5}" --eps "${EPS:-120}" --eps0 "${EPS0:-300}" \
      --epochs "${EPOCHS:-30}" --qat-epochs "${QAT_EPOCHS:-60}" --qat-first-mult "${QAT_FIRST:-3}" --qat-lr "${QAT_LR:-1e-3}" \
      > "results/logs/$name.train.log" 2>&1 || { echo "distill failed: $name"; continue; }
  fi
  if [ ! -f "results/eval/$name.json" ]; then
    ~/projects/bin/sim-run $PY tools/evaluate.py --spec "int:results/models/$name.bin" --out "results/eval/$name.json" \
      --seeds "${SEEDS:-20}" --jobs 2 > "results/logs/$name.eval.log" 2>&1 || echo "eval failed: $name"
  fi
  echo "done $name"
done
