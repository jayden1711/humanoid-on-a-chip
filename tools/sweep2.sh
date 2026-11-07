#!/usr/bin/env bash
# Like sweep.sh but with longer / higher-effort QAT (for low-bit and QR-limit candidates).
#   tools/sweep2.sh h46_w4_L:46,46:4 ...
# Exit status as in sweep.sh: 1 if any config failed or left its expected output missing or empty.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
SIM=${SIM:-$HOME/projects/bin/sim-run}
failed=0
for cfg in "$@"; do
  IFS=: read -r name hidden bits <<< "$cfg"
  ok=1
  if [ ! -s "results/models/$name.bin" ]; then
    $SIM $PY tools/distill.py --name "$name" --hidden "$hidden" --bits "$bits" \
      --float-iters "${FLOAT_ITERS:-6}" --qat-iters "${QAT_ITERS:-5}" --eps "${EPS:-120}" --eps0 "${EPS0:-300}" \
      --epochs "${EPOCHS:-30}" --qat-epochs "${QAT_EPOCHS:-60}" --qat-first-mult "${QAT_FIRST:-3}" --qat-lr "${QAT_LR:-1e-3}" \
      > "results/logs/$name.train.log" 2>&1 || { echo "distill failed: $name"; failed=$((failed + 1)); continue; }
    [ -s "results/models/$name.bin" ] || { echo "distill left no results/models/$name.bin: $name"; failed=$((failed + 1)); continue; }
  fi
  if [ ! -s "results/eval/$name.json" ]; then
    $SIM $PY tools/evaluate.py --spec "int:results/models/$name.bin" --out "results/eval/$name.json" \
      --seeds "${SEEDS:-20}" --jobs 2 > "results/logs/$name.eval.log" 2>&1 || { echo "eval failed: $name"; ok=0; }
    [ -s "results/eval/$name.json" ] || { echo "eval left no results/eval/$name.json: $name"; ok=0; }
  fi
  if [ "$ok" = 1 ]; then echo "done $name"; else failed=$((failed + 1)); fi
done
[ "$failed" -eq 0 ] || { echo "sweep: $failed config(s) failed"; exit 1; }
