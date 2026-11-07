#!/usr/bin/env bash
# Sequential distill + evaluate over a list of "name:hidden:bits" configs. Two worker processes at a time, nice 10.
#   tools/sweep.sh h32_w4:32,32:4 h16_w2:16,16:2 ...
# Env: FLOAT_ITERS, QAT_ITERS, EPS, EPS0, EPOCHS, SEEDS override the defaults.
# A failing config is reported and the sweep moves on, but the script exits 1 at the end if any config failed
# or its expected output (results/models/<name>.bin, results/eval/<name>.json) is missing or empty.
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
      --float-iters "${FLOAT_ITERS:-6}" --qat-iters "${QAT_ITERS:-4}" --eps "${EPS:-150}" --eps0 "${EPS0:-300}" \
      --epochs "${EPOCHS:-30}" > "results/logs/$name.train.log" 2>&1 || { echo "distill failed: $name"; failed=$((failed + 1)); continue; }
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
