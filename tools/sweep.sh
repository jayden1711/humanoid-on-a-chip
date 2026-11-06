#!/usr/bin/env bash
# Sequential distill + evaluate over a list of "name:hidden:bits" configs. Two worker processes at a time, nice 10.
#   tools/sweep.sh h32_w4:32,32:4 h16_w2:16,16:2 ...
# Env: FLOAT_ITERS, QAT_ITERS, EPS, EPS0, EPOCHS, SEEDS override the defaults.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for cfg in "$@"; do
  IFS=: read -r name hidden bits <<< "$cfg"
  if [ ! -f "results/models/$name.bin" ]; then
    ~/projects/bin/sim-run $PY tools/distill.py --name "$name" --hidden "$hidden" --bits "$bits" \
      --float-iters "${FLOAT_ITERS:-6}" --qat-iters "${QAT_ITERS:-4}" --eps "${EPS:-150}" --eps0 "${EPS0:-300}" \
      --epochs "${EPOCHS:-30}" > "results/logs/$name.train.log" 2>&1 || { echo "distill failed: $name"; continue; }
  fi
  if [ ! -f "results/eval/$name.json" ]; then
    ~/projects/bin/sim-run $PY tools/evaluate.py --spec "int:results/models/$name.bin" --out "results/eval/$name.json" \
      --seeds "${SEEDS:-20}" --jobs 2 > "results/logs/$name.eval.log" 2>&1 || echo "eval failed: $name"
  fi
  echo "done $name"
done
