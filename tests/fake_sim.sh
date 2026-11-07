#!/usr/bin/env bash
# Stand-in for "sim-run python tools/{distill,evaluate}.py ..." in tests/test_sweep_exit.sh. Name decides the behaviour:
# *bad* exits 1, *empty* exits 0 without writing the output, anything else writes it.
shift  # the PY word
tool=$(basename "$1"); shift
while [ $# -gt 0 ]; do
  case "$1" in
    --name) name=$2 ;;
    --out) out=$2 ;;
    --spec) spec=$2 ;;
  esac
  shift
done
if [ "$tool" = distill.py ]; then out="results/models/$name.bin"; else name=$(basename "$spec" .bin); fi
case "$name" in
  *bad*) echo "simulated crash"; exit 1 ;;
  *empty*) exit 0 ;;
esac
echo data > "$out"
