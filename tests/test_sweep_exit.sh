#!/usr/bin/env bash
# sweep.sh / sweep2.sh must exit non-zero when any config crashed or left no output. Run: tests/test_sweep_exit.sh
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
W=$(mktemp -d)
fails=0
check() { # label want_nonzero(1|0) rc
  if { [ "$2" = 1 ] && [ "$3" -ne 0 ]; } || { [ "$2" = 0 ] && [ "$3" -eq 0 ]; }; then echo "PASS $1 (rc=$3)"; else echo "FAIL $1 (rc=$3)"; fails=$((fails + 1)); fi
}
chmod +x "$HERE/tests/fake_sim.sh"
for s in sweep sweep2; do
  i=0
  for spec in "good:0 ok_a:8:4" "bad:1 bad_a:8:4 ok_b:8:4" "empty:1 empty_a:8:4" "mixed:1 ok_c:8:4 bad_c:8:4"; do
    i=$((i + 1)); head=${spec%% *}; label=${head%%:*}; want=${head#*:}
    set -- ${spec#* }
    R="$W/$s.$i"
    mkdir -p "$R/tools" "$R/results/models" "$R/results/eval" "$R/results/logs"
    cp "$HERE/tools/$s.sh" "$R/tools/"
    SIM="$HERE/tests/fake_sim.sh" PY=python bash "$R/tools/$s.sh" "$@" > "$R/out.log" 2>&1
    check "$s $label" "$want" $?
  done
done
[ "$fails" -eq 0 ] && echo "all ok" || { echo "$fails failed"; exit 1; }
