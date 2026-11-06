# HANDOFF: humanoid-on-a-chip
Status: first stop point reached on the host (baseline, compression study, equivalence checks, memory fit all done); no firmware, no hardware
Next: owner reviews results/report.md + results/equivalence.md and decides: firmware go (Pico 2 first), or push compression further (2-bit still fails, recurrent students untried)
Updated: 2026-11-06

Read CLAUDE.md, PLAN.md and ~/projects/NUC_RULES.md first. All numbers live in git-ignored `results/` (never in this file).

## What exists (local commits on branch `worktree-overnight-compression`, not pushed)
- `hoc/env.py`: MuJoCo G1 12-DoF env (unitree_rl_gym model, 50 Hz, explicit PD at 500 Hz). Own code, nothing copied from robot-arena-sim.
- `hoc/policies.py`: numpy teacher (original LSTM from `motion.pt`, checked against TorchScript: max diff 1.4e-5) and weight-rounded teacher variants.
- `hoc/evalsuite.py`: seeded scenarios (clean walk / command sequence / standing, sensor dropouts, observation delays, IMU spikes, mixed, falls-only push ladder standing and walking), metrics and the equivalence comparison (paired rollouts, divergence time and attributed event, teacher-forced action error per joint).
- `hoc/pack.py`, `hoc/intref.py`: packed integer model format (header, per-channel requant multipliers, biases, b-bit packed weights, CRC32) and its integer-only reference. Tests: `tests/test_pack.py` (round trip; QAT forward equals integer reference).
- `hoc/student.py`, `tools/distill.py`: DAgger distillation into an MLP, then quantization-aware training to 8/4/3/2-bit weights. Training seeds >= 1000, evaluation seeds < 100.
- `tools/evaluate.py` (full suite + equivalence for one policy spec), `tools/sweep.sh`, `tools/sweep2.sh`, `tools/report.py` (-> results/report.md), `tools/memfit.py` (-> results/memfit.json).
- Third party: unitree_rl_gym fetched at the pinned commit into git-ignored `third_party/` (no remote needed).

## Where to look
- `results/report.md`: size vs walking quality vs push resistance, plus equivalence counts, one row per version.
- `results/eval/<version>.json`: per-episode rows incl. full equivalence records (divergence time, cause, per-joint action error).
- `results/memfit.json`: RP2350 / PY32F002B / QR fit (budget arithmetic; code-size and stack are assumptions, nothing built or measured).
- `results/models/*.bin` packed models; `results/logs/` training and eval logs.

## Findings in words (numbers are in results/)
- The original teacher is the reference; its own float arithmetic with 8-bit rounded weights matches it, 4-bit and 2-bit rounding without retraining does not walk.
- Distilled MLPs with QAT: 8-bit and 4-bit students walk and resist pushes; resistance drops gradually with size. A 4-bit student under the 2,953-byte QR limit exists and walks all clean scenarios; a 3-bit one too. Plain 2-bit (also mixed 2/2/4) did NOT reach walking with this QAT recipe; training MSE grew over DAgger iterations. Untried: bit-width annealing from 4 to 2, knowledge distillation from the 4-bit student, wider/deeper nets.
- Longer QAT (`_L` versions) helped the 4-bit models at the QR size; the smallest versions that still walk with some push resistance are around 1-2 KB; below ~1 KB walking breaks down.
- Equivalence: outcome (fall / no fall) matches the teacher in most episodes of the clean and single-disturbance scenarios, but NO compressed version is trajectory-equivalent: pelvis paths separate by more than 10 cm within ~2 s in essentially every episode (different gait phase/speed), and the mixed-disturbance scenario flips outcomes both ways. The 10 cm threshold is probably too tight for walking; consider a looser, gait-aware metric before using it as a pass/fail gate for firmware.
- Memory fit: every packed candidate fits RP2350 and PY32F002B by arithmetic with an assumed code budget; the original int8 LSTM does not fit PY32F002B flash. Latency not measured.

## Reproduce
```
.venv/bin/python -c "from hoc.policies import extract_teacher; extract_teacher()"
~/projects/bin/sim-run .venv/bin/python tools/evaluate.py --spec teacher --out results/eval/teacher.json --seeds 20 --jobs 2
tools/sweep.sh h46_w4:46,46:4          # distill + evaluate one config
.venv/bin/python tools/report.py
```

## Decisions taken overnight
- The worktree `overnight-compression` already existed (resumed); fast-forwarded it to main (b414778) so PLAN.md was current.
- Fall criterion: pelvis below 0.5 m or tilt over 60 deg. Push test copies the arena's scheme (0.1 s shoves, 50 N steps every 2 s from 2 s) but is this repo's own implementation.
- Compute: CPU only (torch CPU build in `.venv`); nets are tiny, so no gpu-run use. At most 2 worker processes per job, one job at a time, everything through sim-run (machine load was 11-14 from other projects).
- Student family: stateless MLPs (obs already carries last action and gait phase). Recurrent students not tried yet.
- Quantization: per-channel symmetric b-bit weights, int8 activations, Q8 output; requant by 16-bit multiplier and shift (int64 multiply in the reference; firmware needs a plan for that on the M0+).
- Training scenarios use pushes of 30-300 N (450 N pushes made the teacher itself fall in over half the episodes).
- Model selection per run = iterate with fewest falls in its own DAgger rollouts (training distribution), not on eval seeds.

## Blocked / skipped
- A `nohup bash -c ...` command was denied by the permission system; used run_in_background instead.
- No hardware, no firmware (by design). `arm-none-eabi-gcc` is not installed, so no flash/RAM numbers are measured; memory fit is arithmetic.

## Stray files (not deleted, git-ignored)
- `results/models/smoke_h32_w8*` and `results/logs/smoke.log`: smoke-test run.
- `results/models/*.iter*.bin` and `*.final.bin`: per-iteration snapshots of each distillation.
- `.venv/` in the worktree.
- `~/.claude/jobs/52b967bf/tmp/quick.py`, `calib.py`: scratch scripts.

## Open questions for the owner
- Accept stateless MLP students, or try recurrent (GRU) students for the smallest sizes?
- Firmware go-ahead (Pico 2 first)? PLAN's HIL equivalence step needs the integer reference in `hoc/intref.py` ported bit-exactly.
- THIRD_PARTY.md: the 12-DoF model/policy pin is used as listed; Puya HAL license still unconfirmed.
