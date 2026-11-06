# humanoid-on-a-chip: Claude rules

**Read `~/projects/NUC_RULES.md` first.** It covers Quatern priority, sim-run/gpu-run, the results policy and publishing.

A G1 walking policy on a few-dollar microcontroller, controlling a simulated G1 through hardware-in-the-loop.
Plan: PLAN.md (the owner's). State: HANDOFF.md.

- **Claim wording is fixed:** "A full-size humanoid's walking policy runs on a $4 chip." Never stronger
  (not "robot brain"). Credit the prior work listed in PLAN.md wherever the project is described.
- **First stop point:** host baseline + host-only compression study + RP2350 memory-fit check. Then stop
  and report. No firmware work before the owner says go. No hardware is ordered.
- **Don't copy code from robot-arena-sim.** It's private and has no license yet. Fetch the walkers from their
  upstream repos at the pinned commits in THIRD_PARTY.md and write this repo's own adapters and push test
  here. Reading the arena's code for reference is fine; pasting it in isn't, until the owner releases it.
- **Compute:** MuJoCo baseline/push tests through `~/projects/bin/sim-run` (max 4 processes, one thread each).
  GPU use is low (quantization-aware training / distillation only) and always goes through
  `~/projects/bin/gpu-run --need-mib <N> --label humanoid-on-a-chip -- ...`.
- **Hardware-in-the-loop runs where the board is plugged in, probably the Mac.** Keep the host bridge portable
  (macOS + Linux serial). Flashing a board only on the owner's request.
- **Firmware:** integer-only C, fixed-size buffers, no dynamic allocation. Dependencies Apache/BSD/MIT only.
- **Public repo, Apache-2.0.** Code only. Results, logs, videos, compressed weights and tables stay in
  git-ignored `results/` until the owner approves. Nothing goes public without the owner's go-ahead.
