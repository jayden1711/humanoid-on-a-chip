# humanoid-on-a-chip

Goal: a Unitree G1 walking policy runs entirely on a few-dollar microcontroller (Raspberry Pi Pico 2 / RP2350 first, ESP32-S3 second), controlling a simulated G1 in MuJoCo through hardware-in-the-loop. The body is simulated; the chip does all the control.

Claim (exact wording, never stronger): "A full-size humanoid's walking policy runs on a $4 chip." Not "an entire robot brain": there's no vision, planning or language.

Prior work to credit:
- RL locomotion for a sub-centimeter quadruped microrobot on an ARM Cortex-M0 at 5 MHz (arXiv 2512.24740).
- Kansas State thesis: quantized PPO locomotion policy for a legged robot on a microcontroller (TinyML, lunar context).
- Small bipeds running RL policies on single-board computers (Hugging Face / Pollen Microduck on an RK3566).
- tinyworldformer: a world model and actor on a Pi Pico 2 W.
Our new part: a full-size humanoid (G1, 12-29 joints, balance and push recovery) controlled by a microcontroller.

Source policies: reuse the robot-arena walkers (all Apache/BSD/MIT, licenses in robot-arena-sim's THIRD_PARTY.md). Start with the smallest one that controls only the 12 leg joints. Recurrent policies (unitree_rl_gym) keep their memory on the chip; the arena session already re-exported that one with explicit state.

Hardware-in-the-loop loop, 50 Hz:
1. MuJoCo on the host computes the observation (joint positions/velocities, IMU, command).
2. Host sends it to the chip over USB serial.
3. Chip runs the policy, returns joint targets.
4. Host applies them through the PD/actuator model and steps 20 ms.
Two modes: lockstep (sim waits for the chip; exact) and real-time (shows the chip keeps up). The board plugs into whichever machine runs the sim; that will likely be the Mac.

Engineering steps:
1. Baseline: run the chosen walker in MuJoCo on the host, record walking quality and push resistance (reuse the arena's falls-only push test).
2. Compression: int8 quantization; if it doesn't fit or loses too much, distill into a smaller MLP. Keep push resistance as the main quality metric.
3. Firmware: integer-only inference in C for RP2350 (and ESP32-S3), fixed-size buffers, no dynamic allocation, measured latency per step.
4. Host bridge: binary serial protocol with checksums, lockstep and real-time modes, logging.
5. Results table: model size, RAM used, latency per step, walking speed, push resistance vs. the original float policy, for each compression level and each chip.

Demo video: bare chip on the desk with an LED blinking each control step, split screen with the G1 walking and getting pushed in sim; then unplug the USB cable and the robot collapses.

Hardware to buy (not ordered yet): Raspberry Pi Pico 2 (RP2350), optionally an ESP32-S3 dev board (also in pocket-vla's parts list).

First stop point: baseline numbers on the host, the compression study (size vs. push resistance) entirely on the host, and a check that the best candidate fits the RP2350's memory. Then stop and report before any firmware work. No results in git until I approve; nothing posted without my go-ahead.
