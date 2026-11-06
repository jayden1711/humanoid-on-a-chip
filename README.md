# humanoid-on-a-chip

A full-size humanoid's walking policy, running on a $4 microcontroller. The body is simulated; the chip is the brain.

A Unitree G1 walking policy runs on a Raspberry Pi Pico 2 (RP2350), and later an ESP32-S3, and controls a simulated G1 in MuJoCo over USB serial (hardware-in-the-loop). No vision, planning or language: walking and balance only.

Early stage; see [PLAN.md](PLAN.md). Code is Apache-2.0 ([LICENSE](LICENSE)). Third-party policies, models and tools: [THIRD_PARTY.md](THIRD_PARTY.md).
