# Third-party policies, models and tools

Nothing listed here is committed or redistributed. Policies and models are fetched from upstream at the pinned
commit. Compressed/distilled derivatives stay in git-ignored dirs until the owner approves publishing them, and
each derivative keeps its source's license notice. Apache/BSD/MIT only.

## Candidate source policies (same pins as the robot-arena walkers)
| Source | Upstream | Commit | License | Use |
|---|---|---|---|---|
| G1 joystick policy + G1 MJCF | github.com/google-deepmind/mujoco_playground | a4aa0ed8a71a9cc1edd18691541739264ec62ca8 | Apache-2.0 | candidate walker |
| G1 12-DoF LSTM walker (`deploy/pre_train/g1/motion.pt`), `g1_12dof.xml` | github.com/unitreerobotics/unitree_rl_gym | 276801e46c5d433564f24658bac64f254b7d2d4b | BSD-3-Clause | candidate (12 leg joints, recurrent) |
| G1 velocity policy | github.com/unitreerobotics/unitree_rl_mjlab | 1425b15f73bd4095f0df53709d7c389c3eb9e790 | Apache-2.0 | candidate walker |
| G1 velocity policy | github.com/unitreerobotics/unitree_rl_lab | 4960b84732b0c2ec593dccbfe963fda1bcd7b1e3 | Apache-2.0 | candidate walker |
| G1 PPO / FastSAC loco policies | github.com/amazon-far/holosoma | a778c6fdd9ac137c35328cbb42e14ac3453d0561 | Apache-2.0 | candidate walkers |
| Unitree G1 model | github.com/google-deepmind/mujoco_menagerie `unitree_g1` | 98a70c569c4c1b6da467a20012f766ee1f8dc521 | BSD-3-Clause | sim body |

## Tools (planned)
| Tool | License | Use |
|---|---|---|
| MuJoCo (Python) | Apache-2.0 | host sim |
| Raspberry Pi Pico SDK | BSD-3-Clause | RP2350 firmware |
| ESP-IDF | Apache-2.0 | ESP32-S3 firmware |
| CE C/C++ Toolchain (incl. `srldrvce`, `usbdrvce`) | LGPL-3.0, **needs owner decision before publishing anything linked against it** | TI-84 Plus CE target |
