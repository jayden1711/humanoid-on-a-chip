"""MuJoCo host environment for the unitree_rl_gym G1 12-DoF legs model (BSD-3-Clause upstream, pinned in THIRD_PARTY.md).

Conventions follow upstream deploy/deploy_mujoco (dt 0.002, 10 substeps = 50 Hz, explicit PD every substep, 47-dim obs).
This file is this repo's own adapter; no code is copied from other repos.
"""
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "third_party/unitree_rl_gym/resources/robots/g1_description/scene.xml"

KP = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], float)
KD = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], float)
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0])
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25])
PERIOD = 0.8
SUBSTEPS = 10
DT = 0.002
CTRL_DT = DT * SUBSTEPS
OBS_DIM, ACT_DIM = 47, 12
TORQUE_LIMIT = np.array([88, 139, 88, 139, 50, 50] * 2, float)

FALL_HEIGHT = 0.5      # pelvis below this: fell
FALL_GZ = -0.5         # gravity z in the pelvis frame above this (tilt > 60 deg from upright): fell


def gravity_from_quat(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)])


class G1Env:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.m.opt.timestep = DT
        self.d = mujoco.MjData(self.m)
        self.pelvis = self.m.body("pelvis").id
        self.t = 0
        self.last_action = np.zeros(ACT_DIM)
        self.target = DEFAULT.copy()

    def reset(self, seed=0, jitter=True):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        d = self.d
        d.qpos[7:] = DEFAULT
        if jitter:  # small seeded start variation: yaw and joint noise
            yaw = rng.uniform(-0.1, 0.1)
            d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            d.qpos[7:] += rng.normal(0, 0.01, ACT_DIM)
        mujoco.mj_forward(self.m, d)
        self.t = 0
        self.last_action = np.zeros(ACT_DIM)
        self.target = DEFAULT.copy()
        d.xfrc_applied[:] = 0
        return self

    def grav(self):
        return gravity_from_quat(self.d.qpos[3:7])

    def fell(self):
        return bool(self.d.qpos[2] < FALL_HEIGHT or self.grav()[2] > FALL_GZ)

    def obs(self, cmd):
        d = self.d
        ph = (self.t * CTRL_DT) % PERIOD / PERIOD
        return np.hstack([
            d.qvel[3:6] * ANG_VEL_SCALE, gravity_from_quat(d.qpos[3:7]), np.asarray(cmd) * CMD_SCALE,
            d.qpos[7:] - DEFAULT, d.qvel[6:] * DOF_VEL_SCALE, self.last_action,
            [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]]).astype(np.float32)

    def step(self, action, push=None):
        """One 20 ms control step. push = world-frame force vector (N) on the pelvis for the whole step."""
        action = np.asarray(action, float)
        self.last_action = action.copy()
        self.target = action * ACTION_SCALE + DEFAULT
        d, m = self.d, self.m
        d.xfrc_applied[self.pelvis, :3] = 0.0 if push is None else push
        for _ in range(SUBSTEPS):
            tau = (self.target - d.qpos[7:]) * KP - d.qvel[6:] * KD
            d.ctrl[:] = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
            mujoco.mj_step(m, d)
        d.xfrc_applied[self.pelvis, :3] = 0.0
        self.t += 1
