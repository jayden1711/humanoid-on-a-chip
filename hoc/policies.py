"""Policy objects. Interface: reset(); act(obs float32[47]) -> float32[12] (raw action, before the 0.25 scale)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MOTION_PT = ROOT / "third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt"
TEACHER_NPZ = ROOT / "results/models/teacher_lstm.npz"


def extract_teacher(dst=TEACHER_NPZ):
    import torch
    m = torch.jit.load(str(MOTION_PT), map_location="cpu").eval()
    sd = {k: v.numpy().astype(np.float32) for k, v in m.state_dict().items() if "state" not in k}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **sd)
    return sd


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TeacherLSTM:
    """Original unitree_rl_gym G1 policy: LSTM(47->64) + MLP 64->32->12 (ELU), float32 numpy."""
    name = "teacher_float32"

    def __init__(self, weights=None):
        w = dict(np.load(TEACHER_NPZ)) if weights is None else weights
        self.w = {k: np.asarray(v, np.float32) for k, v in w.items()}
        self.reset()

    def reset(self):
        self.h = np.zeros(64, np.float32)
        self.c = np.zeros(64, np.float32)

    def act(self, obs):
        w = self.w
        g = w["memory.weight_ih_l0"] @ obs + w["memory.weight_hh_l0"] @ self.h + w["memory.bias_ih_l0"] + w["memory.bias_hh_l0"]
        i, f, gg, o = g[:64], g[64:128], g[128:192], g[192:]
        self.c = (sigmoid(f) * self.c + sigmoid(i) * np.tanh(gg)).astype(np.float32)
        self.h = (sigmoid(o) * np.tanh(self.c)).astype(np.float32)
        x = elu(w["actor.0.weight"] @ self.h + w["actor.0.bias"])
        return (w["actor.2.weight"] @ x + w["actor.2.bias"]).astype(np.float32)


def fake_quant_sym(w, bits, axis=1):
    """Symmetric per-output-row weight fake-quantization (float arithmetic, quantized values)."""
    qmax = 2 ** (bits - 1) - 1 if bits > 2 else 1
    qmin = -(2 ** (bits - 1))
    s = np.abs(w).max(axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1, s)
    return (np.clip(np.round(w / s), qmin, qmax) * s).astype(np.float32)


class TeacherFakeQuant(TeacherLSTM):
    """Teacher with weights rounded to `bits` bits (per-row scales), float arithmetic. Not an integer reference."""

    def __init__(self, bits):
        w = dict(np.load(TEACHER_NPZ))
        for k in w:
            if k.endswith("weight") or "weight_" in k:
                w[k] = fake_quant_sym(w[k], bits)
        super().__init__(w)
        self.name = f"teacher_ptq_w{bits}_float_arith"
