"""Integer-only reference inference for a packed model (hoc.pack). This is the numerics the firmware must reproduce
bit for bit; the on-chip HIL check compares against it. Accumulators are int32-range values (asserted); the requant
multiply is done in int64 here (firmware needs a 32x16->48 bit multiply or a pre-shift; to be settled in firmware work).
"""
import numpy as np
from . import pack


class IntPolicy:
    def __init__(self, q: pack.QModel, name="int"):
        self.q = q
        self.name = name
        self.scale = (2.0 ** q.in_exp.astype(np.float64))
        self.max_acc = 0
        self.reset()

    @classmethod
    def from_file(cls, path):
        return cls(pack.from_bytes(open(path, "rb").read()), name=f"int:{path}")

    def reset(self):
        pass

    def quant_obs(self, obs):
        return np.clip(np.rint(obs.astype(np.float64) * self.scale), -128, 127).astype(np.int64)

    def forward_int(self, x):
        L = len(self.q.layers)
        for i, l in enumerate(self.q.layers):
            acc = l.W.astype(np.int64) @ x + l.B
            self.max_acc = max(self.max_acc, int(np.abs(acc).max()))
            y = (acc * l.M + (1 << (l.S - 1))) >> l.S
            x = np.clip(y, 0, 127) if i < L - 1 else np.clip(y, -32768, 32767)
        return x

    def act(self, obs):
        return (self.forward_int(self.quant_obs(obs)) / 256.0).astype(np.float32)
