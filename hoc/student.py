"""Distilled student MLP: float training, quantization-aware training (QAT) and export to the packed integer model.

Input transform (fixed, part of the format): x_int = clip(round(obs * 2^e_i), -128, 127), network input x = x_int / 128.
Hidden activations are int8 in [0, 127] with a learned per-layer scale; weights are per-output-channel b-bit integers
with learned scales; biases live on the accumulator grid; output is Q8 (action = int16 / 256).
"""
import math
import numpy as np
import torch
import torch.nn as nn
from . import pack

OBS_DIM, ACT_DIM = 47, 12


def choose_in_exp(obs, pct=99.9):
    m = np.percentile(np.abs(obs), pct, axis=0)
    m = np.maximum(m, 1e-3)
    return np.clip(np.floor(np.log2(127.0 / m)), -8, 7).astype(int)


def rnd(x):
    return x + (torch.round(x) - x).detach()


def gscale(s, g):
    return s * g + (s - s * g).detach()


class QMLP(nn.Module):
    def __init__(self, dims, bits, in_exp):
        super().__init__()
        self.dims, self.bits = list(dims), list(bits)
        self.in_exp = np.asarray(in_exp, int)
        self.register_buffer("in_mul", torch.tensor(2.0 ** self.in_exp, dtype=torch.float32))
        self.W = nn.ParameterList(); self.b = nn.ParameterList()
        self.log_sw = nn.ParameterList(); self.log_sa = nn.ParameterList()
        for i in range(len(dims) - 1):
            lin = nn.Linear(dims[i], dims[i + 1])
            nn.init.kaiming_uniform_(lin.weight, a=math.sqrt(5))
            self.W.append(lin.weight); self.b.append(lin.bias)
            self.log_sw.append(nn.Parameter(torch.zeros(dims[i + 1])))
        for i in range(len(dims) - 2):  # one learned activation scale per hidden layer output
            self.log_sa.append(nn.Parameter(torch.tensor(math.log(4.0 / 127))))
        self.quant = False

    @staticmethod
    def qrange(bits):
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1

    def x_in(self, obs):
        x = obs * self.in_mul
        if self.quant:
            return torch.clamp(rnd(x), -128, 127) / 128.0
        return torch.clamp(x, -128, 127) / 128.0

    def forward(self, obs):
        x = self.x_in(obs)
        L = len(self.W)
        sx = 1.0 / 128.0
        for i in range(L):
            W, b = self.W[i], self.b[i]
            if self.quant:
                qmin, qmax = self.qrange(self.bits[i])
                g = 1.0 / math.sqrt(W.shape[1] * qmax)
                sw = gscale(torch.exp(self.log_sw[i]), g)[:, None]
                Wh = torch.clamp(rnd(W / sw), qmin, qmax) * sw
                acc_s = sw[:, 0] * sx
                bh = torch.clamp(rnd(b / acc_s), -32768, 32767) * acc_s
            else:
                Wh, bh = W, b
            z = x @ Wh.T + bh
            if i < L - 1:
                if self.quant:
                    sa = gscale(torch.exp(self.log_sa[i]), 1.0 / math.sqrt(z.shape[1] * 127))
                    x = torch.clamp(rnd(z / sa), 0, 127) * sa
                    sx = sa
                else:
                    x = torch.relu(z)
            else:
                x = torch.clamp(rnd(z * 256.0), -32768, 32767) / 256.0 if self.quant else z
        return x

    @torch.no_grad()
    def init_scales(self, obs_batch):
        """Set weight scales (per-row MSE search) and activation scales (99.5 percentile) from float weights."""
        was = self.quant
        self.quant = False
        x = self.x_in(obs_batch)
        for i in range(len(self.W)):
            W = self.W[i]
            qmin, qmax = self.qrange(self.bits[i])
            best_s = None; best_e = None
            for c in np.linspace(0.25, 1.2, 40):
                s = (W.abs().amax(1, keepdim=True) / qmax * c).clamp_min(1e-8)
                e = ((torch.clamp(torch.round(W / s), qmin, qmax) * s - W) ** 2).sum(1)
                if best_s is None:
                    best_s, best_e = s.clone(), e.clone()
                else:
                    m = e < best_e
                    best_s[m] = s[m]; best_e[m] = e[m]
            self.log_sw[i].copy_(torch.log(best_s[:, 0]))
            z = x @ W.T + self.b[i]
            if i < len(self.W) - 1:
                x = torch.relu(z)
                self.log_sa[i].copy_(torch.log(torch.quantile(x.flatten()[::7], 0.995).clamp_min(1e-3) / 127))
        self.quant = was

    @torch.no_grad()
    def export(self):
        L = len(self.W)
        layers = []
        sx = 1.0 / 128.0
        for i in range(L):
            qmin, qmax = self.qrange(self.bits[i])
            sw = torch.exp(self.log_sw[i]).double()
            Wq = torch.clamp(torch.round(self.W[i].double() / sw[:, None]), qmin, qmax).numpy().astype(np.int64)
            acc_s = sw * sx
            Bq = torch.clamp(torch.round(self.b[i].double() / acc_s), -32768, 32767).numpy().astype(np.int64)
            sy = 1.0 / 256.0 if i == L - 1 else float(torch.exp(self.log_sa[i]))
            real = (acc_s / sy).numpy()
            S = int(np.clip(np.floor(np.log2(65535.0 / real.max())), 0, 31))
            M = np.clip(np.rint(real * 2.0 ** S), 0, 65535).astype(np.int64)
            layers.append(pack.QLayer(self.bits[i], Wq, Bq, M, S))
            sx = sy
        return pack.QModel(self.in_exp.copy(), layers)

    def save_float(self, path):
        d = {"dims": np.array(self.dims), "in_exp": self.in_exp}
        for i in range(len(self.W)):
            d[f"W{i}"] = self.W[i].detach().numpy(); d[f"b{i}"] = self.b[i].detach().numpy()
        np.savez(path, **d)


class FloatStudent:
    """Numpy float32 inference of a distilled float MLP (no quantization anywhere except the clamp of the input)."""
    name = "student_float32"

    def __init__(self, d):
        self.dims = list(d["dims"]); self.mul = (2.0 ** d["in_exp"]).astype(np.float32)
        self.W = [d[f"W{i}"].astype(np.float32) for i in range(len(self.dims) - 1)]
        self.b = [d[f"b{i}"].astype(np.float32) for i in range(len(self.dims) - 1)]

    @classmethod
    def load(cls, path):
        return cls(dict(np.load(path)))

    def reset(self):
        pass

    def act(self, obs):
        x = np.clip(obs * self.mul, -128, 127) / 128.0
        for i in range(len(self.W)):
            x = self.W[i] @ x + self.b[i]
            if i < len(self.W) - 1:
                x = np.maximum(x, 0)
        return x.astype(np.float32)
