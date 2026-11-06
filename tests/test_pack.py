"""Pack/unpack round trip, and QAT forward vs the integer reference. Run: .venv/bin/python tests/test_pack.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from hoc import pack
from hoc.student import QMLP
from hoc.intref import IntPolicy


def test_roundtrip():
    rng = np.random.default_rng(0)
    for bits in (2, 3, 4, 8):
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        layers = [pack.QLayer(bits, rng.integers(lo, hi + 1, (7, 47)), rng.integers(-3000, 3000, 7),
                              rng.integers(1, 65535, 7), 15),
                  pack.QLayer(bits, rng.integers(lo, hi + 1, (12, 7)), rng.integers(-3000, 3000, 12),
                              rng.integers(1, 65535, 12), 14)]
        q = pack.QModel(rng.integers(-8, 8, 47), layers)
        q2 = pack.from_bytes(pack.to_bytes(q))
        assert (q2.in_exp == q.in_exp).all()
        for a, b in zip(q.layers, q2.layers):
            assert a.bits == b.bits and a.S == b.S
            assert (a.W == b.W).all() and (a.B == b.B).all() and (a.M == b.M).all()
        assert len(pack.to_bytes(q)) == pack.size_breakdown(q)["total"]


def test_qat_matches_int():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    obs = rng.normal(0, 0.5, (4000, 47)).astype(np.float32)
    for bits in (8, 4, 2):
        m = QMLP([47, 24, 24, 12], [bits] * 3, np.full(47, 2))
        m.init_scales(torch.from_numpy(obs)); m.quant = True
        pol = IntPolicy(m.export())
        with torch.no_grad():
            ref = m(torch.from_numpy(obs[:200])).numpy()
        got = np.array([pol.act(o) for o in obs[:200]])
        err = np.abs(ref - got)
        print(f"bits {bits}: max |QAT forward - integer ref| = {err.max():.4f}, mean {err.mean():.5f}, max|out| {np.abs(ref).max():.2f}")
        assert err.mean() < 0.02, err.mean()


if __name__ == "__main__":
    test_roundtrip(); print("roundtrip ok")
    test_qat_matches_int(); print("qat vs int ok")
