"""Packed integer model format (the thing that would be stored in flash / a QR code) and its exact byte accounting.

Layout (little endian):
  'H','C' | version=1 | L (layers) | dims[L+1] u16 | bits[L] u8 | shift S[L] u8 | in_exp[47] as signed nibbles (24 B)
  per layer: M[m] u16 (requant multipliers) | B[m] i16 (biases in accumulator units) | W packed (offset codes, LSB first)
  crc32 (4 B)

Inference (integer only):  x = clip(round(obs * 2^in_exp), -128, 127)   (done on the host; the chip receives int8)
  acc = W x + B      (int32)
  hidden: y = clip((acc*M + 2^(S-1)) >> S, 0, 127)         last: y = clip((acc*M + 2^(S-1)) >> S, -32768, 32767)
  action = y_last / 256
"""
import struct
import zlib
from dataclasses import dataclass, field
import numpy as np

MAGIC = b"HC"
VERSION = 1
OBS_DIM = 47


@dataclass
class QLayer:
    bits: int
    W: np.ndarray      # (m, n) int, values in [-2^(b-1), 2^(b-1)-1]
    B: np.ndarray      # (m,) int16
    M: np.ndarray      # (m,) uint16
    S: int


@dataclass
class QModel:
    in_exp: np.ndarray                       # (47,) int in [-8, 7]
    layers: list = field(default_factory=list)

    @property
    def dims(self):
        return [self.layers[0].W.shape[1]] + [l.W.shape[0] for l in self.layers]

    def n_weights(self):
        return int(sum(l.W.size for l in self.layers))


def _pack_codes(codes, bits):
    """codes: 1-D uint array, packed LSB first, `bits` per value."""
    n = len(codes)
    total = (n * bits + 7) // 8
    out = np.zeros(total, np.uint8)
    bitpos = np.arange(n) * bits
    for b in range(bits):
        bit = ((codes >> b) & 1).astype(np.uint8)
        pos = bitpos + b
        np.add.at(out, pos // 8, (bit << (pos % 8)).astype(np.uint8))
    return out.tobytes()


def _unpack_codes(buf, n, bits):
    arr = np.frombuffer(buf, np.uint8)
    codes = np.zeros(n, np.int64)
    bitpos = np.arange(n) * bits
    for b in range(bits):
        pos = bitpos + b
        codes |= ((arr[pos // 8] >> (pos % 8)) & 1).astype(np.int64) << b
    return codes


def to_bytes(q: QModel):
    dims = q.dims
    L = len(q.layers)
    out = bytearray(MAGIC + bytes([VERSION, L]))
    out += struct.pack(f"<{L + 1}H", *dims)
    out += bytes(l.bits for l in q.layers)
    out += bytes(l.S for l in q.layers)
    e = (np.asarray(q.in_exp, int) & 0xF).astype(np.uint8)
    if len(e) % 2:
        e = np.append(e, 0)
    out += bytes((e[0::2] | (e[1::2] << 4)).tolist())
    for l in q.layers:
        out += struct.pack(f"<{len(l.M)}H", *[int(v) for v in l.M])
        out += struct.pack(f"<{len(l.B)}h", *[int(v) for v in l.B])
        codes = (l.W.astype(np.int64) - (-(1 << (l.bits - 1)))).reshape(-1)
        out += _pack_codes(codes, l.bits)
    out += struct.pack("<I", zlib.crc32(bytes(out)) & 0xFFFFFFFF)
    return bytes(out)


def from_bytes(buf: bytes):
    if zlib.crc32(buf[:-4]) & 0xFFFFFFFF != struct.unpack("<I", buf[-4:])[0]:
        raise ValueError("bad CRC")
    assert buf[:2] == MAGIC and buf[2] == VERSION
    L = buf[3]
    p = 4
    dims = struct.unpack_from(f"<{L + 1}H", buf, p); p += 2 * (L + 1)
    bits = list(buf[p:p + L]); p += L
    S = list(buf[p:p + L]); p += L
    nb = (OBS_DIM + 1) // 2
    nib = np.frombuffer(buf[p:p + nb], np.uint8); p += nb
    e = np.empty(2 * nb, int); e[0::2] = nib & 0xF; e[1::2] = nib >> 4
    e = e[:OBS_DIM]; e = np.where(e >= 8, e - 16, e)
    layers = []
    for l in range(L):
        n, m = dims[l], dims[l + 1]
        M = np.array(struct.unpack_from(f"<{m}H", buf, p), np.int64); p += 2 * m
        B = np.array(struct.unpack_from(f"<{m}h", buf, p), np.int64); p += 2 * m
        wb = (n * m * bits[l] + 7) // 8
        codes = _unpack_codes(buf[p:p + wb], n * m, bits[l]); p += wb
        W = (codes + (-(1 << (bits[l] - 1)))).reshape(m, n)
        layers.append(QLayer(bits[l], W, B, M, S[l]))
    assert p == len(buf) - 4
    return QModel(e, layers)


def size_breakdown(q: QModel):
    L = len(q.layers)
    dims = q.dims
    hdr = 4 + 2 * (L + 1) + L + L + (OBS_DIM + 1) // 2 + 4
    scales = sum(2 * len(l.M) for l in q.layers)
    biases = sum(2 * len(l.B) for l in q.layers)
    weights = sum((l.W.size * l.bits + 7) // 8 for l in q.layers)
    return {"total": hdr + scales + biases + weights, "header_and_crc": hdr, "requant_multipliers": scales,
            "biases": biases, "weights": weights, "n_weights": q.n_weights(), "dims": dims,
            "bits": [l.bits for l in q.layers]}
