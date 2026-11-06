"""Memory-fit check of a packed model against the targets. Arithmetic only: no firmware has been built or measured.

  .venv/bin/python tools/memfit.py results/models/x.bin [more.bin ...] [--out results/memfit.json]

Code-size and stack figures are budgets (assumptions), not measurements; they are stated in the output.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hoc import pack

TARGETS = {
    "RP2350 (Pico 2)": {"flash": 4 * 1024 * 1024, "ram": 520 * 1024, "code_budget": 8 * 1024, "stack": 2048},
    "PY32F002B": {"flash": 24 * 1024, "ram": 3 * 1024, "code_budget": 6 * 1024, "stack": 512},
}
QR_LIMIT = 2953  # QR version 40, error correction L, binary mode
SERIAL_RX = 47 + 8   # int8 obs + frame header/CRC
SERIAL_TX = 24 + 8   # 12 x int16 + frame header/CRC


def fit(q, nbytes):
    dims = q.dims
    macs = sum(l.W.size for l in q.layers)
    act_ram = max(dims[i] + dims[i + 1] for i in range(len(dims) - 1))  # ping-pong int8 buffers (largest adjacent pair)
    out = {"model_bytes": nbytes, "dims": dims, "bits": [l.bits for l in q.layers], "macs_per_step": macs,
           "qr": {"limit_bytes": QR_LIMIT, "fits": nbytes <= QR_LIMIT, "margin": QR_LIMIT - nbytes}}
    for name, t in TARGETS.items():
        flash = nbytes + t["code_budget"]
        ram = act_ram * 1 + 4 * max(dims[1:]) + SERIAL_RX + SERIAL_TX + t["stack"]  # int8 acts + int32 acc row + buffers + stack
        out[name] = {"flash_used": flash, "flash_total": t["flash"], "flash_ok": flash <= t["flash"],
                     "ram_used": ram, "ram_total": t["ram"], "ram_ok": ram <= t["ram"],
                     "assumed_code_budget": t["code_budget"], "assumed_stack": t["stack"]}
    return out


def teacher_fit():
    n = 4 * 64 * (47 + 64) + 2 * 4 * 64 + 64 * 32 + 32 + 32 * 12 + 12
    flash = n * 4
    ram = 2 * 64 * 4 + (47 + 256 + 64 + 32 + 12) * 4 + 2048
    r = {"params": n, "float32_flash_bytes": flash, "int8_flash_bytes": n, "ram_state_and_scratch_float32": ram}
    for name, t in TARGETS.items():
        r[name] = {"float32_fits_flash": flash + t["code_budget"] <= t["flash"], "int8_fits_flash": n + t["code_budget"] <= t["flash"],
                   "ram_fits": ram <= t["ram"]}
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {"teacher_original": teacher_fit(), "models": {}}
    for p in a.models:
        buf = Path(p).read_bytes()
        res["models"][Path(p).stem] = fit(pack.from_bytes(buf), len(buf))
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
