"""Distill the teacher LSTM into a small MLP with DAgger, then quantization-aware fine-tuning to b-bit weights.

  sim-run .venv/bin/python tools/distill.py --name h32_w4 --hidden 32,32 --bits 4

Writes results/models/<name>.bin (packed integer model), <name>_float.npz (float student) and <name>.train.json.
Training seeds are >= 1000; evaluation seeds are < 100, so there is no overlap.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

OUT = Path(__file__).resolve().parent.parent / "results/models"


def _init():
    os.nice(10)
    torch.set_num_threads(1)


def collect(args):
    spec, seeds, beta, noise = args
    from hoc.registry import load_policy
    from hoc.evalsuite import rollout, random_train_scenario
    teacher = load_policy("teacher")
    pol = teacher if spec == "teacher" else load_policy(spec)
    obs, lab, falls, steps = [], [], 0, 0
    for s in seeds:
        rng = np.random.default_rng(s)
        sc = random_train_scenario(s)
        r = rollout(pol, sc, label=teacher, beta=beta, act_noise=noise * rng.uniform(0.3, 1.0), rng=rng)
        obs.append(r["obs"]); lab.append(r["label"])
        falls += r["fell_t"] is not None
        steps += r["n"]
    return np.concatenate(obs).astype(np.float32), np.concatenate(lab).astype(np.float32), falls, steps


def run_collect(ex, spec, seed0, n, beta, noise, jobs):
    chunks = [list(range(seed0 + i, seed0 + n, jobs)) for i in range(jobs)]
    res = list(ex.map(collect, [(spec, c, beta, noise) for c in chunks]))
    return (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]),
            sum(r[2] for r in res), sum(r[3] for r in res))


def train(model, X, Y, epochs, lr, bs=2048, wd=0.0, log=None):
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    n = len(X)
    steps = epochs * ((n + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    last = None
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(model(Xt[idx]), Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss) * len(idx)
        last = tot / n
    return last


@torch.no_grad()
def mse_on(model, X, Y):
    return float(torch.nn.functional.mse_loss(model(torch.from_numpy(X)), torch.from_numpy(Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hidden", required=True, help="e.g. 32,32")
    ap.add_argument("--bits", required=True, help="one value for all layers or one per layer, e.g. 4 or 8,4,4")
    ap.add_argument("--float-iters", type=int, default=4)
    ap.add_argument("--qat-iters", type=int, default=6)
    ap.add_argument("--eps", type=int, default=100, help="episodes per DAgger iteration")
    ap.add_argument("--eps0", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--init-float", default=None, help="reuse a float student npz + its dataset (skips float stage)")
    a = ap.parse_args()
    from hoc.student import QMLP, choose_in_exp
    from hoc.intref import IntPolicy
    from hoc import pack
    torch.manual_seed(0)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    hidden = [int(h) for h in a.hidden.split(",")]
    dims = [47] + hidden + [12]
    L = len(dims) - 1
    bl = [int(b) for b in a.bits.split(",")]
    bits = bl * L if len(bl) == 1 else bl
    assert len(bits) == L
    log = {"name": a.name, "dims": dims, "bits": bits, "iters": []}
    t0 = time.time()
    seed_ctr = 1000
    with ProcessPoolExecutor(a.jobs, initializer=_init) as ex:
        # --- iteration 0: teacher drives (with action noise for state coverage)
        X, Y, f, st = run_collect(ex, "teacher", seed_ctr, a.eps0, 1.0, 0.15, a.jobs); seed_ctr += a.eps0
        print(f"[{time.time()-t0:5.0f}s] data0 {len(X)} samples, teacher falls {f}/{a.eps0}", flush=True)
        in_exp = choose_in_exp(X)
        model = QMLP(dims, bits, in_exp)
        # --- float DAgger
        fl_path = OUT / f"{a.name}_float.npz"
        for it in range(a.float_iters + 1):
            ep = a.epochs * 2 if it == 0 else a.epochs
            tr = train(model, X, Y, ep, 3e-3 if it == 0 else 1.5e-3)
            model.save_float(fl_path)
            beta = max(0.0, 0.3 - 0.15 * it)
            if it == a.float_iters:
                beta = 0.0
            Xn, Yn, f, st = run_collect(ex, f"float:{fl_path}", seed_ctr, a.eps, beta, 0.0, a.jobs); seed_ctr += a.eps
            X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn])
            print(f"[{time.time()-t0:5.0f}s] float it{it}: train mse {tr:.5f}, student falls {f}/{a.eps} (beta {beta}), data {len(X)}", flush=True)
            log["iters"].append({"stage": "float", "it": it, "mse": tr, "falls": f, "eps": a.eps, "beta": beta})
        # --- QAT
        model.init_scales(torch.from_numpy(X[::5]))
        model.quant = True
        best = None
        for it in range(a.qat_iters + 1):
            tr = train(model, X, Y, a.epochs, 1e-3 if it == 0 else 6e-4)
            qm = model.export()
            blob = pack.to_bytes(qm)
            p = OUT / f"{a.name}.iter{it}.bin"
            p.write_bytes(blob)
            beta = 0.0
            Xn, Yn, f, st = run_collect(ex, f"int:{p}", seed_ctr, a.eps, beta, 0.0, a.jobs); seed_ctr += a.eps
            X, Y = np.concatenate([X, Xn]), np.concatenate([Y, Yn])
            sz = pack.size_breakdown(qm)["total"]
            print(f"[{time.time()-t0:5.0f}s] qat it{it}: train mse {tr:.5f}, int student falls {f}/{a.eps}, {sz} bytes", flush=True)
            log["iters"].append({"stage": "qat", "it": it, "mse": tr, "falls": f, "eps": a.eps, "bytes": sz})
            if best is None or f < best[0]:
                best = (f, it)
        # final: fine-tune on all data with a last low-lr pass and export it too, then pick the best iterate by falls
        tr = train(model, X, Y, a.epochs, 3e-4)
        qm = model.export(); blob = pack.to_bytes(qm)
        pf = OUT / f"{a.name}.final.bin"; pf.write_bytes(blob)
        Xn, Yn, f, st = run_collect(ex, f"int:{pf}", seed_ctr, a.eps, 0.0, 0.0, a.jobs)
        print(f"[{time.time()-t0:5.0f}s] final: train mse {tr:.5f}, int student falls {f}/{a.eps}", flush=True)
        log["iters"].append({"stage": "final", "mse": tr, "falls": f, "eps": a.eps})
        if f <= best[0]:
            chosen = pf
        else:
            chosen = OUT / f"{a.name}.iter{best[1]}.bin"
        (OUT / f"{a.name}.bin").write_bytes(chosen.read_bytes())
    q = pack.from_bytes((OUT / f"{a.name}.bin").read_bytes())
    log["chosen"] = str(chosen.name); log["size"] = pack.size_breakdown(q); log["wall_s"] = time.time() - t0
    (OUT / f"{a.name}.train.json").write_text(json.dumps(log, indent=1, default=str))
    print("chosen", chosen.name, log["size"])


if __name__ == "__main__":
    main()
