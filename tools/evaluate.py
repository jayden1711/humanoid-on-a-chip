"""Full evaluation of one policy spec: walking quality, push resistance, and the equivalence check against the teacher.

  sim-run .venv/bin/python tools/evaluate.py --spec teacher --out results/eval/teacher.json
  sim-run .venv/bin/python tools/evaluate.py --spec int:results/models/x.bin --out results/eval/x.json

Output goes to results/ (git-ignored). Teacher rollouts are recomputed per run (cheap: ~0.5 s per 20 s episode).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from hoc.registry import load_policy
from hoc.evalsuite import (SCENARIOS_ALL, make_scenario, rollout, walk_metrics, push_score, equivalence, events)


def one(args):
    spec, name, seed, with_eq = args
    os.nice(10)
    pol = load_policy(spec)
    sc = make_scenario(name, seed)
    r = rollout(pol, sc)
    row = {"scenario": name, "seed": seed, "fell_t": r["fell_t"]}
    row.update(push_score(r) if name.endswith("push") else walk_metrics(r))
    if with_eq and spec != "teacher":
        ref = load_policy("teacher")
        row["equiv"] = equivalence(ref, pol, sc)
    return row


def summarize(rows):
    out = {}
    for name in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == name]
        d = {"episodes": len(rs), "falls": sum(r["fell_t"] is not None for r in rs) if not name.endswith("push") else None}
        if name.endswith("push"):
            v = [r["max_survived_N"] for r in rs]
            d.update(median_max_survived_N=float(np.median(v)), mean_max_survived_N=float(np.mean(v)),
                     min=float(np.min(v)), max=float(np.max(v)), per_seed=v)
        else:
            ok = [r for r in rs if not r["fell"]]
            d["falls"] = len(rs) - len(ok)
            for k in ("vx_err", "vy_err", "wz_err", "vx_mean", "height", "tilt"):
                if ok:
                    d[k] = float(np.mean([r[k] for r in ok]))
        eq = [r["equiv"] for r in rs if "equiv" in r]
        if eq:
            d["equiv"] = {
                "outcomes": {o: sum(e["outcome"] == o for e in eq) for o in
                             ("both_stand", "both_fall", "cand_falls_ref_stands", "cand_stands_ref_falls")},
                "episodes_diverged_gt10cm": sum(e["t_diverge_s"] is not None for e in eq),
                "median_t_diverge_s": float(np.median([e["t_diverge_s"] for e in eq if e["t_diverge_s"] is not None]))
                if any(e["t_diverge_s"] is not None for e in eq) else None,
                "tf_action_err_mean": float(np.mean([e["tf_action_err_mean"] for e in eq])),
                "tf_action_err_max": float(np.max([e["tf_action_err_max"] for e in eq])),
                "max_pelvis_xy_drift_m": float(np.max([e["max_pelvis_xy_drift_m"] for e in eq])),
                "mean_rms_joint_diff_rad": float(np.mean([e["rms_joint_diff_rad_mean"] for e in eq])),
            }
        out[name] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--no-equiv", action="store_true")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS_ALL))
    a = ap.parse_args()
    jobs = [(a.spec, n, s, not a.no_equiv) for n in a.scenarios.split(",") for s in range(a.seeds)]
    t0 = time.time()
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(one, jobs))
    else:
        rows = [one(j) for j in jobs]
    res = {"spec": a.spec, "seeds": a.seeds, "summary": summarize(rows), "rows": rows, "wall_s": time.time() - t0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for n, d in res["summary"].items():
        brief = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k not in ("per_seed", "equiv")}
        print(f"{n:14s}", brief, ("EQ " + str(d["equiv"]["outcomes"])) if "equiv" in d else "")


if __name__ == "__main__":
    main()
