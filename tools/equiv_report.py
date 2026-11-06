"""Equivalence report per version: where and why the compressed policy diverges from the teacher.

  .venv/bin/python tools/equiv_report.py h46_w4_L h32_w4 ...   -> results/equivalence.md
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"


def section(name):
    d = json.loads((R / "eval" / f"{name}.json").read_text())
    eq = [(r["scenario"], r["seed"], r["equiv"]) for r in d["rows"] if "equiv" in r]
    L = [f"## {name}  ({d['spec']})", ""]
    L.append("| scenario | stand/stand | fall/fall | cand falls, ref stands | cand stands, ref falls | diverged >10 cm | median t_diverge (s) | mean teacher-forced action err | max pelvis drift (m) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sc in sorted({e[0] for e in eq}):
        es = [e[2] for e in eq if e[0] == sc]
        oc = Counter(e["outcome"] for e in es)
        td = [e["t_diverge_s"] for e in es if e["t_diverge_s"] is not None]
        L.append(f"| {sc} | {oc['both_stand']} | {oc['both_fall']} | {oc['cand_falls_ref_stands']} | {oc['cand_stands_ref_falls']} | "
                 f"{len(td)}/{len(es)} | {np.median(td):.2f} | {np.mean([e['tf_action_err_mean'] for e in es]):.3f} | "
                 f"{max(e['max_pelvis_xy_drift_m'] for e in es):.2f} |" if td else
                 f"| {sc} | {oc['both_stand']} | {oc['both_fall']} | {oc['cand_falls_ref_stands']} | {oc['cand_stands_ref_falls']} | 0/{len(es)} | - | "
                 f"{np.mean([e['tf_action_err_mean'] for e in es]):.3f} | {max(e['max_pelvis_xy_drift_m'] for e in es):.2f} |")
    L.append("")
    cause = Counter()
    for sc, seed, e in eq:
        if e["outcome"] in ("cand_falls_ref_stands", "cand_stands_ref_falls"):
            cause[(e["outcome"], e["diverge_cause"] or "no 10 cm divergence before fall")] += 1
    L.append("Where fall/no-fall outcomes differ (outcome, first >10 cm divergence attributed to):")
    L.append("")
    if cause:
        for (o, c), n in cause.most_common(12):
            L.append(f"- {n}x {o}: {c.split(' at t=')[0]}")
    else:
        L.append("- none")
    L.append("")
    worst = sorted(eq, key=lambda x: -x[2]["tf_action_err_max"])[:3]
    L.append("Largest single-step teacher-forced action errors (same observations fed to both policies):")
    L.append("")
    for sc, seed, e in worst:
        L.append(f"- {sc} seed {seed}: |dAction| max {e['tf_action_err_max']:.2f} at t={e['tf_worst_step']['t']:.2f}s")
    jm = np.mean([e["tf_action_err_per_joint_mean"] for _, _, e in eq], axis=0)
    L.append("")
    L.append("Mean teacher-forced |dAction| per joint (L hip pitch/roll/yaw, knee, ankle pitch/roll, then right): "
             + ", ".join(f"{v:.3f}" for v in jm))
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    out = ["# Behavioral equivalence vs the original float teacher (host, seeds 0-19 per scenario)", "",
           "Same seeds, commands, pushes and sensor corruption for both policies. 'Teacher-forced' = candidate fed the teacher's"
           " recorded observations. Divergence = pelvis xy differs by more than 10 cm between the two closed-loop rollouts.", ""]
    for n in sys.argv[1:]:
        out.append(section(n))
    (R / "equivalence.md").write_text("\n".join(out))
    print("\n".join(out))
