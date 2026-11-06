"""Build results/report.md from results/eval/*.json (+ model sizes): size vs walking quality vs push resistance, and the
per-version equivalence summary. Output stays in results/ (git-ignored).

  .venv/bin/python tools/report.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

R = Path(__file__).resolve().parent.parent / "results"


def model_size(spec):
    if spec.startswith("int:"):
        from hoc import pack
        q = pack.from_bytes(Path(spec[4:]).read_bytes() if Path(spec[4:]).is_absolute() else (R.parent / spec[4:]).read_bytes())
        s = pack.size_breakdown(q)
        return s["total"], "x".join(map(str, s["dims"])) + " w" + "/".join(map(str, s["bits"]))
    if spec == "teacher":
        return 31388 * 4, "LSTM64+MLP32 f32"
    if spec.startswith("ptqW"):
        return 31388 * int(spec[4:]) // 8, f"LSTM64+MLP32 w{spec[4:]} (float arith)"
    if spec.startswith("float:"):
        d = np.load(R.parent / spec[6:] if not Path(spec[6:]).is_absolute() else spec[6:])
        n = sum(d[f"W{i}"].size + d[f"b{i}"].size for i in range(len(d["dims"]) - 1))
        return n * 4, "x".join(map(str, d["dims"])) + " f32"
    return 0, spec


def main():
    rows = []
    for p in sorted((R / "eval").glob("*.json")):
        d = json.loads(p.read_text())
        sz, desc = model_size(d["spec"])
        s = d["summary"]
        g = lambda sc, k: s.get(sc, {}).get(k)
        eq = lambda sc: s.get(sc, {}).get("equiv", {}).get("outcomes", {})
        n = d["seeds"]
        nfall_clean = sum((g(sc, "falls") or 0) for sc in ("walk_clean", "seq_clean", "stand_clean"))
        nfall_dist = sum((g(sc, "falls") or 0) for sc in ("walk_dropout", "walk_delay", "walk_imu", "walk_mix"))
        div = sum(s[sc]["equiv"]["outcomes"]["cand_falls_ref_stands"] for sc in s if "equiv" in s[sc])
        rev = sum(s[sc]["equiv"]["outcomes"]["cand_stands_ref_falls"] for sc in s if "equiv" in s[sc])
        rows.append({"file": p.stem, "spec": d["spec"], "desc": desc, "bytes": sz,
                     "walk_vx_err": g("walk_clean", "vx_err"), "falls_clean": nfall_clean, "falls_dist": nfall_dist,
                     "stand_push": g("stand_push", "median_max_survived_N"), "walk_push": g("walk_push", "median_max_survived_N"),
                     "cand_falls_ref_stands": div, "cand_stands_ref_falls": rev, "n": n})
    rows.sort(key=lambda r: -r["bytes"])
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    out = ["# Compression study: size vs quality (host only, MuJoCo)", "",
           "Seeds per scenario: %d. falls_clean = falls over walk/seq/stand clean episodes; falls_dist = over dropout/delay/IMU/mix walking;"
           " push = median over seeds of the largest shove survived (N). cand_falls = episodes where the candidate fell but the teacher stood"
           " (equivalence check, all scenarios); cand_stands = the reverse." % (rows[0]["n"] if rows else 0), "",
           "| version | bytes | walk vx err (m/s) | falls clean | falls dist | stand push N | walk push N | cand falls / ref stands | cand stands / ref falls |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"| {r['file']} ({r['desc']}) | {r['bytes']} | {f(r['walk_vx_err'])} | {r['falls_clean']} | {r['falls_dist']} | "
                   f"{f(r['stand_push'], 0)} | {f(r['walk_push'], 0)} | {r['cand_falls_ref_stands']} | {r['cand_stands_ref_falls']} |")
    (R / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
