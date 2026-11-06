"""Scenarios (seeded disturbance schedules), closed-loop rollouts, walking/push metrics and the equivalence comparison.

A scenario is a plain dict, fully determined by (name, seed), so a teacher and a compressed policy see exactly the same
commands, pushes and sensor corruption.
"""
import numpy as np
from .env import G1Env, CTRL_DT

PUSH_LEVELS = list(range(50, 901, 50))
PUSH_FIRST, PUSH_EVERY, PUSH_DUR = 2.0, 2.0, 0.1


def _cmd_fn(kind):
    if kind == "stand":
        return lambda t: (0.0, 0.0, 0.0)
    if kind == "walk":
        return lambda t: (0.5, 0.0, 0.0)
    if kind == "seq":  # forward, forward+turn, stop, sideways, back
        def f(t):
            if t < 4: return (0.5, 0, 0)
            if t < 8: return (0.4, 0, 0.5)
            if t < 11: return (0, 0, 0)
            if t < 15: return (0, 0.3, 0)
            return (-0.3, 0, 0)
        return f
    raise KeyError(kind)


def make_scenario(name, seed):
    """name in: walk_clean, seq_clean, stand_push, walk_push, walk_dropout, walk_delay, walk_imu, walk_mix, stand_clean"""
    rng = np.random.default_rng(1000 * sum(map(ord, name)) + seed)
    sc = {"name": name, "seed": seed, "cmd": "walk", "T": 20.0, "pushes": [], "dropouts": [], "delays": [], "spikes": []}
    if name == "stand_clean":
        sc.update(cmd="stand", T=10.0)
    elif name == "walk_clean":
        pass
    elif name == "seq_clean":
        sc.update(cmd="seq", T=20.0)
    elif name in ("stand_push", "walk_push"):
        sc.update(cmd="stand" if name == "stand_push" else "walk")
        sc["T"] = PUSH_FIRST + len(PUSH_LEVELS) * PUSH_EVERY
        prng = np.random.default_rng(10_000 + seed)  # same push directions for stand and walk (arena-style ladder)
        for k, F in enumerate(PUSH_LEVELS):
            a = prng.uniform(-np.pi, np.pi)
            sc["pushes"].append({"t": PUSH_FIRST + k * PUSH_EVERY, "dur": PUSH_DUR, "force": float(F),
                                 "dir": [float(np.cos(a)), float(np.sin(a))]})
    if name in ("walk_dropout", "walk_mix"):
        for _ in range(4):  # frozen observation for 3-8 control steps
            sc["dropouts"].append({"t": float(rng.uniform(2, 18)), "steps": int(rng.integers(3, 9))})
    if name in ("walk_delay", "walk_mix"):
        for _ in range(2):  # observation arrives k steps late for 1-2 s
            sc["delays"].append({"t": float(rng.uniform(2, 16)), "dur": float(rng.uniform(1, 2)), "k": int(rng.integers(1, 3))})
    if name in ("walk_imu", "walk_mix"):
        for _ in range(5):  # IMU spike: gyro and gravity vector jolted for 1-2 steps
            sc["spikes"].append({"t": float(rng.uniform(2, 18)), "steps": int(rng.integers(1, 3)),
                                 "gyro": rng.normal(0, 4.0, 3).tolist(), "tilt": rng.normal(0, 0.25, 2).tolist()})
    if name in ("walk_mix", "walk_pushmix"):
        for _ in range(4):
            a = rng.uniform(-np.pi, np.pi)
            sc["pushes"].append({"t": float(rng.uniform(3, 18)), "dur": 0.1, "force": float(rng.uniform(80, 250)),
                                 "dir": [float(np.cos(a)), float(np.sin(a))]})
    sc["pushes"].sort(key=lambda p: p["t"])
    return sc


SCENARIOS_CLEAN = ["walk_clean", "seq_clean", "stand_clean"]
SCENARIOS_DISTURB = ["walk_dropout", "walk_delay", "walk_imu", "walk_mix"]
SCENARIOS_PUSH = ["stand_push", "walk_push"]
SCENARIOS_ALL = SCENARIOS_CLEAN + SCENARIOS_DISTURB + SCENARIOS_PUSH


def events(sc):
    """Time-ordered list of (t_start, t_end, label) disturbance events, for attributing divergences."""
    ev = []
    for p in sc["pushes"]:
        ev.append((p["t"], p["t"] + p["dur"], f"push {p['force']:.0f} N"))
    for d in sc["dropouts"]:
        ev.append((d["t"], d["t"] + d["steps"] * CTRL_DT, f"sensor dropout {d['steps']} steps"))
    for d in sc["delays"]:
        ev.append((d["t"], d["t"] + d["dur"], f"obs delay {d['k']} steps"))
    for d in sc["spikes"]:
        ev.append((d["t"], d["t"] + d["steps"] * CTRL_DT, "IMU spike"))
    return sorted(ev)


def rollout(policy, sc, record=True):
    """Closed-loop episode. Returns dict of per-step arrays. The corrupted observation is what the policy sees."""
    env = G1Env().reset(sc["seed"])
    policy.reset()
    cmdf = _cmd_fn(sc["cmd"])
    n = int(round(sc["T"] / CTRL_DT))
    hist = []  # clean observation history for delays
    last_good = None
    frozen = 0
    rec = {k: [] for k in ("t", "qpos", "qvel", "obs", "act", "cmd", "push")} if record else None
    fell_t = None
    for k in range(n):
        t = k * CTRL_DT
        cmd = cmdf(t)
        o = env.obs(cmd)
        hist.append(o)
        # sensor corruption
        for d in sc["dropouts"]:
            if abs(t - d["t"]) < CTRL_DT / 2:
                frozen = d["steps"]
        if frozen > 0 and last_good is not None:
            o_in = last_good
            frozen -= 1
        else:
            o_in = o
            for d in sc["delays"]:
                if d["t"] <= t < d["t"] + d["dur"]:
                    o_in = hist[max(0, k - d["k"])]
            for s in sc["spikes"]:
                if s["t"] <= t < s["t"] + s["steps"] * CTRL_DT:
                    o_in = o_in.copy()
                    o_in[0:3] += np.array(s["gyro"], np.float32) * 0.25
                    o_in[3:5] += np.array(s["tilt"], np.float32)
            last_good = o_in
        a = policy.act(o_in)
        push = None
        for p in sc["pushes"]:
            if p["t"] <= t < p["t"] + p["dur"]:
                push = np.array([p["dir"][0], p["dir"][1], 0.0]) * p["force"]
        if rec is not None:
            rec["t"].append(t); rec["qpos"].append(env.d.qpos.copy()); rec["qvel"].append(env.d.qvel[:6].copy())
            rec["obs"].append(o_in); rec["act"].append(np.asarray(a, np.float32)); rec["cmd"].append(cmd)
            rec["push"].append(0.0 if push is None else float(np.linalg.norm(push)))
        env.step(a, push)
        if env.fell():
            fell_t = (k + 1) * CTRL_DT
            break
    out = {"fell_t": fell_t, "n": k + 1, "scenario": sc}
    if rec is not None:
        out.update({k_: np.array(v) for k_, v in rec.items()})
    return out


def yaw_of(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def walk_metrics(r, t0=3.0):
    """Walking quality of a rollout: survived, mean forward speed vs command, lateral/yaw tracking error."""
    if r["fell_t"] is not None:
        return {"fell": True, "fell_t": r["fell_t"]}
    t = r["t"]; m = t >= t0
    yaw = yaw_of(r["qpos"])
    c, s = np.cos(yaw), np.sin(yaw)
    vx_w, vy_w = r["qvel"][:, 0], r["qvel"][:, 1]
    vb_x = c * vx_w + s * vy_w
    vb_y = -s * vx_w + c * vy_w
    cmd = r["cmd"]
    return {"fell": False, "fell_t": None,
            "vx_err": float(np.mean(np.abs(vb_x[m] - cmd[m, 0]))), "vy_err": float(np.mean(np.abs(vb_y[m] - cmd[m, 1]))),
            "wz_err": float(np.mean(np.abs(r["qvel"][m, 5] - cmd[m, 2]))),
            "vx_mean": float(np.mean(vb_x[m])), "height": float(np.mean(r["qpos"][m, 2])),
            "tilt": float(np.mean(np.degrees(np.arccos(np.clip(-r["obs"][m, 5], -1, 1)))))}


def push_score(r):
    """Largest shove survived: still standing when the next one arrives (arena-style falls-only scoring)."""
    sc = r["scenario"]
    ps = sc["pushes"]
    ft = r["fell_t"]
    surv = [p["force"] for p in ps if ft is None or p["t"] + PUSH_EVERY <= ft]
    culprit = None
    if ft is not None:
        before = [p for p in ps if p["t"] <= ft]
        culprit = before[-1]["force"] if before else None
    return {"max_survived_N": max(surv) if surv else 0.0, "fell_t": ft, "failed_after_N": culprit}


# ----------------------------------------------------------------------------------------------- equivalence
def equivalence(ref, cand, sc, ref_run=None, div_thresh=0.10):
    """Compare candidate to reference on one scenario.

    - closed-loop: both policies drive their own rollouts under the identical scenario; compare pelvis trajectory,
      joint states, fall / no-fall, and the first time the pelvis positions differ by more than div_thresh metres.
    - teacher-forced: the candidate is fed the reference's recorded observations (no drift) and its action error per
      joint and step is measured, isolating the policy difference from trajectory drift.
    """
    a = ref_run if ref_run is not None else rollout(ref, sc)
    b = rollout(cand, sc)
    n = min(a["n"], b["n"])
    ev = events(sc)
    pa, pb = a["qpos"][:n, 0:2], b["qpos"][:n, 0:2]
    dpos = np.linalg.norm(pa - pb, axis=1)
    dq = np.sqrt(np.mean((a["qpos"][:n, 7:] - b["qpos"][:n, 7:]) ** 2, axis=1))
    dyaw = np.abs(np.angle(np.exp(1j * (yaw_of(a["qpos"][:n]) - yaw_of(b["qpos"][:n])))))
    dact = np.abs(a["act"][:n] - b["act"][:n])  # closed-loop action difference (includes drift effects)
    idx = np.nonzero(dpos > div_thresh)[0]
    t_div = float(a["t"][idx[0]]) if len(idx) else None
    cause = None
    if t_div is not None:
        near = [e for e in ev if e[0] - 0.5 <= t_div <= e[1] + 1.0]
        cause = near[-1][2] + f" at t={near[-1][0]:.2f}s" if near else "no disturbance event nearby (gait-phase drift)"
    # teacher-forced action error
    cand.reset()
    tf = np.array([cand.act(o) for o in a["obs"]])
    err_tf = np.abs(tf - a["act"])
    return {"scenario": sc["name"], "seed": sc["seed"],
            "ref_fell_t": a["fell_t"], "cand_fell_t": b["fell_t"],
            "outcome": ("both_stand" if a["fell_t"] is None and b["fell_t"] is None else
                        "both_fall" if a["fell_t"] is not None and b["fell_t"] is not None else
                        "cand_falls_ref_stands" if b["fell_t"] is not None else "cand_stands_ref_falls"),
            "t_diverge_s": t_div, "diverge_cause": cause,
            "max_pelvis_xy_drift_m": float(dpos.max()), "final_pelvis_xy_drift_m": float(dpos[-1]),
            "rms_joint_diff_rad_mean": float(dq.mean()), "max_yaw_diff_rad": float(dyaw.max()),
            "closed_loop_action_absdiff_mean": float(dact.mean()),
            "tf_action_err_mean": float(err_tf.mean()), "tf_action_err_max": float(err_tf.max()),
            "tf_action_err_per_joint_mean": err_tf.mean(axis=0).round(4).tolist(),
            "tf_worst_step": {"t": float(a["t"][int(err_tf.max(axis=1).argmax())]), "err": float(err_tf.max())}}
