"""Generate Sugeno and Mamdani baseline trajectories for the four viewer episodes.

Identical arena, noise_sigma, seed, dt and t_max as the recorded hybrid run — the
controller is the only manipulated variable, which is the whole point of the
comparison. Settings recovered by reproducing each hybrid run exactly (recover.py).

Emits baselines.json: per episode, per controller, the path polyline plus the
episode-level metrics.
"""
import json, re, io
import numpy as np
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import pandas as pd
import robot_sim as R
import fis_controllers as F
from fim_engine import build_implication

V2 = "../viewer/robot_viewer_v2.html"
src = io.open(V2, encoding="utf-8").read()
EPS = json.loads(re.search(
    r'<script type="application/json" id="epdata">(.*?)</script>', src, re.S).group(1))

# recovered in recover.py — verified by exact metric reproduction on all 4 episodes
RUN = {"cluttered": 1, "corridor": 0, "u_trap": 0, "moving": 1}
SIGMA, DT, TMAX = 0.05, 0.05, 90.0

# the recordings are hybrid v3 ("hybridB"): champion implication #8 + T_Luk
CH = pd.read_csv("implication_champions.csv").iloc[8]
I8 = build_implication(CH["structure"], CH["S"], CH["T"], CH["N"],
                       lam=CH["lambda"], w=CH["w"])
T_LUK = lambda a, b: np.maximum(a + b - 1.0, 0.0)
V3 = dict(beta=2.803, kappa=5.665, brake_gain=0.153)

def arena_of(e):
    a = e["arena"]
    return R.Arena(name=a["name"], segments=[tuple(s) for s in a["segments"]],
                   circles=a["circles"], start=tuple(a["start"]),
                   goal=tuple(a["goal"]), size=tuple(a["size"]))

CTRLS = [("sugeno",  F.sugeno_controller),
         ("mamdani", F.mamdani_controller)]

out = {}
for e in EPS:
    name = e["name"]
    seed = RUN[name]
    ar = arena_of(e)

    # sanity: reproduce the recorded v3 run itself, so a mismatch is caught loudly
    hyb = R.simulate(ar, F.hybrid_controller_v3(I8, T_LUK, **V3), noise_sigma=SIGMA,
                     seed=seed, dt=DT, t_max=TMAX, record=True)
    for k in ("time", "path_len", "min_clearance", "smoothness_tv"):
        assert abs(hyb[k] - e["metrics"][k]) < 5e-4, (name, k, hyb[k], e["metrics"][k])
    assert len(hyb["trace"]) == len(e["trace"]), (name, len(hyb["trace"]), len(e["trace"]))

    out[name] = {}
    for cname, factory in CTRLS:
        r = R.simulate(ar, factory(), noise_sigma=SIGMA, seed=seed, dt=DT,
                       t_max=TMAX, record=True)
        # x, y AND heading — the ghost-race rendering orients a translucent robot
        # along the path, so theta is needed; rays/obs stay out (3x the size, unused)
        path = [[round(f["x"], 3), round(f["y"], 3), round(f["th"], 4)]
                for f in r["trace"]]
        out[name][cname] = {
            "path": path,
            "success": bool(r["success"]),
            "collided": bool(r["collided"]),
            "metrics": {"time": round(r["time"], 3),
                        "path_len": round(r["path_len"], 3),
                        "path_ratio": round(r["path_ratio"], 3),
                        "min_clearance": round(r["min_clearance"], 3),
                        "smoothness_tv": round(r["smoothness_tv"], 3)},
        }
        m = r["metrics"] if "metrics" in r else r
        status = "collided" if r["collided"] else ("reached" if r["success"] else "timeout")
        print(f"{name:10s} {cname:8s} {status:9s} t={r['time']:6.2f}s  "
              f"len={r['path_len']:6.2f}m  clr={r['min_clearance']:6.3f}m  "
              f"tv={r['smoothness_tv']:8.2f}  frames={len(path)}")
    h = e["metrics"]
    hstat = "collided" if e.get("collided") else ("reached" if e["success"] else "timeout")
    print(f"{name:10s} {'hybrid':8s} {hstat:9s} t={h['time']:6.2f}s  "
          f"len={h['path_len']:6.2f}m  clr={h['min_clearance']:6.3f}m  "
          f"tv={h['smoothness_tv']:8.2f}  frames={len(e['trace'])}   <- recorded")
    print()

json.dump(out, open("baselines.json", "w"), separators=(",", ":"))
raw = len(json.dumps(out, separators=(",", ":")))
print(f"baselines.json written: {raw/1024:.1f} KB")
