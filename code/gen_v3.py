"""Regenerate the four viewer episodes with hybrid v3 — the controller the campaign
figures actually describe (run_final_validation.py: kind 'hybridB').

Identical arena, noise_sigma, seed, dt to the existing recordings, so the ONLY change
is the inference engine. That also keeps the Sugeno/Mamdani comparison baselines valid
without regeneration, since they were produced under exactly these settings.
"""
import json, re, io, sys
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

# --- exactly as run_final_validation.py builds 'hybridB' ----------------------
CHAMPS = pd.read_csv("implication_champions.csv")
R8 = CHAMPS.iloc[8]
T_LUK = lambda a, b: np.maximum(a + b - 1.0, 0.0)
CFG = dict(beta=2.803, kappa=5.665, brake_gain=0.153)
I8 = build_implication(R8["structure"], R8["S"], R8["T"], R8["N"],
                       lam=R8["lambda"], w=R8["w"])

# sanity: the champion implication must be the I8 the viewer already displays
for x in (0.0, 0.25, 0.5, 0.75, 1.0):
    for y in (0.0, 0.25, 0.5, 0.75, 1.0):
        want = 1.0 - min(x, 1.0 - y)
        got = float(I8(np.array(x), np.array(y)))
        assert abs(got - want) < 1e-12, (x, y, got, want)
print("champion #8 == 1 - min(x, 1-y)  [matches the viewer's I8 panel]\n")

RUN = {"cluttered": 1, "corridor": 0, "u_trap": 0, "moving": 1}
SIGMA, DT = 0.05, 0.05
TMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0

def arena_of(e):
    a = e["arena"]
    return R.Arena(name=a["name"], segments=[tuple(s) for s in a["segments"]],
                   circles=a["circles"], start=tuple(a["start"]),
                   goal=tuple(a["goal"]), size=tuple(a["size"]))

out = []
print(f"t_max={TMAX}  sigma={SIGMA}  dt={DT}   (arena/seed unchanged)\n")
print(f"{'episode':10s} {'controller':12s} {'outcome':9s} {'time':>7s} {'len':>7s} "
      f"{'clear':>7s} {'tv':>8s}  frames")
for e in EPS:
    name = e["name"]
    ar = arena_of(e)
    ctl = F.hybrid_controller_v3(I8, T_LUK, **CFG)
    r = R.simulate(ar, ctl, noise_sigma=SIGMA, seed=RUN[name], dt=DT,
                   t_max=TMAX, record=True)
    status = "collided" if r["collided"] else ("reached" if r["success"] else "TIMEOUT")
    old = e["metrics"]
    ostat = "collided" if e.get("collided") else ("reached" if e["success"] else "timeout")
    print(f"{name:10s} {'hybrid base':12s} {ostat:9s} {old['time']:7.2f} {old['path_len']:7.2f} "
          f"{old['min_clearance']:7.3f} {old['smoothness_tv']:8.2f}  {len(e['trace'])}")
    print(f"{name:10s} {'hybrid v3':12s} {status:9s} {r['time']:7.2f} {r['path_len']:7.2f} "
          f"{r['min_clearance']:7.3f} {r['smoothness_tv']:8.2f}  {len(r['trace'])}")
    print()
    out.append({
        "name": name,
        "success": bool(r["success"]),
        "collided": bool(r["collided"]),
        "metrics": {"time": round(r["time"], 3), "path_len": round(r["path_len"], 3),
                    "path_ratio": round(r["path_ratio"], 3),
                    "min_clearance": round(r["min_clearance"], 3),
                    "smoothness_tv": round(r["smoothness_tv"], 3)},
        "arena": e["arena"],          # unchanged
        "trace": r["trace"],
    })

json.dump(out, open("epdata_v3.json", "w"), separators=(",", ":"))
print(f"epdata_v3.json written: {len(json.dumps(out, separators=(',', ':')))/1024:.1f} KB")
