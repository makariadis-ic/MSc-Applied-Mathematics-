"""run_ablation.py — attribution ablation for the v3 hybrid (config B).

Variants (all share beta=2.803, kappa=5.665, brake=0.153, T=luk, imp stratum 8):
  full       : implication activation + implication brake      (the published config)
  act_only   : implication activation, no brake
  brake_only : plain Sugeno activation, implication brake
  gain_ctrl  : confidence as plain multiplicative gain (mu*c) + linear brake
               -> same information, no implication semantics anywhere
  classic_imp: full, but Lukasiewicz implication instead of generated #8
  sugeno     : baseline
Cells: cluttered + moving x sigma {0.15, 0.3}, fresh seeds 5000-5099.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from multiprocessing import Pool

from fim_engine import build_implication
from robot_sim import make_arena, simulate
from fis_controllers import sugeno_controller, hybrid_controller_v3

CHAMPS = pd.read_csv("implication_champions.csv")
T_LUK = lambda a, b: np.maximum(a + b - 1.0, 0.0)
R8 = CHAMPS.iloc[8]
LK = ("S(N(x),y)", "min(x+y,1)", "max(x+y-1,0)", "1-x")
P = dict(beta=2.803, kappa=5.665, brake_gain=0.153)


def make_ctl(kind):
    if kind == "sugeno":
        return sugeno_controller()
    if kind == "classic_imp":
        I = build_implication(*LK)
        return hybrid_controller_v3(I, T_LUK, **P)
    I = build_implication(R8["structure"], R8["S"], R8["T"], R8["N"],
                          lam=R8["lambda"], w=R8["w"])
    if kind == "full":
        return hybrid_controller_v3(I, T_LUK, **P)
    if kind == "act_only":
        return hybrid_controller_v3(I, T_LUK, beta=P["beta"], kappa=P["kappa"],
                                    brake_gain=0.0)
    if kind == "brake_only":
        return hybrid_controller_v3(I, T_LUK, **P, act_mode="plain")
    if kind == "gain_ctrl":
        return hybrid_controller_v3(I, T_LUK, **P, act_mode="gain", brake_mode="linear")
    raise ValueError(kind)


def one_run(job):
    kind, ar, nz, sd = job
    arena = make_arena(ar, np.random.default_rng(100 + sd))
    m = simulate(arena, make_ctl(kind), noise_sigma=nz, seed=sd, t_max=150)
    m.pop("trace", None)
    return {"kind": kind, "arena": ar, "noise": nz, "seed": sd, **m}


def main():
    kinds = ("sugeno", "full", "act_only", "brake_only", "gain_ctrl", "classic_imp")
    jobs = [(k, ar, nz, sd) for k in kinds
            for ar in ("cluttered", "moving") for nz in (0.15, 0.3)
            for sd in range(5000, 5100)]
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    with Pool(5) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs, chunksize=12)):
            rows.append(r)
            if (i+1) % 500 == 0:
                print(f"{i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("ablation_results.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
