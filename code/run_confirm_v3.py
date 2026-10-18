"""run_confirm_v3.py — v3 confirmatory on fresh seeds 3000-3199.

Pre-registered from the 64-config random search (tuning seeds 0-15):
  cfg A (search #38): imp stratum 8 (N(T(x,N(y))), S_LK/T_M/N_std, sig 1001111),
                      T=luk, beta=3.010, kappa=5.778, brake=0.777
  cfg B (search #50): same implication, T=luk, beta=2.803, kappa=5.665, brake=0.153
Cells: cluttered+moving x sigma {0.15, 0.3}. Baselines: sugeno, mamdani.
One-sided McNemar vs sugeno.
"""
import sys, os, time
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from multiprocessing import Pool

from fim_engine import build_implication
from robot_sim import make_arena, simulate
from fis_controllers import sugeno_controller, mamdani_controller, hybrid_controller_v3

CHAMPS = pd.read_csv("implication_champions.csv")
T_LUK = lambda a, b: np.maximum(a + b - 1.0, 0.0)
R8 = CHAMPS.iloc[8]

CONFIGS = {
    "hybridA": dict(beta=3.010, kappa=5.778, brake_gain=0.777),
    "hybridB": dict(beta=2.803, kappa=5.665, brake_gain=0.153),
}
SEEDS = range(3000, 3200)


def make_ctl(kind):
    if kind == "sugeno":
        return sugeno_controller()
    if kind == "mamdani":
        return mamdani_controller()
    I = build_implication(R8["structure"], R8["S"], R8["T"], R8["N"],
                          lam=R8["lambda"], w=R8["w"])
    return hybrid_controller_v3(I, T_LUK, **CONFIGS[kind])


def one_run(job):
    kind, ar, nz, sd = job
    arena = make_arena(ar, np.random.default_rng(100 + sd))
    m = simulate(arena, make_ctl(kind), noise_sigma=nz, seed=sd, t_max=150)
    m.pop("trace", None)
    return {"kind": kind, "arena": ar, "noise": nz, "seed": sd, **m}


def main():
    jobs = [(k, ar, nz, sd)
            for k in ("sugeno", "mamdani", "hybridA", "hybridB")
            for ar in ("cluttered", "moving") for nz in (0.15, 0.3)
            for sd in SEEDS]
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    with Pool(5) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs, chunksize=12)):
            rows.append(r)
            if (i+1) % 500 == 0:
                print(f"{i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("confirmatory_v3_results.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
