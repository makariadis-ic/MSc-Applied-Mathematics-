"""run_final_validation.py — hybridB full-benchmark validation, fresh seeds 4000-4099.
All 4 arenas x sigma {0, 0.1, 0.2, 0.3} x sugeno/mamdani/hybridB."""
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
    "hybridB": dict(beta=2.803, kappa=5.665, brake_gain=0.153),
}
SEEDS = range(4000, 4100)


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
            for k in ("sugeno", "mamdani", "hybridB")
            for ar in ("cluttered", "corridor", "u_trap", "moving")
            for nz in (0.0, 0.1, 0.2, 0.3)
            for sd in SEEDS]
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    with Pool(5) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs, chunksize=12)):
            rows.append(r)
            if (i+1) % 500 == 0:
                print(f"{i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("final_validation_results.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
