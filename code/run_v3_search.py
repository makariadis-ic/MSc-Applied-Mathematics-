"""run_v3_search.py — broad random search over the v3 design space.

Search dims: implication (all 13 strata) x beta x kappa x brake_gain x T-norm.
Objective: mean(success) - 0.5*mean(collided) over 16 tuning seeds x
{cluttered, moving} x sigma {0.15, 0.3}. 64 sampled configs + sugeno reference.
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
T_NORMS = {"min": lambda a,b: np.minimum(a,b),
           "prod": lambda a,b: a*b,
           "luk": lambda a,b: np.maximum(a+b-1.0, 0.0)}
rng = np.random.default_rng(7)

CONFIGS = []
for ci in range(64):
    CONFIGS.append({
        "cfg": ci,
        "imp": int(rng.integers(0, len(CHAMPS))),
        "tnorm": ["min","prod","luk"][int(rng.integers(0,3))],
        "beta": float(rng.uniform(0.5, 4.0)),
        "kappa": float(rng.uniform(2.0, 8.0)),
        "brake": float(rng.uniform(0.0, 0.8)),
    })

SEEDS = range(16)
CELLS = [(ar, nz) for ar in ("cluttered","moving") for nz in (0.15, 0.3)]


def one_run(job):
    kind, cfg, ar, nz, sd = job
    if kind == "sugeno":
        ctl = sugeno_controller()
        meta = {"cfg": -1}
    else:
        r = CHAMPS.iloc[cfg["imp"]]
        I = build_implication(r["structure"], r["S"], r["T"], r["N"],
                              lam=r["lambda"], w=r["w"])
        ctl = hybrid_controller_v3(I, T_NORMS[cfg["tnorm"]], beta=cfg["beta"],
                                   kappa=cfg["kappa"], brake_gain=cfg["brake"])
        meta = dict(cfg)
    arena = make_arena(ar, np.random.default_rng(100 + sd))
    m = simulate(arena, ctl, noise_sigma=nz, seed=sd, t_max=150)
    m.pop("trace", None)
    return {**meta, "kind": kind, "arena": ar, "noise": nz, "seed": sd, **m}


def main():
    jobs = []
    for ar, nz in CELLS:
        for sd in SEEDS:
            jobs.append(("sugeno", None, ar, nz, sd))
            for cfg in CONFIGS:
                jobs.append(("hybrid", cfg, ar, nz, sd))
    print(f"{len(jobs)} runs", flush=True)
    t0 = time.time()
    with Pool(5) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(one_run, jobs, chunksize=12)):
            rows.append(r)
            if (i+1) % 500 == 0:
                print(f"{i+1}/{len(jobs)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv("v3_search_results.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
