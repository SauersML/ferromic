#!/usr/bin/env python
"""Extreme-flux extension: push m up to 1e-4 to locate the breakdown point.
Uses the most flux-sensitive, best-baseline cells:
  recurrent, rho=1e-8, depths young & recent (good baseline power).
  single,    rho=1e-8, depths young & recent (low baseline FPR).
"""
import time, json
from multiprocessing import Pool
import numpy as np
import flux_sim as fs

TIME_DEPTHS = {
    "young":  dict(t01_23=250000, t0_1=100000, t2_3=50000,  t_inv=100000),
    "recent": dict(t01_23=100000, t0_1=50000,  t2_3=25000,  t_inv=50000),
}
FLUX = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]


def _one(arg):
    sc, depth, rho, m, mw, seed = arg
    G, lab, _ = fs.simulate(sc, 0.1, 240, rho, mw, m, TIME_DEPTHS[depth], seed)
    return fs.classify(G, lab)


if __name__ == "__main__":
    reps = 60
    pool = Pool(7)
    results = []
    seed = 500000
    t0 = time.time()
    for sc in ("single", "recurrent"):
        for depth in ("young", "recent"):
            for m in FLUX:
                jobs = [(sc, depth, 1e-8, m, 1e-8, seed + r) for r in range(reps)]
                ev = np.array(pool.map(_one, jobs))
                seed += reps
                res = dict(scenario=sc, depth=depth, rho=1e-8, m_flux=m, reps=reps,
                           mean_events=float(ev.mean()),
                           recurrent_call_rate=float((ev >= 2).mean()))
                results.append(res)
                metric = "FPR" if sc == "single" else "power"
                print(f"{sc:9s} {depth:6s} m={m:.0e} {metric}={res['recurrent_call_rate']:.2f} "
                      f"mean_ev={res['mean_events']:.2f} ({time.time()-t0:.0f}s)", flush=True)
                with open("sweep_extreme.json", "w") as fh:
                    json.dump(results, fh, indent=2)
    pool.close(); pool.join()
    print("DONE", flush=True)
