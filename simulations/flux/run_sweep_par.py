#!/usr/bin/env python
"""Parallel flux sweep: reps distributed across a process pool."""
import sys, time, json, argparse
from multiprocessing import Pool
import numpy as np
import flux_sim as fs

TIME_DEPTHS = {
    "young":  dict(t01_23=250000, t0_1=100000, t2_3=50000,  t_inv=100000),
    "recent": dict(t01_23=100000, t0_1=50000,  t2_3=25000,  t_inv=50000),
    "old":    dict(t01_23=500000, t0_1=250000, t2_3=100000, t_inv=250000),
}
FLUX = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]


def _one(arg):
    scenario, depth, rho, m_flux, m_within, seed = arg
    times = TIME_DEPTHS[depth]
    G, lab = fs.simulate(scenario, 0.1, 240, rho, m_within, m_flux, times, seed)
    return fs.classify(G, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--depths", default="recent,young,old")
    ap.add_argument("--rhos", default="0,1e-8,1e-6")
    ap.add_argument("--scenarios", default="single,recurrent")
    ap.add_argument("--m_within", type=float, default=1e-8)
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--out", default="sweep_results.json")
    args = ap.parse_args()

    depths = args.depths.split(",")
    rhos = [float(x) for x in args.rhos.split(",")]
    scenarios = args.scenarios.split(",")

    cells = []
    seed = 10_000
    for sc in scenarios:
        for depth in depths:
            for rho in rhos:
                for m in FLUX:
                    cells.append((sc, depth, rho, m, seed))
                    seed += args.reps

    results = []
    t0 = time.time()
    pool = Pool(args.procs)
    for ci, (sc, depth, rho, m, base_seed) in enumerate(cells):
        jobs = [(sc, depth, rho, m, args.m_within, base_seed + r)
                for r in range(args.reps)]
        events = np.array(pool.map(_one, jobs))
        call = (events >= 2)
        res = dict(scenario=sc, depth=depth, rho=rho, m_flux=m, reps=args.reps,
                   mean_events=float(events.mean()),
                   median_events=float(np.median(events)),
                   recurrent_call_rate=float(call.mean()),
                   events_hist={int(k): int(v) for k, v in
                                zip(*np.unique(events, return_counts=True))})
        results.append(res)
        el = time.time() - t0
        metric = "FPR" if sc == "single" else "power"
        print(f"[{ci+1}/{len(cells)}] {sc:9s} {depth:6s} rho={rho:.0e} "
              f"m={m:.0e}  {metric}={res['recurrent_call_rate']:.2f} "
              f"mean_ev={res['mean_events']:.2f}  ({el:.0f}s)", flush=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    pool.close(); pool.join()
    print(f"DONE {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
