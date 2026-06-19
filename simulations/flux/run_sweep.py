#!/usr/bin/env python
"""
Flux sweep driver.

For each (scenario, time-depth, recomb, flux) cell, simulate `reps` loci,
run the parsimony classifier, and record:
  - single scenarios:   false-positive rate = P(inferred events >= 2)
  - recurrent scenarios: power = P(inferred events >= 2)
                         and the distribution of inferred #events.

Flux m is symmetric migration between opposite-orientation demes (per lineage
per generation): the gene-conversion / double-crossover analog.
Swept: m in {0, 1e-9, 1e-8, 1e-7, 1e-6}.

Time depths (years; from the manuscript manifests):
  young:  Tsp_p01_p23=250000  (recurrent: 250000/100000/50000 ; single t_inv=100000)
  recent: 100000              (recurrent: 100000/50000/25000  ; single t_inv=50000)
  old:    500000              (recurrent: 500000/250000/100000; single t_inv=250000)
Recomb rho in {0, 1e-8, 1e-6}.  inv_freq=0.1, sample_hap=240, m_within=1e-8.
"""
import sys, time, json, argparse
import numpy as np
import flux_sim as fs

TIME_DEPTHS = {
    "young":  dict(t01_23=250000, t0_1=100000, t2_3=50000,  t_inv=100000),
    "recent": dict(t01_23=100000, t0_1=50000,  t2_3=25000,  t_inv=50000),
    "old":    dict(t01_23=500000, t0_1=250000, t2_3=100000, t_inv=250000),
}
FLUX = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]


def run_cell(scenario, depth, rho, m_flux, reps, m_within, base_seed):
    times = TIME_DEPTHS[depth]
    events = np.empty(reps, dtype=int)
    for r in range(reps):
        G, lab = fs.simulate(scenario, 0.1, 240, rho, m_within, m_flux,
                             times, seed=base_seed + r)
        events[r] = fs.classify(G, lab)
    call_recurrent = (events >= 2)
    return dict(
        scenario=scenario, depth=depth, rho=rho, m_flux=m_flux, reps=reps,
        mean_events=float(events.mean()),
        median_events=float(np.median(events)),
        recurrent_call_rate=float(call_recurrent.mean()),
        events_hist={int(k): int(v) for k, v in
                     zip(*np.unique(events, return_counts=True))},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--depths", default="recent,young,old")
    ap.add_argument("--rhos", default="0,1e-8,1e-6")
    ap.add_argument("--scenarios", default="single,recurrent")
    ap.add_argument("--m_within", type=float, default=1e-8)
    ap.add_argument("--out", default="sweep_results.json")
    args = ap.parse_args()

    depths = args.depths.split(",")
    rhos = [float(x) for x in args.rhos.split(",")]
    scenarios = args.scenarios.split(",")

    results = []
    t0 = time.time()
    seed = 10_000
    total = len(scenarios) * len(depths) * len(rhos) * len(FLUX)
    done = 0
    for sc in scenarios:
        for depth in depths:
            for rho in rhos:
                for m in FLUX:
                    res = run_cell(sc, depth, rho, m, args.reps,
                                   args.m_within, seed)
                    seed += args.reps
                    results.append(res)
                    done += 1
                    el = time.time() - t0
                    metric = ("FPR" if sc == "single" else "power")
                    print(f"[{done}/{total}] {sc:9s} {depth:6s} rho={rho:.0e} "
                          f"m={m:.0e}  {metric}={res['recurrent_call_rate']:.2f} "
                          f"mean_ev={res['mean_events']:.2f}  "
                          f"({el:.0f}s elapsed)", flush=True)
                    with open(args.out, "w") as fh:
                        json.dump(results, fh, indent=2)
    print(f"\nDONE  total {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
