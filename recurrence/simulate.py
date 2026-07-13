"""Stage 1 (simulate): generate the labelled structured-coalescent training set.

Across the manuscript's Fig-1 axes (time depth, recombination rate, between-orientation
gene-conversion flux, inverted-allele frequency), at the real-data haplotype sample
size, generate single-origin (label 0) and 3-event recurrent (label 1) inversions and
compute the classifier feature vector for each.

The coalescent model is the repo's ``simulations/flux/flux_sim.py`` (msprime; reused
verbatim, not re-vendored). Deterministic: every grid cell gets a fixed base seed
derived from its grid index. Writes the gzip'd training set consumed by ``fit``.

Requires msprime (install the ``recurrence`` extra). Every downstream stage and every
test runs from the committed ``data/sim_features.csv.gz`` without this stage.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os

import numpy as np
import pandas as pd

from . import paths
from .features import FEATURE_NAMES, extract_features, popgen_group_stats
from .transferable import TRANSFERABLE_FEATURES, from_group_stats

TIME_DEPTHS = {
    "young":  dict(t01_23=250000, t0_1=100000, t2_3=50000,  t_inv=100000),
    "recent": dict(t01_23=100000, t0_1=50000,  t2_3=25000,  t_inv=50000),
    "old":    dict(t01_23=500000, t0_1=250000, t2_3=100000, t_inv=250000),
}
RHOS = [0.0, 1e-8, 1e-6]
FLUX = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]
FREQS = [0.05, 0.10, 0.20, 0.35, 0.50]
SCENARIOS = ["single", "recurrent"]
M_WITHIN = 1e-8
SAMPLE_HAP = 88


def _load_flux_sim():
    """Load the repo's coalescent model (simulations/flux/flux_sim.py) by path;
    simulations/ is not an importable package."""
    fp = os.path.join(paths.REPO_ROOT, "simulations", "flux", "flux_sim.py")
    spec = importlib.util.spec_from_file_location("ferromic_flux_sim", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_grid(reps):
    """Deterministic ordered list of (scenario, depth, rho, m_flux, freq, seed)."""
    grid = []
    idx = 0
    for sc, depth, rho, m, freq in itertools.product(SCENARIOS, TIME_DEPTHS, RHOS, FLUX, FREQS):
        base = 1_000_000 + idx * 10_000
        for r in range(reps):
            grid.append((sc, depth, rho, m, freq, base + r))
        idx += 1
    return grid


def _run_one(args_and_seqlen):
    (sc, depth, rho, m, freq, seed), seq_len, _fs = args_and_seqlen
    times = TIME_DEPTHS[depth]
    G, lab, meta = _fs.simulate(sc, freq, SAMPLE_HAP, rho, M_WITHIN, m, times, seed=seed)
    feats = extract_features(G, lab)
    stats = popgen_group_stats(G, lab, seq_len)
    tfeats = from_group_stats(stats)
    fI = meta.get("fI")
    observable = not (sc == "recurrent" and fI is not None and (fI == 0.0 or fI == 1.0))
    row = dict(scenario=sc, label=1 if sc == "recurrent" else 0, depth=depth,
               rho=rho, m_flux=m, inv_freq=freq, seed=seed, n_sites=G.shape[1],
               fI=fI, observable_recurrence=bool(observable))
    row.update(feats)
    row.update({"bp_" + k: v for k, v in stats.items()})
    row.update({"tf_" + k: v for k, v in tfeats.items()})
    return row


def simulate(reps=25, procs=8, out=None):
    """Generate the training set. Returns the DataFrame and (if ``out``) writes it."""
    fs = _load_flux_sim()
    seq_len = fs.SEQ_LEN
    grid = build_grid(reps)
    payload = [(g, seq_len, fs) for g in grid]
    rows = []
    if procs > 1:
        from multiprocessing import Pool
        with Pool(procs) as pool:
            for i, row in enumerate(pool.imap_unordered(_run_one, payload, chunksize=16)):
                rows.append(row)
                if (i + 1) % 500 == 0 or (i + 1) == len(grid):
                    print(f"  {i + 1}/{len(grid)}", flush=True)
    else:
        for i, p in enumerate(payload):
            rows.append(_run_one(p))
    df = pd.DataFrame(rows)
    base = ["scenario", "label", "depth", "rho", "m_flux", "inv_freq", "seed",
            "n_sites", "fI", "observable_recurrence"]
    tf_cols = ["tf_" + k for k in TRANSFERABLE_FEATURES]
    bp_cols = [c for c in df.columns if c.startswith("bp_")]
    df = df[base + FEATURE_NAMES + tf_cols + bp_cols]
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows -> {out}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the recurrence training set (msprime).")
    ap.add_argument("--reps", type=int, default=25, help="replicates per grid cell")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=paths.SIM_FEATURES)
    args = ap.parse_args(argv)
    simulate(reps=args.reps, procs=args.procs, out=args.out)


if __name__ == "__main__":
    main()
