"""Stage 1 (simulate): generate the labelled structured-coalescent training set.

Across the manuscript's Fig-1 axes (time depth, recombination rate,
between-orientation gene-conversion flux, inverted-allele frequency), at the
real-data haplotype sample size, generate single-origin (label 0) and recurrent
(label 1) inversions and compute the classifier feature vector for each.

Both the coalescent model and the origin count come from the reference pipeline
in ``simulations/refsim`` -- a faithful port of ``hsiehphLab/inversionSimulation``.
``tree_n_events`` is therefore ``minMutHomoplasy``: the Fitch parsimony minimum
number of orientation state changes on an IQ-TREE maximum-likelihood haplotype
tree with the ancestral outgroup collapsed, exactly as upstream scores it.

Requires msprime, Biopython and an IQ-TREE binary. The grid is ~11k loci and a
few seconds of IQ-TREE each, so it is meant to be run sharded on a cluster:

    # on the cluster, one array task per shard
    python simulations/refsim/run_grid.py --task trainset \\
        --shard $SLURM_ARRAY_TASK_ID --nshards 8 --procs 64 \\
        --out out/trainset_shard$SLURM_ARRAY_TASK_ID.csv

    # then, locally, merge the shards into the committed training set
    python -m recurrence.cli simulate --merge 'out/trainset_shard*.csv'

``--merge`` is the normal path. Running this module without ``--merge`` executes
the same grid in-process, which is only practical for small ``--reps``.

Every downstream stage and every test runs from the committed
``data/sim_features.csv.gz`` without this stage.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os

import pandas as pd

from . import paths
from .features import FEATURE_NAMES
from .transferable import TRANSFERABLE_FEATURES

BASE_COLUMNS = ["scenario", "label", "depth", "rho", "m_flux", "inv_freq",
                "seed", "n_sites", "frac_admix_i", "frac_admix_d",
                "call_recurrent"]


def _load_run_grid():
    """Load ``simulations/refsim/run_grid.py`` by path (simulations/ is not a package)."""
    fp = os.path.join(paths.REPO_ROOT, "simulations", "refsim", "run_grid.py")
    spec = importlib.util.spec_from_file_location("ferromic_refsim_grid", fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _order_columns(df):
    tf_cols = ["tf_" + k for k in TRANSFERABLE_FEATURES]
    bp_cols = sorted(c for c in df.columns if c.startswith("bp_"))
    lead = [c for c in BASE_COLUMNS if c in df.columns]
    feat = [c for c in FEATURE_NAMES if c in df.columns]
    tf_cols = [c for c in tf_cols if c in df.columns]
    keep = lead + feat + tf_cols + bp_cols
    return df[keep]


def merge(patterns, out=None):
    """Merge sharded ``run_grid.py --task trainset`` CSVs into the training set."""
    files = sorted(f for pat in patterns for f in glob.glob(pat))
    if not files:
        raise SystemExit(f"no shard CSVs matched {patterns!r}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n_all = len(df)
    if "error" in df.columns:
        failed = df["error"].astype(str).str.strip().replace("nan", "")
        df = df[failed == ""].drop(columns=["error"])
    if len(df) != n_all:
        print(f"dropped {n_all - len(df)} failed replicates")
    df = df.sort_values("seed", kind="mergesort").reset_index(drop=True)
    df = _order_columns(df)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {len(df)} rows x {df.shape[1]} cols -> {out}")
    return df


def simulate(reps=25, procs=8, out=None):
    """Run the reference trainset grid in-process (no sharding). Slow; see ``merge``."""
    grid_mod = _load_run_grid()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        shard_csv = os.path.join(tmp, "trainset_shard0.csv")
        grid_mod.main(["--task", "trainset", "--shard", "0", "--nshards", "1",
                       "--procs", str(procs), "--reps", str(reps),
                       "--out", shard_csv])
        return merge([shard_csv], out=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the recurrence training set.")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="glob(s) of sharded run_grid.py --task trainset CSVs")
    ap.add_argument("--reps", type=int, default=25, help="replicates per grid cell")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=paths.SIM_FEATURES)
    args = ap.parse_args(argv)

    if args.merge:
        merge(args.merge, out=args.out)
    else:
        simulate(reps=args.reps, procs=args.procs, out=args.out)


if __name__ == "__main__":
    main()
