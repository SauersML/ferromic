#!/usr/bin/env python
"""Shardable driver for the reference recurrence pipeline (``refsim.py``).

Two grids, both scored by the upstream classifier (IQ-TREE ML tree + Fitch
parsimony on the orientation trait, ``minMutHomoplasy >= 2`` = recurrent):

``flux``
    The between-orientation gene-flux sweep. Axes: scenario (single /
    recurrent) x time depth x recombination rate x flux rate, at the upstream
    manifest's ``sampleHaploSize = 240``, ``inv_freq = 0.1``, ``mig_const = 1e-8``.
    ``m_flux = 0`` is upstream's own model, so the first flux column is the
    reference false-positive rate / power.

``trainset``
    The labelled training set for ``recurrence/``: the same axes plus inverted
    allele frequency, at the real-data haplotype count (88), emitting the full
    per-locus feature vector with the reference origin count as ``tree_n_events``.

Sharding: ``--shard k --nshards N`` takes grid rows with ``index % N == k``, so a
SLURM array covers the grid exactly once. Within a shard ``--procs`` replicates
run concurrently, each IQ-TREE single-threaded.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refsim  # noqa: E402

TIME_DEPTHS = refsim.TIME_DEPTHS
RHOS = [0.0, 1e-8, 1e-6]
FLUX = [0.0, 1e-9, 1e-8, 1e-7, 1e-6]
# Upstream has one demography; a locus is single-event when the inverted
# admixture draw lands on 0 or 1. "single_upstream" is that locus.
# single_repo constrains the direct draw to the non-sister clade, which is
# what Fig. 1A depicts; single_upstream leaves it free, as the script does.
SCENARIOS = ["single_upstream", "single_repo", "recurrent"]
M_WITHIN = 1e-8

FLUX_SAMPLE_HAP = 240
FLUX_INV_FREQ = 0.1
FLUX_REPS = 60
FLUX_SEED0 = 10_000

TRAIN_SAMPLE_HAP = 88
TRAIN_FREQS = [0.05, 0.10, 0.20, 0.35, 0.50]
TRAIN_REPS = 25
TRAIN_SEED0 = 1_000_000

# Extreme-flux extension: push m two orders of magnitude past the sweep's top
# value to locate where the classifier actually breaks down. Restricted to the
# rho = 1e-8 cells at the two shallower depths, which have the best baseline
# power (recurrent) and the lowest baseline FPR (single).
EXTREME_FLUX = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
EXTREME_DEPTHS = ["young", "recent"]
EXTREME_RHO = 1e-8
EXTREME_REPS = 60
EXTREME_SEED0 = 500_000

# Gene-flux sweep for Reviewer 1. This is the ``replicate`` grid exactly -- same
# two arms, same four depths, same three recombination rates, same six
# frequencies -- with one axis added: a symmetric between-orientation migration
# rate laid over both models. m_flux = 0 reproduces the replication itself, so
# every flux level is read against a baseline generated under identical
# conditions rather than against a previous run.
#
# The ladder is in multiples of the model's own within-orientation gene flow
# (``M_WITHIN`` = 1e-8): none, 1x, 10x, 100x.
FLUXSWEEP_FLUX = [0.0, 1e-8, 1e-7, 1e-6]
FLUXSWEEP_SEED0 = 3_000_000
# Generous; the run is bounded by --max-seconds, not by this. Replicate is the
# outer loop, so whatever the wall clock allows is a balanced grid.
FLUXSWEEP_REPS = 20

# Second flux sweep, on ONE demography for both arms.
#
# The first sweep pairs "single" (the two-deme model read literally off the
# Methods paragraph) with "upstream" (the nine-deme manifest run with both
# admixture draws free). Those are different demographies, and "upstream" is a
# mixture that also contains genuinely single-origin replicates, so its call
# rate is a detection rate rather than power against a recurrent truth. Reading
# a false-positive rate off one arm and a power off the other means the two
# halves of the figure do not describe the same model, which is exactly the
# ambiguity flagged in review.
#
# "fluxsweep2" fixes both arms to the nine-deme upstream demography and lets the
# inverted admixture draw decide the truth: single_upstream conditions it on a
# single inverted origin, recurrent conditions it on both. The false-positive
# rate and the power are then measured on the same model, and one demography
# figure describes both panels.
FLUXSWEEP2_SCENARIOS = ["single_upstream", "recurrent"]
FLUXSWEEP2_REPS = 20
FLUXSWEEP2_SEED0 = 6_000_000

# Third flux sweep, on the pair the MANUSCRIPT depicts.
#
# fluxsweep2 uses single_upstream, which is what the upstream *script* does: the
# inverted draw is conditioned on one origin and the direct draw is left free.
# Manuscript Fig. 1A draws something more specific -- the direct sample taken
# from the non-sister deme, i.e. fD = 1 - fI -- which is `single_repo`. Keeping
# direct lineages out of the inverted clade's ancestral group lowers the
# false-positive rate, and the manuscript reports a rate below 5%, which neither
# single_upstream (0.26) nor the two-deme model (0.007, and exactly zero at every
# no-flux cell) reproduces. This grid runs the depicted pair so the sweep's
# false-positive arm is the model the published figure describes.
FLUXSWEEP3_SCENARIOS = ["single_repo", "recurrent"]
FLUXSWEEP3_REPS = 20
FLUXSWEEP3_SEED0 = 9_000_000

# Frequency-trajectory comparison for Reviewer 1. Same axes as the replication,
# minus the flux ladder, run twice: once on the published constant-size
# single-event model and once on the model in which the inversion actually
# rises in frequency from one haplotype to its observed value. The two arms occupy
# different cells of the same grid, so they get different seeds and the
# comparison is between independent samples, not paired.
GROWTH_SCENARIOS = ["single", "single_growth"]
GROWTH_REPS = 50
GROWTH_SEED0 = 4_000_000

# The same comparison on the recurrent side, where the quantity at stake is
# power rather than the false-positive rate.
RGROWTH_SCENARIOS = ["recurrent", "recurrent_growth"]
RGROWTH_REPS = 50
RGROWTH_SEED0 = 5_000_000

# Replication of the manuscript's Fig. 1 power analysis, at its own parameters.
# The manifest frequencies are Fig. 1G's x axis; the Methods give 100 replicates
# per model, mig_const 1e-8 and sampleHaploSize 240.
#
# Two arms, matching what the manuscript describes rather than what one script
# happens to emit. "single" is the two-population model of Fig. 1A at the FIRST
# event of each triple -- 500/250/100/50 kya, exactly the four depths the Methods
# list. "upstream" is the manifest run as-is, both admixture draws free, which is
# what produces the recurrent power curve.
REPL_FREQS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
REPL_SCENARIOS = ["single", "upstream"]
REPL_REPS = 100
REPL_SEED0 = 2_000_000


def build_grid(task, reps=None):
    """Deterministic ordered grid of job dicts (index order is the shard key)."""
    rows = []
    if task == "flux":
        reps = reps or FLUX_REPS
        for cell, (sc, depth, rho, m) in enumerate(
                itertools.product(SCENARIOS, TIME_DEPTHS, RHOS, FLUX)):
            for r in range(reps):
                rows.append(dict(scenario=sc, depth=depth, rho=rho, m_flux=m,
                                 inv_freq=FLUX_INV_FREQ,
                                 sample_size=FLUX_SAMPLE_HAP,
                                 seed=FLUX_SEED0 + cell * reps + r))
    elif task == "trainset":
        reps = reps or TRAIN_REPS
        for cell, (sc, depth, rho, m, freq) in enumerate(
                itertools.product(SCENARIOS, TIME_DEPTHS, RHOS, FLUX, TRAIN_FREQS)):
            base = TRAIN_SEED0 + cell * 10_000
            for r in range(reps):
                rows.append(dict(scenario=sc, depth=depth, rho=rho, m_flux=m,
                                 inv_freq=freq, sample_size=TRAIN_SAMPLE_HAP,
                                 seed=base + r))
    elif task == "replicate":
        reps = reps or REPL_REPS
        for cell, (sc, depth, rho, freq) in enumerate(
                itertools.product(REPL_SCENARIOS, TIME_DEPTHS, RHOS, REPL_FREQS)):
            for r in range(reps):
                rows.append(dict(scenario=sc, depth=depth, rho=rho, m_flux=0.0,
                                 inv_freq=freq, sample_size=FLUX_SAMPLE_HAP,
                                 seed=REPL_SEED0 + cell * reps + r))
    elif task == "fluxsweep":
        reps = reps or FLUXSWEEP_REPS
        # Replicate is the OUTER loop so the grid is swept a full replicate at a
        # time. A run cut short by the wall clock then loses whole replicates
        # spread evenly over every cell, rather than whole cells off the end.
        cells = list(itertools.product(REPL_SCENARIOS, TIME_DEPTHS, RHOS,
                                       FLUXSWEEP_FLUX, REPL_FREQS))
        for r in range(reps):
            for cell, (sc, depth, rho, m, freq) in enumerate(cells):
                rows.append(dict(scenario=sc, depth=depth, rho=rho,
                                 m_flux=m, inv_freq=freq,
                                 sample_size=FLUX_SAMPLE_HAP,
                                 seed=FLUXSWEEP_SEED0 + cell * reps + r))
    elif task == "fluxsweep2":
        reps = reps or FLUXSWEEP2_REPS
        cells = list(itertools.product(FLUXSWEEP2_SCENARIOS, TIME_DEPTHS, RHOS,
                                       FLUXSWEEP_FLUX, REPL_FREQS))
        for r in range(reps):
            for cell, (sc, depth, rho, m, freq) in enumerate(cells):
                rows.append(dict(scenario=sc, depth=depth, rho=rho,
                                 m_flux=m, inv_freq=freq,
                                 sample_size=FLUX_SAMPLE_HAP,
                                 seed=FLUXSWEEP2_SEED0 + cell * reps + r))
    elif task == "fluxsweep3":
        reps = reps or FLUXSWEEP3_REPS
        cells = list(itertools.product(FLUXSWEEP3_SCENARIOS, TIME_DEPTHS, RHOS,
                                       FLUXSWEEP_FLUX, REPL_FREQS))
        for r in range(reps):
            for cell, (sc, depth, rho, m, freq) in enumerate(cells):
                rows.append(dict(scenario=sc, depth=depth, rho=rho,
                                 m_flux=m, inv_freq=freq,
                                 sample_size=FLUX_SAMPLE_HAP,
                                 seed=FLUXSWEEP3_SEED0 + cell * reps + r))
    elif task == "growth":
        reps = reps or GROWTH_REPS
        cells = list(itertools.product(GROWTH_SCENARIOS, TIME_DEPTHS, RHOS,
                                       REPL_FREQS))
        for r in range(reps):
            for cell, (sc, depth, rho, freq) in enumerate(cells):
                rows.append(dict(scenario=sc, depth=depth, rho=rho, m_flux=0.0,
                                 inv_freq=freq, sample_size=FLUX_SAMPLE_HAP,
                                 seed=GROWTH_SEED0 + cell * reps + r))
    elif task == "rgrowth":
        reps = reps or RGROWTH_REPS
        cells = list(itertools.product(RGROWTH_SCENARIOS, TIME_DEPTHS, RHOS,
                                       REPL_FREQS))
        for r in range(reps):
            for cell, (sc, depth, rho, freq) in enumerate(cells):
                rows.append(dict(scenario=sc, depth=depth, rho=rho, m_flux=0.0,
                                 inv_freq=freq, sample_size=FLUX_SAMPLE_HAP,
                                 seed=RGROWTH_SEED0 + cell * reps + r))
    elif task == "extreme":
        reps = reps or EXTREME_REPS
        for cell, (sc, depth, m) in enumerate(
                itertools.product(SCENARIOS, EXTREME_DEPTHS, EXTREME_FLUX)):
            for r in range(reps):
                rows.append(dict(scenario=sc, depth=depth, rho=EXTREME_RHO,
                                 m_flux=m, inv_freq=FLUX_INV_FREQ,
                                 sample_size=FLUX_SAMPLE_HAP,
                                 seed=EXTREME_SEED0 + cell * reps + r))
    else:
        raise ValueError(task)
    return rows


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_BACKBONE = None
_TASK = None
_SCRATCH = None


def _init(task, scratch):
    global _BACKBONE, _TASK, _SCRATCH
    _BACKBONE = refsim.load_reference()
    _TASK = task
    _SCRATCH = scratch


def _genotypes(mts, n_hap):
    """(n_hap, n_site) 0/1 matrix over the sites the reference pipeline retains."""
    import numpy as np

    cols = []
    for _pos, ref, hap_alleles in refsim.site_table(mts):
        cols.append([0 if a == ref else 1 for a in hap_alleles])
    if not cols:
        return np.zeros((n_hap, 0), dtype=np.uint8)
    return np.asarray(cols, dtype=np.uint8).T


def _run_one(job):
    t0 = time.time()
    times = TIME_DEPTHS[job["depth"]]
    workdir = tempfile.mkdtemp(prefix="refsim_", dir=_SCRATCH)
    try:
        mts, sample_ids, meta = refsim.simulate(
            job["scenario"], times["t01_23"], times["t0_1"], times["t2_3"],
            job["sample_size"], job["inv_freq"], job["rho"],
            job.get("m_const", M_WITHIN),
            job["seed"], m_flux=job["m_flux"], t_inv_years=times.get("t_inv"),
            flux_scope=job.get("flux_scope", "leaves"))
        mapping = refsim.mapping_hap_SV(sample_ids)
        aln = os.path.join(workdir, "locus.fa")
        n_sites = refsim.write_fasta(aln, mts, sample_ids, _BACKBONE)
        if n_sites == 0:
            # A locus with no retained segregating site has no identifiable tree
            # (IQ-TREE refuses an alignment with no variable column). The
            # orientation trait can always be explained by a single change, which
            # is the same convention parsimony.classify uses.
            n_events = 1
        else:
            treefile = refsim.run_iqtree(aln, os.path.join(workdir, "locus"),
                                         seed=job["seed"])
            n_events = int(refsim.min_mutations(treefile, mapping))
    except Exception as exc:                        # noqa: BLE001
        return dict(job, error=repr(exc)[:300])
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    row = dict(job)
    row.update(n_sites=n_sites, tree_n_events=n_events,
               call_recurrent=int(n_events >= 2),
               frac_admix_i=meta["frac_admix_i"],
               frac_admix_d=meta["frac_admix_d"],
               label=1 if job["scenario"] == "recurrent" else 0,
               secs=round(time.time() - t0, 1), error="")

    if _TASK == "trainset":
        import numpy as np

        from recurrence.features import extract_features, popgen_group_stats
        from recurrence.transferable import from_group_stats

        n_hap = 2 * len(sample_ids)
        G = _genotypes(mts, n_hap)
        labels = np.array([1] * (2 * meta["n_inv_sample"])
                          + [0] * (n_hap - 2 * meta["n_inv_sample"]))
        feats = extract_features(G, labels, tree_n_events=n_events)
        stats = popgen_group_stats(G, labels, refsim.SEQ_LENGTH)
        row.update(feats)
        row.update({"bp_" + k: v for k, v in stats.items()})
        row.update({"tf_" + k: v for k, v in from_group_stats(stats).items()})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=["flux", "trainset", "extreme", "replicate", "fluxsweep", "fluxsweep2", "fluxsweep3", "growth", "rgrowth"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--flux-scope", default="leaves", choices=["leaves", "all"],
                    help="where between-orientation flux acts (see "
                         "refsim.demography). The grid and its seeds are "
                         "unchanged, so a 'leaves' and an 'all' run are paired "
                         "locus for locus.")
    ap.add_argument("--m-const", type=float, default=None,
                    help="override within-orientation migration (default 1e-8). "
                         "The legacy manifest's panmixia rows use 1, which "
                         "collapses each same-orientation pair into one "
                         "population and makes the sample single-origin.")
    ap.add_argument("--flux", default=None,
                    help="comma-separated subset of m_flux values to run (the "
                         "grid and its seeds are unchanged; other rows are "
                         "skipped). Use to get the no-flux baseline first.")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated subset of scenarios to run "
                         "(the grid and its seeds are unchanged; other rows are skipped)")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="stop cleanly once this much wall clock has elapsed; "
                         "every locus finished by then is already on disk")
    ap.add_argument("--depths", default=None,
                    help="comma-separated subset of time depths to run "
                         "(the grid and its seeds are unchanged; other rows are skipped)")
    ap.add_argument("--rhos", default=None,
                    help="comma-separated subset of recombination rates to run "
                         "(the grid and its seeds are unchanged; other rows are skipped)")
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    grid = build_grid(args.task, args.reps)
    for j in grid:
        j["flux_scope"] = args.flux_scope
        if args.m_const is not None:
            j["m_const"] = args.m_const
    mine = [j for i, j in enumerate(grid) if i % args.nshards == args.shard]
    if args.scenarios:
        want = set(args.scenarios.split(","))
        mine = [j for j in mine if j["scenario"] in want]
    if args.depths:
        want = set(args.depths.split(","))
        mine = [j for j in mine if j["depth"] in want]
    if args.rhos:
        keep = [float(x) for x in args.rhos.split(",")]
        mine = [j for j in mine
                if any(abs(j["rho"] - r) < 1e-30 for r in keep)]
    if args.flux:
        keep = [float(x) for x in args.flux.split(",")]
        mine = [j for j in mine
                if any(abs(j["m_flux"] - m) < 1e-30 for m in keep)]
    print(f"[shard {args.shard}/{args.nshards}] {len(mine)}/{len(grid)} jobs, "
          f"{args.procs} procs", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    rows = []
    # Rows are written as they complete. Failures that arrive before the first
    # successful result are buffered until the complete output schema is known.
    fh = open(args.out, "w", newline="")
    w = None
    pending = []
    from multiprocessing import Pool
    with Pool(args.procs, initializer=_init,
              initargs=(args.task, args.scratch)) as pool:
        it = pool.imap_unordered(_run_one, mine, chunksize=1)
        for i, row in enumerate(it):
            if w is None and row.get("error"):
                pending.append(row)
            elif w is None:
                fields = sorted(row)
                lead = [f for f in ("scenario", "label", "depth", "rho", "m_flux",
                                    "flux_scope", "inv_freq", "sample_size",
                                    "seed", "tree_n_events", "call_recurrent",
                                    "n_sites") if f in fields]
                w = csv.DictWriter(fh, fieldnames=lead + [f for f in fields
                                                          if f not in lead])
                w.writeheader()
                for buffered in pending:
                    w.writerow(buffered)
                pending.clear()
            rows.append(row)
            if w is not None:
                w.writerow(row)
            el = time.time() - t0
            if (i + 1) % 50 == 0 or (i + 1) == len(mine):
                fh.flush()
                print(f"  {i + 1}/{len(mine)}  {el:.0f}s  "
                      f"({el / (i + 1):.1f}s/job)", flush=True)
            if args.max_seconds and el > args.max_seconds:
                print(f"  wall-clock limit {args.max_seconds}s reached after "
                      f"{i + 1} loci; stopping cleanly", flush=True)
                pool.terminate()
                break
    if w is None:
        fields = sorted({key for row in pending for key in row})
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in pending:
            w.writerow(row)
    fh.flush()
    fh.close()
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"wrote {len(rows)} rows ({n_err} errors) -> {args.out} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
