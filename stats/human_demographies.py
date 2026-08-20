"""Turn every published human demographic model into a time-change spec.

Motivation: the envelope test's null (stats/inversion_selection_envelope.py) was
a CONSTANT-SIZE coalescent, and 17q21.31's p_balance turned out to swing across
four orders of magnitude depending on the size history assumed -- from below
1e-4 under recent expansion to 0.26 under a deep bottleneck. So the history is
not a nuisance to validate against, it is the dominant term, and picking one by
hand is not defensible. This script replaces the hand-picked caricature with the
published catalogue: every HomSap demographic model in stdpopsim, converted to
the piecewise-constant relative-size spec that the null's `--demography` flag
consumes.

Why the IICR and not "the model's Ne": most of these models are structured, with
several demes and migration, so there is no single N(t) in them. What our null
needs is the panmictic history that reproduces the same coalescent, and for
pairwise coalescence that object is exactly the inverse instantaneous
coalescence rate (Mazet et al. 2016): if a random pair from our sampling
configuration coalesces at instantaneous rate R(t), then

    nu(t) = R(0) / R(t)

is the relative size a panmictic population would need to generate that rate.
R(t) comes from msprime's DemographyDebugger.coalescence_rate_trajectory, which
computes it exactly from the model -- no simulation.

Scope caveat, stated once and loudly: the IICR is exact for PAIRWISE coalescence
and an approximation for the higher-order tree shape our n=80 conditioned-branch
null also depends on. It is nonetheless the right reduction to make, because the
alternative on offer is assuming a constant size, which is the same
approximation with nu fixed at 1.

Two sampling configurations per model, because our panel is multi-ancestry and
the honest answer is a range, not a point:

  pooled   lineages spread evenly over the demes that can be sampled at time 0
           -- the analogue of a panel spanning several superpopulations
  <deme>   all lineages in one contemporary deme -- the single-ancestry reading

Output: a TSV of model/config -> spec string, consumed by --demography.
Run this with stdpopsim importable (project-space pylibs on PYTHONPATH); it does
NOT import the envelope module, so its numpy can differ from the analysis env.
"""

import argparse
import sys

import numpy as np


# Contemporary present-day human demes. stdpopsim reports sampling_time == 0 for
# archaic and ancient-DNA demes too (Neanderthal, Denisovan, Loschbour, the
# AncientEurope stages), and a null that samples our 44 present-day haplotypes
# from a Neanderthal deme is not a model of our data -- it is a different
# question. Pooled configurations are built from this list only, so the
# multi-ancestry analogue does not silently include archaics.
MODERN_DEMES = {
    "YRI", "CEU", "CHB", "JPT", "AFR", "EUR", "ASIA", "ADMIX",
    "African_Americans", "generic", "Han", "Sardinian", "Mbuti",
    "ME", "J", "WAJ", "EAJ", "Papuan",
}


def iicr_spec(demography, lineages, n_grid, t_max_gen, tol):
    """Piecewise-constant relative size from the pairwise coalescence rate.

    Returns (spec_string, n_epochs_kept, diagnostics dict). Time is emitted in
    the null's own unit -- one unit is the time in which a pair coalesces at
    rate 1 at the present, i.e. 1/R(0) generations -- so the absolute scale
    never has to be agreed on. The mutation layer calibrates to each locus's
    observed pi_dir, so only the SHAPE of nu matters."""
    import msprime  # noqa: F401  (imported by caller's env)

    dbg = demography.debug()
    # log-spaced grid: the interesting structure is recent, but the deep end
    # has to reach past the oldest split or the tail of the tree sees a
    # flat-lined nu that the model never actually asserts.
    steps = np.concatenate([[0.0], np.logspace(0, np.log10(t_max_gen), n_grid)])
    rate, _prob = dbg.coalescence_rate_trajectory(steps=steps,
                                                  lineages=lineages,
                                                  double_step_validation=False)
    rate = np.asarray(rate, dtype=float)
    ok = np.isfinite(rate) & (rate > 0)
    if not ok[0]:
        raise ValueError("coalescence rate at t=0 is not positive")
    r0 = rate[0]
    nu = np.where(ok, r0 / np.where(ok, rate, 1.0), np.nan)
    # forward-fill any non-finite tail with the last good value: beyond the
    # deepest event the model is constant anyway.
    last = 1.0
    for i in range(len(nu)):
        if np.isfinite(nu[i]):
            last = nu[i]
        else:
            nu[i] = last
    t_our = steps * r0                      # generations -> null time units

    # Thin: keep a breakpoint only where nu actually moves. 200 grid points
    # would otherwise make a 4 kB spec string for a history with 6 epochs.
    keep = [0]
    for i in range(1, len(nu)):
        if abs(np.log(nu[i] / nu[keep[-1]])) > tol:
            keep.append(i)
    if keep[-1] != len(nu) - 1:
        keep.append(len(nu) - 1)

    # pair i encodes epoch [t_{i-1}, t_i) with size nu(t_{i-1})
    pairs = []
    for j in range(1, len(keep)):
        t_right = t_our[keep[j]]
        nu_left = nu[keep[j - 1]]
        pairs.append(f"{t_right:.6g},{nu_left:.6g}")
    spec = "pw:" + ";".join(pairs)
    diag = {"nu_min": float(np.min(nu)), "nu_max": float(np.max(nu)),
            "r0_per_gen": float(r0), "n_grid_kept": len(keep),
            "t_max_our_units": float(t_our[-1])}
    return spec, len(pairs), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lineages", type=int, default=80,
                    help="sample size to match the locus being tested")
    ap.add_argument("--n-grid", type=int, default=300)
    ap.add_argument("--t-max-gen", type=float, default=400_000,
                    help="deep end of the grid in generations (30 yr/gen -> "
                         "400k generations is ~12 Myr, past every split)")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="keep a breakpoint when log nu moves by more than this")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import stdpopsim

    sp = stdpopsim.get_species("HomSap")
    rows = []
    for model in sp.demographic_models:
        # only present-day human demes can stand in for our samples
        modern = [p.name for p in model.populations
                  if getattr(p, "sampling_time", 0) == 0
                  and p.name in MODERN_DEMES]
        if not modern:
            print(f"  {model.id}: no contemporary human deme, skipped")
            continue
        dem = model.model
        configs = {}
        # pooled: spread as evenly as possible over contemporary demes. Skipped
        # for single-deme models, where pooled is identical to the single-deme
        # config and would only duplicate a row.
        if len(modern) > 1:
            base, extra = divmod(a.n_lineages, len(modern))
            pooled = {p: base + (1 if i < extra else 0)
                      for i, p in enumerate(modern)}
            configs["pooled:" + "+".join(modern)] = {k: v for k, v in
                                                     pooled.items() if v > 0}
        for p in modern:
            configs[p] = {p: a.n_lineages}
        for cfg_name, lineages in configs.items():
            try:
                spec, nep, diag = iicr_spec(dem, lineages, a.n_grid,
                                            a.t_max_gen, a.tol)
            except Exception as e:                     # noqa: BLE001
                print(f"  {model.id} [{cfg_name}]: FAILED {type(e).__name__}: "
                      f"{e}")
                continue
            rows.append({"model": model.id, "config": cfg_name,
                         "n_epochs": nep, **diag, "spec": spec})
            print(f"  {model.id:34s} {cfg_name:28s} epochs={nep:3d} "
                  f"nu in [{diag['nu_min']:.3g}, {diag['nu_max']:.3g}]")

    if not rows:
        sys.exit("no demographies extracted")
    cols = ["model", "config", "n_epochs", "nu_min", "nu_max", "r0_per_gen",
            "n_grid_kept", "t_max_our_units", "spec"]
    with open(a.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {len(rows)} demography specs -> {a.out}")


if __name__ == "__main__":
    main()
