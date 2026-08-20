#!/usr/bin/env python
"""Checks on the frequency-trajectory single-event model (``demography_growth``).

Reviewer 1 objects that the published model never lets an inversion rise in
frequency: the
inverted class sits at a constant ``N_a / 100`` regardless of the inversion's
frequency or age, so frequency enters only through how many haplotypes are drawn,
and there is no interval over which the inverted haplotypes accumulate diversity.
``demography_growth`` replaces that with a trajectory from one founding haplotype
to the observed present-day frequency.

These are the checks that the replacement behaves as claimed, run before any
production grid:

1. **Trajectory endpoints.** The inverted class is ``N_a x_0`` today and exactly
   one haplotype at the inversion event, for every frequency and every age.
2. **Total size.** The two orientations sum to ``N_a`` throughout, so the model
   redistributes the population rather than inflating it at high frequency.
3. **Single origin is enforced.** Every inverted lineage coalesces at or before
   ``t_inv``; the inverted sample is monophyletic in the true tree.
4. **Diversity accumulates.** Nucleotide diversity within inverted haplotypes
   rises with both frequency and age, against the published model where it is
   flat in both. This is the reviewer's substantive point, so it is measured
   rather than asserted.
5. **Agreement with theory.** Pairwise coalescence time within the inverted class
   is compared with the trajectory's own prediction,
   ``E[T2] = int_0^T exp(-int_0^t ds / (2 N_a x(s))) dt`` plus the residual
   ancestral contribution, so a wrong growth rate shows up as a broken ratio
   rather than a plausible-looking number.

    python check_growth.py [--reps 40]
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import refsim  # noqa: E402

FREQS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
DEPTHS = ["old", "young", "recent", "very_recent"]


def _sizes(de, pop, t):
    """Deme size of ``pop`` at time ``t`` generations ago, via the debugger."""
    dbg = de.debug()
    for ep in dbg.epochs:
        if ep.start_time <= t < ep.end_time:
            for p in ep.populations:
                if p.name == pop:
                    return p.start_size * math.exp(-p.growth_rate
                                                   * (t - ep.start_time))
    return float("nan")


def check_trajectory():
    print("\n" + "=" * 74)
    print("CHECK 1-2  trajectory endpoints and conservation of N_a")
    print("=" * 74)
    print(f"{'depth':<12}{'freq':>7}{'N_I today':>12}{'want':>10}"
          f"{'N_I at t_inv':>14}{'want':>8}{'N_I+N_D':>10}")
    ok = True
    for depth in DEPTHS:
        t_inv = refsim.TIME_DEPTHS[depth]["t_inv"]
        t_gen = t_inv / refsim.GENERATION_TIME
        for x0 in FREQS:
            de = refsim.demography_growth(t_inv, x0)
            now_i = _sizes(de, "P_I", 0.0)
            now_d = _sizes(de, "P_D", 0.0)
            # A hair before the split, where the class is one haplotype.
            end_i = _sizes(de, "P_I", t_gen * (1 - 1e-9))
            want_now = refsim.N_A * x0
            want_end = 0.5                      # one haplotype = 0.5 diploid
            good = (abs(now_i - want_now) < 1e-6 * want_now
                    and abs(end_i - want_end) < 1e-3
                    and abs(now_i + now_d - refsim.N_A) < 1e-6 * refsim.N_A)
            ok &= good
            print(f"{depth:<12}{x0:>7.2f}{now_i:>12.1f}{want_now:>10.1f}"
                  f"{end_i:>14.4f}{want_end:>8.1f}{now_i + now_d:>10.1f}"
                  f"{'' if good else '   <-- FAIL'}")
    print(f"\n  trajectory endpoints and conservation: {'PASS' if ok else 'FAIL'}")
    return ok


def _e_t2_theory(t_inv, x0, n_steps=20000):
    """E[pairwise coalescence time] for two inverted lineages under x(t).

    Coalescence hazard at time t is 1 / (2 N_a x(t)). Lineages that survive to
    the inversion event join the ancestral population and coalesce there, adding
    2 N_a to the expectation weighted by the survival probability.
    """
    t_gen = t_inv / refsim.GENERATION_TIME
    alpha = refsim.growth_rate(t_inv, x0)
    dt = t_gen / n_steps
    surv, e_t = 1.0, 0.0
    for i in range(n_steps):
        t = (i + 0.5) * dt
        n_i = max(0.5, refsim.N_A * x0 * math.exp(-alpha * t))
        haz = 1.0 / (2 * n_i)
        e_t += surv * haz * t * dt
        surv *= math.exp(-haz * dt)
    return e_t + surv * (t_gen + 2 * refsim.N_A)


def check_coalescent(reps):
    """Simulate ancestries only (no mutation, no tree inference) and measure."""
    import msprime

    print("\n" + "=" * 74)
    print("CHECK 3-5  single origin enforced, diversity accumulates, theory match")
    print("=" * 74)
    print(f"{'depth':<12}{'freq':>6}{'TMRCA_inv/t_inv':>17}{'mono':>7}"
          f"{'E[T2] sim':>11}{'theory':>10}{'ratio':>7}{'pi_inv/pi_dir':>15}")
    all_ok = True
    for depth in DEPTHS:
        t_inv = refsim.TIME_DEPTHS[depth]["t_inv"]
        t_gen = t_inv / refsim.GENERATION_TIME
        for x0 in FREQS:
            n_inv = max(2, int(240 * x0) // 2)
            n_dir = (240 - int(240 * x0)) // 2
            de = refsim.demography_growth(t_inv, x0)
            tmrca, mono, t2_inv, pi_i, pi_d = [], 0, [], [], []
            for r in range(reps):
                ts = msprime.sim_ancestry(
                    samples=[msprime.SampleSet(n_inv, population="P_I", ploidy=2),
                             msprime.SampleSet(n_dir, population="P_D", ploidy=2)],
                    demography=de, sequence_length=refsim.SEQ_LENGTH,
                    recombination_rate=0.0, random_seed=9_000_000 + r)
                inv = list(range(2 * n_inv))
                dirn = list(range(2 * n_inv, 2 * (n_inv + n_dir)))
                tree = ts.first()
                tmrca.append(tree.tmrca(*inv) if len(inv) > 1 else 0.0)
                # Monophyly: the MRCA of the inverted set subtends only that set.
                mrca = tree.mrca(*inv)
                mono += int(sum(1 for s in tree.samples(mrca)) == len(inv))
                t2_inv.append(tree.tmrca(inv[0], inv[1]))
                mts = msprime.sim_mutations(ts, rate=refsim.MU,
                                            random_seed=9_000_000 + r)
                pi_i.append(mts.diversity(sample_sets=[inv]).item())
                pi_d.append(mts.diversity(sample_sets=[dirn]).item())
            m_tmrca = sum(tmrca) / len(tmrca)
            m_t2 = sum(t2_inv) / len(t2_inv)
            theory = _e_t2_theory(t_inv, x0)
            ratio = m_t2 / theory if theory else float("nan")
            m_pi_i = sum(pi_i) / len(pi_i)
            m_pi_d = sum(pi_d) / len(pi_d)
            # TMRCA of the inverted sample must not exceed the inversion age by
            # more than the ancestral tail a straggler lineage can contribute.
            ok = m_tmrca <= t_gen * 1.05 and mono == reps
            all_ok &= ok
            pi_ratio = (m_pi_i / m_pi_d) if m_pi_d else float("nan")
            flag = "" if ok else "  <-- FAIL"
            print(f"{depth:<12}{x0:>6.2f}{m_tmrca / t_gen:>17.3f}"
                  f"{mono:>4}/{reps:<3}{m_t2:>11.0f}{theory:>10.0f}"
                  f"{ratio:>7.2f}{pi_ratio:>15.3f}{flag}")
    print(f"\n  single origin enforced in every replicate: "
          f"{'PASS' if all_ok else 'FAIL'}")
    return all_ok


def check_published_contrast(reps):
    """The same measurement on the published constant-size model, for contrast."""
    import msprime

    print("\n" + "=" * 74)
    print("CONTRAST  published constant-size model, same sampling")
    print("=" * 74)
    print("Nucleotide diversity within inverted haplotypes, pi_inv (x 1e-4)\n")
    print(f"{'depth':<12}" + "".join(f"{x:>9.2f}" for x in FREQS)
          + "   model")
    for depth in DEPTHS:
        t_inv = refsim.TIME_DEPTHS[depth]["t_inv"]
        for label, maker in (("published", lambda f: refsim.demography_single(t_inv)),
                             ("trajectory", lambda f: refsim.demography_growth(t_inv, f))):
            vals = []
            for x0 in FREQS:
                n_inv = max(2, int(240 * x0) // 2)
                n_dir = (240 - int(240 * x0)) // 2
                de = maker(x0)
                acc = []
                for r in range(reps):
                    ts = msprime.sim_ancestry(
                        samples=[msprime.SampleSet(n_inv, population="P_I", ploidy=2),
                                 msprime.SampleSet(n_dir, population="P_D", ploidy=2)],
                        demography=de, sequence_length=refsim.SEQ_LENGTH,
                        recombination_rate=0.0, random_seed=9_500_000 + r)
                    mts = msprime.sim_mutations(ts, rate=refsim.MU,
                                                random_seed=9_500_000 + r)
                    acc.append(mts.diversity(
                        sample_sets=[list(range(2 * n_inv))]).item())
                vals.append(1e4 * sum(acc) / len(acc))
            print(f"{depth:<12}" + "".join(f"{v:>9.2f}" for v in vals)
                  + f"   {label}")
    print("\n  The published row is flat in frequency by construction; the "
          "trajectory row\n  should rise with both frequency and age, which is "
          "the reviewer's point.")


def check_recurrent(reps):
    """The recurrent trajectory model: one founder per event, three origins.

    Each class that begins with an orientation change descends from a single
    haplotype, so each must be monophyletic in the true tree -- the analogue of
    the single-event check. Fitch parsimony on the true tree must also require at
    least two orientation changes, otherwise the locus is not recurrent at all
    and the power measurement would be scoring the wrong thing.
    """
    import msprime

    print("\n" + "=" * 74)
    print("CHECK 6-7  recurrent trajectory: one founder per event, >= 2 origins")
    print("=" * 74)
    print(f"{'depth':<12}{'freq':>6}{'P0_D mono':>11}{'P2_I mono':>11}"
          f"{'Pa_I clade':>12}{'parsimony >= 2':>16}{'mean changes':>14}")
    all_ok = True
    f_i = f_d = 0.5
    for depth in DEPTHS:
        d = refsim.TIME_DEPTHS[depth]
        for x0 in FREQS:
            de = refsim.demography_recurrent_growth(
                d["t01_23"], d["t0_1"], d["t2_3"], x0, f_i, f_d,
                m_const=1e-8)
            n_inv = max(2, int(240 * x0) // 2)
            n_dir = (240 - int(240 * x0)) // 2
            # Split each orientation between its two origins as f_I / f_D do.
            n1 = max(1, round(n_inv * f_i)); n2 = max(1, n_inv - n1)
            n0 = max(1, round(n_dir * f_d)); n3 = max(1, n_dir - n0)
            # P1_I alone need NOT be monophyletic: P0_D descends from the
            # same ancestor Pa_I, so its lineages legitimately nest inside the
            # inverted clade. What must hold is that each class founded by one
            # haplotype is monophyletic, and that the whole Pa_I clade is.
            mono = {"P0_D": 0, "P2_I": 0, "Pa_I clade": 0}
            changes = []
            for r in range(reps):
                ts = msprime.sim_ancestry(
                    samples=[msprime.SampleSet(n1, population="P1_I", ploidy=2),
                             msprime.SampleSet(n2, population="P2_I", ploidy=2),
                             msprime.SampleSet(n0, population="P0_D", ploidy=2),
                             msprime.SampleSet(n3, population="P3_D", ploidy=2)],
                    demography=de, sequence_length=refsim.SEQ_LENGTH,
                    recombination_rate=0.0, random_seed=9_700_000 + r)
                tree = ts.first()
                off, sets = 0, {}
                for name, k in (("P1_I", n1), ("P2_I", n2),
                                ("P0_D", n0), ("P3_D", n3)):
                    sets[name] = list(range(off, off + 2 * k))
                    off += 2 * k
                groups = {"P0_D": sets["P0_D"], "P2_I": sets["P2_I"],
                          "Pa_I clade": sets["P1_I"] + sets["P0_D"]}
                for name, members in groups.items():
                    mrca = tree.mrca(*members)
                    mono[name] += int(len(list(tree.samples(mrca)))
                                      == len(members))
                # Fitch parsimony on the true tree, orientation as the trait.
                geno = [0] * off
                for name in ("P1_I", "P2_I"):
                    for s in sets[name]:
                        geno[s] = 1
                _anc, muts = tree.map_mutations(geno, ["D", "I"])
                changes.append(len(muts))
            ok = (all(v == reps for v in mono.values())
                  and min(changes) >= 2)
            all_ok &= ok
            print(f"{depth:<12}{x0:>6.2f}{mono['P0_D']:>8}/{reps:<3}"
                  f"{mono['P2_I']:>8}/{reps:<3}{mono['Pa_I clade']:>9}/{reps:<3}"
                  f"{sum(c >= 2 for c in changes):>13}/{reps:<3}"
                  f"{statistics.fmean(changes):>14.2f}"
                  f"{'' if ok else '  <-- FAIL'}")
    print(f"\n  one founder per event and recurrence present: "
          f"{'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=40)
    args = ap.parse_args(argv)
    ok = check_trajectory()
    ok &= check_coalescent(args.reps)
    ok &= check_recurrent(args.reps)
    check_published_contrast(args.reps)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
