#!/usr/bin/env python
"""Reference recurrence pipeline: a faithful port of hsiehphLab/inversionSimulation.

This module reproduces, step for step, the recurrence inference of
``hsiehphLab/inversionSimulation`` (the pipeline behind the manuscript's Fig. 1G),
so that every recurrence number in this repository is produced by the *same*
method rather than by an approximation of it.

Correspondence to the original simulation sources
-------------------------------------------------
=========================================  ==================================
upstream file                              function here
=========================================  ==================================
scripts/recurrentINV_m1.2pop.py            ``demography`` / ``simulate``
scripts/singleINV_m1.py                     ``demography_single`` / ``simulate``
scripts/ancestralstateInVCF.forINVsim.py   ``site_table`` / ``mapping_hap_SV``
scripts/phasedVCF2Fasta.py                 ``write_fasta``
scripts/rmFASTAseqs.py                     ``write_fasta`` (``seq2beRM`` filter)
Snakefile.full.snake rule ``IQTree``       ``run_iqtree``
scripts/computeMinMutations.py             ``min_mutations``
=========================================  ==================================

The classification is therefore, exactly as upstream:

1. structured-coalescent simulation under the 9-deme model of
   ``recurrentINV_m1.2pop.py`` (N_a = 6000, mu = 1.25e-8, 25 y/generation);
2. VCF-equivalent site table with multiallelic sites dropped and the ancestral
   allele set to REF (``AA=REF``, as upstream);
3. a full-length nucleotide alignment built on the upstream reference backbone
   (``inputFiles/temp.fa``, vendored here byte-for-byte), one sequence per
   haplotype plus the ancestral/outgroup sequence ``CMP_CMP_0``;
4. an IQ-TREE maximum-likelihood tree over that alignment, rooted on the
   outgroup (``-m MFPMERGE -keep-ident -o CMP_CMP_0``);
5. the outgroup collapsed, orientation mapped onto the tips as a binary trait
   (``A`` = direct, ``T`` = inverted), and the minimum number of orientation
   state changes scored with Biopython's ``ParsimonyScorer`` (Fitch).

``min_mutations`` returns that count -- upstream's ``minMutHomoplasy``. The
recurrence call is ``count >= 2``.

Deliberate deviations, all of which leave the ``.treefile`` and therefore the
parsimony score unchanged:

* ``--date/--date-options/--clock-sd/--date-tip/--date-ci`` are dropped. Upstream
  those flags produce ``{locus}.timetree.nex``, which is consumed only by the
  plotting rules; ``computeMinMutations.py`` reads ``{locus}.treefile``.
* ``-bb 1000`` is dropped. It generates bootstrap-support outputs after the
  maximum-likelihood search, while recurrence is scored only from
  ``{locus}.treefile``.
* ``-nt 1`` and an explicit ``-seed``: replicates are parallelised one per core,
  and a fixed seed makes each replicate's tree search reproducible.

Single vs. recurrent origin
---------------------------
The original analysis has two distinct generators. ``singleINV_m1.py`` is a
two-population model with one inversion event. ``recurrentINV_m1.2pop.py`` is a
nine-population model with three inversion events. The latter independently
draws the contribution from each of the two inverted demes and each of the two
direct demes as ``random.randint(0, 10) / 10``. Those draws are preserved
exactly: the production analysis does not condition either mixture or force a
sample to come from a particular descendant population.

The pinned public GitHub commit contains both generators. The single-event
model uses its one-split topology and inverted-deme size exactly: ``N_a/100``.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile

# ---------------------------------------------------------------------------
# Fixed parameters -- recurrentINV_m1.2pop.py, verbatim
# ---------------------------------------------------------------------------
CHROM_ID = "chr1"
N_A = 6000
MU = 1.25e-8
GENERATION_TIME = 25
SEQ_LENGTH = 200_000

# config.yaml
OUTGROUP = "CMP_CMP_0"
SEQ2BERM = "CMP_CMP_1"

_HERE = os.path.dirname(os.path.abspath(__file__))
REF_FASTA = os.path.join(_HERE, "inputFiles", "temp.fa")

def iqtree_binary():
    """Resolve the IQ-TREE executable: ``$IQTREE_BIN``, else the first of
    ``iqtree2`` / ``iqtree3`` / ``iqtree`` on ``PATH``."""
    env = os.environ.get("IQTREE_BIN")
    if env:
        return env
    for name in ("iqtree2", "iqtree3", "iqtree"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        "no IQ-TREE binary found; install iqtree2 or set IQTREE_BIN")


def load_reference(path=REF_FASTA, length=SEQ_LENGTH):
    """The upstream ``inputFiles/temp.fa`` backbone, as ``bytes``.

    ``phasedVCF2Fasta.py`` slices ``reference_seq[chrom][pos_start-1:pos_end]``
    for ``locus = chr1:1-200000``, i.e. the first ``SEQ_LENGTH`` bases. Held as
    bytes rather than a list of characters so that the per-haplotype copies stay
    at one byte per base -- at 240 haplotypes x 200 kbp the list-of-str form
    costs ~400 MB per worker, which does not fit a full node of workers.
    """
    seq = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq.append(line.strip())
    return "".join(seq)[:length].encode("ascii")


# ---------------------------------------------------------------------------
# 1. Demography -- recurrentINV_m1.2pop.py
# ---------------------------------------------------------------------------
def _flux_pairs(Tsp_p01_p23, Tsp_p0_p1, Tsp_p2_p3):
    """Every opposite-orientation deme pair that is ever simultaneously active.

    Yields ``(deme_a, deme_b, t_start)`` in generations, where ``t_start`` is the
    oldest-going time at which both demes of the pair exist. A population's
    orientation is the state it carries during that interval: the first event
    creates ``Pa_I`` and ``Pa_D``; the later events create the four descendant
    populations. Enumerating overlapping lifetimes keeps flux on continuously
    as population splits replace ancestors with descendants.
    """
    spans = {                          # name: (orientation, start, end)
        "P0_D": ("D", 0.0, Tsp_p0_p1), "P1_I": ("I", 0.0, Tsp_p0_p1),
        "P2_I": ("I", 0.0, Tsp_p2_p3), "P3_D": ("D", 0.0, Tsp_p2_p3),
        "Pa_I": ("I", Tsp_p0_p1, Tsp_p01_p23),
        "Pa_D": ("D", Tsp_p2_p3, Tsp_p01_p23),
    }
    names = list(spans)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (oa, sa, ea), (ob, sb, eb) = spans[a], spans[b]
            if oa == ob:
                continue
            lo, hi = max(sa, sb), min(ea, eb)
            if hi > lo:
                pairs.append((a, b, lo))
    return pairs


def demography(t01_23_years, t0_1_years, t2_3_years, m_const, frac_admix_i,
               frac_admix_d, m_flux=0.0):
    """The upstream 9-deme structured-coalescent model.

    ``m_flux`` is this repository's *only* addition: symmetric migration between
    opposite-orientation demes (the gene-conversion / double-crossover analogue).
    At ``m_flux = 0`` the demography is upstream's, unchanged.

    Flux acts between every opposite-orientation pair for the entire interval in
    which both demes exist, including ancestral demes (see ``_flux_pairs``).
    """
    import msprime

    Tsp_p01_p23 = t01_23_years / GENERATION_TIME
    Tsp_p0_p1 = t0_1_years / GENERATION_TIME
    Tsp_p2_p3 = t2_3_years / GENERATION_TIME

    de = msprime.Demography()
    de.add_population(name="P_I", description="Final INV group", initial_size=0.1 * N_A)
    de.add_population(name="P_D", description="Final DIR group", initial_size=N_A)
    de.add_population(name="P0_D", description="Pop1, DIR", initial_size=0.01 * N_A)
    de.add_population(name="P1_I", description="Pop2, INV", initial_size=0.1 * N_A)
    de.add_population(name="P2_I", description="Pop3, INV", initial_size=0.1 * N_A)
    de.add_population(name="P3_D", description="Pop4, DIR", initial_size=N_A)
    de.add_population(name="Pa_I", description="Ancestral INV group", initial_size=0.1 * N_A)
    de.add_population(name="Pa_D", description="Ancestral DIR group", initial_size=N_A)
    de.add_population(name="P00", description="Ancestral group", initial_size=N_A)

    de.set_symmetric_migration_rate(["P0_D", "P3_D"], m_const)
    de.set_symmetric_migration_rate(["P1_I", "P2_I"], m_const)

    # --- this repository's flux extension (upstream has no between-orientation
    # --- migration; m_flux = 0 reproduces upstream exactly) ---
    if m_flux > 0:
        for a, b, t_start in _flux_pairs(Tsp_p01_p23, Tsp_p0_p1, Tsp_p2_p3):
            if t_start <= 0:
                de.set_symmetric_migration_rate([a, b], m_flux)
            else:
                de.add_symmetric_migration_rate_change(
                    time=t_start, populations=[a, b], rate=m_flux)
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)

    de.add_admixture(time=0.00001, derived="P_I", ancestral=["P1_I", "P2_I"],
                     proportions=[frac_admix_i, 1 - frac_admix_i])
    de.add_admixture(time=0.00001, derived="P_D", ancestral=["P0_D", "P3_D"],
                     proportions=[frac_admix_d, 1 - frac_admix_d])
    de.add_population_split(time=Tsp_p2_p3, derived=["P2_I", "P3_D"], ancestral="Pa_D")
    de.add_population_split(time=Tsp_p0_p1, derived=["P0_D", "P1_I"], ancestral="Pa_I")
    de.add_population_split(time=Tsp_p01_p23, derived=["Pa_I", "Pa_D"], ancestral="P00")
    de.sort_events()
    return de


def demography_recurrent_growth(t01_23_years, t0_1_years, t2_3_years,
                                inv_freq, frac_admix_i, frac_admix_d,
                                m_const=0.0, m_flux=0.0, n_steps=96):
    """Recurrent model in which each of the three events starts as one haplotype.

    The published recurrent model gives every deme a fixed size -- 0.1 N_a for
    the inverted classes, 0.01 N_a for the direct class descended from them --
    so, exactly as in the single-event model, an inversion is a fully formed
    subpopulation from the instant it appears and its frequency never enters the
    coalescent. Reviewer 1's objection applies here too.

    Here each newly arising orientation carries its own trajectory, and the
    admixture proportions do double duty: ``frac_admix_i`` sets both the share of
    sampled inverted haplotypes drawn from ``P1_I`` and the size of that deme, so
    a deme that contributes a tenth of the sample is a tenth the size rather than
    the same size as one contributing nine tenths.

    Present-day sizes, which sum to ``N_a``::

        P1_I = N_a x f_I          P2_I = N_a x (1 - f_I)
        P0_D = N_a (1 - x) f_D    P3_D = N_a (1 - x)(1 - f_D)

    Each class that *begins* with an orientation change grows from a single
    haplotype at its own event: the inverted clade (``Pa_I`` then ``P1_I``) from
    the first event, ``P0_D`` from the second, ``P2_I`` from the third. The
    direct class ``P3_D``/``Pa_D`` is the ancestral orientation and has no
    founding bottleneck; it is stepped to take whatever ``N_a`` leaves over, so
    the four classes conserve the population at every point.
    """
    import msprime

    T1 = t01_23_years / GENERATION_TIME
    T2 = t0_1_years / GENERATION_TIME
    T3 = t2_3_years / GENERATION_TIME
    x0 = float(inv_freq)
    founder = 0.5                                   # one haplotype

    n1I = max(1.0, N_A * x0 * frac_admix_i)
    n2I = max(1.0, N_A * x0 * (1.0 - frac_admix_i))
    n0D = max(1.0, N_A * (1.0 - x0) * frac_admix_d)
    n3D = max(1.0, N_A * (1.0 - x0) * (1.0 - frac_admix_d))
    a1I = math.log(n1I / founder) / T1
    a0D = math.log(n0D / founder) / T2
    a2I = math.log(n2I / founder) / T3

    def inv_clade(t):
        return n1I * math.exp(-a1I * t)

    def p0d(t):
        return n0D * math.exp(-a0D * t) if t < T2 else 0.0

    def p2i(t):
        return n2I * math.exp(-a2I * t) if t < T3 else 0.0

    de = msprime.Demography()
    de.add_population(name="P_I", description="Final INV group", initial_size=N_A * x0)
    de.add_population(name="P_D", description="Final DIR group",
                      initial_size=N_A * (1.0 - x0))
    de.add_population(name="P0_D", description="Pop1, DIR",
                      initial_size=n0D, growth_rate=a0D)
    de.add_population(name="P1_I", description="Pop2, INV",
                      initial_size=n1I, growth_rate=a1I)
    de.add_population(name="P2_I", description="Pop3, INV",
                      initial_size=n2I, growth_rate=a2I)
    de.add_population(name="P3_D", description="Pop4, DIR", initial_size=n3D)
    de.add_population(name="Pa_I", description="Ancestral INV group",
                      initial_size=n1I, growth_rate=a1I)
    de.add_population(name="Pa_D", description="Ancestral DIR group",
                      initial_size=N_A)
    de.add_population(name="P00", description="Ancestral group", initial_size=N_A)

    # The ancestral direct class takes the remainder, stepped. Below T3 that is
    # P3_D; above it, P2_I has merged in and the remainder belongs to Pa_D.
    for i in range(1, n_steps):
        t = T1 * i / n_steps
        rest = max(1.0, N_A - inv_clade(t) - p0d(t) - p2i(t))
        de.add_population_parameters_change(
            time=t, population=("P3_D" if t < T3 else "Pa_D"),
            initial_size=rest)

    de.set_symmetric_migration_rate(["P0_D", "P3_D"], m_const)
    de.set_symmetric_migration_rate(["P1_I", "P2_I"], m_const)
    if m_flux > 0:
        for a, b, t_start in _flux_pairs(T1, T2, T3):
            if t_start <= 0:
                de.set_symmetric_migration_rate([a, b], m_flux)
            else:
                de.add_symmetric_migration_rate_change(
                    time=t_start, populations=[a, b], rate=m_flux)
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)

    de.add_admixture(time=0.00001, derived="P_I", ancestral=["P1_I", "P2_I"],
                     proportions=[frac_admix_i, 1 - frac_admix_i])
    de.add_admixture(time=0.00001, derived="P_D", ancestral=["P0_D", "P3_D"],
                     proportions=[frac_admix_d, 1 - frac_admix_d])
    de.add_population_split(time=T3, derived=["P2_I", "P3_D"], ancestral="Pa_D")
    de.add_population_split(time=T2, derived=["P0_D", "P1_I"], ancestral="Pa_I")
    de.add_population_split(time=T1, derived=["Pa_I", "Pa_D"], ancestral="P00")
    de.sort_events()
    return de


def draw_admixture(rng):
    """Draw the two mixture proportions exactly as the recurrent generator."""
    frac_i = rng.randint(0, 10) / 10
    frac_d = rng.randint(0, 10) / 10
    return frac_i, frac_d


# Inverted-deme size in hsiehphLab/inversionSimulation's public single-event
# generator (``singleINV_m1.py``: ``initial_size=N_a/100``).
SINGLE_INV_FRACTION = 1 / 100


def demography_single(t_inv_years, m_flux=0.0):
    """The manuscript's single-event model: one divergence, nothing else.

    Methods: "an inversion event creates a subpopulation (e.g., inverted
    haplotypes) diverged from the ancestral population ... Each divergence
    introduces a bottleneck (90% reduction) in the new subpopulation". Single-event
    models are run at t_inv in {500, 250, 100, 50} kya. There is no second inverted
    deme and no sister direct deme -- those exist only in the recurrent model.

    The inverted deme is ``N_a / 100`` and the deme retaining the ancestral
    orientation is ``N_a``, exactly as in the public upstream generator. Direct
    lineages never enter the inverted deme, so the tree carries exactly one
    orientation change.
    """
    import msprime

    de = msprime.Demography()
    de.add_population(name="P_I", description="INV group",
                      initial_size=N_A * SINGLE_INV_FRACTION)
    de.add_population(name="P_D", description="DIR group", initial_size=N_A)
    de.add_population(name="P00", description="Ancestral group", initial_size=N_A)
    if m_flux > 0:
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)
    de.add_population_split(time=t_inv_years / GENERATION_TIME,
                            derived=["P_I", "P_D"], ancestral="P00")
    de.sort_events()
    return de


def growth_rate(t_inv_years, inv_freq, n_anc=None):
    """Per-generation rate taking one haplotype at ``t_inv`` to ``inv_freq`` today.

    The inversion is a single mutation, so it starts as one copy out of ``2 N_a``
    and rises to its present frequency. Writing the inverted class as a
    subpopulation of size ``N_a x(t)``, an exponential trajectory
    ``x(t) = x_0 exp(-alpha t)`` measured backwards from the present satisfies
    ``x(T) = 1 / (2 N_a)`` when

        alpha = ln(2 N_a x_0) / T.

    That is the growth msprime wants: a population declining backwards in time to
    a single founding lineage exactly at the inversion event.
    """
    n_anc = N_A if n_anc is None else n_anc
    t_gen = t_inv_years / GENERATION_TIME
    return math.log(2 * n_anc * float(inv_freq)) / t_gen


def demography_growth(t_inv_years, inv_freq, m_flux=0.0, n_steps=64):
    """Single-event model in which the inversion actually rises to ``inv_freq``.

    The published model gives the inverted class a constant size of ``N_a / 100``
    whatever the inversion's frequency or age, so frequency never enters the
    coalescent -- it only decides how many haplotypes are drawn. Reviewer 1 is
    right that this leaves no room for the inversion to accumulate diversity while
    it does so (Charlesworth 2023).

    Here the inverted class instead carries its own frequency trajectory. It is
    one haplotype at ``t_inv`` and ``N_a * inv_freq`` diploids today, growing at
    ``growth_rate`` in between; the direct class takes the remainder,
    ``N_a * (1 - x(t))``, in ``n_steps`` piecewise-constant pieces, so the two
    orientations together conserve ``N_a`` at every point rather than the total
    population quietly changing size with the inversion's frequency.

    Two consequences, both of them the point of the exercise. Frequency is now a
    parameter of the model and not of the sampling: a 50% inversion has an
    inverted effective size of ``0.5 N_a`` and a 1% inversion ``0.01 N_a``,
    differing fifty-fold in the diversity they can carry. And every inverted
    lineage is forced to coalesce at or before ``t_inv``, because the class
    narrows to a single founder there -- which is what a single origin means.
    """
    import msprime

    x0 = float(inv_freq)
    t_gen = t_inv_years / GENERATION_TIME
    alpha = growth_rate(t_inv_years, x0)

    de = msprime.Demography()
    de.add_population(name="P_I", description="INV group",
                      initial_size=N_A * x0, growth_rate=alpha)
    de.add_population(name="P_D", description="DIR group",
                      initial_size=N_A * (1.0 - x0))
    de.add_population(name="P00", description="Ancestral group", initial_size=N_A)
    # The direct class is N_a(1 - x(t)), which is not exponential, so it is
    # stepped. Steps are uniform in t, which is uniform in log x for an
    # exponential trajectory, so they are finest where x moves fastest.
    for i in range(1, n_steps):
        t = t_gen * i / n_steps
        x = x0 * math.exp(-alpha * t)
        de.add_population_parameters_change(
            time=t, population="P_D", initial_size=N_A * (1.0 - x))
    if m_flux > 0:
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)
    de.add_population_split(time=t_gen, derived=["P_I", "P_D"], ancestral="P00")
    de.sort_events()
    return de


# ---------------------------------------------------------------------------
# 2. Simulation -- recurrentINV_m1.2pop.py
# ---------------------------------------------------------------------------
def simulate(scenario, t01_23_years, t0_1_years, t2_3_years, sample_size,
             inv_freq, rho, m_const, seed, m_flux=0.0, seq_length=SEQ_LENGTH,
             t_inv_years=None):
    """Return ``(tree_sequence, sample_ids, meta)``.

    ``sample_size`` is a haplotype count (upstream ``sampleHaploSize``); the
    inverted / direct split and the diploid rounding follow upstream exactly.

    ``scenario="single"`` uses the historical one-divergence model at
    ``t_inv_years`` (see ``demography_single``). ``scenario="recurrent"`` uses
    the public nine-deme generator and preserves both independent random mixture
    draws exactly.

    In every model, gene flux acts for the full interval during which opposite
    orientations coexist.
    """
    import random

    import msprime

    rng = random.Random(seed)

    num_inv = int(sample_size * inv_freq)
    num_inv_sample = round(num_inv / 2)
    num_direct = sample_size - num_inv
    num_direct_sample = round(num_direct / 2)

    if scenario in ("single", "single_growth"):
        if t_inv_years is None:
            raise ValueError(f"scenario {scenario!r} requires t_inv_years")
        if scenario == "single_growth":
            de = demography_growth(t_inv_years, inv_freq, m_flux=m_flux)
        else:
            de = demography_single(t_inv_years, m_flux=m_flux)
        frac_i, frac_d = 1.0, 0.0
    else:
        if scenario not in ("recurrent", "recurrent_growth"):
            raise ValueError(f"unknown scenario {scenario!r}")
        frac_i, frac_d = draw_admixture(rng)
        if scenario == "recurrent_growth":
            de = demography_recurrent_growth(
                t01_23_years, t0_1_years, t2_3_years, inv_freq, frac_i, frac_d,
                m_const=m_const, m_flux=m_flux)
        else:
            de = demography(t01_23_years, t0_1_years, t2_3_years, m_const,
                            frac_i, frac_d, m_flux=m_flux)

    sample_ids = []
    for i, v in enumerate([num_inv_sample, num_direct_sample]):
        for ii in range(v):
            if i not in [0]:
                sample_ids.append("D%s%s_D%s%s" % (i, ii, i, ii))
            else:
                sample_ids.append("I%s%s_I%s%s" % (i, ii, i, ii))

    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(round(num_inv_sample), population="P_I", ploidy=2),
                 msprime.SampleSet(round(num_direct_sample), population="P_D", ploidy=2)],
        demography=de, sequence_length=seq_length, recombination_rate=rho,
        random_seed=seed)
    mts = msprime.sim_mutations(ts, rate=MU, random_seed=seed)
    meta = dict(frac_admix_i=frac_i, frac_admix_d=frac_d,
                n_inv_sample=num_inv_sample, n_direct_sample=num_direct_sample)
    return mts, sample_ids, meta


# ---------------------------------------------------------------------------
# 3. VCF-equivalent site table -- ancestralstateInVCF.forINVsim.test.py
# ---------------------------------------------------------------------------
def site_table(mts):
    """Yield ``(pos_1based, ref_allele, [allele_per_haplotype])`` per retained site.

    Mirrors ``ancestralstateInVCF.forINVsim.test.py``: multiallelic records
    (``len(ALT) > 1`` in the VCF ALT field) are dropped and ``AA`` is set to
    ``REF``, which for an msprime VCF is the site's ancestral state.
    """
    for var in mts.variants():
        alleles = var.alleles
        ref = alleles[0]
        alt = [a for a in alleles[1:] if a is not None]
        # VCF ALT field is the comma-joined ALT alleles; upstream drops the
        # record when that field is longer than one character.
        if len(",".join(alt)) > 1:
            continue
        if not alt:
            continue
        # msprime's own write_vcf rounds the position and does NOT add one, so
        # POS 612 is written for a site at 612.0. Adding one put every variant a
        # base to the right of where upstream's VCF-fed pipeline puts it. Every
        # sequence shifted together so the tree was unaffected, but the alignment
        # was not the one upstream builds.
        pos = int(round(var.site.position))       # msprime write_vcf POS
        yield pos, ref, [alleles[g] for g in var.genotypes]


def mapping_hap_SV(sample_ids):
    """``mapping_hap_SV.txt``: ``(hapID, SV, orig_hapID)`` rows.

    Upstream reads the orientation off the first character of each half of the
    sample name (``I...`` inverted, ``D...`` direct).
    """
    rows = []
    for sample in sample_ids:
        for i, hap in enumerate(sample.split("_")):
            hap_id = "%s_%s" % (sample, i + 1)
            rows.append((hap_id, "TRUE" if hap[0] == "I" else "FALSE", hap_id))
    return rows


# ---------------------------------------------------------------------------
# 4. Alignment -- phasedVCF2Fasta.py + rmFASTAseqs.py
# ---------------------------------------------------------------------------
def write_fasta(path, mts, sample_ids, backbone, seq2berm=SEQ2BERM,
                outgroup=OUTGROUP):
    """Write the full-length alignment upstream feeds to IQ-TREE.

    One sequence per haplotype, each the reference backbone with its own alleles
    substituted at the retained sites, preceded by the ancestral (``AA``)
    sequence named ``outgroup``. Sequences whose name matches ``seq2berm`` are
    dropped, as ``rmFASTAseqs.py`` does.
    """
    n_hap = 2 * len(sample_ids)
    aa_seq = bytearray(backbone)
    hap_seqs = [bytearray(backbone) for _ in range(n_hap)]

    n_sites = 0
    for pos, ref, hap_alleles in site_table(mts):
        idx = pos - 1                              # locus starts at position 1
        aa_seq[idx] = ord(ref)                     # AA=REF, upstream
        for h in range(n_hap):
            hap_seqs[h][idx] = ord(hap_alleles[h])
        n_sites += 1

    hap_names = ["%s_%s" % (s, j) for s in sample_ids for j in (1, 2)]
    records = [(outgroup, aa_seq)] + list(zip(hap_names, hap_seqs))
    with open(path, "wb") as fh:
        for name, seq in records:
            if seq2berm and re.search(seq2berm, name):
                continue                           # rmFASTAseqs.py
            fh.write(b">" + name.encode("ascii") + b"\n" + bytes(seq) + b"\n")
    return n_sites


# ---------------------------------------------------------------------------
# 5. ML tree -- Snakefile rule IQTree
# ---------------------------------------------------------------------------
def run_iqtree(aln_path, prefix, outgroup=OUTGROUP, seed=1, threads=1,
               binary=None, max_attempts=3):
    """Run IQ-TREE with the upstream flags and return the ``.treefile`` path.

    IQ-TREE 2.1.2 fails for some large ``-seed`` values with "Tree taxa and
    alignment sequence do not match": its initial parsimony tree comes back
    missing taxa. The alignment is fine -- the same file succeeds at a different
    seed. On failure the seed is therefore folded into a small range and retried.
    Only loci that would otherwise fail outright are affected; every locus whose
    first attempt succeeds keeps the seed it was given, so results are unchanged.
    """
    binary = binary or iqtree_binary()
    last = None
    for attempt in range(max_attempts):
        this_seed = seed if attempt == 0 else (int(seed) % 90_000) + attempt
        cmd = [binary, "-safe", "-s", aln_path, "-keep-ident", "-redo",
               "-m", "MFPMERGE", "-pre", prefix, "-o", outgroup,
               "-nt", str(threads), "-seed", str(this_seed), "-quiet"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)
            return prefix + ".treefile"
        except subprocess.CalledProcessError as exc:
            last = exc
    raise last


# ---------------------------------------------------------------------------
# 6. Parsimony score -- computeMinMutations.py
# ---------------------------------------------------------------------------
def min_mutations(treefile, mapping_rows, outgroup=OUTGROUP):
    """``minMutHomoplasy``: Fitch minimum number of orientation state changes.

    A line-for-line port of ``computeMinMutations.py``: read the newick tree,
    collapse the outgroup tip, build the binary orientation alignment
    (``A`` = direct, ``T`` = inverted) over ``orig_hapID``, and score it with
    Biopython's ``ParsimonyScorer``.
    """
    from Bio import Phylo
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import ParsimonyScorer
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    tree = Phylo.read(treefile, "newick")
    for _i, v in enumerate(tree.get_terminals()):
        if re.search(outgroup, str(v)):
            tree.collapse(outgroup)

    trait_aln = MultipleSeqAlignment([
        SeqRecord(Seq("T" if sv == "TRUE" else "A"), id=orig_hap_id)
        for _hap_id, sv, orig_hap_id in mapping_rows
    ])
    return ParsimonyScorer().get_score(tree, trait_aln)


# ---------------------------------------------------------------------------
# One locus, end to end
# ---------------------------------------------------------------------------
def classify_locus(scenario, times, sample_size, inv_freq, rho, m_const, seed,
                   m_flux=0.0, workdir=None, backbone=None, keep=False,
                   iqtree_binary=None):
    """Simulate one locus and return the upstream recurrence result.

    Returns a dict with ``n_events`` (``minMutHomoplasy``), ``call_recurrent``
    (``n_events >= 2``), the realised admixture proportions and the number of
    retained segregating sites.
    """
    backbone = backbone if backbone is not None else load_reference()
    mts, sample_ids, meta = simulate(
        scenario, times["t01_23"], times["t0_1"], times["t2_3"], sample_size,
        inv_freq, rho, m_const, seed, m_flux=m_flux,
        t_inv_years=times.get("t_inv"))
    mapping = mapping_hap_SV(sample_ids)

    tmp = workdir or tempfile.mkdtemp(prefix="refsim_")
    try:
        os.makedirs(tmp, exist_ok=True)
        aln = os.path.join(tmp, "locus.fa")
        n_sites = write_fasta(aln, mts, sample_ids, backbone)
        treefile = run_iqtree(aln, os.path.join(tmp, "locus"), seed=seed,
                              binary=iqtree_binary)
        n_events = min_mutations(treefile, mapping)
    finally:
        if workdir is None and not keep:
            shutil.rmtree(tmp, ignore_errors=True)

    return dict(scenario=scenario, seed=seed, rho=rho, m_flux=m_flux,
                inv_freq=inv_freq, n_sites=n_sites, n_events=int(n_events),
                call_recurrent=bool(n_events >= 2), **meta)


TIME_DEPTHS = {
    # The manifest columns Tsp_p01_p23 / Tsp_p0_p1 / Tsp_p2_p3, in years. Upstream
    # ships the first three; the Methods describe a fourth, "very recent", which
    # the repository does not include. It matters: a single-event locus dates from
    # the *first* event (Tsp_p01_p23), so the four single-event depths the Methods
    # list -- 500 / 250 / 100 / 50 kya -- are the first columns of these four rows,
    # and 50 kya, where the manuscript reports its highest false-positive rate,
    # exists only in the row upstream omits.
    #
    # ``t_inv`` belongs to the two-deme ``single`` sensitivity model only.
    # t_inv is the single-event divergence, and it is the FIRST event of the
    # triple, not the second. A single-origin sample never visits the demes below
    # that split, so its history is one divergence at Tsp_p01_p23 -- which is why
    # the Methods list single-event models at 500 / 250 / 100 / 50 kya, exactly
    # this column. Running them at t0_1 makes every single-event locus about half
    # its proper age, and a younger inverted deme has less time to coalesce, so
    # inverted monophyly breaks more often and the false-positive rate inflates.
    "old":         dict(t01_23=500_000, t0_1=250_000, t2_3=100_000, t_inv=500_000),
    "young":       dict(t01_23=250_000, t0_1=100_000, t2_3=50_000,  t_inv=250_000),
    "recent":      dict(t01_23=100_000, t0_1=50_000,  t2_3=25_000,  t_inv=100_000),
    "very_recent": dict(t01_23=50_000,  t0_1=25_000,  t2_3=10_000,  t_inv=50_000),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="single",
                    choices=["single", "recurrent"])
    ap.add_argument("--depth", default="young", choices=sorted(TIME_DEPTHS))
    ap.add_argument("--sample-size", type=int, default=240)
    ap.add_argument("--inv-freq", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--m-const", type=float, default=1e-8)
    ap.add_argument("--m-flux", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)

    res = classify_locus(args.scenario, TIME_DEPTHS[args.depth],
                         args.sample_size, args.inv_freq, args.rho,
                         args.m_const, args.seed, m_flux=args.m_flux,
                         keep=args.keep)
    print(res)


if __name__ == "__main__":
    main()
