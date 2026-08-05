#!/usr/bin/env python
"""Reference recurrence pipeline: a faithful port of hsiehphLab/inversionSimulation.

This module reproduces, step for step, the recurrence inference of
``hsiehphLab/inversionSimulation`` (the pipeline behind the manuscript's Fig. 1G),
so that every recurrence number in this repository is produced by the *same*
method rather than by an approximation of it.

Correspondence to the upstream repository
-----------------------------------------
=========================================  ==================================
upstream file                              function here
=========================================  ==================================
scripts/recurrentINV_m1.2pop.py            ``demography`` / ``simulate``
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
   outgroup (``-m MFPMERGE -keep-ident -bb 1000 -o CMP_CMP_0``);
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
* ``-nt 1`` and an explicit ``-seed``: replicates are parallelised one per core,
  and a fixed seed makes each replicate's tree search reproducible.

Single vs. recurrent origin
---------------------------
Upstream has **one** demography and one script. Whether a replicate is a
single-event or a recurrent locus is not a property of the model but of the
sample, decided by a single line of ``recurrentINV_m1.2pop.py``::

    frac_admixI = random.randint(0,10)/10

The sampled inverted deme ``P_I`` is admixed from ``P1_I`` and ``P2_I`` in
proportions ``[fI, 1 - fI]``. When that draw lands on 0 or 1 -- two of its eleven
values -- every sampled inverted haplotype descends from one inverted deme, which
is one inversion origin. Otherwise both origins contribute. So a single-event
locus is an upstream replicate whose draw came up 0 or 1, and no separate
demography is needed. Manuscript Fig. 1A draws exactly that: three demes, two
direct and one inverted, because the unsampled inverted deme contributes nothing.

``scenario`` therefore selects a sampling regime, not a model:

``single_upstream``
    ``fI`` in {0, 1}, ``fD`` left free -- the manuscript's single-event locus.
``recurrent``
    ``fI`` in the interior, so both origins are sampled.
``upstream``
    both draws untouched; the mixture upstream actually produces.
``single_repo``
    ``fI`` in {0, 1} *and* ``fD = 1 - fI``. Keeping the direct sample out of the
    inverted clade's ancestral group lowers the false-positive rate, but upstream
    does not do this, so it is a sensitivity rather than the reference.

``demography_single`` is a two-deme, one-split model read literally off the
Methods paragraph. It is **not** what Fig. 1G scored, and it behaves differently:
a 50-kya divergence gives inverted lineages only 2,000 generations to coalesce,
so incomplete lineage sorting scatters them across the tree and the
false-positive rate rises far above the upstream model's, where the inverted
sample has sat in a small deme since the first event. It is kept, as
``scenario="single"``, only as a sensitivity on that reading.
"""
from __future__ import annotations

import argparse
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
    oldest-going time at which both demes of the pair exist. The four leaf pairs
    start at 0; the ancestral pairs start when the younger of the two appears.

    The ancestral demes carry the orientation of upstream's own names: ``Pa_I``
    is its "Ancestral INV group", ``Pa_D`` its "Ancestral DIR group". That label
    is a clade label rather than a per-lineage state -- ``Pa_I`` is the ancestor
    of both ``P1_I`` (inverted) and ``P0_D`` (direct) -- so treating it as an
    inverted deme is a modelling choice, not something the upstream model
    asserts. It is the choice the model's own naming implies, and the one the
    demography figure draws.
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
               frac_admix_d, m_flux=0.0, flux_scope="leaves"):
    """The upstream 9-deme structured-coalescent model.

    ``m_flux`` is this repository's *only* addition: symmetric migration between
    opposite-orientation demes (the gene-conversion / double-crossover analogue).
    At ``m_flux = 0`` the demography is upstream's, unchanged.

    ``flux_scope`` decides *when* that flux acts:

    ``"leaves"``
        Only between the four sampled demes, which is where ``set_symmetric_
        migration_rate`` puts it and therefore what every committed sweep ran.
        Because msprime zeroes a deme's migration rates when it merges into its
        ancestor, flux switches off deme by deme as the splits are passed: with
        the ``young`` depths it acts on all four pairs for 0-50 kya, on
        ``P0_D``-``P1_I`` alone for 50-100 kya, and **not at all** from 100 kya
        back to the root at 250 kya, where only ``Pa_I`` and ``Pa_D`` are left.

    ``"all"``
        Flux between every opposite-orientation pair of demes for the whole time
        both exist, ancestral demes included (see ``_flux_pairs``). The two
        orientations are then partially connected over the model's entire
        history rather than only over its most recent stretch.
    """
    import msprime

    if flux_scope not in ("leaves", "all"):
        raise ValueError(f"unknown flux_scope {flux_scope!r}")

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
            if t_start <= 0:                       # the four sampled-deme pairs
                de.set_symmetric_migration_rate([a, b], m_flux)
            elif flux_scope == "all":
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


def draw_admixture(scenario, rng):
    """``frac_admixI`` / ``frac_admixD`` as drawn upstream, constrained by scenario.

    Upstream: ``random.randint(0, 10) / 10`` for both.

    ``recurrent`` constrains ``fI`` to the interior so both inverted origins are
    sampled. ``upstream`` leaves both draws alone.

    ``single_upstream`` is the manuscript's single-event locus. Upstream has one
    demography and one script; whether a replicate is single-event or recurrent
    is decided by ``frac_admixI`` alone. When that draw lands on 0 or 1 -- two of
    its eleven values -- every sampled inverted haplotype descends from a single
    inverted deme, which is a single inversion origin. Conditioning on those two
    values is the same thing as running upstream and keeping those replicates.

    Crucially the direct draw is left alone, because upstream leaves it alone.
    Constraining it to the non-sister clade (``fD = 1 - fI``) keeps direct
    lineages out of the inverted clade's ancestor and lowers the false-positive
    rate, but that is our idea, not upstream's, and a locus simulated that way is
    not the locus Fig. 1G scored. ``single_repo`` keeps that constrained variant
    available as a sensitivity.
    """
    frac_d = rng.randint(0, 10) / 10
    if scenario == "recurrent":
        frac_i = rng.randint(1, 9) / 10            # interior -- two origins
    elif scenario == "upstream":
        frac_i = rng.randint(0, 10) / 10           # upstream's unconstrained draw
    elif scenario == "single_upstream":
        frac_i = float(rng.randint(0, 1))          # 0.0 or 1.0 -- one origin
    elif scenario == "single_repo":
        frac_i = float(rng.randint(0, 1))
        frac_d = 1.0 - frac_i                      # direct from the non-sister deme
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return frac_i, frac_d


def demography_single(t_inv_years, m_flux=0.0):
    """The manuscript's single-event model: one divergence, nothing else.

    Methods: "an inversion event creates a subpopulation (e.g., inverted
    haplotypes) diverged from the ancestral population ... Each divergence
    introduces a bottleneck (90% reduction) in the new subpopulation". Single-event
    models are run at t_inv in {500, 250, 100, 50} kya. There is no second inverted
    deme and no sister direct deme -- those exist only in the recurrent model.

    Deme sizes follow upstream's convention: the inverted deme is 0.1 * N_a, the
    direct deme and the ancestor are N_a.
    """
    import msprime

    de = msprime.Demography()
    de.add_population(name="P_I", description="INV group", initial_size=0.1 * N_A)
    de.add_population(name="P_D", description="DIR group", initial_size=N_A)
    de.add_population(name="P00", description="Ancestral group", initial_size=N_A)
    if m_flux > 0:
        de.set_symmetric_migration_rate(["P_I", "P_D"], m_flux)
    de.add_population_split(time=t_inv_years / GENERATION_TIME,
                            derived=["P_I", "P_D"], ancestral="P00")
    de.sort_events()
    return de


# ---------------------------------------------------------------------------
# 2. Simulation -- recurrentINV_m1.2pop.py
# ---------------------------------------------------------------------------
def simulate(scenario, t01_23_years, t0_1_years, t2_3_years, sample_size,
             inv_freq, rho, m_const, seed, m_flux=0.0, seq_length=SEQ_LENGTH,
             t_inv_years=None, flux_scope="leaves"):
    """Return ``(tree_sequence, sample_ids, meta)``.

    ``sample_size`` is a haplotype count (upstream ``sampleHaploSize``); the
    inverted / direct split and the diploid rounding follow upstream exactly.

    ``scenario="single"`` uses the manuscript's one-divergence model at
    ``t_inv_years`` (see ``demography_single``). ``"single_repo"`` instead keeps
    the full upstream demography and constrains the admixture draws so only one
    inverted origin is sampled -- a cross-check that the two agree.

    ``flux_scope`` is a no-op for ``scenario="single"``: that model has exactly
    two demes, both alive for the whole interval between the present and the one
    divergence, and only ``P00`` above it. Flux is therefore already acting
    everywhere two demes coexist, so ``"leaves"`` and ``"all"`` give bit-identical
    single-origin loci at every seed. Only the 9-deme recurrent model changes.
    """
    import random

    import msprime

    rng = random.Random(seed)

    num_inv = int(sample_size * inv_freq)
    num_inv_sample = round(num_inv / 2)
    num_direct = sample_size - num_inv
    num_direct_sample = round(num_direct / 2)

    if scenario == "single":
        if t_inv_years is None:
            raise ValueError("scenario 'single' requires t_inv_years")
        # Drawn and discarded so the RNG stream matches the other scenarios.
        rng.randint(0, 10)
        de = demography_single(t_inv_years, m_flux=m_flux)
        frac_i, frac_d = 1.0, 0.0
    else:
        frac_i, frac_d = draw_admixture(scenario, rng)
        de = demography(t01_23_years, t0_1_years, t2_3_years, m_const,
                        frac_i, frac_d, m_flux=m_flux, flux_scope=flux_scope)

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
        cmd = [binary, "-safe", "-s", aln_path, "-keep-ident", "-bb", "1000",
               "-redo", "-m", "MFPMERGE", "-pre", prefix, "-o", outgroup,
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
                   iqtree_binary=None, flux_scope="leaves"):
    """Simulate one locus and return the upstream recurrence result.

    Returns a dict with ``n_events`` (``minMutHomoplasy``), ``call_recurrent``
    (``n_events >= 2``), the realised admixture proportions and the number of
    retained segregating sites.
    """
    backbone = backbone if backbone is not None else load_reference()
    mts, sample_ids, meta = simulate(
        scenario, times["t01_23"], times["t0_1"], times["t2_3"], sample_size,
        inv_freq, rho, m_const, seed, m_flux=m_flux,
        t_inv_years=times.get("t_inv"), flux_scope=flux_scope)
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
    ap.add_argument("--scenario", default="single_upstream",
                    choices=["single_upstream", "single", "single_repo",
                             "recurrent", "upstream"])
    ap.add_argument("--depth", default="young", choices=sorted(TIME_DEPTHS))
    ap.add_argument("--sample-size", type=int, default=240)
    ap.add_argument("--inv-freq", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--m-const", type=float, default=1e-8)
    ap.add_argument("--m-flux", type=float, default=0.0)
    ap.add_argument("--flux-scope", default="leaves", choices=["leaves", "all"],
                    help="'leaves': flux only between the four sampled demes "
                         "(what every committed sweep ran). 'all': flux between "
                         "every opposite-orientation pair for as long as both "
                         "demes exist, ancestral demes included.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args(argv)

    res = classify_locus(args.scenario, TIME_DEPTHS[args.depth],
                         args.sample_size, args.inv_freq, args.rho,
                         args.m_const, args.seed, m_flux=args.m_flux,
                         keep=args.keep, flux_scope=args.flux_scope)
    print(res)


if __name__ == "__main__":
    main()
