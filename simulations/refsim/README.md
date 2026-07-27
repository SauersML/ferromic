# `simulations/refsim/` — the reference recurrence pipeline, plus the gene-flux sweep

This directory is a faithful port of
[`hsiehphLab/inversionSimulation`](https://github.com/hsiehphLab/inversionSimulation) —
the structured-coalescent simulation and recurrence classifier behind the manuscript's
Fig. 1G — with exactly one addition: a **between-orientation gene-flux** term.

Every recurrence number in this repository is produced by the code here, so simulated
false-positive rates and power are directly comparable to the upstream pipeline's.

## The classifier

Identical to upstream, step for step:

1. structured-coalescent simulation under the 9-deme model of
   `scripts/recurrentINV_m1.2pop.py` (N<sub>a</sub> = 6000, µ = 1.25 × 10⁻⁸,
   25 y/generation, 200 kbp locus);
2. VCF-equivalent site table with multiallelic records dropped and `AA` set to `REF`
   (`scripts/ancestralstateInVCF.forINVsim.test.py`);
3. a **full-length nucleotide alignment** on upstream's own reference backbone
   (`inputFiles/temp.fa`, vendored here byte-for-byte), one sequence per haplotype plus
   the ancestral outgroup `CMP_CMP_0` (`scripts/phasedVCF2Fasta.py`, `scripts/rmFASTAseqs.py`);
4. an **IQ-TREE maximum-likelihood tree** over that alignment rooted on the outgroup —
   `-safe -keep-ident -bb 1000 -m MFPMERGE -o CMP_CMP_0` (Snakefile rule `IQTree`);
5. the outgroup collapsed, orientation mapped onto the tips as a binary trait
   (`A` = direct, `T` = inverted), and the minimum number of orientation state changes
   scored with Biopython's Fitch `ParsimonyScorer` (`scripts/computeMinMutations.py`).

The returned count is upstream's `minMutHomoplasy`. **A locus is called recurrent iff
the count is ≥ 2.**

`refsim.py` documents the file-by-file correspondence in its module docstring, including
the two deliberate deviations — dropping `--date*` (which only produces the timetree the
plotting rules consume, never `.treefile`) and pinning `-nt 1 -seed N` so replicates
parallelise one per core and reproduce exactly.

### Single vs. recurrent origin

Upstream has **one** demography. Whether the sampled inverted haplotypes carry one or two
independent inversion origins is decided by `frac_admixI = random.randint(0, 10) / 10`:
the final inverted deme `P_I` is drawn from the two independently derived inverted demes
`P1_I` and `P2_I` in proportions `[fI, 1 − fI]`. At `fI ∈ {0, 1}` every inverted lineage
descends from a single inverted deme — a **single-origin** locus. At `0 < fI < 1` both
origins contribute — a **recurrent** locus.

`simulate(scenario=...)` therefore constrains only `fI`. Demography, deme sizes, split
times and sampling are untouched, so the two labels are the upstream model conditioned on
its own admixture draw rather than a separate hand-built model.

## The one addition: between-orientation flux

`m_flux` is symmetric migration between opposite-orientation demes (every inverted deme
↔ every direct deme) — the gene-conversion / double-crossover analogue, in **migrants per
lineage per generation**. Upstream has no such term, so **`m_flux = 0` reproduces
upstream exactly** and is the reference column in every table below.

## Layout

```
refsim.py          the reference pipeline (simulate -> alignment -> IQ-TREE -> parsimony)
run_grid.py        shardable driver for the three grids
make_report.py     per-replicate CSVs -> flux_results.{csv,md}, sweep_full.json, figure
refsim.sbatch      SLURM array wrapper
inputFiles/        upstream's reference backbone, vendored byte-for-byte
out/               per-replicate CSVs (one per shard)
```

## Grids

| task | axes | loci |
|---|---|---|
| `flux` | scenario × depth (recent/young/old) × ρ (0, 10⁻⁸, 10⁻⁶) × m (0, 10⁻⁹, 10⁻⁸, 10⁻⁷, 10⁻⁶), 240 haplotypes, f<sub>inv</sub> = 0.1 | 5,400 |
| `extreme` | scenario × depth (young/recent) × m (10⁻⁶ … 10⁻⁴) at ρ = 10⁻⁸ | 1,200 |
| `trainset` | the `flux` axes × f<sub>inv</sub> ∈ {0.05, 0.1, 0.2, 0.35, 0.5}, 88 haplotypes, full feature vector | 11,250 |

`trainset` is the labelled training set consumed by `recurrence/` and carries the
reference origin count as its `tree_n_events` feature.

## Results

**Main sweep** (5,400 loci, 240 haplotypes, f<sub>inv</sub> = 0.1), marginal over the
nine depth × ρ cells. The single-origin half is the corrected one-divergence model; the
recurrent half is unchanged, since the single-event bug never touched it:

| m_flux | 0 | 10⁻⁹ | 10⁻⁸ | 10⁻⁷ | 10⁻⁶ |
|---|---|---|---|---|---|
| single-origin FPR | 0.113 | 0.102 | 0.115 | 0.094 | 0.106 |
| recurrent power | 0.891 | 0.889 | 0.857 | 0.874 | 0.907 |

Pooled endpoints: FPR 0.113 → 0.106 (z = −0.39, **p = 0.70**); power 0.891 → 0.907
(z = 0.91, p = 0.36).

**Neither the false-positive rate nor power moves detectably across the swept flux
range.** This is the manuscript's claim, measured with the manuscript's own classifier.
An earlier version of this file reported the FPR rising 0.157 → 0.207 at p = 0.033; that
was produced by the mis-specified single-event scenario and is withdrawn.

**Corrected single-event model — 2,700 loci.** The earlier single-event scenario was
mis-specified (it let direct haplotypes come from the deme sister to the inverted one;
see `refsim.py`). `single` is now the Methods' own one-divergence model at
t_inv ∈ {50, 100, 250} kya for recent/young/old. Overall false-positive rate **0.106**.

By depth × ρ, at m = 0 (upstream's own model, no flux):

| depth (t_inv) | ρ = 0 | ρ = 10⁻⁸ | ρ = 10⁻⁶ |
|---|---|---|---|
| recent (50 kya) | 0.533 | 0.333 | 0.000 |
| young (100 kya) | 0.133 | 0.017 | 0.000 |
| old (250 kya) | 0.000 | 0.000 | 0.000 |

Two things match the manuscript and two do not.

*Matches:* the false-positive rate is driven by inversion age in the direction the
manuscript reports — it is worst at the shallowest depth and zero at the oldest, and
the manuscript likewise puts its highest rate at the 50-kya model. Deep or recombining
loci are at or below 5%.

*Does not match:* at 50 kya with no recombination the rate here is 0.53, not 0.04; and
the ordering in ρ is inverted — we get 0.000 at ρ = 10⁻⁶ where the manuscript reports
its *maximum* of 4%. The mechanism for our direction is straightforward: at ρ = 0 the
whole 200 kbp is a single genealogy, so incomplete lineage sorting in a 600-individual
inverted deme only 2,000 generations old frequently breaks inverted monophyly, while at
ρ = 10⁻⁶ the alignment averages over many genealogies and the ML tree separates the
orientations cleanly. That is expected coalescent behaviour, so the residual gap is not
explained by the tree method or by the single-event model, and remains open.

**An earlier version of this section claimed 1.6%, "under < 5%".** That was inferred
from the subset of the *old* 9-deme run with no sister-deme admixture, which is not the
same model: in that subset the inverted lineages sit in a small deme for the full
100 kya to the P00 split, giving far more time to coalesce than the clean 50-kya model
allows. Not comparable, and the claim is withdrawn.

**Extreme extension** (1,200 loci at ρ = 10⁻⁸), corrected single-origin model,
marginal over the two depths:

| m_flux | 10⁻⁶ | 3×10⁻⁶ | 10⁻⁵ | 3×10⁻⁵ | 10⁻⁴ |
|---|---|---|---|---|---|
| single-origin FPR | 0.192 | 0.333 | 0.492 | 0.800 | 0.983 |
| recurrent power | 0.900 | 0.883 | 0.883 | 0.900 | 0.942 |

Detection holds at 0.88–0.94 throughout while the false-positive rate climbs to
0.98 (z = 12.5). Past m ≈ 10⁻⁵ the classifier calls nearly everything recurrent and
the two scenarios stop being distinguishable. **The breakdown sits between 10⁻⁶ and
10⁻⁵ — above the entire range the manuscript sweeps**, which is why the main sweep
is flat.

## Reproduce

Needs `msprime`, `biopython`, and an IQ-TREE binary (`iqtree2`, `iqtree3` or `iqtree` on
`PATH`, or `$IQTREE_BIN`). One locus is a few seconds of IQ-TREE, so the grids are meant
to be sharded:

```bash
# one array task per shard; --nshards must equal the array size
sbatch --array=0-7 --export=ALL,TASK=flux     refsim.sbatch
sbatch --array=0-7 --export=ALL,TASK=trainset refsim.sbatch
sbatch --array=0-3 --export=ALL,TASK=extreme  refsim.sbatch

# aggregate the flux sweep
python make_report.py 'out/flux_shard*.csv'

# fold the training set into recurrence/data/sim_features.csv.gz
python -m recurrence.cli simulate --merge 'simulations/refsim/out/trainset_shard*.csv'
```

A single locus, end to end, for a smoke test:

```bash
python refsim.py --scenario single --depth young --rho 0 --seed 1
```
