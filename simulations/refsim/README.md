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
nine depth × ρ cells:

| m_flux | 0 | 10⁻⁹ | 10⁻⁸ | 10⁻⁷ | 10⁻⁶ |
|---|---|---|---|---|---|
| single-origin FPR | 0.157 | 0.183 | 0.189 | 0.170 | 0.207 |
| recurrent power | 0.891 | 0.889 | 0.857 | 0.874 | 0.907 |

Pooled endpoints: FPR 0.157 → 0.207 (z = 2.13, **p = 0.033**); power 0.891 → 0.907
(z = 0.91, p = 0.36).

**Power is unaffected by flux across the swept range. The false-positive rate is
not — it rises by about a third, and the rise is detectable.** This is a change
from the previous NJ-based sweep, which showed no significant movement in either.

**The single-origin false-positive rate is < 5%, and the earlier claim that it was
not is withdrawn.** The first version of this port built its single-event scenario by
constraining only `frac_admixI`, which does *not* give a single-origin sample — the
direct haplotypes could still be drawn from the deme sister to the inverted one, and
those lineages force extra orientation state changes. Splitting the committed
single-origin replicates by how much weight `fD` placed on that sister deme:

| weight on the sister direct deme | 0.0 | 0.1–0.9 | 1.0 |
|---|---|---|---|
| false-positive rate | **0.016** | ~0.20 | 0.132 |

At zero weight — a genuine single-origin sample — the rate is **1.6%** (n = 254),
under the manuscript's < 5%. By depth × ρ in that clean subset, the only cell with any
false positives is `recent`/ρ = 0 (0.120, n = 25); everything else is 0.000 except
recent/ρ = 10⁻⁸ at 0.032. `recent` is the 50-kya model the manuscript calls "very
recent", which is exactly where it reports its highest rate. The 0.157 figure this
README previously carried was an artefact of the mis-specified scenario.

`single` is now the manuscript's own one-divergence model (see `refsim.py`); the
tables above are from the superseded run and are being regenerated.

**Extreme extension** (1,200 loci at ρ = 10⁻⁸), marginal over the two depths:

| m_flux | 10⁻⁶ | 3×10⁻⁶ | 10⁻⁵ | 3×10⁻⁵ | 10⁻⁴ |
|---|---|---|---|---|---|
| single-origin FPR | 0.300 | 0.383 | 0.567 | 0.808 | 0.942 |
| recurrent power | 0.900 | 0.883 | 0.883 | 0.900 | 0.942 |

Detection holds at 0.88–0.94 while the false-positive rate climbs to 0.94: past
m ≈ 10⁻⁵ the classifier calls nearly everything recurrent and the two scenarios stop
being distinguishable. The breakdown sits between 10⁻⁶ and 10⁻⁵, above the entire
range the manuscript sweeps.

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
