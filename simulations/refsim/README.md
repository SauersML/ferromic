# Reference recurrence simulations and gene-flux analysis

This directory contains the structured-coalescent simulation and recurrence
classifier used for the manuscript response. The implementation is a port of
the immutable upstream source at commit
`6ff2ce3c77f056bec013f64dce1efec963468bfc`, verified through
`reproducibility/manuscript_sources.json`.

## Recurrence classifier

Each simulated locus is processed as follows:

1. simulate the public one-split single-event topology with its `N_a/100`
   inverted deme, or the public nine-population
   recurrent model, with `msprime`;
2. retain biallelic sites and construct a full-length haplotype alignment on
   the upstream `inputFiles/temp.fa` reference backbone;
3. infer the maximum-likelihood tree with IQ-TREE 2.1.2 using the upstream
   `-safe -keep-ident -m MFPMERGE -o CMP_CMP_0` search options;
4. collapse the outgroup and score direct/inverted tip labels with Fitch
   parsimony; and
5. classify a locus as recurrent when `minMutHomoplasy >= 2`.

The classifier uses the best IQ-TREE `.treefile`. The upstream `-bb 1000` step
is omitted because it generates bootstrap-support outputs that never enter the
recurrence call.

## Reported gene-flux analysis

The response's gene-flux analysis comes from the `gene_flux` grid:

- `single`: the public `scripts/singleINV_m1.py` two-population,
  one-divergence topology with its `N_a/100` inverted deme;
- `recurrent`: the public `recurrentINV_m1.2pop.py` nine-population model.

The recurrent arm preserves the public generator's sampling without any
constraint: the inverted mixture and direct mixture are two independent draws
from `random.randint(0, 10) / 10`. The single-event arm has no mixture because
there is exactly one direct population and one inverted population.

Symmetric between-orientation migration is added between every pair of
opposite-orientation populations for the entire interval in which both exist,
including ancestral populations. This is the analogue of gene conversion and
double crossing over. The grid uses:

- flux `m`: 0, 1e-8, 1e-7, and 1e-6 per lineage per generation;
- inversion frequencies: 0.01, 0.02, 0.05, 0.10, 0.25, and 0.50;
- four event-age models and three recombination rates;
- 20 replicates per complete parameter cell, or 120 replicates after pooling
  the six inversion frequencies; and
- 240 sampled haplotypes and seeds 9,000,000 through 9,011,519.

The exact counts, Wilson intervals, and two-sided Cochran-Armitage trend tests
are regenerated from the per-locus results by `make_report.py`; they are not
inputs to the analysis. `verify_reported_flux.py` additionally checks every
recurrent row against the two independent mixture draws implied by its seed.

## Reproduction

`run_grid.py` constructs every grid deterministically. `validate_grid.py` fails
on missing, duplicated, extra, or errored loci and records SHA-256 checksums for
every input shard. The GitHub Actions workflow
`.github/workflows/refsim_simulations.yml` runs and validates all manuscript
simulation grids from source. On MSI/Sioux, the reported sweep can be run with:

```bash
sbatch --array=0-7 --export=ALL,TASK=gene_flux,TAG=gene_flux \
  refsim.sbatch
python validate_grid.py --task gene_flux \
  --inputs 'out/gene_flux_shard*.csv' \
  --provenance gene_flux_provenance.json
python make_report.py 'out/gene_flux_shard*.csv' --prefix gene_flux
python verify_reported_flux.py --rows 'out/gene_flux_shard*.csv'
```

The MSI wrapper pins the historical MSI IQ-TREE 2.1.2 module and verifies its
binary checksum before running. GitHub Actions uses the official IQ-TREE 2.1.2
release archive with its independently pinned archive checksum.

## Files

- `refsim.py`: demographies, simulation, alignment, IQ-TREE, and parsimony call
- `run_grid.py`: deterministic sharded grids
- `validate_grid.py`: strict completeness and provenance validation
- `make_report.py`: gene-flux tables, statistics, and figure
- `replicate_manuscript.py`: zero-flux power-analysis report
- `make_growth_report.py`: frequency-trajectory sensitivity summaries
- `refsim.sbatch`: Sioux production wrapper
- `setup_sioux.sbatch`: pinned Python environment setup

## Single-event inverted-deme size

`hsiehphLab/inversionSimulation` sets the single-event inverted deme to
`N_a / 100`. The production model has no alternate size: the code, model audit,
GitHub Actions workflow, and MSI wrapper all use the public 1% value. The
recurrent generator's 10% child demes are unchanged because they belong to the
distinct recurrent model.
