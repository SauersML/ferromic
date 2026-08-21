# Reference recurrence simulations and gene-flux analysis

This directory contains the structured-coalescent simulation and recurrence
classifier used for the manuscript response. The implementation is a port of
the immutable upstream source at commit
`3f845a2b4e017842d1d6648210f148faab616e17`, verified through
`reproducibility/manuscript_sources.json`.

## Recurrence classifier

Each simulated locus is processed as follows:

1. simulate the nine-deme structured coalescent with `msprime`;
2. retain biallelic sites and construct a full-length haplotype alignment on
   the upstream `inputFiles/temp.fa` reference backbone;
3. infer the maximum-likelihood tree with IQ-TREE 2.1.2 using the upstream
   `-safe -keep-ident -bb 1000 -m MFPMERGE -o CMP_CMP_0` options;
4. collapse the outgroup and score direct/inverted tip labels with Fitch
   parsimony; and
5. classify a locus as recurrent when `minMutHomoplasy >= 2`.

The classifier uses the best IQ-TREE `.treefile`. Bootstrap trees are generated
by IQ-TREE but are not used to make the recurrence call.

## Reported gene-flux analysis

The response's gene-flux numbers come from the `fluxsweep` grid. Both arms use
the shared nine-deme demography:

- `single_repo`: one inverted origin is sampled and direct haplotypes are
  sampled from the opposite ancestral clade (`fD = 1 - fI`);
- `recurrent`: both inverted origins are represented in the sample.

Symmetric between-orientation migration is added among extant leaf demes as the
analogue of gene conversion and double crossing over. The grid uses:

- flux `m`: 0, 1e-8, 1e-7, and 1e-6 per lineage per generation;
- inversion frequencies: 0.01, 0.02, 0.05, 0.10, 0.25, and 0.50;
- four event-age models and three recombination rates;
- 20 replicates per complete parameter cell, or 120 replicates after pooling
  the six inversion frequencies; and
- 240 sampled haplotypes and seeds 9,000,000 through 9,011,519.

This is the analysis that gives a maximum-flux single-event false-positive rate
of 4.6% (95% Wilson CI 3.6-5.8%), a two-sided Cochran-Armitage trend p = 0.0071,
and no trend in recurrent-event power (p = 0.3059). These values are regenerated
by `make_report.py`; they are not hard-coded into that script.

The literal two-deme `single` model is retained only for the separate
frequency-trajectory sensitivity analysis. It is not the source of the reported
gene-flux results.

## Reproduction

`run_grid.py` constructs every grid deterministically. `validate_grid.py` fails
on missing, duplicated, extra, or errored loci and records SHA-256 checksums for
every input shard. The GitHub Actions workflow
`.github/workflows/refsim_simulations.yml` runs and validates all manuscript
simulation grids from source. On MSI/Sioux, the reported sweep can be run with:

```bash
sbatch --array=0-7 --export=ALL,TASK=fluxsweep,TAG=fluxsweep \
  refsim.sbatch
python validate_grid.py --task fluxsweep \
  --inputs 'out/fluxsweep_shard*.csv' \
  --provenance fluxsweep_provenance.json
python make_report.py 'out/fluxsweep_shard*.csv' --prefix gene_flux
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
