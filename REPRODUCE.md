# Reproducing the analyses

Everything except the All of Us PheWAS runs in GitHub Actions from committed
inputs, and commits its outputs back. Nothing needs a local machine or a cluster.

## What runs where

| Workflow | What it regenerates | Trigger |
|---|---|---|
| `revision_analyses.yml` | every table added or corrected for the revision (below) | push to `stats/**`, or manual |
| `refsim_simulations.yml` | the reference recurrence simulations, 20-way sharded, then refit + rescore | manual, or a change to `simulations/refsim/*.py` |
| `recurrence.yml` | the `recurrence/` reproduction and QC tests, with IQ-TREE installed | push to `recurrence/**` |
| `replicate_manuscript_statistics.yml` | the main statistics report and CDS tables | push to `stats/**` or `data/**` |
| `generate_supplementary_tables.yml` | `supplementary_tables.xlsx` | push |
| `run_analysis.yml`, `analysis_pipeline.yml`, `manual_run_vcf.yml` | π / F<sub>ST</sub> / per-site tracks from the VCFs | manual |
| `combine_paml.yml` | the PAML clade-model results | manual |
| `fixed.yml` | fixed-difference outputs | manual |

## Revision analyses, by reviewer comment

| Comment | Script | Output |
|---|---|---|
| R1 — circularity of the recurrence calls | `stats/recurrence_sd_architecture.py` | `data/recurrence_sd_{calls,summary}.tsv` |
| R1 — 4-fold degenerate diversity | `stats/four_fold_pi.py --from-table`, `stats/four_fold_pi_correlations.py` | `data/four_fold_pi_{tests,correlations}.tsv` |
| R1 — π<sub>N</sub>/π<sub>S</sub> | `stats/pin_pis.py --from-table` | `data/pin_pis_tests.tsv` |
| R2 #1 — CDS conservation inference | `stats/robust_cds_reanalysis.py`, `stats/cds_conservation_calibration.py` | `data/robust_cds_reanalysis_*`, `data/cds_conservation_calibration.tsv` |
| R2 #2 — extreme clade-model ω | `stats/paml_extreme_omega_check.py` | `data/paml_extreme_omega_check.tsv` |
| R2 #3 — external PheWAS replication | `stats/finngen_replication.py` | `data/finngen_replication.tsv` |
| R2 #5 — why 93 of 292 | `stats/table_s5_exclusion_reasons.py` | `data/table_s5_exclusion_reasons.tsv` |
| R2 #9 — AGES beyond one tagging SNP | `stats/ages_multi_tag_snps.py` | `data/ages_multi_tag_snps.tsv` |
| R2 #11 — consistent inversion naming | `stats/generate_tables.py` | canonical `inversion` column in every locus-bearing sheet |
| R3 #2 — flux robustness of the classifier | `simulations/refsim/` | `simulations/refsim/{flux,extreme}_results.*` |
| per-gene power context | `stats/cds_haplotype_counts.py` | `data/cds_haplotype_counts.tsv` |
| expression evidence at the loci | `stats/inversion_eqtl_lookup.py` | `data/inversion_eqtl.tsv` |
| number reconciliations | `stats/decay_spearman_variants.py`, `stats/imputation_threshold_summary.py` | `data/decay_spearman_variants.tsv`, `data/imputation_threshold_summary.tsv` |

`stats/RELEASED_NUMBERS_NOTE.md` records, for every number where the manuscript
and the repository disagreed, which one the committed code produces and why.

## Guards

`revision_analyses.yml` will fail rather than quietly rewrite the record if:

* **any analysis input is still chimp-polarized.** The June 2026 polarization
  cutover was reverted on 2026-07-01 but left its regenerated outputs behind; that
  is what made the PAML ω directions contradict the manuscript for a month. The
  guard checks the last commit touching each input against the polarization commit
  list.
* **a headline number moves.** SD agreement 0.7957, SD interaction p = 0.000356,
  93 of 292 analysed, and FDFT1 / BLK ω in the inverted / direct clade respectively.
* **a tracked gzip stops decompressing** (checked before the simulation commit).
  A CI text-fixer once rewrote three committed `.csv.gz` files as if they were
  text; `fixer.yml` now refuses to touch binaries and verifies afterwards.

## What is not reproducible here

* **The All of Us PheWAS.** Individual-level data live in the AoU Researcher
  Workbench and cannot be exported. The imputation model, the tagging SNPs and
  every downstream summary are committed; the association step is not runnable
  outside the workbench.
* **The per-CDS PHYLIP alignments** (`phy_outputs.zip`, 1.2 GB) were pruned from
  the tree. `four_fold_pi.py` and `pin_pis.py` recover them from the git-LFS object
  when it is present, and otherwise run with `--from-table`, which regenerates the
  downstream tests exactly from the committed per-inversion tables.
