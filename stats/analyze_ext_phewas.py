"""Design specification for external-PheWAS replication matching.

This module is intentionally documentation-only: it records how a row in
``data/phewas_results.tsv`` is matched to the corresponding external summary-statistics
row(s) in ``data/aggregated_phenotype_results.tsv``. The executable implementation of
this matching lives in ``stats/phewas_ext_replicate.py``; keep the two in sync.

Previously this file began with raw, un-commented prose and pasted TSV tables, so it
raised ``SyntaxError`` on import (a reproducibility defect). The spec is preserved below
as a module docstring so the file is valid Python and importable.

Inputs
------
* aggregated external PheWAS (UK-Biobank-style), columns:
  ``phenostring  phenocode  chrom  pos  ref  alt  rsids  nearest_genes  consequence
  pval  beta  sebeta  af  case_af  control_af  tstat``
  https://raw.githubusercontent.com/SauersML/ferromic/refs/heads/main/data/aggregated_phenotype_results.tsv
* our PheWAS results: ``data/phewas_results.tsv`` (keyed by ``Phenotype`` and ``Inversion``).
* phenotype name mapping: ``data/mappings_final.tsv``
  (``Source_Phenotype  Has_Good_Match  Best_Match  All_Matches  Reasoning  Source_ICD10``).
* inversion -> tag-SNP coordinate mapping: ``data/best_tagging_snps_qvalues.tsv``
  (provides the hg38 ``chr:pos`` plus ``REF``/``ALT`` for each inversion region).

Matching procedure
------------------
1. For each ``phewas_results`` row, look up its ``Phenotype`` in ``mappings_final`` via an
   exact match on ``Source_Phenotype``; warn if ``Has_Good_Match`` is not ``True``.
   The candidate external phenostrings are in ``All_Matches`` (semicolon-separated).
2. Map the row's ``Inversion`` to an hg38 ``chr:pos`` (and ``REF``/``ALT``) via
   ``best_tagging_snps_qvalues`` (``hg38`` column, format ``chr:pos``).
3. Exact-match ``chrom``/``pos`` of the hg38 coordinate against
   ``aggregated_phenotype_results`` to find the external row(s); then verify ``REF``/``ALT``
   agree with the external ``ref``/``alt`` and warn on mismatch.
4. Restrict the candidate external rows to those whose ``phenostring`` is in the mapped
   ``All_Matches`` set, and record which mapped phenotype is the ``Best_Match``.

Notes
-----
A single ``phewas_results`` row may match one or several external rows. The matching keys
are deliberately exact (coordinate + REF/ALT and curated phenotype name mapping) to avoid
spurious replication claims.
"""

# Documentation-only module; see stats/phewas_ext_replicate.py for the implementation.
