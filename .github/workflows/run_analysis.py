name: run-analysis

on:
  workflow_dispatch:

jobs:
  fetch-and-head:
    runs-on: ubuntu-latest
    steps:
      - name: Download files over HTTPS and print head
        shell: bash
        run: |
          set -euo pipefail

          base='https://sharedspace.s3.msi.umn.edu'

          # Each file path is specified explicitly
          mapfile -t URLS <<'EOF'
          https://sharedspace.s3.msi.umn.edu/public_internet/variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/pairwise_glm_contrasts_fdr.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/pairwise_results_fdr.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/paml_results.checkpoint.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/perm_identical_pairs.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/perm_pairwise_identity.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/phy_metadata.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/region_identical_proportions.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/category_means_at_mean_covariates.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/category_summary.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_emm.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_emm_adjusted.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_emm_nocov.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_identical_proportions.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_pairwise.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_pairwise_adjusted.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/cds_pairwise_nocov.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/fixed_diff_summary.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/gene_direct_inverted.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/gene_inversion_direct_inverted.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/gene_inversion_fixed_differences.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/gene_inversion_permutation.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/glm_category_coefs.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/hudson_fst_results.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inv_info.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inversion_fst_estimates.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inversion_level_counts.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inversion_level_medians.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/kruskal_result.tsv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/fst_tests_summary.csv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inversion_statistical_results.csv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/inv_info.csv
          https://sharedspace.s3.msi.umn.edu/public_internet/ferromic/output.csv
          EOF

          mkdir -p downloaded
          : > filelist.txt

          for url in "${URLS[@]}"; do
            rel="${url#${base}/}"
            out="downloaded/${rel}"
            mkdir -p "$(dirname "$out")"
            echo "Fetching $url"
            curl -fSL --retry 3 "$url" -o "$out"
            echo "$out" >> filelist.txt
          done

          while IFS= read -r f; do
            echo "===== HEAD: $f ====="
            head -n 10 "$f"
            echo
          done < filelist.txt
