# Ferromic Source Tree

The `src/` directory contains the Rust crate that powers the Ferromic command-line tools and Python bindings. Key binaries include:

- `run_vcf`: Streams variants from chromosome VCFs to generate diversity metrics, PCA tables, FST estimates, and PHYLIP exports.
- `ferromic`: Concatenates per-chromosome VCFs in coordinate order.
- `vcf_merge`: Merges VCF shards with memory-aware buffering.

## Example analysis run

Use the main analysis driver to reproduce a typical haplotype-aware diversity scan:

```bash
cargo run --release --bin run_vcf -- \
    --vcf_folder ../vcfs \
    --config_file ../variants_freeze4inv_sv_inv_hg38_processed_arbigent_filtered_manualDotplot_filtered_PAVgenAdded_withInvCategs_syncWithWH.fixedPH.simpleINV.mod.tsv \
    --mask_file ../hardmask.hg38.v4_acroANDsdOnly.over99.bed \
    --reference ../hg38.no_alt.fa \
    --gtf ../gencode.v47.basic.annotation.gtf \
    --fst
```

The command expects phased genotypes in the configuration TSV, a matching reference FASTA, and a GTF/GFF annotation. Outputs are written next to the working directory and mirror the CLI description in the top-level `README.md`.
