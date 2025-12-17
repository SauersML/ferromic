# Inversion polymorphism paper
- The imputation pipeline can be found in [/imputation](https://github.com/SauersML/ferromic/tree/main/imputation).
- The PheWAS pipeline can be found in [/phewas](https://github.com/SauersML/ferromic/tree/main/phewas).
- Assorted analysis and plotting scripts can be found within [/stats](https://github.com/SauersML/ferromic/tree/main/stats).
- The pipeline to run the PAML analysis can be found within [/cds](https://github.com/SauersML/ferromic/tree/main/cds)

---

# Ferromic

[![PyPI](https://img.shields.io/pypi/v/ferromic)](https://pypi.org/project/ferromic/) [![Build Status](https://github.com/SauersML/ferromic/actions/workflows/CI.yml/badge.svg)](https://github.com/SauersML/ferromic/actions/workflows/CI.yml) [![License](https://img.shields.io/crates/l/ferromic)](LICENSE.md)

Ferromic is a Rust-accelerated population genetics toolkit built for haplotype-aware studies on large variant cohorts. It offers batteries-included CLI workflows alongside polished Python bindings so the same core algorithms can be reused in notebooks and scripted pipelines.

## Highlights

- **Purpose built for haplotype-aware studies.** Separates per-haplotype diversity metrics, supports inversion-aware sample groupings, and ships with Hudson and Weir & Cockerham FST estimators.
- **Designed for big cohorts.** Rayon-powered multithreading, streaming VCF readers, progress bars, and resumable temporary directories keep terabyte-scale runs responsive.
- **Rich output surface.** Generates region summaries, per-base FASTA-style tracks, PCA tables, PHYLIP files, and optional Hudson TSV exports ready for downstream notebooks.
- **Python ergonomics.** A PyO3-powered module exposes the same core statistics to Python, NumPy, and pandas workflows without sacrificing performance.
- **Memory-aware dense matrices.** Detects ploidy, compresses genotypes into dense representations, and caches population summaries to accelerate repeated scans.

## Quick start

### Rust command-line pipeline

1. **Prepare inputs**
   - Place bgzipped or plain-text VCFs for each chromosome in a directory.
   - Supply a reference FASTA and matching GTF/GFF annotation.
   - Describe regions of interest in a TSV file (see [Regional configuration file](#regional-configuration-file)).
   - (Optional) Prepare mask or allow BEDs and an FST population map if you plan to enable `--fst`.
2. **Invoke the main driver**

   ```bash
   cargo run --release --bin run_vcf -- \
       --vcf_folder ./vcfs \
       --reference ./reference/hg38.no_alt.fa \
       --gtf ./reference/hg38.knownGene.gtf \
       --config_file ./regions.tsv \
       --mask_file ./hardmask.bed \
       --pca --fst
   ```

   The command streams each region, honours mask/allow lists, writes a CSV summary, and (with `--pca` or `--fst`) emits additional PCA and FST artefacts.
3. **Review results**
   - `output.csv` captures haplotype-specific diversity statistics.
   - `per_site_diversity_output.falsta` and `per_site_fst_output.falsta` contain base-wise tracks for plotting or heatmaps.
   - Optional PCA and Hudson tables land next to the main CSV.

### Python API

Install the wheel with `pip install ferromic`, then compute diversity statistics in-memory:

```python
import numpy as np
import ferromic as fm

genotypes = np.array([
    [[0, 0], [0, 1], [1, 1]],
    [[0, 1], [0, 0], [1, 1]],
], dtype=np.uint8)

population = fm.Population.from_numpy(
    "demo",
    genotypes=genotypes,
    positions=[101, 202],
    haplotypes=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    sequence_length=1000,
    sample_names=["sampleA", "sampleB", "sampleC"],
)

print("Segregating sites:", population.segregating_sites())
print("Nucleotide diversity:", population.nucleotide_diversity())

pca = fm.chromosome_pca(
    variants=[
        {"position": 101, "genotypes": [[0, 0], [0, 1], [1, 1]]},
        {"position": 202, "genotypes": [[0, 1], [0, 0], [1, 1]]},
    ],
    sample_names=["sampleA", "sampleB", "sampleC"],
)

print("PCA components shape:", pca.coordinates.shape)
```

The Python surface mirrors the Rust crate: Hudson-style populations, per-site diversity iterators, PCA utilities, and sequence-length helpers are available under the top-level `ferromic` namespace. The bindings favour "plain" Python collections—variants can be dictionaries, dataclasses, or any object exposing `position` and `genotypes`, while haplotypes accept tuples such as `(sample_index, "L")` or `(sample_index, 1)`. All heavy lifting happens in Rust, so interactive workflows retain native performance, and return values are rich Python objects with cached attributes (for example, `FstEstimate.value`, `FstEstimate.sum_a`, `FstEstimate.sum_b`, and `FstEstimate.sites`).

#### High-level API surface

| Object | Description |
| --- | --- |
| `ferromic.Population` | Container with cached diversity metrics for a haplotype group; backs Hudson-style comparisons and exposes `from_numpy(id, genotypes, positions, haplotypes, sequence_length, sample_names)` for ergonomic construction. |
| `ferromic.segregating_sites(variants)` | Count polymorphic sites for a cohort or region. |
| `ferromic.nucleotide_diversity(variants, haplotypes, sequence_length)` | Compute π with optional BED-style masks. |
| `ferromic.watterson_theta(segregating_sites, sample_count, sequence_length)` | Closed-form θ estimator mirroring the CLI output. |
| `ferromic.per_site_diversity(variants, haplotypes, region=None)` | Iterator over per-position π/θ values that underpins `per_site_diversity_output.falsta`. |
| `ferromic.wc_fst(...)` | Weir & Cockerham FST results returned as a `WcFstResult` object containing pairwise matrices and per-site components. |
| `ferromic.hudson_fst(pop1, pop2)` / `hudson_dxy` | Hudson-style FST and D<sub>xy</sub> between arbitrary `Population` objects, returned as structured `HudsonFstResult` instances. |
| `ferromic.chromosome_pca(...)` family | Memory-aware PCA helpers that stream per-chromosome loadings using Faer-backed SVD, matching the CLI `--pca` artefacts. |
| Utility helpers | Functions such as `adjusted_sequence_length` and `inversion_allele_frequency` mirror CLI adjustments for masked bases and inversion calls. |

Consult `src/pytests` for end-to-end regression suites that exercise PCA, Hudson, and Weir & Cockerham pipelines directly from Python.

## Installation

### Use the prebuilt binaries

Download the latest release assets or run the helper script:

```bash
curl -fsSL https://raw.githubusercontent.com/SauersML/ferromic/main/install.sh | bash
```

The script pulls platform-appropriate tarballs for `ferromic`, `vcf_stats`, and `vcf_merge`, expands them in-place, marks them executable, and prints `--help` summaries for each tool.

### Build from source

1. Install Rust nightly (Ferromic targets edition 2024 features):

   ```bash
   rustup toolchain install nightly
   rustup override set nightly
   ```

2. Clone and build the project:

   ```bash
   git clone https://github.com/SauersML/ferromic.git
   cd ferromic
   cargo build --release
   ```

   The compiled binaries live under `target/release/`.

   Use `cargo run --bin run_vcf -- --help` to confirm the toolchain is set up correctly.

Environment variable `RAMDISK_PATH` can be set to redirect temporary directories to a specific high-speed volume (defaults to `/dev/shm`).

### Install the Python wheel

```bash
pip install ferromic
```

To develop against the local checkout use [maturin](https://github.com/PyO3/maturin):

```bash
python -m pip install "maturin[patchelf]"
maturin develop --release
```

Set `PYO3_PYTHON` or pass `--python` to target a specific interpreter.

When the wheel is installed via `pip`, the Rust extensions are compiled in release mode by default. During development the `maturin develop` workflow above produces editable installs that stay in sync with local code changes.

## Input requirements

- **VCF folder (`--vcf_folder`)** – one file per chromosome (plain or gzipped). Header validation enforces matching sample layouts across files.
- **Reference FASTA (`--reference`)** – used to reconstruct PHYLIP sequences and to mask non-callable positions.
- **GTF/GFF (`--gtf`)** – provides CDS definitions; overlapping transcripts trigger PHYLIP exports for both haplotype groups.
- **Region definition** – either a single `--region` (`chr:start-end`) or a multi-region TSV via `--config_file`.

### Regional configuration file

Tab-delimited TSV with a header containing **seven metadata columns** (which must be present even if blank) followed immediately by one column per haplotype sample:

| Column | Description |
| --- | --- |
| `seqnames` | Chromosome identifier (with or without `chr`). |
| `start` | 1-based inclusive start coordinate for the region window. |
| `end` | 1-based inclusive end coordinate for the region window. |
| `POS` | Representative variant used for provenance. |
| `orig_ID` | Region identifier carried into outputs. |
| `verdict` | Manual or automated verdict flag. |
| `categ` | Category label for stratified summaries. |
| `sample…` | One column per sample containing phased genotypes such as `0|0`, `0|1`, or `1|1`. |

Values to the left/right of the `|` assign each haplotype to group 0 or 1. Suffixes like `_lowconf` persist in unfiltered counts but are removed from filtered analyses.

#### Coordinate conventions

Ferromic consumes several genomics formats and keeps their native coordinate systems:

- Input VCFs use **1-based inclusive** coordinates as defined in the VCF specification.
- Input BED masks/allow lists are **0-based half-open** intervals.
- Config TSV entries expect **1-based inclusive** coordinates for start/end/POS fields.
- GTF/GFF annotations are interpreted as **1-based inclusive** when extracting CDS spans.
- Outputs use **1-based inclusive** coordinates for CSV/TSV reports, and PHYLIP filenames encode `start`/`end` in the same 1-based inclusive system (for example, `start100_end200`).

### Optional masks and group definitions

- **Mask BED (`--mask_file`)** – 0-based half-open intervals to exclude.
- **Allow BED (`--allow_file`)** – 0-based half-open intervals to whitelist.
- **FST populations (`--fst_populations`)** – CSV where each row names a population followed by sample identifiers for Weir & Cockerham contrasts.

## Running analyses with `run_vcf`

### CLI options

| Flag | Required | Description |
| --- | --- | --- |
| `--vcf_folder <path>` | ✓ | Directory containing chromosome VCFs. |
| `--reference <path>` | ✓ | Reference genome FASTA. |
| `--gtf <path>` | ✓ | Gene annotation GTF/GFF. |
| `--chr <id>` | | Restrict processing to a single chromosome. |
| `--region <start-end>` | | Analyse one region instead of a TSV batch. |
| `--config_file <file>` | | TSV of regions/haplotypes (see above). |
| `--output_file <file>` | | Override the default `output.csv`. |
| `--min_gq <int>` | | Genotype quality threshold (default 30). |
| `--mask_file <bed>` | | Exclude intervals from all statistics. |
| `--allow_file <bed>` | | Only consider variants inside these intervals. |
| `--pca` | | Emit chromosome-level PCA TSVs for filtered haplotypes. |
| `--pca_components <int>` | | Number of principal components (default 10). |
| `--pca_output <file>` | | Combined PCA summary filename (default `pca_results.tsv`). |
| `--fst` | | Enable Hudson and Weir & Cockerham FST outputs. |
| `--fst_populations <file>` | | Optional CSV describing named populations for FST. |

### Example end-to-end run

```bash
run_vcf \
  --vcf_folder ./vcfs \
  --reference ./reference/hg38.no_alt.fa \
  --gtf ./reference/hg38.knownGene.gtf \
  --config_file ./regions.tsv \
  --mask_file ./hardmask.bed \
  --allow_file ./accessibility.bed \
  --output_file diversity_summary.csv \
  --min_gq 35 \
  --pca \
  --fst
```

On startup Ferromic prints a status box summarising version, CPU threads, and timestamps, then streams variants chromosome-by-chromosome. Temporary FASTA slices and PHYLIP files are staged in a RAM-backed directory when available.

### Principal components and FST outputs

- **PCA** – `pca_per_chr_outputs/chr_<id>.tsv` hold per-chromosome coordinates with haplotype labels; `pca_results.tsv` aggregates global PCA computed via SVD using the Faer linear algebra backend for high-performance CPU execution.
- **Weir & Cockerham** – CSV columns prefixed with `haplotype_` cover overall FST, between/within population variance, and informative site counts.
- **Hudson** – Summary columns `hudson_fst_hap_group_0v1`, `hudson_dxy_hap_group_0v1`, and per-group π values are produced, with an optional `hudson_fst_results.tsv.gz` listing every pairwise comparison.

## Output artefacts

- **Summary tables:**
  - `output.csv` – per-region statistics: raw/adjusted sequence lengths, segregating site counts, Watterson’s θ, nucleotide diversity π, inversion allele frequencies, and haplotype counts for both filtered and unfiltered tracks.
  - `phy_metadata.tsv` – index of generated PHYLIP files linking transcript IDs, gene names, genomic coordinates, and spliced CDS lengths to the corresponding alignment paths.
- **Track files (`.falsta`):**
  - `per_site_diversity_output.falsta` – per-base arrays for π and Watterson’s θ stored as FASTA-like tracks (headers such as `>per_site_diversity_pi`).
  - `per_site_fst_output.falsta` – per-base Weir & Cockerham and Hudson components with headers like `>hudson_pairwise_fst_hap_0v1_num` and `>wc_weighted_fst_denominator` to ease parsing and genome-browser visualisation.
- **Alignments:**
  - `*.phy.gz` – PHYLIP-formatted CDS alignments for every transcript overlapping a region, phased by haplotype group (`group_{0|1}_{transcript_id}_chr_<chr>_start_<start>_end_<end>_combined.phy`).
  - Optional `hudson_fst_results.tsv.gz` when `--fst` is active, listing Hudson components per comparison for the same 1-based inclusive coordinates.

## Additional binaries

| Binary | Purpose |
| --- | --- |
| `ferromic` | High-throughput VCF concatenator (distinct from the library) with chromosome-aware ordering, async writers, and Rayon chunking controls. |
| `vcf_merge` | Memory-aware VCF merge utility with optional RAM ceilings, mmap-assisted buffering, and per-chromosome progress readouts. |
| `run_vcf` | Primary Ferromic CLI that streams regions, emits diversity/FST summaries, PCA tables, per-base FASTA tracks, and PHYLIP CDS exports. |

The concatenation and merge utilities (`ferromic`, `vcf_merge`) share `--input <dir>` and `--output <file>` flags and are optimised for large cohorts through Rayon parallelism and Tokio-based async writers. The analysis driver `run_vcf` exposes the richer CLI documented in [Running analyses with `run_vcf`](#running-analyses-with-run_vcf).

## Project layout and helper scripts

- `src/` – Rust crate providing parsing, progress reporting, statistics, transcript handling, and CLI entry points.
- `scripts/` – Python utilities for downstream tasks (e.g., deduplication, dN/dS calculations, PHYLIP conversions).
- `stats/` – Exploratory analysis notebooks and plotting scripts for diversity, FST, PCA, and inversion studies.
- `data/` – Example metadata including `callset.tsv`, significant phenotypes, `inv_properties.tsv` (the current inversion metadata
  filename; legacy `inv_info.tsv` is no longer used), and support files for tutorials.
- `phewas/` – PheWAS modelling pipeline with helper modules and automation scripts.

## Development

1. Format and lint Rust code with `cargo fmt --all` and `cargo clippy --all-targets`.
2. Run the test suite:

   ```bash
   cargo test
   ```

3. Validate the Python bindings (optional):

   ```bash
   maturin develop --release
   pytest -q
   ```

Benchmarks such as `cargo bench --bench pca` quantify PCA throughput. Contributions are welcome via pull requests; please include reproducible commands and a short description of data requirements.

## License

Ferromic is released under the MIT License. See [LICENSE.md](LICENSE.md) for details.
