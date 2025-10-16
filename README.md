# Ferromic

Rust-accelerated population genetics toolkit for large variant datasets with batteries-included CLI workflows and Python bindings.

## Table of contents

- [Highlights](#highlights)
- [Quick start](#quick-start)
  - [Rust command-line pipeline](#rust-command-line-pipeline)
  - [Python API](#python-api)
- [Installation](#installation)
  - [Use the prebuilt binaries](#use-the-prebuilt-binaries)
  - [Build from source](#build-from-source)
  - [Install the Python wheel](#install-the-python-wheel)
- [Input requirements](#input-requirements)
  - [Regional configuration file](#regional-configuration-file)
  - [Optional masks and group definitions](#optional-masks-and-group-definitions)
- [Running analyses with `run_vcf`](#running-analyses-with-run_vcf)
  - [CLI options](#cli-options)
  - [Example end-to-end run](#example-end-to-end-run)
  - [Principal components and FST outputs](#principal-components-and-fst-outputs)
- [Output artefacts](#output-artefacts)
- [Additional binaries](#additional-binaries)
- [Phenome-wide association (PheWAS) pipeline](#phenome-wide-association-phewas-pipeline)
- [Project layout and helper scripts](#project-layout-and-helper-scripts)
- [Development](#development)
- [License](#license)

## Highlights

- **Purpose built for haplotype-aware studies.** Separates per-haplotype diversity metrics, supports inversion-aware sample groupings, and ships with Hudson and Weir & Cockerham FST estimators.
- **Designed for big cohorts.** Rayon-powered multithreading, streaming VCF readers, progress bars, and resumable temporary directories keep terabyte-scale runs responsive.
- **Rich output surface.** Generates region summaries, per-base FASTA-style tracks, PCA tables, PHYLIP files, and optional Hudson TSV exports ready for downstream notebooks.
- **First-class Python ergonomics.** A PyO3-powered module exposes the same core statistics to Python, NumPy, and pandas workflows without sacrificing performance.

## Quick start

### Rust command-line pipeline

1. **Prepare inputs**
   - Place bgzipped or plain-text VCFs for each chromosome in a directory.
   - Supply a reference FASTA and matching GTF/GFF annotation.
   - Describe regions of interest in a TSV file (see [Regional configuration file](#regional-configuration-file)).
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

The Python surface mirrors the Rust crate: Hudson-style populations, per-site diversity iterators, PCA utilities, and sequence-length helpers are available under the top-level `ferromic` namespace. The bindings favour "plain" Python collections—variants can be dictionaries, dataclasses, or any object exposing `position` and `genotypes`, while haplotypes accept tuples such as `(sample_index, "L")` or `(sample_index, 1)`.

#### High-level API surface

| Object | Description |
| --- | --- |
| `ferromic.Population` | Container with cached diversity metrics for a haplotype group; backs Hudson-style comparisons. |
| `ferromic.segregating_sites(variants)` | Count polymorphic sites for a cohort or region. |
| `ferromic.nucleotide_diversity(variants, haplotypes, sequence_length)` | Compute π with optional BED-style masks. |
| `ferromic.watterson_theta(segregating_sites, sample_count, sequence_length)` | Closed-form θ estimator mirroring the CLI output. |
| `ferromic.per_site_diversity(variants, haplotypes, region=None)` | Iterator over per-position π/θ values that underpins `per_site_diversity_output.falsta`. |
| `ferromic.wc_fst(...)` | Weir & Cockerham FST estimates with per-site and aggregate components. |
| `ferromic.hudson_fst(pop1, pop2)` / `hudson_dxy` | Hudson-style FST and D<sub>xy</sub> between arbitrary `Population` objects. |
| `ferromic.chromosome_pca(...)` family | Memory-aware PCA helpers that stream per-chromosome loadings, matching the CLI `--pca` artefacts. |
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

## Input requirements

- **VCF folder (`--vcf_folder`)** – one file per chromosome (plain or gzipped). Header validation enforces matching sample layouts across files.
- **Reference FASTA (`--reference`)** – used to reconstruct PHYLIP sequences and to mask non-callable positions.
- **GTF/GFF (`--gtf`)** – provides CDS definitions; overlapping transcripts trigger PHYLIP exports for both haplotype groups.
- **Region definition** – either a single `--region` (`chr:start-end`) or a multi-region TSV via `--config_file`.

### Regional configuration file

Tab-delimited TSV with a header containing seven metadata columns followed by one column per haplotype sample:

| Column | Description |
| --- | --- |
| `seqnames` | Chromosome identifier (with or without `chr`). |
| `start` / `end` | 1-based inclusive coordinates for the region window. |
| `POS` | Representative variant used for provenance. |
| `orig_ID` | Region identifier carried into outputs. |
| `verdict` | Manual or automated verdict flag. |
| `categ` | Category label for stratified summaries. |
| `sample…` | One column per sample containing phased genotypes such as `0|0`, `0|1`, or `1|1`. |

Values to the left/right of the `|` assign each haplotype to group 0 or 1. Suffixes like `_lowconf` persist in unfiltered counts but are removed from filtered analyses.

#### Coordinate conventions

Ferromic consumes several genomics formats and keeps their native coordinate systems:

- VCF positions are treated as **1-based inclusive**.
- BED masks/allow lists are **0-based half-open** intervals.
- TSV configuration files expect **1-based inclusive** coordinates.
- GTF/GFF annotations are interpreted as **1-based inclusive** when extracting CDS spans.

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

- **PCA** – `pca_per_chr_outputs/chr_<id>.tsv` hold per-chromosome coordinates with haplotype labels; `pca_results.tsv` aggregates global PCA.
- **Weir & Cockerham** – CSV columns prefixed with `haplotype_` cover overall FST, between/within population variance, and informative site counts.
- **Hudson** – Summary columns `hudson_fst_hap_group_0v1`, `hudson_dxy_hap_group_0v1`, and per-group π values are produced, with an optional `hudson_fst_results.tsv` listing every pairwise comparison.

## Output artefacts

- **Main CSV** – per-region statistics: raw/adjusted sequence lengths, segregating site counts, Watterson’s θ, nucleotide diversity π, inversion allele frequencies, and haplotype counts for both filtered and unfiltered tracks.
- **Per-site FASTA-style files** – `per_site_diversity_output.falsta` exposes site-level π/θ arrays, while `per_site_fst_output.falsta` stores Weir & Cockerham and Hudson components per base.
- **Hudson TSV (optional)** – When `--fst` is active, `hudson_fst_results.tsv` lists 0-based coordinates and FST components for each population comparison.
- **PHYLIP alignments** – For every transcript overlapping a region Ferromic writes `group_{0|1}_{transcript_id}_chr_<chr>_start_<start>_end_<end>_combined.phy` containing reference-adjusted CDS sequences with `_L`/`_R` haplotype suffixes.

## Additional binaries

| Binary | Purpose |
| --- | --- |
| `ferromic` | High-throughput VCF concatenator with chromosome-aware ordering, async writers, and Rayon chunking controls. |
| `vcf_merge` | Memory-aware VCF merge utility with optional RAM ceilings, mmap-assisted buffering, and per-chromosome progress readouts. |
| `run_vcf` | Primary Ferromic CLI that streams regions, emits diversity/FST summaries, PCA tables, per-base FASTA tracks, and PHYLIP CDS exports. |

The concatenation and merge utilities (`ferromic`, `vcf_merge`) share `--input <dir>` and `--output <file>` flags and are optimised for large cohorts through Rayon parallelism and Tokio-based async writers. The analysis driver `run_vcf` exposes the richer CLI documented in [Running analyses with `run_vcf`](#running-analyses-with-run_vcf).

## Phenome-wide association (PheWAS) pipeline

The `phewas/` directory ships a batteries-included association testing stack that mirrors the production workflow used to scan inversion calls against thousands of phenotypes:

- **Orchestration (`run.py`)** – coordinates Google BigQuery cohort pulls (optional), phenotype preprocessing, and stage-1 logistic score/LRT tests. A system monitor thread reports container CPU/RAM budgets while a global exception hook surfaces worker crashes immediately.
- **Resource governance (`pipes.py`)** – manages shared-memory budgets across multiprocessing pools, honours cgroup limits, exposes progress snapshots, and caches intermediate fit metadata to avoid redundant recalculations.
- **Modelling (`models.py`)** – implements penalised logistic and linear models with automatic fallbacks (ridge, Firth bias correction, bootstrap score tests) plus sequential stopping rules for rare traits.
- **Phenotype engineering (`pheno.py`, `categories.py`)** – loads callset metadata, derives case/control cohorts, and encodes ancestry-aware covariates and phenotype groupings.
- **Automation utilities (`run.py`, `testing.py`, `tests.py`, `test_setup.sh`)** – provide harnesses for dry-run validation, regression testing, and continuous-integration smoke checks.

Configuration is driven by a context dictionary (`ctx`) that records cache directories, population filters, minimum case/control thresholds per ancestry, and bootstrap budgets. Cached JSON manifests ensure that stale intermediate results are automatically invalidated when the cohort, cache tag, or inversion target changes.

## Project layout and helper scripts

- `src/` – Rust crate providing parsing, progress reporting, statistics, transcript handling, and CLI entry points.
- `scripts/` – Python utilities for downstream tasks (e.g., deduplication, dN/dS calculations, PHYLIP conversions).
- `stats/` – Exploratory analysis notebooks and plotting scripts for diversity, FST, PCA, and inversion studies.
- `data/` – Example metadata including `callset.tsv`, significant phenotypes, and support files for tutorials.
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
