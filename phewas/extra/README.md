# Supplemental PheWAS Outputs

This folder houses ancillary PheWAS figures and follow-up instructions. The lead image compares odds ratios derived from imputed inversion dosages with those obtained from tagging SNP dosages.

## Polygenic score control follow-up

To reproduce the polygenic score sensitivity analysis:

1. **Install the `gnomon` toolkit:**
   ```bash
   git clone https://github.com/SauersML/gnomon
   cd gnomon
   cargo build --release
   cd ..
   ```
2. **Download the microarray PLINK inputs:**
   ```bash
   gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://vwb-aou-datasets-controlled/v8/microarray/plink/* .
   ```
3. **Compute scores for the 17q21 region (example invocation):**
   ```bash
   ./gnomon/target/release/gnomon score "PGS004378 | chr17:45535159-46342045, PGS005198 | chr17:45535159-46342045, PGS004146 | chr17:45535159-46342045, PGS004229 | chr17:45535159-46342045, PGS004869 | chr17:45535159-46342045, PGS000507 | chr17:45535159-46342045" ./arrays
   ```
4. **Rename the output for downstream scripts:**
   ```bash
   cp arrays.sscore scores.tsv
   ```

Adjust the score list and genomic intervals as needed for other regions or panels.

## Ancestry-specific principal components

For the AoU v8 production run restricted to the 37 previously significant phenotypes,
use the checkpointed wrapper from the repository root:

```bash
bash phewas/run_aou_within_ancestry_hits.sh
```

The wrapper installs the latest published build from `gnomon` main at the start of
every invocation, so it never uses a commit-pinned or stale local executable.

It builds the seven required inversion dosages, downloads the array PLINK files once to
the VM's local disk, and then fits and analyzes EUR, AFR, EAS, AMR, SAS, and MID in
sequence. It runs only the within-ancestry-PC arm; the existing global-PC analyses are
not repeated. Completed dosage, array-download, PCA, and PheWAS outputs are validated
and reused on restart.

The PheWAS adjusts for the principal components published alongside the callset. Those
come from projecting participants onto a cross-ancestry reference, and the same file
supplies the ancestry labels, so the two are views of one computation. They separate
continental groups well and carry very little variance *inside* a group, which means they
cannot control fine-scale (within-continent) structure. This workflow fits components
separately within each ancestry group and re-runs the association models with them.

It is an additional sensitivity analysis. The pooled multi-ancestry PheWAS is unchanged
and remains the primary result: global components plus ancestry indicators are the right
instrument for between-ancestry structure.

### Why it must be stratified

Each group's components come from a separate eigendecomposition, on a separate sample
set, with a separate variant set and its own component count. `WPC3` in one group and
`WPC3` in another are different axes. Stacking them into a column named `PC3` would ask
one coefficient to describe a covariate whose meaning changes by row, so
`--pc-source within-ancestry` requires `--pop-label` and fails without it.

### Stage 1 — localize the array and fit the components

The controlled dataset and workspace mounts are GCSFuse-backed. The production wrapper
uses a resumable, sliced `gcloud storage cp` download to put the v8 array PLINK trio on
the VM's actual local disk. Every ancestry fit reuses that local dataset; PLINK does not
rewrite a second marker subset because Gnomon performs its own indexed marker selection.

```bash
LOCAL=/home/jupyter/aou-phewas
ARRAYS=/home/jupyter/aou-phewas/source/arrays

AUX=/home/jupyter/workspace/
AUX+=vwb-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux

ANCESTRY="$AUX/ancestry/echo_v4_r2.ancestry_preds.tsv"
RELATED="$AUX/relatedness/samples_relatedness_flagged_samples.tsv"

# Autosomes minus every long-range LD region and every tested inversion (+/- 1 Mb).
python -m phewas.extra.within_ancestry_pca sites \
    --bim "$ARRAYS.bim" \
    --out "$LOCAL/sites/include_sites.tsv"

# One group at a time. Defaults are the production settings shown below.
python -m phewas.extra.within_ancestry_pca fit \
    --genotypes "$ARRAYS" \
    --sites "$LOCAL/sites/include_sites.tsv" \
    --ancestry "$ANCESTRY" \
    --cohort "$PWD/imputed_inversion_dosages.tsv" \
    --related "$RELATED" \
    --group eur \
    --out-dir "$PWD/within_ancestry_pcs" \
    --gnomon gnomon \
    --dosages "$PWD/imputed_inversion_dosages.tsv"
```

The producer calls current `gnomon main` as a single indexed PLINK fit with 16
components, four worker threads, MAF ≥ 0.01, sample and variant missingness ≤ 0.05,
and LD normalization. The producer explicitly requests an evenly spaced 100,000-marker
budget and a 500 kbp physical window. Marker thinning and a physical-distance window
are paired deliberately: after thinning, a window measured in neighboring sites no
longer has a stable genomic meaning. The LD work is local rather than all-pairs, but it
can still dominate runtime because each local system grows with the number of retained
markers in that window; the marker budget keeps that density bounded. Current main also applies
`--threads 4` to both its fit-local and process-wide Rayon pools, so the producer no
longer needs its own CPU-affinity workaround.

Convergence is strict. The producer never passes `--allow-unconverged`, and current main
therefore refuses to create a model when the eigensolver does not meet its tolerance.
The producer independently requires `converged=true` in `hwe_summary.tsv` before it
writes PheWAS covariates. The JSON sidecar records the full command, gnomon version and
executable digest, resolved LD policy, and solver diagnostics.

`--cohort` takes any file whose first column is a participant id, so the imputed dosage
table can be passed directly: the components must be fit on the same people the
association models are fit on, or the two arms are not comparable.

Excluding the inversion loci from the variant list is the step that matters most. A
component that loaded on them would partly *be* the exposure, and adjusting for it would
regress away the signal being measured. Passing `--dosages` turns that from an assumption
into a check: the sidecar records the largest absolute correlation between any component
and any inversion dosage, which should be near zero.

Current main's native `--out PREFIX` writes every artifact independently of the genotype
path. The producer assigns a distinct prefix to each ancestry, so concurrent fits cannot
overwrite one another and the shared BED/BIM/FAM trio is neither copied nor symlinked.

Sample missingness QC is computed after the ancestry/cohort keep list, as in PLINK. If it
removes even one participant, the producer stops: silently accepting the reduced score
table would change the treatment-arm PheWAS cohort relative to its matched global-PC
run. Resolve such a mismatch upstream and use the same participant set in both arms.
There is no global-PC fallback for a requested within-ancestry run.

### Stage 2 — run both arms

This reruns only the 37 phenotypes that were significant in the uploaded pooled
analysis, not all 1,089 phenotypes.

```bash
for pop in afr amr eas eur mid sas; do
    python3 -m phewas.cli \
        --pop-label "$pop" \
        --pheno-file phewas/data/significant_phenotypes.txt \
        --min-cases-controls 100

    python3 -m phewas.cli \
        --pop-label "$pop" \
        --pc-source within-ancestry \
        --pheno-file phewas/data/significant_phenotypes.txt \
        --min-cases-controls 100
done
```

The components are read from `WITHIN_ANCESTRY_PCS_URI`, which defaults to the layout the
fitting step writes and is overridden with `FERROMIC_WITHIN_ANCESTRY_PCS_URI` when they
live somewhere else, such as a workspace bucket. It is configured like every other data
source the pipeline reads rather than as a command-line argument.

Run **both** arms. Stratifying costs power and changes the model on its own, so without
the control arm a shifted estimate cannot be attributed to the components rather than to
stratification. The two arms differ in exactly one thing, which is what makes the contrast
interpretable. Result filenames carry the population and the PC source, and the covariate
cache key includes both, so the arms cannot share cached covariates.

### Stage 3 — meta-analyse

```bash
python stats/phewas_within_ancestry_meta.py --results-dir <dir>
```

Inverse-variance meta-analysis across groups within each arm, with Cochran's Q, I², and a
DerSimonian-Laird random-effects estimate, plus the genomic control factor per inversion
per arm. The quantity that answers the stratification question is the ratio of the
within-ancestry meta-analysed effect to the pooled effect: near one means structure is not
driving the association.

Read effect-size concordance and the meta-analysis, not per-stratum significance. Power
falls sharply in the smaller groups, and the 17q21.31 inversion is close to
European-specific, so associations there will lose significance in other groups for
reasons that have nothing to do with confounding.
