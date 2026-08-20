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

### Stage 1 — fit the components

Stage the microarray PLINK trio locally first; streaming genotypes over the network
dominates the runtime of everything else.

```bash
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://vwb-aou-datasets-controlled/v8/microarray/plink/* .

# Autosomes minus every long-range LD region and every tested inversion (+/- 1 Mb).
python -m phewas.extra.within_ancestry_pca sites \
    --bim arrays.bim --out sites/include_sites.tsv

# One group at a time. Defaults are the production settings shown below.
python -m phewas.extra.within_ancestry_pca fit \
    --genotypes ./arrays \
    --sites sites/include_sites.tsv \
    --ancestry ancestry_preds.tsv \
    --cohort imputed_inversion_dosages.tsv \
    --related relatedness_flagged_samples.tsv \
    --group eur \
    --out-dir within_ancestry_pcs \
    --gnomon /path/to/gnomon/target/release/gnomon \
    --dosages imputed_inversion_dosages.tsv
```

The producer calls current `gnomon main` as a single indexed PLINK fit with 16
components, four worker threads, MAF ≥ 0.01, sample and variant missingness ≤ 0.05,
and LD normalization. Current main makes `--ld` safe for a biobank array by applying an
evenly spaced 100,000-marker budget and a 500 kbp physical window when no expert
override is supplied. Marker thinning and a physical-distance window are paired
deliberately: after thinning, a window measured in neighboring sites no longer has a
stable genomic meaning. The LD work is local rather than all-pairs, but it can still
dominate runtime because each local system grows with the number of retained markers in
that window; the marker budget keeps that density bounded. Current main also applies
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

```bash
for pop in eur afr amr eas sas; do
    python3 -m phewas.cli --pop-label "$pop"                                # control arm
    python3 -m phewas.cli --pop-label "$pop" --pc-source within-ancestry    # treatment arm
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
