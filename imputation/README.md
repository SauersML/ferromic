# Inversion dosage imputation

The inference pipeline applies published partial-least-squares models to SNP dosages.
It preserves each model's predictor order, aligns effect alleles, fills missing calls
with ancestry-specific means, and constrains predictions to the biological range 0–2.

## AoU v8 PheWAS cohort

Run from the repository root inside an All of Us Workbench runtime. Keep generated
matrices on the VM's local disk; the controlled dataset mount is GCSFuse-backed.

Create a small inference environment once. `--no-deps` deliberately preserves the
Workbench's coherent NumPy/SciPy stack while replacing its older scikit-learn and
adding the PLINK reader required here.

```bash
python3 -m venv --system-site-packages \
  /home/jupyter/aou-phewas/venv

/home/jupyter/aou-phewas/venv/bin/pip install \
  --only-binary=:all: \
  --no-deps \
  'scikit-learn==1.7.2' \
  'threadpoolctl==3.6.0' \
  bed-reader

source /home/jupyter/aou-phewas/venv/bin/activate
```

```bash
LOCAL=/home/jupyter/aou-phewas

V8=/home/jupyter/workspace/
V8+=vwb-aou-datasets-controlled/v8

ACAF="$V8/wgs/short_read/snpindel/"
ACAF+=acaf_threshold/plink_bed

ANCESTRY="$V8/wgs/short_read/snpindel/aux/"
ANCESTRY+=ancestry/echo_v4_r2.ancestry_preds.tsv

mkdir -p "$LOCAL"

python3 -m imputation.prepare_data_for_infer \
  --plink-dir "$ACAF" \
  --output-dir "$LOCAL/genotype_matrices" \
  --threads 4

python3 -m imputation.infer_dosage \
  --genotype-dir "$LOCAL/genotype_matrices" \
  --ancestry "$ANCESTRY" \
  --model-dir "$LOCAL/models" \
  --output "$PWD/imputed_inversion_dosages.tsv" \
  --threads 4
```

With no `--target` arguments, both commands use the canonical seven-inversion PheWAS
set in `imputation/targets.py`. Predictor preparation reads only the required variants
from chromosomes 4, 6, 8, 10, 12, and 17. It does not scan every genotype in each BED
and does not copy the chromosome shards.

Preparation writes one Fortran-order int8 matrix per model, `sample_ids.tsv`, and a
provenance report. It verifies identical participant order across all six chromosome
shards and estimates disk use before allocation. Inference requires at least 90% of a
model's predictors to have a call rate of at least 1%; lower coverage fails closed.

The trained models and SNP specifications are release assets:

<https://github.com/SauersML/ferromic/releases/tag/imputation-models-v1>

Both commands fetch only the seven requested model assets through the release manifest.

## Method

Models were trained on observed inversion dosages and SNP dosages inside each inversion
and within 50 kbp of its breakpoints. Synthetic diploid genomes constructed from phased
reference haplotypes expanded the observed haplotype combinations. Nested
cross-validation selected the number of PLS components and evaluated out-of-sample
performance without allowing a held-out sample's haplotypes into synthetic training
genomes.

The preparation command matches model predictors by chromosome, GRCh38 position, and
effect allele. Ambiguous or absent predictors remain explicitly missing. During
inference, missing genotypes are filled with per-ancestry predictor means; participants
without a recognized ancestry label receive global means. The output is a TSV with one
row per WGS participant and one column per inversion dosage.

Training is implemented in `linked.py`; release packaging is implemented in
`pack_models.py`. `benchmark_hsinv0284.py` applies the same preparation logic to the
experimentally genotyped 6q24.1 inversion.
