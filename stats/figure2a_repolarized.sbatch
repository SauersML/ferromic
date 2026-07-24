#!/bin/bash
#SBATCH --job-name=fig2a-repolarized
#SBATCH --partition=msismall
#SBATCH --account=hsiehph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/figure2a-%j.out
#SBATCH --error=logs/figure2a-%j.err

set -euo pipefail

module load python3/3.10.9_anaconda2023.03_libmamba
python -s stats/figure2a_repolarized.py \
  --output-csv data/output.csv \
  --inv-properties data/inv_properties.tsv \
  --polarity-json data/chimp_alignment_responses.json \
  --outdir results/figure2a_repolarized
