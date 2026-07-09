#!/bin/bash
#SBATCH --job-name=pav_dry
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=msismall
#SBATCH -o /projects/standard/hsiehph/sauer354/di/pav_run/pav_dry_%j.out
#SBATCH -e /projects/standard/hsiehph/sauer354/di/pav_run/pav_dry_%j.err
set -uo pipefail
PAV=/projects/standard/hsiehph/shared/software/pipelines/pav
ENV=/projects/standard/hsiehph/shared/conda_shared/envs/pav5
export PATH="$ENV/bin:$PAV:$PATH"
cd /projects/standard/hsiehph/sauer354/di/pav_run
echo "=== snakemake version ==="; $ENV/bin/snakemake --version
echo "=== DRY RUN ==="
$ENV/bin/snakemake -s $PAV/Snakefile --configfile config.json -n -p 2>&1 | tail -40
