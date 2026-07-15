#!/bin/bash
#SBATCH --job-name=slco-svbyeye-fixenv
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=a100
set -euo pipefail
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye
export MAMBA_ROOT_PREFIX=$ART/env/mamba_root
echo "=== install missing SVbyEye deps (gggenes, BSgenome) into existing env ==="
micromamba install -y -p "$ENVP" -c conda-forge -c bioconda \
  r-gggenes bioconductor-bsgenome bioconductor-rtracklayer
echo "=== install SVbyEye (deps now present) ==="
"$ENVP/bin/Rscript" -e '
  options(repos=c(CRAN="https://cloud.r-project.org"))
  remotes::install_github("daewoooo/SVbyEye", upgrade="never", dependencies=FALSE)
  stopifnot(requireNamespace("SVbyEye", quietly=TRUE))
  suppressPackageStartupMessages(library(SVbyEye))
  cat("SVbyEye OK:", as.character(packageVersion("SVbyEye")), "\n")
  cat("plotMiro exported:", exists("plotMiro"), "\n")
'
echo "FIXENV_DONE"
