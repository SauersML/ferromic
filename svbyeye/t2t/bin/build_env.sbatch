#!/bin/bash
#SBATCH --job-name=slco-svbyeye-env
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=a100
set -euo pipefail
ART=${ART:?set ART to your svbyeye/t2t working directory (holds ref/ paf/ stageb/ plots/ pipeline/)}
ENVP=$ART/env/svbyeye
export MAMBA_ROOT_PREFIX=$ART/env/mamba_root
mkdir -p "$MAMBA_ROOT_PREFIX"
echo "=== creating micromamba env at $ENVP ==="
micromamba create -y -p "$ENVP" -c conda-forge -c bioconda \
  "minimap2=2.30" samtools \
  "r-base>=4.3" r-remotes r-biocmanager \
  r-ggplot2 r-dplyr r-data.table r-scales r-gtools r-wesanderson \
  r-rcolorbrewer r-tibble r-magrittr r-ggnewscale r-ggforce \
  r-randomcolor r-minpack.lm \
  bioconductor-genomicranges bioconductor-genomicalignments \
  bioconductor-biostrings bioconductor-rsamtools bioconductor-iranges \
  bioconductor-s4vectors bioconductor-genomeinfodb
echo "=== base env built ==="
"$ENVP/bin/minimap2" --version
"$ENVP/bin/samtools" --version | head -1
"$ENVP/bin/Rscript" -e 'cat(R.version.string,"\n")'
echo "=== installing SVbyEye from GitHub ==="
"$ENVP/bin/Rscript" -e '
  options(repos=c(CRAN="https://cloud.r-project.org"))
  remotes::install_github("daewoooo/SVbyEye", upgrade="never", dependencies=FALSE)
  stopifnot(requireNamespace("SVbyEye", quietly=TRUE))
  cat("SVbyEye OK:", as.character(packageVersion("SVbyEye")), "\n")
'
echo "ENV_DONE"
