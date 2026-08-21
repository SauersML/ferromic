#!/bin/bash -l
#SBATCH --job-name=refsim-setup
#SBATCH --account=hsiehph
#SBATCH --partition=sioux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=logs/refsim_setup_%j.out

set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}
cd "$ROOT"
mkdir -p logs

module load python3/3.12.4_anaconda2024.06-1_libmamba
python3 -m venv env
env/bin/python -m pip install --upgrade pip
env/bin/python -m pip install --requirement requirements.txt

env/bin/python - <<'PY'
import Bio
import msprime
import numpy
import pandas
import scipy
import sklearn

print("biopython", Bio.__version__)
print("msprime", msprime.__version__)
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
print("scikit-learn", sklearn.__version__)
PY
module load iqtree2/2.1.2-gcc-13.1.0-xsnl63s
IQTREE_BIN=$(command -v iqtree2)
test "$(sha256sum "$IQTREE_BIN" | cut -d' ' -f1)" = \
  81a16f2958e63d25cf6d67e2b64222b2ba24b34aa76069a6225a2df36ce166a2
"$IQTREE_BIN" --version | head -1
