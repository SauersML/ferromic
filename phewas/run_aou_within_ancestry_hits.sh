#!/usr/bin/env bash
set -euo pipefail

readonly REPO="/home/jupyter/repos/ferromic"
readonly LOCAL="/home/jupyter/aou-phewas"
readonly VENV="$LOCAL/venv"
readonly GNOMON_INSTALLER="https://raw.githubusercontent.com/"\
"SauersML/gnomon/main/install.sh"

cd "$REPO"
mkdir -p "$LOCAL"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/pip" install \
  --disable-pip-version-check \
  --only-binary=:all: \
  --no-deps \
  scikit-learn==1.7.2 \
  threadpoolctl==3.6.0 \
  bed-reader==1.1.0

curl -fsSL "$GNOMON_INSTALLER" | bash -s -- --binary gnomon
export PATH="$HOME/.local/bin:$PATH"
hash -r

exec "$VENV/bin/python" \
  -m phewas.aou_within_ancestry_hits
