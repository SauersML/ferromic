#!/bin/bash
set -euo pipefail

ARCH=$(uname -m)
OS=$(uname -s)

# Release tag to install. The published release assets live under this tag; override with
# e.g. `VERSION=v0.0.2 bash install.sh`. Keep this in sync with the tag that actually has
# uploaded binary assets (NOT necessarily the in-development Cargo version).
VERSION="${VERSION:-v0.0.2}"
BASE_URL="https://github.com/SauersML/ferromic/releases/download/${VERSION}"

# Binaries to install. These must be real Cargo binary targets (see [[bin]] in Cargo.toml):
# ferromic and vcf_merge. (The old script also fetched a non-existent `vcf_stats` target.)
BINARIES=("ferromic" "vcf_merge")

download_and_extract() {
  local BINARY_NAME=$1
  local URL=$2
  curl -fL "$URL" -o "${BINARY_NAME}.tar.gz"
  tar -xzvf "${BINARY_NAME}.tar.gz"
  chmod +x "$BINARY_NAME"
}

# Detect architecture and OS to choose the right release artifact triple.
if [[ "$ARCH" == "x86_64" ]]; then
  if [[ "$OS" == "Linux" ]]; then
    TRIPLE="x86_64-unknown-linux-gnu"
  elif [[ "$OS" == "Darwin" ]]; then
    TRIPLE="x86_64-apple-darwin"
  fi
elif [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  if [[ "$OS" == "Linux" ]]; then
    TRIPLE="aarch64-unknown-linux-gnu"
  elif [[ "$OS" == "Darwin" ]]; then
    TRIPLE="aarch64-apple-darwin"
  fi
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

if [[ -z "${TRIPLE:-}" ]]; then
  echo "Unsupported OS/architecture combination: $OS/$ARCH"
  exit 1
fi

for BIN in "${BINARIES[@]}"; do
  download_and_extract "$BIN" "${BASE_URL}/${BIN}-${TRIPLE}.tar.gz"
done

# Sanity check: run each installed binary with --help.
for BIN in "${BINARIES[@]}"; do
  "./${BIN}" --help
done
