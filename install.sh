# /bin/bash
ARCH=$(uname -m)
OS=$(uname -s)

download_and_extract() {
  local BINARY_NAME=$1
  local URL=$2
  curl -L $URL -o "${BINARY_NAME}.tar.gz"
  tar -xzvf "${BINARY_NAME}.tar.gz"
  chmod +x $BINARY_NAME
}

# Detect architecture and OS. For ferromic, vcf_stats, and vcf_merge
if [[ "$ARCH" == "x86_64" ]]; then
  if [[ "$OS" == "Linux" ]]; then
    FERROMIC_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/ferromic-x86_64-unknown-linux-gnu.tar.gz"
    VCF_STATS_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_stats-x86_64-unknown-linux-gnu.tar.gz"
    VCF_MERGE_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_merge-x86_64-unknown-linux-gnu.tar.gz"
  elif [[ "$OS" == "Darwin" ]]; then
    FERROMIC_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/ferromic-x86_64-apple-darwin.tar.gz"
    VCF_STATS_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_stats-x86_64-apple-darwin.tar.gz"
    VCF_MERGE_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_merge-x86_64-apple-darwin.tar.gz"
  fi
elif [[ "$ARCH" == "aarch64" ]]; then
  if [[ "$OS" == "Linux" ]]; then
    FERROMIC_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/ferromic-aarch64-unknown-linux-gnu.tar.gz"
    VCF_STATS_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_stats-aarch64-unknown-linux-gnu.tar.gz"
    VCF_MERGE_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_merge-aarch64-unknown-linux-gnu.tar.gz"
  elif [[ "$OS" == "Darwin" ]]; then
    FERROMIC_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/ferromic-aarch64-apple-darwin.tar.gz"
    VCF_STATS_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_stats-aarch64-apple-darwin.tar.gz"
    VCF_MERGE_URL="https://github.com/SauersML/ferromic/releases/download/v0.0.2/vcf_merge-aarch64-apple-darwin.tar.gz"
  fi
else
  echo "Unsupported architecture: $ARCH"
  exit 1
fi

# Download and extract ferromic, vcf_stats, and vcf_merge
download_and_extract "ferromic" $FERROMIC_URL
download_and_extract "vcf_stats" $VCF_STATS_URL
download_and_extract "vcf_merge" $VCF_MERGE_URL

# Run binaries with --help
./ferromic --help
./vcf_stats --help
./vcf_merge --help
