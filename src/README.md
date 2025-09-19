```
set -euo pipefail

TARGET="./src"
echo "[DEBUG] PWD: $(pwd)"
echo "[DEBUG] Target directory: $TARGET"
mkdir -p "$TARGET"

echo "[DEBUG] Truncating existing *.rs under $TARGET (if any)..."
if find "$TARGET" -type f -name '*.rs' -print -quit >/dev/null; then
  while IFS= read -r -d '' f; do
    : > "$f"
    echo "[DEBUG] truncated: $f"
  done < <(find "$TARGET" -type f -name '*.rs' -print0)
else
  echo "[INFO] No existing *.rs files found under $TARGET to truncate."
fi

echo "[DEBUG] Removing existing *.rs under $TARGET (if any)..."
if find "$TARGET" -type f -name '*.rs' -print -quit >/dev/null; then
  while IFS= read -r -d '' f; do
    rm -- "$f"
    echo "[DEBUG] removed: $f"
  done < <(find "$TARGET" -type f -name '*.rs' -print0)
else
  echo "[INFO] No existing *.rs files found under $TARGET to remove."
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "[DEBUG] Using temp dir: $TMP"

TARBALL_URL='https://api.github.com/repos/SauersML/ferromic/tarball/main'
echo "[DEBUG] Downloading tarball: $TARBALL_URL"
curl -fL "$TARBALL_URL" -o "$TMP/repo.tar.gz"
echo "[DEBUG] Tarball downloaded: $(du -h "$TMP/repo.tar.gz" | cut -f1)"

echo "[DEBUG] Extracting tarball..."
tar -xzf "$TMP/repo.tar.gz" -C "$TMP"

# The tarball unpacks to a single top-level folder; find the root src/ there.
SRCDIR="$(find "$TMP" -mindepth 2 -maxdepth 2 -type d -name src -print -quit || true)"
if [ -z "${SRCDIR:-}" ]; then
  echo "[ERROR] Could not locate top-level src/ directory in the ferromic repo tarball."
  echo "[ERROR] Searched under: $TMP"
  exit 1
fi
echo "[DEBUG] Found repo src dir: $SRCDIR"

echo "[DEBUG] Searching for .rs files under repo src/..."
FOUND=0
COPIED=0
SKIPPED=0

# Copy only .rs files, preserving subdirectory structure under ./src.
while IFS= read -r -d '' FILE; do
  FOUND=$((FOUND+1))
  REL="${FILE#$SRCDIR/}"               # path relative to repo's src/
  DEST="$TARGET/$REL"
  DESTDIR="$(dirname "$DEST")"
  mkdir -p "$DESTDIR"

  if [ -e "$DEST" ]; then
    echo "[SKIP] $DEST already exists; not overwriting."
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  cp "$FILE" "$DEST"
  chmod 0644 "$DEST"
  echo "[OK] Copied: $DEST (from: $FILE)"
  COPIED=$((COPIED+1))
done < <(find "$SRCDIR" -type f -name '*.rs' -print0)

echo "[DEBUG] .rs files found under repo src/: $FOUND"
echo "[DEBUG] .rs files copied to $TARGET: $COPIED"
echo "[DEBUG] .rs files skipped (already existed): $SKIPPED"

echo "[DEBUG] Final tree of $TARGET:"
# 'tree' may not exist in minimal containers; fall back to find.
if command -v tree >/dev/null 2>&1; then
  tree "$TARGET" || true
else
  find "$TARGET" -print | sed "s|^|[TREE] |"
fi
```
