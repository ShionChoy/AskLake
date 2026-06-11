#!/usr/bin/env bash
# Download IMDb (non-commercial) into ./data (gitignored).
# Raw data is NEVER committed.
set -euo pipefail

DATA_DIR="${1:-data}"
IMDB_DIR="$DATA_DIR/imdb/raw"
mkdir -p "$IMDB_DIR"

IMDB_BASE="https://datasets.imdbws.com"
IMDB_FILES=(
  title.basics.tsv.gz
  title.ratings.tsv.gz
  title.crew.tsv.gz
  name.basics.tsv.gz
  title.principals.tsv.gz
  title.akas.tsv.gz
  title.episode.tsv.gz
)
echo ">> IMDb -> $IMDB_DIR"
for f in "${IMDB_FILES[@]}"; do
  curl -fL --retry 3 -o "$IMDB_DIR/$f" "$IMDB_BASE/$f"
done

echo ">> Done. IMDb is non-commercial (no redistribution). Themes are built from current Wikipedia (CC BY-SA) at graph-build time."
