#!/usr/bin/env bash
# Fetch the real-documentation corpus used by data/golden/fastapi.jsonl.
#
# The toy corpus is 7 documents and 16 chunks, which is too small for a
# retrieval number to mean much: with 16 candidates, retrieval barely has to
# discriminate. This pulls the FastAPI documentation (MIT) - 138 files, 673
# chunks - so the same harness can be run against a corpus that is 42x larger
# and written by people who were not thinking about this evaluation.
#
# Not vendored into the repo: it is someone else's documentation, it changes,
# and a shallow sparse clone is a few seconds.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/corpus-real"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "cloning fastapi docs..."
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git "$TMP/fastapi" >/dev/null 2>&1
git -C "$TMP/fastapi" sparse-checkout set docs/en/docs >/dev/null 2>&1

mkdir -p "$DEST"

# Flatten by path, not by basename. The docs tree has index.md in fourteen
# directories plus duplicate first-steps.md, middleware.md and websockets.md, so
# a plain `cp` into one folder silently drops 17 of 155 files - an 11% smaller
# corpus, with no error to notice.
SRC="$TMP/fastapi/docs/en/docs"
find "$SRC" -name '*.md' | while read -r file; do
  rel="${file#"$SRC"/}"
  cp "$file" "$DEST/${rel//\//__}"
done

found=$(find "$SRC" -name '*.md' | wc -l)
wrote=$(ls "$DEST" | wc -l)
echo "wrote $wrote files to data/corpus-real (found $found)"
if [ "$found" -ne "$wrote" ]; then
  echo "WARNING: $((found - wrote)) files did not survive the copy" >&2
fi
echo
echo "evaluate against it with:"
echo "  python -m eval.run_eval --golden data/golden/fastapi.jsonl --corpus data/corpus-real"
