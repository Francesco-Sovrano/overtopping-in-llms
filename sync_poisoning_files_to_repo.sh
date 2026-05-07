#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
	echo "Usage: $0 /path/to/overtopping-phenomenology" >&2
	exit 1
fi

TARGET_REPO="$1"
MANIFEST="poisoning_github_files.txt"

if [[ ! -d "$TARGET_REPO" ]]; then
	echo "Target repo does not exist: $TARGET_REPO" >&2
	exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
	echo "Missing manifest: $MANIFEST" >&2
	exit 1
fi

while IFS= read -r relpath; do
	[[ -n "$relpath" ]] || continue
	[[ "$relpath" != \#* ]] || continue
	if [[ ! -f "$relpath" ]]; then
		echo "Missing source file: $relpath" >&2
		exit 1
	fi
	mkdir -p "$TARGET_REPO/$(dirname "$relpath")"
	cp "$relpath" "$TARGET_REPO/$relpath"
	echo "copied $relpath"
done < "$MANIFEST"

echo
echo "Copied poisoning experiment files into $TARGET_REPO"
echo "Next:"
echo "  cd $TARGET_REPO"
echo "  git status --short"
echo "  git add \$(cat $MANIFEST)"
