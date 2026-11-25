#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# No tracked source file over 2,000 lines. Files already over would be listed in
# scripts/file-size-baseline.txt and may only shrink; none exist yet.
set -euo pipefail
max=2000
bad=0
while IFS= read -r -d '' f; do
  n=$(wc -l < "$f")
  if [ "$n" -gt "$max" ] && ! grep -qxF "$f" scripts/file-size-baseline.txt 2>/dev/null; then
    echo "file-size: $f has $n lines (max $max)" >&2; bad=1
  fi
done < <(git ls-files -z -- '*.rs')
[ "$bad" -eq 0 ] && echo "file-size: OK"
exit "$bad"
