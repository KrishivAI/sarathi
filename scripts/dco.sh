#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# DCO: every non-merge commit in BASE..HEAD has a Signed-off-by trailer.   dco.sh BASE HEAD
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 BASE HEAD" >&2; exit 2; }
bad=0
shas=$(git rev-list --no-merges "$1..$2") || { echo "dco: cannot list commits in range $1..$2" >&2; exit 1; }
for sha in $shas; do
  git log -1 --format=%B "$sha" | grep -qE '^Signed-off-by: .+ <.+>$' \
    || { echo "dco: $(git rev-parse --short "$sha") has no Signed-off-by (use git commit -s)" >&2; bad=1; }
done
[ "$bad" -eq 0 ] && echo "dco: OK ($1..$2)"
exit "$bad"
