#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Every tracked .rs, .sh and .py file carries an SPDX-License-Identifier: Apache-2.0 line in its first 5 lines.
set -euo pipefail
bad=0
while IFS= read -r -d '' f; do
  head -n 5 "$f" | grep -q 'SPDX-License-Identifier: Apache-2.0' || { echo "spdx: missing header: $f" >&2; bad=1; }
done < <(git ls-files -z -- '*.rs' '*.sh' '*.py')
[ "$bad" -eq 0 ] && echo "spdx: OK"
exit "$bad"
