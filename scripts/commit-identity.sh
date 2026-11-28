#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Commit identity: every commit is authored and committed by a maintainer
# identity listed below, and every Co-authored-by / Signed-off-by trailer names
# one. A look-alike address belongs to someone else, and GitHub credits them.
# GitHub's own address is allowed as a committer only (merges made on the site).
#   commit-identity.sh BASE HEAD     CI: every commit in BASE..HEAD, plus
#                                    PR_TITLE and PR_BODY when those are set
#   commit-identity.sh --revs REV... pre-push hook: the commits `git rev-list`
#                                    lists for REV...
set -euo pipefail

authors='44522021+techgopal@users.noreply.github.com
343708768+shehatathe@users.noreply.github.com'
committers="$authors
noreply@github.com"

bad=0

# allowed EMAIL LIST: EMAIL is one whole line of LIST, ignoring case.
allowed() {
  grep -qixF -- "$1" <<< "$2"
}

report() {
  echo "commit-identity: $1" >&2
  bad=1
}

# check_trailers LABEL TEXT: every sign-off and co-author line names an allowed
# identity in the form `Name <email>`.
check_trailers() {
  local label=$1 line email
  local trailer='^[[:space:]]*(co-authored-by|signed-off-by)[[:space:]]*:'
  local form='^[^<>]*<([^<>[:space:]]+)>[[:space:]]*$'
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    email=$(sed -nE "s/$form/\1/p" <<< "$line")
    if [ -z "$email" ]; then
      report "$label: no readable email in '$line'"
    elif ! allowed "$email" "$authors"; then
      report "$label: trailer email '$email' is not an allowed identity"
    fi
  done <<< "$(grep -iE "$trailer" <<< "$2" || true)"
}

check_commit() {
  local sha=$1 short author committer message
  short=$(git rev-parse --short "$sha")
  author=$(git log -1 --no-show-signature --format=%ae "$sha")
  committer=$(git log -1 --no-show-signature --format=%ce "$sha")
  message=$(git log -1 --no-show-signature --format=%B "$sha")
  allowed "$author" "$authors" \
    || report "commit $short: author email '$author' is not an allowed identity"
  allowed "$committer" "$committers" \
    || report "commit $short: committer email '$committer' is not an allowed identity"
  check_trailers "commit $short" "$message"
}

if [ "${1:-}" = "--revs" ] && [ $# -ge 2 ]; then
  shift
  range="$*"
elif [ $# -eq 2 ]; then
  range="$1..$2"
  set -- "$range"
else
  echo "usage: $0 BASE HEAD | --revs REV..." >&2
  exit 2
fi

shas=$(git rev-list "$@") || {
  echo "commit-identity: cannot list commits in range $range" >&2
  exit 1
}
for sha in $shas; do
  check_commit "$sha"
done
[ -z "${PR_TITLE+set}" ] || check_trailers "PR title" "$PR_TITLE"
[ -z "${PR_BODY+set}" ] || check_trailers "PR body" "$PR_BODY"

if [ "$bad" -ne 0 ]; then
  echo "commit-identity: allowed: the identities listed in scripts/commit-identity.sh" >&2
  exit 1
fi
echo "commit-identity: OK ($range)"
