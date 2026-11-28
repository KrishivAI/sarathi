#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Tests scripts/commit-identity.sh and .githooks/pre-push on throwaway repositories.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
check="$root/scripts/commit-identity.sh"
hook="$root/.githooks/pre-push"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE PR_TITLE PR_BODY
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1

one='44522021+techgopal@users.noreply.github.com'
two='343708768+shehatathe@users.noreply.github.com'
web='noreply@github.com'
other='gopal@users.noreply.github.com'

git init -q -b main "$tmp/repo"
mkdir "$tmp/repo/scripts"
cp "$check" "$tmp/repo/scripts/" 2>/dev/null || true
cd "$tmp/repo"

# commit AUTHOR_EMAIL COMMITTER_EMAIL MESSAGE: an empty commit; prints its id.
commit() {
  GIT_AUTHOR_NAME=a GIT_AUTHOR_EMAIL=$1 GIT_COMMITTER_NAME=c GIT_COMMITTER_EMAIL=$2 \
    git commit -q --allow-empty -m "$3"
  git rev-parse HEAD
}

failed=0
# expect STATUS TEXT NAME COMMAND...: COMMAND ends with STATUS and prints TEXT.
expect() {
  local want=$1 text=$2 name=$3 out got=0
  shift 3
  out=$("$@" 2>&1) || got=$?
  if [ "$got" -eq "$want" ] && grep -qF -- "$text" <<< "$out"; then
    echo "ok: $name"
  else
    echo "FAIL: $name: status $got (want $want), want text '$text', got:" >&2
    echo "$out" >&2
    failed=1
  fi
}

base=$(commit "$one" "$one" base)

# case_commit NAME STATUS TEXT AUTHOR COMMITTER MESSAGE: one commit on top of base.
case_commit() {
  local head
  git reset -q --hard "$base"
  head=$(commit "$4" "$5" "$6")
  [ "$2" -eq 0 ] || expect "$2" "$(git rev-parse --short "$head")" "$1 (names the commit)" "$check" "$base" "$head"
  expect "$2" "$3" "$1" "$check" "$base" "$head"
}

case_commit "first identity" 0 "commit-identity: OK" "$one" "$one" "work

Signed-off-by: a <$one>"
case_commit "second identity with a co-author" 0 "commit-identity: OK" "$two" "$two" "work

Co-authored-by: a <$one>
Signed-off-by: b <$two>"
case_commit "web-flow committer" 0 "commit-identity: OK" "$one" "$web" "work"
case_commit "email case does not matter" 0 "commit-identity: OK" "44522021+TechGopal@users.noreply.github.com" "$one" "work"
case_commit "bad author" 1 "author email '$other'" "$other" "$one" "work"
case_commit "bad committer" 1 "committer email '$other'" "$one" "$other" "work"
case_commit "web-flow is not an author" 1 "author email '$web'" "$web" "$web" "work"
case_commit "bad co-author" 1 "trailer email '$other'" "$one" "$one" "work

Co-authored-by: x <$other>"
case_commit "bad sign-off, odd case and indent" 1 "trailer email '$other'" "$one" "$one" "work

  SIGNED-OFF-BY : x <$other>"
case_commit "web-flow is not a trailer identity" 1 "trailer email '$web'" "$one" "$one" "work

Co-authored-by: x <$web>"
case_commit "trailer without an email" 1 "no readable email" "$one" "$one" "work

Co-authored-by: somebody"
case_commit "allowed address inside a longer one" 1 "trailer email 'x$one'" "$one" "$one" "work

Signed-off-by: x <x$one>"
case_commit "two addresses on one trailer" 1 "no readable email" "$one" "$one" "work

Co-authored-by: x <$other> <$one>"

git reset -q --hard "$base"
bad=$(commit "$other" "$one" "bad, below a good one")
good=$(commit "$one" "$one" good)
expect 1 "$(git rev-parse --short "$bad")" "a bad commit below the head" "$check" "$base" "$good"
expect 0 "commit-identity: OK" "commits outside the range are not read" "$check" "$bad" "$good"
expect 1 "cannot list commits" "unknown range" "$check" "$base" 0123456789012345678901234567890123456789

git reset -q --hard "$base"
head=$(commit "$one" "$one" good)
expect 1 "PR body: trailer email '$other'" "bad trailer in the PR body" \
  env PR_BODY="text"$'\r\n\r\n'"Co-authored-by: x <$other>"$'\r' "$check" "$base" "$head"
expect 1 "PR title: trailer email '$other'" "bad trailer in the PR title" \
  env PR_TITLE="Signed-off-by: x <$other>" "$check" "$base" "$head"
expect 0 "commit-identity: OK" "good trailer with CRLF in the PR body" \
  env PR_BODY="Co-authored-by: a <$one>"$'\r' "$check" "$base" "$head"

# The pre-push hook gets one "local-ref local-id remote-ref remote-id" line per ref on stdin.
zero=0000000000000000000000000000000000000000
push() { printf 'refs/heads/b %s refs/heads/b %s\n' "$1" "$2" | "$hook" origin unused; }

git reset -q --hard "$base"
good=$(commit "$one" "$one" good)
bad=$(commit "$other" "$one" bad)
expect 0 "commit-identity: OK" "push: new branch, good commits" push "$good" "$zero"
expect 1 "author email '$other'" "push: new branch with a bad commit" push "$bad" "$zero"
expect 1 "author email '$other'" "push: update with a bad commit" push "$bad" "$good"
expect 1 "author email '$other'" "push: remote id not present locally" push "$bad" 0123456789012345678901234567890123456789
expect 0 "" "push: deleting a branch" push "$zero" "$bad"
git update-ref refs/remotes/origin/old "$bad"
after=$(commit "$one" "$one" after)
expect 0 "commit-identity: OK" "push: commits the remote already has are not read" push "$after" "$zero"

[ "$failed" -eq 0 ] && echo "commit-identity-test: all passed"
exit "$failed"
