#!/usr/bin/env bash
# Keeps each fork (mistral.rs and candle, submodules at a pinned upstream ref) at its pin plus the series in
# patches/, one patch per body of work. The fork trees are the working copy: Oscar's changes live there as
# uncommitted edits and cargo builds from there. patches/ is the only tracked copy, so an edit is durable once
# --sync has written it there, and setup never discards edits of yours unless --reset asks it to.
#
#   ./setup.sh           apply patches/ after a clone or a pull; a tree that already holds it is left alone
#   ./setup.sh --check   non-zero unless every tree holds exactly patches/ at the pinned ref; says what differs
#   ./setup.sh --sync    write the trees' edits into patches/ for the commit (the pre-commit hook insists)
#   ./setup.sh --reset   discard the trees, edits of yours included, and apply patches/ afresh
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
FORKS=("third_party/oscar-mistral.rs:mistral.rs" "third_party/oscar-candle:candle")
GEN="python3 $ROOT/scripts/make_patch_series.py"

case "${1-}" in
  "") mode=apply ;;
  --check|--sync|--reset) [ $# = 1 ] && mode=${1#--} ;;
esac
[ -n "${mode-}" ] || { echo "usage: $0 [--check | --sync | --reset]" >&2; exit 2; }

fail(){ echo "setup: $*" >&2; exit 1; }
# The tree holds exactly the series in $1: the generator rebuilds a series from the tree and compares.
holds(){ [ -d "$1" ] && $GEN --check --fork "$name" --against "$1" >/dev/null 2>&1; }
dirty(){ [ -n "$(git -C "$ROOT/$dir" status --porcelain --untracked-files=all 2>/dev/null | head -1)" ]; }
applies_to(){ sed -n 's/^Applies to: //p' "$1"/*.patch | head -1; }
# Back to the pin, the index included (`apply --3way` stages what it applies).
discard(){ git -C "$ROOT/$dir" reset -q --hard && git -C "$ROOT/$dir" clean -fdq; }
# A copy of the series in $1 kept in the fork's git directory: what this tree holds, so a series a pull changed
# is told from edits of yours.
record(){ rm -rf "$applied"; mkdir -p "$applied"; cp -R "$1"/. "$applied"/; }

# Every patch of $1 in order, `--3way` so a moved pin is a conflict rather than a refusal; git's output is shown
# for the failing patch alone (it is chatty about fallbacks and whitespace when everything lands).
apply_series(){
  local p out
  for p in "$1"/*.patch; do
    out=$(git -C "$ROOT/$dir" apply --3way "$p" 2>&1) && { echo "  applied $(basename "$p")"; continue; }
    echo "$out" >&2
    echo "setup: $name: $(basename "$p") did not apply at ${pin:0:12}" >&2
    return 1
  done
}

if [ "$mode" = sync ]; then
  $GEN -q || fail "patches/ not written"
  git -C "$ROOT" status --short patches/ | sed 's/^/  /'
  echo "setup: commit patches/ to keep the work"
  exit 0
fi

status=0
for entry in "${FORKS[@]}"; do
  dir=${entry%%:*}; name=${entry##*:}
  patchdir=$ROOT/patches/$name
  [ -e "$ROOT/$dir/.git" ] || fail "$dir is not a checkout; run: git submodule update --init"
  applied=$(git -C "$ROOT/$dir" rev-parse --absolute-git-dir 2>/dev/null)/oscar-patched.d
  pin=$(git -C "$ROOT/$dir" rev-parse HEAD 2>/dev/null) || fail "cannot read $dir HEAD"
  # The series is cut against one upstream ref, named in every patch's header; oscar records the same ref as the
  # submodule's commit.
  want=$(applies_to "$patchdir")
  [ -n "$want" ] || fail "patches/$name has no patch with an 'Applies to:' header"
  wantsha=$(git -C "$ROOT/$dir" rev-parse "$want^{commit}" 2>/dev/null) || wantsha=$want
  recorded=$(git -C "$ROOT" rev-parse -q --verify "HEAD:$dir" 2>/dev/null)
  [ "$pin" = "$wantsha" ] || [ "$recorded" = "$wantsha" ] \
    || fail "$dir is at ${pin:0:12}, but patches/$name applies to $want, which oscar does not record for $dir either.
       If the pin moved on purpose, re-cut the series against it: scripts/make_patch_series.py --verify"

  if [ "$mode" = check ]; then
    if [ "$pin" != "$wantsha" ]; then
      echo "setup: $dir is at ${pin:0:12}, patches/$name applies to $want; run ./setup.sh" >&2; status=1
    elif ! holds "$patchdir"; then
      $GEN --check --fork "$name" >&2
      if holds "$applied"; then
        echo "setup: $dir holds the series applied before; patches/$name has changed since; run ./setup.sh" >&2
      else
        echo "setup: $dir has edits that are not in patches/$name: ./setup.sh --sync writes them there, ./setup.sh --reset discards them" >&2
      fi
      status=1
    fi
    continue
  fi
  if [ "$mode" = apply ] && [ "$pin" = "$wantsha" ] && holds "$patchdir"; then
    record "$patchdir"
    echo "setup: $name already patched at ${pin:0:12}"
    continue
  fi
  if dirty; then
    if [ "$mode" = reset ]; then
      echo "setup: $name: the tree discarded as --reset asks"
    elif holds "$applied"; then
      echo "setup: $name: patches/$name changed since it was applied; applying it afresh"
    else
      fail "$dir has local edits that re-applying patches/$name would erase. They are neither patches/$name nor the series setup applied here, so setup cannot tell work in progress from leftovers and stops.
$(printf '\n  %-48s%s' "List the edited files:" "./setup.sh --check" "Keep the edits as a new series, then commit:" "./setup.sh --sync" "Discard the edits and apply patches/$name:" "./setup.sh --reset")"
    fi
    discard
  fi
  if [ "$pin" != "$wantsha" ]; then
    # A `git pull` moved the pin and left the checkout behind.
    git -C "$ROOT" submodule update --init -- "$dir" || fail "git submodule update failed for $dir"
    pin=$(git -C "$ROOT/$dir" rev-parse HEAD 2>/dev/null) || fail "cannot read $dir HEAD"
    [ "$pin" = "$wantsha" ] || fail "$dir is at ${pin:0:12} after the update, not at $want"
  fi
  if ! apply_series "$patchdir"; then
    # The series applied before back in place, at its own pin, so the tree still builds.
    if [ -d "$applied" ]; then
      discard && git -C "$ROOT/$dir" checkout -q "$(applies_to "$applied")" && apply_series "$applied" >/dev/null \
        && echo "setup: $name: the series applied before is back in place" >&2
    fi
    fail "$name: patches/$name did not apply at ${pin:0:12}; fix that patch, or re-cut the series against the pin."
  fi
  record "$patchdir"
  echo "setup: $name patched at ${pin:0:12} ($(ls "$patchdir"/*.patch | wc -l | tr -d ' ') patches)"
done
[ "$mode" = check ] && [ $status = 0 ] && echo "setup: both forks hold patches/ at their pinned refs"

# The tracked pre-commit hook (a commit is refused while a fork tree holds work patches/ does not) linked into the
# hook directory this clone uses; a hook already there is left alone.
if [ "$mode" != check ] && [ -f "$ROOT/.githooks/pre-commit" ]; then
  hookdir=$(git -C "$ROOT" config --local --get core.hooksPath || echo "$ROOT/.git/hooks")
  case "$hookdir" in /*) ;; *) hookdir=$ROOT/$hookdir ;; esac
  if [ -d "$hookdir" ] && [ ! -e "$hookdir/pre-commit" ]; then
    ln -s "$ROOT/.githooks/pre-commit" "$hookdir/pre-commit" && echo "setup: pre-commit hook installed at $hookdir/pre-commit"
  fi
fi
exit $status
