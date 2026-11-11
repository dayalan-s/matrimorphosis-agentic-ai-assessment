#!/usr/bin/env bash
# Everything except the free paths (src/app/, README.md, .gitignore) must match the template manifest (.checksums).
#   check_protected.sh        -> checks what is staged (pre-commit)
#   check_protected.sh HEAD   -> checks that commit (pre-push, CI)
set -uo pipefail
REF="${1:-}"
MANIFEST=".checksums"
. "$(dirname "$0")/_lib.sh"

if [ -n "$REF" ]; then
    actual_raw=$(git ls-tree -r "$REF" | sed -E 's/^[0-9]+ blob ([0-9a-f]+)\t(.*)$/\1  \2/')
else
    actual_raw=$(git ls-files -s | sed -E 's/^[0-9]+ ([0-9a-f]+) [0-9]\t(.*)$/\1  \2/')
fi
if ! expected=$(git show "${REF}:${MANIFEST}" 2>/dev/null); then
    echo "[eval] $MANIFEST is missing - restore it with: git restore --staged --worktree $MANIFEST"
    exit 1
fi

# free paths (listed in the manifest header) and the manifest itself are not compared
free=$(printf '%s\n' "$expected" | sed -n 's/^# free: //p')
if [ -z "$free" ]; then echo "[eval] $MANIFEST has no '# free:' header"; exit 1; fi
# shellcheck disable=SC2086
actual=$(printf '%s\n' "$actual_raw" | grep -v -E "$(free_pattern $free)")

problems=$(awk -F'  ' '
    NR == FNR { if (NF >= 2) e[$2] = $1; next }
    { if (NF >= 2) a[$2] = $1 }
    END {
        for (p in e) { if (!(p in a)) print "  missing:      " p; else if (a[p] != e[p]) print "  modified:     " p }
        for (p in a) if (!(p in e)) print "  not allowed:  " p
    }' <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") | sort)

if [ -n "$problems" ]; then
    echo "[eval] Locked files differ from the template (only the files listed in docs/TASK.md, README.md and .gitignore are editable):"
    echo "$problems"
    echo ""
    echo "[eval] Only edit the files listed in docs/TASK.md. To fix:"
    echo "         modified/missing -> git restore --staged --worktree <file>"
    echo "         not allowed      -> git rm -f --cached <file> (and delete it)"
    exit 1
fi
echo "[eval] Protected files OK"
