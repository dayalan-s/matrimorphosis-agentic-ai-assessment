#!/usr/bin/env bash
# Template author tool: rewrite .checksums from the staged tree (git add -A first).
# Lists every locked file as "<git blob sha>  <path>"; free paths go in the header.
# Keep FREE equal to `studentPaths` in the assessment config (grav sync checks this).
set -euo pipefail
. "$(dirname "$0")/_lib.sh"
FREE=(src/app/agent.py src/app/tools_s1.py src/app/kb/tools.py src/app/persistence.py src/app/main.py src/app/context.py src/app/custom_tools.py
      src/app/mcp_server/server.py src/app/migrations/ src/app/dashboards/ README.md .gitignore)
{
    echo "# free: ${FREE[*]}"
    git ls-files -s \
        | sed -E 's/^[0-9]+ ([0-9a-f]+) [0-9]\t(.*)$/\1  \2/' \
        | grep -v -E "$(free_pattern "${FREE[@]}")" \
        | sort -k2
} > .checksums
echo "Wrote .checksums ($(grep -vc '^#' .checksums) locked files)"
