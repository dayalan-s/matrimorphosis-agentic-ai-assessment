# shared by check_protected.sh and generate_checksums.sh
# free_pattern <paths...>: grep -E pattern matching "<sha>  <path>" lines that are NOT locked
# (a trailing / means a whole directory; otherwise an exact file; .checksums itself is always free)
free_pattern() {
    local pat='^[0-9a-f]+  (\.checksums$' f esc
    for f in "$@"; do
        esc=$(printf '%s' "$f" | sed 's/[.[\*^$]/\\&/g')
        case "$f" in */) pat="$pat|$esc" ;; *) pat="$pat|$esc\$" ;; esac
    done
    printf '%s)' "$pat"
}
