#!/usr/bin/env bash
# Shell helpers intentionally limited to syntax available in Bash 3.2.

poisoning_is_true() {
  case "${1-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]) return 0 ;;
    *) return 1 ;;
  esac
}
