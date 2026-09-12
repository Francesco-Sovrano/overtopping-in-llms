#!/usr/bin/env bash
# Shell helpers intentionally limited to syntax available in Bash 3.2.

poisoning_is_true() {
  case "${1-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]) return 0 ;;
    *) return 1 ;;
  esac
}

# Success iff the most recent guarded pipeline call for this endpoint stopped
# intentionally after Stage 5 because no valid circuit was discovered.
poisoning_pipeline_skipped_no_circuit() {
  local output_data_dir="$1"
  python3 - "$output_data_dir/pipeline_status.json" <<'PYNOCK'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "skipped_no_circuit" else 1)
PYNOCK
}
