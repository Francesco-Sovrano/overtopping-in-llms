#!/usr/bin/env bash
# Shared trigger-lift / CHA analysis defaults.
#
# Canonical names are generic where the setting belongs to CHA itself.  Legacy
# POISONING_* aliases remain accepted so existing launch commands keep working.

_set_alias() {
  local canonical="$1" legacy="$2" default="$3"
  local cval="${!canonical-}" lval="${!legacy-}" value
  if [[ -n "$cval" && -n "$lval" && "$cval" != "$lval" ]]; then
    echo "ERROR: conflicting settings: $canonical=$cval but legacy $legacy=$lval" >&2
    return 2
  fi
  if [[ -n "$cval" ]]; then value="$cval"; elif [[ -n "$lval" ]]; then value="$lval"; else value="$default"; fi
  printf -v "$canonical" '%s' "$value"
  export "$canonical"
}
_set_alias CHA_REFERENCE_N_PER_SIDE POISONING_REFERENCE_CHA_SIDE 64 || return 2
_set_alias CHA_TAU POISONING_CHA_TAU 0.3 || return 2
_set_alias CHA_LOW_DATA_POLICY POISONING_LOW_DATA_POLICY skip || return 2
_set_alias CHA_MIN_ACTUAL_N_PER_SIDE POISONING_MIN_ACTUAL_CHA_SIDE 16 || return 2
_set_alias CHA_PRUNE_ALPHA POISONING_CHA_PRUNE_ALPHA 0.05 || return 2
_set_alias CHA_MAX_N_PER_SIDE POISONING_MAX_DISCOVERY_SIDE "$CHA_REFERENCE_N_PER_SIDE" || return 2

# Reuse the pipeline's existing stage-7 cap rather than introducing another
# stage-7 argument. _run_pipeline.sh has used REFINE_SAMPLING_MAX_POINTS=10000
# as its default cap; poisoning uses that same value for held-out/all singleton
# evaluation.  Older poisoning-specific all-points names are accepted only as
# aliases for this existing setting.
if [[ -n "${REFINE_SAMPLING_MAX_POINTS-}" ]]; then
  _stage7_cap="$REFINE_SAMPLING_MAX_POINTS"
elif [[ -n "${TRIGGER_LIFT_ALL_POINTS_MAX_ROWS-}" ]]; then
  _stage7_cap="$TRIGGER_LIFT_ALL_POINTS_MAX_ROWS"
elif [[ -n "${POISONING_ALL_POINTS_MAX_ROWS-}" ]]; then
  _stage7_cap="$POISONING_ALL_POINTS_MAX_ROWS"
else
  _stage7_cap=10000
fi
for _alias_name in TRIGGER_LIFT_ALL_POINTS_MAX_ROWS POISONING_ALL_POINTS_MAX_ROWS; do
  _alias_val="${!_alias_name-}"
  if [[ -n "$_alias_val" && "$_alias_val" != "$_stage7_cap" ]]; then
    echo "ERROR: conflicting settings: REFINE_SAMPLING_MAX_POINTS=$_stage7_cap but $_alias_name=$_alias_val" >&2
    return 2
  fi
done
export REFINE_SAMPLING_MAX_POINTS="$_stage7_cap"
export TRIGGER_LIFT_ALL_POINTS_MAX_ROWS="$_stage7_cap"
export POISONING_ALL_POINTS_MAX_ROWS="$_stage7_cap"
unset _stage7_cap _alias_name _alias_val

# Trigger-lift candidate acquisition is task-specific, but by default it uses
# the same 10k ceiling as stage 7.  The source pools are already shuffled by a
# deterministic task seed, so a capped scan takes the first N rows of that fixed
# order.  No spectral or hash-based acquisition is introduced.
_set_alias TRIGGER_LIFT_SCAN_MAX_ROWS POISONING_CAUSAL_SCAN_MAX_ROWS "$REFINE_SAMPLING_MAX_POINTS" || return 2
_set_alias TRIGGER_LIFT_SCAN_CHUNK POISONING_CAUSAL_SCAN_CHUNK 2048 || return 2
export TRIGGER_LIFT_SCAN_EARLY_STOP="${TRIGGER_LIFT_SCAN_EARLY_STOP:-0}"

# Backward-compatible aliases exported for older local scripts/task modules.
export POISONING_REFERENCE_CHA_SIDE="$CHA_REFERENCE_N_PER_SIDE"
export POISONING_CHA_TAU="$CHA_TAU"
export POISONING_LOW_DATA_POLICY="$CHA_LOW_DATA_POLICY"
export POISONING_MIN_ACTUAL_CHA_SIDE="$CHA_MIN_ACTUAL_N_PER_SIDE"
export POISONING_CHA_PRUNE_ALPHA="$CHA_PRUNE_ALPHA"
export POISONING_MAX_DISCOVERY_SIDE="$CHA_MAX_N_PER_SIDE"
export POISONING_CAUSAL_SCAN_MAX_ROWS="$TRIGGER_LIFT_SCAN_MAX_ROWS"
export POISONING_CAUSAL_SCAN_CHUNK="$TRIGGER_LIFT_SCAN_CHUNK"
export POISONING_ALL_POINTS_MAX_ROWS="$REFINE_SAMPLING_MAX_POINTS"
