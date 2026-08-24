#!/usr/bin/env bash
# Shared trigger-lift / CHA analysis defaults.

export CHA_REFERENCE_N_PER_SIDE="${CHA_REFERENCE_N_PER_SIDE:-64}"
export CHA_TAU="${CHA_TAU:-0.3}"
export CHA_LOW_DATA_POLICY="${CHA_LOW_DATA_POLICY:-skip}"
export CHA_MIN_ACTUAL_N_PER_SIDE="${CHA_MIN_ACTUAL_N_PER_SIDE:-16}"
export CHA_PRUNE_ALPHA="${CHA_PRUNE_ALPHA:-0.05}"
export CHA_MAX_N_PER_SIDE="${CHA_MAX_N_PER_SIDE:-$CHA_REFERENCE_N_PER_SIDE}"

# Stage 7 and trigger-lift scanning share the same default 10k ceiling. A value
# of 0 is interpreted by the downstream stage as unlimited where supported.
export REFINE_SAMPLING_MAX_POINTS="${REFINE_SAMPLING_MAX_POINTS:-10000}"
export TRIGGER_LIFT_SCAN_MAX_ROWS="${TRIGGER_LIFT_SCAN_MAX_ROWS:-$REFINE_SAMPLING_MAX_POINTS}"
export TRIGGER_LIFT_SCAN_CHUNK="${TRIGGER_LIFT_SCAN_CHUNK:-2048}"
export TRIGGER_LIFT_SCAN_MIN_ROWS="${TRIGGER_LIFT_SCAN_MIN_ROWS:-0}"
export TRIGGER_LIFT_SCAN_EARLY_STOP="${TRIGGER_LIFT_SCAN_EARLY_STOP:-0}"
