#!/usr/bin/env bash
# Shared trigger-lift / CHA analysis defaults.

export CHA_REFERENCE_N_PER_SIDE="${CHA_REFERENCE_N_PER_SIDE:-64}"
export CHA_TAU="${CHA_TAU:-0.3}"
export CHA_LOW_DATA_POLICY="${CHA_LOW_DATA_POLICY:-skip}"
export CHA_MIN_ACTUAL_N_PER_SIDE="${CHA_MIN_ACTUAL_N_PER_SIDE:-16}"
export CHA_PRUNE_ALPHA="${CHA_PRUNE_ALPHA:-0.05}"
export CHA_MAX_N_PER_SIDE="${CHA_MAX_N_PER_SIDE:-$CHA_REFERENCE_N_PER_SIDE}"

# The three caps define different populations and therefore have independent
# defaults. A value of 0 is interpreted as unlimited where the downstream stage
# supports exhaustive evaluation.
export REFINE_SAMPLING_MAX_POINTS="${REFINE_SAMPLING_MAX_POINTS:-10000}"
export TRIGGER_LIFT_SCAN_MAX_ROWS="${TRIGGER_LIFT_SCAN_MAX_ROWS:-10000}"
export NORMAL_TASK_SCAN_MAX_ROWS="${NORMAL_TASK_SCAN_MAX_ROWS:-10000}"
export TRIGGER_LIFT_SCAN_CHUNK="${TRIGGER_LIFT_SCAN_CHUNK:-2048}"
export TRIGGER_LIFT_SCAN_MIN_ROWS="${TRIGGER_LIFT_SCAN_MIN_ROWS:-0}"
export TRIGGER_LIFT_SCAN_EARLY_STOP="${TRIGGER_LIFT_SCAN_EARLY_STOP:-0}"

# Poisoning checkpoint sweeps include legitimate no-circuit states. By default,
# discovery is allowed to run, but a completed no-circuit Stage-5 outcome stops
# circuit-dependent downstream analyses.  Keep that policy separate from the
# generic SKIP_IF_NO_CIRCUIT flag, whose semantics are now *preflight/reuse-only*.
export POISONING_SKIP_IF_NO_CIRCUIT="${POISONING_SKIP_IF_NO_CIRCUIT:-1}"
export SKIP_DOWNSTREAM_IF_NO_CIRCUIT="${SKIP_DOWNSTREAM_IF_NO_CIRCUIT:-$POISONING_SKIP_IF_NO_CIRCUIT}"
export SKIP_IF_NO_CIRCUIT="${SKIP_IF_NO_CIRCUIT:-false}"

# Optional poisoning execution policies.
# - POISONING_SKIP_CIRCUIT_DISCOVERY=1: do not run EAP at all; use an explicit
#   full-network Stage-6 candidate space in a separate *_full_ablation namespace.
# - POISONING_FULL_ABLATION_IF_NO_CIRCUIT=1: try normal discovery first and, only
#   when it completes with no usable circuit, rerun the causal analysis using
#   that explicit full-network candidate space.
export POISONING_SKIP_CIRCUIT_DISCOVERY="${POISONING_SKIP_CIRCUIT_DISCOVERY:-0}"
export POISONING_FULL_ABLATION_IF_NO_CIRCUIT="${POISONING_FULL_ABLATION_IF_NO_CIRCUIT:-0}"
