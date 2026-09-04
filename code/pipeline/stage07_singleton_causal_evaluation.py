"""Canonical Stage-7 entry point: held-out singleton causal evaluation.

The implementation remains in ``stage07_refine_neuron_anchored_rules`` for
backward compatibility with historical caches/scripts.  New code should invoke
this module; rule extraction is an optional legacy add-on and is disabled by
default in ``run_pipeline.sh``.
"""

from pipeline.stage07_refine_neuron_anchored_rules import main


if __name__ == "__main__":
    main()
