# Overtopping experiment catalogue

The overtopping study uses an explicit experiment catalogue rather than a Cartesian product of task, model, phase, and intervention settings.

## Suites

`code/studies/overtopping/experiments/run_experiments.py` exposes:

- `paper-primary`: the 28 settings used by the primary manuscript profile;
- `paper-auxiliary`: targeted supporting comparisons;
- `all`: both suites.

The primary profile name is `iclr-28`.

## Listing and dry runs

From the repository root:

```bash
./run_overtopping_experiments.sh --list
./run_overtopping_experiments.sh --dry-run
```

The Python catalogue runner also supports filters:

```bash
cd code
python3 -m studies.overtopping.experiments.run_experiments \
  --suite paper-primary \
  --task arithmetic \
  --model EleutherAI/pythia-1b \
  --dry-run
```

Available filters include task, model, intervention, mode, phase of execution, and evaluation split.

## Execution phases

`--phase` accepts:

- `pipeline`: run the shared causal pipeline only;
- `analysis`: run study-level analysis for existing pipeline outputs;
- `all`: run both.

The default evaluation split is `test`. `train` and `all` must be requested explicitly.

## Outputs

Each catalogue entry writes under the task/model data hierarchy used by the shared pipeline. Aggregate manuscript analysis is generated separately by `generate_results.sh`.

Scientific configuration values that must coexist on disk are encoded in run paths, including spectral sample size, phase, anchoring mode, threshold, evaluation split, and non-default point caps.

## Catalogue maintenance

A primary-catalogue change changes the manuscript population and must be made in the explicit experiment definitions rather than inferred from directories on disk. Reporting validates the primary matrix against the selected profile and does not treat arbitrary top-level experiment directories as additional settings.
