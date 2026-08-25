# Reporting

`reporting/` contains orchestration whose scope spans both studies.

`generate_final_results.py` invokes the overtopping analysis stages, aggregates canonical poisoning runs when available, and writes the final result manifest. `result_paths.py` defines the aggregate output directories used by that orchestration.

Run from `code/`:

```bash
python3 -m reporting.generate_final_results --help
```

Scientific metric implementations remain in `studies/overtopping/analysis/`, `studies/poisoning/`, or shared statistical modules in `core/`.
