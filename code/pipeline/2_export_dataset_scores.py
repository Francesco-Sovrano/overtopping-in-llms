"""Export the cached task dataset to scores.csv without feature engineering.

Used by poisoning causal runs: the downstream spectral circuit code only needs
prompt/target/output columns plus task metadata, not proposed/tested features.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from lib.feature_extraction_runner import resolve_task_spec

p = argparse.ArgumentParser()
p.add_argument('--task_module', required=True)
p.add_argument('--prompts_answers_pkl_file', required=True)
p.add_argument('--features_scores_dir', required=True)
args = p.parse_args()

task = resolve_task_spec(args.task_module)
df = task.load_dataset_from_cache(args.prompts_answers_pkl_file)
if not isinstance(df, pd.DataFrame):
    raise SystemExit('task.load_dataset_from_cache() must return a pandas DataFrame')
if df.empty:
    raise SystemExit('cached poisoning dataset is empty')
required = [task.DEFAULT_INPUT, task.DEFAULT_TARGETS[0], task.DEFAULT_OUTPUT]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f'cached dataset missing required columns: {missing}')
out = Path(args.features_scores_dir)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / 'scores.csv', index=False)
# Generic stage 7 expects features.json to be a JSON list.  Poisoning does
# not define semantic features, so the scientifically correct list is empty.
# Keep the dataset-only provenance in a separate metadata file instead of
# pretending that dataframe columns are engineered features.
(out / 'features.json').write_text('[]\n', encoding='utf-8')
(out / 'dataset_export_metadata.json').write_text(json.dumps({
    'mode': 'dataset_only',
    'feature_engineering': False,
    'reason': 'poisoning causal localization uses the behavioral target directly',
    'columns': list(df.columns),
}, indent=2) + '\n', encoding='utf-8')
print(f'[poisoning-direct] exported {len(df)} cached rows to {out / "scores.csv"}; semantic feature engineering disabled')
