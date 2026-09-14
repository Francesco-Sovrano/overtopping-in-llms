RQ1 readability update:
- final rendered labels now also have semi-transparent white background boxes
- label placement searches farther from crowded point clusters
- point exclusion zones are larger so labels avoid sitting on top of markers
- Pythia checkpoint names remain P1-0 / P1-48k / P1-96k

Run:
python -m reporting.generate_manuscript_figures --results-root ../results
