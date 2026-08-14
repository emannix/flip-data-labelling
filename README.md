# installing environment

uv venv
source .venv/bin/activate
uv pip sync pyproject.toml


source .venv/bin/activate

# building labelling workbooks

`make_spreadsheet.py` turns the dataset builder's `dataset.csv` into a two-sheet
relabelling workbook (one row per farm, one row per image). Farms that already appear in
a workbook under `labelled_sheets/` are dropped, so each run only covers what is left to
do — move a completed workbook into `labelled_sheets/` and it will be excluded next time.

Input defaults to:

    /home/mannixe/FLIP/flip-geoimage-dataset-builder/original_new_2026_07_24_generalisation/dataset.csv

## NSW case studies

Bega, Caniaba, Freemans, Mangrove and Nowra — the default `--sources`:

    python make_spreadsheet.py \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_nsw.xlsx

With Bega/Nowra/Freemans already in `labelled_sheets/`, this yields the 70 remaining
farms (250 images) across Mangrove and Caniaba.

## VIC case studies

Bacchus Marsh, Balliang, Gisborne and Wyuna — roughly 2,400 images, so consider running
one reach at a time:

    python make_spreadsheet.py \
        --sources bacchusmarsh balliang gisborne wyuna \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_vic.xlsx

Per reach, e.g.:

    python make_spreadsheet.py \
        --sources bacchusmarsh \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_bacchusmarsh.xlsx
    python make_spreadsheet.py \
        --sources balliang \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_balliang.xlsx
    python make_spreadsheet.py \
        --sources gisborne \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_gisborne.xlsx
    python make_spreadsheet.py \
        --sources wyuna \
        --output output/2026_07_24_generalisation_relabel_farm_and_image_wyuna.xlsx

## options

- `--input` — `dataset.csv`, or one of the original flat `.xlsx` relabelling sheets.
- `--sources` — reaches to keep, matched case-insensitively against the start of the
  `source` shapefile name. Pass with no values to keep everything.
- `--exclude-labelled` — directory of completed workbooks whose farms are already done
  (default `labelled_sheets/`). Pass `""` to keep everything.

# scoring the models

`gen_evaluation.py` scores the ComFe generalisation runs against the completed workbooks
in `labelled_sheets/` and writes everything to `output_eval/`;
`gen_evaluation_dashboard.py` turns those CSVs into a self-contained HTML dashboard. Run
them in that order — the dashboard reads only the CSVs, never the run directories.

    python gen_evaluation.py
    python gen_evaluation_dashboard.py
    xdg-open output_eval/gen_evaluation_dashboard.html

The evaluation takes a couple of minutes, almost all of it the bootstrap; drop
`--bootstrap` to a few hundred for a quick pass. The run directories and `dataset.csv`
paths are constants at the top of `gen_evaluation.py` (`MODELS`, `DATASET`) — edit those
to point at a different sweep.

## what comes out

    output_eval/gen_evaluation.csv        per-class AP, AUC, bootstrap intervals, paired deltas
    output_eval/scores_image.csv          one row per labelled crop: annotation + every model score
    output_eval/scores_farm.csv           one row per labelled farm: X marks + aggregated scores
    output_eval/confusion_image.{csv,png} annotated crop label vs model argmax
    output_eval/confusion_farm.{csv,png}  annotated farm class vs the farm's top class
    output_eval/models.csv                which runs were scored, and with what settings
    output_eval/gen_evaluation_dashboard.html

The two `scores_*.csv` carry both the ensemble (`old|dairy`) and each individual seed
run (`old#1|dairy`, `old#2|dairy`, …), which is what lets the dashboard draw the seed
spread without re-reading the runs.

## how the numbers are aggregated

- **Over seeds — probabilities, not scores.** A model's headline column is the mean of
  its seed runs' `y_hat`, and average precision is computed once on that ensemble. It is
  *not* the mean of the per-seed APs. The per-seed APs are reported alongside
  (`AP seed mean` / `AP seed sd` in the CSV, one dot or thin curve per seed in the
  dashboard) so the seed spread stays visible next to the ensemble.
- **Over crops within a farm.** A farm's score for a class is the `--aggregation` over
  its crops, `max` by default — the question being asked is whether the class would
  surface for the farm at all. The same aggregation is applied to each seed separately.
- **Over classes.** The macro row is the unweighted mean of the evaluable per-class APs,
  so a class with six positives counts as much as one with a hundred and twenty.
  `aqua` (no positives) and `goat` (no model output) are left blank rather than counted
  as zero.
- **Uncertainty.** 95 % percentile intervals from resampling the evaluation units — crops
  or farms — and rescoring the ensemble. The old and new models see the *same* resamples,
  so the `delta AP` interval is paired.

## options

`gen_evaluation.py`:

- `--aggregation {max,mean}` — how a farm's per-crop scores become one farm score
  (default `max`).
- `--min-confidence {High,Medium,Low}` — drop farm-level X marks the labeller was less
  sure of, turning them into negatives. A sensitivity check, not the headline.
- `--bootstrap` — resamples per interval (default 2000).
- `--seed` — bootstrap seed (default 0).
- `--output` — where the CSVs and PNGs land (default `output_eval/`).

`gen_evaluation_dashboard.py`:

- `--output` — the directory to read the CSVs from (default `output_eval/`).
- `--file` — where to write the HTML (default `<output>/gen_evaluation_dashboard.html`).
