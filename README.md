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
