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

    /home/mannixe/FLIP/flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation/dataset.csv

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

Latest pull

    python make_spreadsheet.py \
        --sources bacchusmarsh \
        --output output/2026_08_21_generalisation_relabel_farm_and_image_bacchusmarsh.xlsx
    python make_spreadsheet.py \
        --sources gisborne \
        --output output/2026_08_21_generalisation_relabel_farm_and_image_gisborne.xlsx

## Generalisatoin extra data

A second builder run covering nine new `Lot_*` reaches alongside more farms from the four
Victorian ones. None of its 1,420 farms appear in the main pull, so every reach is worth
building. The `Lot_*` shapefiles need the `lot_` prefix on `--sources`, and the outputs
are named `..._extra_...` so they do not collide with the runs above.

    EXTRA=/home/mannixe/FLIP/flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation_extra/dataset.csv

    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_bega \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_bega.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_caniaba \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_caniaba.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_casino \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_casino.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_corowa \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_corowa.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_freemans \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_freemans.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_hanwood \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_hanwood.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_mangrove \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_mangrove.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_nowra \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_nowra.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources lot_redlands \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_redlands.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources bacchusmarsh \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_bacchusmarsh.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources balliang \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_balliang.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources gisborne \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_gisborne.xlsx
    python make_spreadsheet.py \
        --input $EXTRA \
        --sources wyuna \
        --output output/2026_08_21_generalisation_extra_relabel_farm_and_image_wyuna.xlsx

What they write, and how big each one is:

    2026_08_21_generalisation_extra_relabel_farm_and_image_bega.xlsx           188 farms   226 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_caniaba.xlsx         11 farms    49 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_casino.xlsx         264 farms   331 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_corowa.xlsx         100 farms   157 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_freemans.xlsx        57 farms   108 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_hanwood.xlsx        266 farms   413 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_mangrove.xlsx        19 farms    37 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_nowra.xlsx           13 farms    44 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_redlands.xlsx        13 farms   101 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_bacchusmarsh.xlsx   234 farms   515 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_balliang.xlsx        55 farms   191 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_gisborne.xlsx        69 farms   145 images
    2026_08_21_generalisation_extra_relabel_farm_and_image_wyuna.xlsx          131 farms   305 images

## Generalisation extra test (VicPICs supplement, September 2026)

A third builder run, `output_gen_extra_test/`, over the September 2026 VicPICs supplement:
314 Victorian PIC farms cropped from the VIC training rasters, sheep/poultry/pigs only, and
wholly held out (see the builder's README, "generalisation_extra_test"). The builder only
writes `dataset.csv` once the crops are back on the workstation, so run that step there
first if the directory has no csv yet:

    cd /home/mannixe/FLIP/flip-geoimage-dataset-builder
    .venv/bin/python extract_imagery_aerial_csv.py --output-dir output_gen_extra_test \
        --csv output_gen_extra_test/dataset.csv --keep-drop-classes

That drops the 784 blank crops (of 4,096) and leaves 3,312 over 301 farms. The single
`source` value is `Vic_PICs_generalisation_extras_sept26`, so `--sources vic_pics` keeps
everything; none of its farms appear in `labelled_sheets/`, and `source_image_path` is
blank throughout because these crops have no source photograph.

    TEST=/home/mannixe/FLIP/flip-geoimage-dataset-builder/output_gen_extra_test/dataset.csv

    python make_spreadsheet.py \
        --input $TEST \
        --sources vic_pics \
        --output output/2026_09_18_generalisation_extra_test_relabel_farm_and_image_vicpics.xlsx

    2026_09_18_generalisation_extra_test_relabel_farm_and_image_vicpics.xlsx   301 farms  3312 images

Two things to know when labelling it. Every farm was cropped from *every* indexed raster
over it, not only its named `WMS_NAME` capture, so 179 of the 301 farms carry two to four
epochs of the same buildings; the csv's `wms_match` column marks the named one, and 32
farms have no crop from it at all because that raster (wimmera 2016) is not on the share.
And 45 of the farms overlap a farm in the training join, 35 of them by more than half
(`training_overlap_frac` in the csv), so they are not all new farms. The review page
below makes both visible.


## options

- `--input` — `dataset.csv`, or one of the original flat `.xlsx` relabelling sheets.
- `--sources` — reaches to keep, matched case-insensitively against the start of the
  `source` shapefile name. Pass with no values to keep everything.
- `--exclude-labelled` — directory of completed workbooks whose farms are already done
  (default `labelled_sheets/`). Pass `""` to keep everything.

# reviewing farms in the browser

`make_spreadsheet_html.py` renders the farms of a workbook as one HTML page so they can be
looked at without opening the GeoTIFFs one at a time. It takes the same `--input`,
`--sources` and `--exclude-labelled` as `make_spreadsheet.py` and selects the same farms
in the same order, so row *n* of the "Farm labels" sheet is farm *n* on the page. For each
farm it shows (1) the builder's `{farm_uid}_metadata.json` — stock counts, PIC, property
attributes, WMS raster — with the empty fields dropped and a satellite map link when the
farm has a Lat/Long, and (2) every crop in `dataset.csv` for that farm, grouped by the
raster it was cut from, the farm's own WMS raster first and then by capture date. A filter
box at the top matches on the uid, PIC, PFI, class or any metadata value; clicking a crop
opens it large in a lightbox on the same page, the arrow keys step through the farm's
crops, and Esc closes it.

The page lives with the data rather than in this repo: it is written as `review.html` at
the top of the build directory, beside `dataset.csv`.

    python make_spreadsheet_html.py --input $TEST --sources vic_pics
    xdg-open /home/mannixe/FLIP/flip-geoimage-dataset-builder/output_gen_extra_test/review.html

Browsers cannot show the JPEG-in-GeoTIFF crops, so each one gets a JPEG preview beside it
(`..._building_0.tif` -> `..._building_0.jpg`) that the page references by relative path;
nothing is copied anywhere else, and the page and previews travel with the build directory
when it is tarred or synced. Re-running only makes the previews that are missing, so a
rebuild of the page after `labelled_sheets/` changes takes a few seconds. The builder's
own tooling globs `*.tif` and `*_metadata.json`, so the previews do not disturb it. For
the 3,312 crops above the previews come to about 690 MB at the default size and take
25 s on 8 processes; only the rows of `dataset.csv` are shown, the builder having already
dropped the blank crops.

- `--output` — write the html somewhere else (the previews stay beside the crops and are
  referenced relatively, so it still works from another directory on the same machine).
- `--thumbnail` — longest side of each preview in px (default 1000; the crops are 2001).
- `--quality` — JPEG quality (default 82); `--workers` — preview processes (default 8).
- `--force` — remake previews that already exist, e.g. after changing `--thumbnail`.
- `--embed` — inline the previews as data URIs for a single self-contained file; at the
  default size that is most of a GB for a whole build, so pair it with a small
  `--thumbnail` or use it on one reach at a time.
- `--title` — page title (default `<build directory> review`).

# building the crop-level dataset

`gen_dataset_croplevel.py` turns the completed workbooks in `labelled_sheets/` back into
a dataset the models can train on. The builder's `dataset.csv` labels every crop with its
*farm's* `Farm_type`, which is wrong for most of them — a farm's ten crops are usually
one shed and nine paddocks — so this reads the per-crop `Label` from each workbook's
"Image labels" sheet and writes `dataset_relabelled.csv` keyed on that instead.

    python gen_dataset_croplevel.py --workbooks '2026_07_24_*'
    python gen_dataset_croplevel.py --workbooks '2026_08_21_generalisation_extra_*' \
        --dataset /home/mannixe/FLIP/flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation_extra/dataset.csv

Defaults to the same `original_new_2026_08_21_generalisation/dataset.csv` as
`make_spreadsheet.py`, and writes into that same directory — beside the build's own
`dataset.csv`, never over it, and with the relative `image_path` values still resolving
so no imagery is moved or copied. Of the build's 2,913 crops, the 1,705 that have been
labelled come through less the 76 marked `Ambiguous`, for 1,629. The rest (bacchusmarsh
and gisborne) are waiting on their workbooks.

The second run does the same for the `generalisation_extra` build — the `Lot_*` parcels
on the same reaches plus Casino, Corowa, Hanwood and Redlands, whose workbooks are the
`2026_08_21_generalisation_extra_*` ones. The two builds share no farm and no crop, and
each build's `image_path` is relative to its own directory, so each is relabelled in
place and `gen_dataset_master.py` below combines them as two sources. `--workbooks`
narrows `labelled_sheets/` to one build's workbooks; without it every workbook is read
and the other build's crops are reported as unmatched. The extra build's NSW shapefiles
are named `Lot_<reach>_clean_clip.shp`, so a leading `lot_` is ignored when a source is
matched to a region. Of its 2,622 crops the 1,135 labelled so far (the eight NSW reaches)
come through less 118 `Ambiguous`, for 1,017 — all train/val, since its VIC workbooks
are not done yet. Redlands is in Queensland rather than NSW; it is not VIC and so joins
the train/val pool, and only nine of its crops are anything but `Ambiguous`.

    original_new_2026_08_21_generalisation_extra/
        dataset_relabelled.csv     1,017 crops over 650 farms
        relabelled_train_df.csv      800 crops
        relabelled_val_df.csv        217 crops
        relabelled_test_df.csv         0 crops

Nearly all of it is background — 339 residential, 296 other/industrial, 322 paddock —
with 60 livestock crops. The parcels carry no `Farm_type`, so the crop-label-versus-
farm-label comparison the script prints is empty for this build.

The join is on `(farm_uid, ecw_stem, building_cluster)`, not `image_path`: the workbooks
were written against the PFI-keyed collection, so their paths still read
`<PFI>/buildings/…` while the current build is keyed on the farm UID.

    dataset_relabelled.csv     all 1,629 crops, with the split column
    relabelled_train_df.csv      397 crops
    relabelled_val_df.csv         86 crops
    relabelled_test_df.csv     1,146 crops

## splits

Geographic, so the held-out set is a different landscape rather than a different farm in
the same one:

    train/val   NSW   bega, caniaba, freemans, mangrove, nowra      483 crops, 156 farms
    test        VIC   bacchusmarsh, balliang, gisborne, wyuna     1,146 crops, 242 farms

Every crop is in exactly one of the three — there is no `excluded` split. This is a
build from scratch, not an increment on a historical one, so there is nothing to stay
compatible with and no reason to write rows the loaders would have to know to skip.

Within NSW the train/val cut follows `extract_imagery_aerial_csv.py` in the dataset
builder: grouped on `farm_uid` so a farm's crops never straddle the two, stratified,
`VAL_FRACTION` (0.20) of the farms, `RANDOM_STATE` 42. The one difference is what it
stratifies *on* — that script takes the farm's single `processed_class`, but a crop-level
farm has many crop labels, so the stratum here is the farm's most common non-background
class (its most common label outright for a farm that is all paddock).

Note how uneven the class split is by region: poultry is 49 crops in NSW against 3 in
VIC, beef 7 against 49. That is what a geographic hold-out costs, and it is worth
reading the per-class test numbers with the `crops per split x class` table in hand.

## labels

The workbook dropdown offers the eleven classes plus four catch-alls.

`Paddock` and `Other/Industrial` are kept as classes in their own right — half this
collection is paddock, and a crop-level classifier has to be able to say so — normalised
to `paddock` and `other_industrial`.

`Multiple Classes` (17 crops) is not a class but a pointer to the comment, where the
labeller names what they saw: `Poultry/Freerange pigs`, `Dairy/Horse`, `dairy/ backyard
pig`. These are **genuinely multi-label** — two real livestock classes in the one crop,
not something a most-specific rule could collapse — so the crop carries both, and
`binary_*` says so, exactly as the builder already handles a farm labelled `dairy,beef`.
Two balliang crops had the label but no comment; both sit on farms X-marked
`commercialpig` + `freerangepig` on the "Farm labels" sheet, so the farm sheet resolves
them. The resolved combinations:

    4  dairy,horse            2  backyardpig,dairy         2  poultry,residential
    4  freerangepig,poultry   2  commercialpig,freerangepig (from the farm sheet)
    2  horse,poultry          1  backyardpig,commercialpig

Class order within `crop_classes` is canonical, not as written: the same labeller wrote
both `Dairy/Horse` and `Horse/Dairy` for the same pair, so the comment's order carries no
information and keeping it would split one combination across two spellings. The raw
comment is kept in `crop_comments` regardless.

`Ambiguous` (76 crops) means the labeller could not call it at all, so there is nothing
to train or test on and those are dropped outright. `--drop-labels` with no values keeps
them.

## columns

Each row carries the class list and both the old and new labels, so nothing is lost:

- `crop_classes` — the crop's classes, comma-separated. The primary column.
- `n_classes` — 1 for all but the 17 multi-label crops.
- `processed_class` — first entry of `crop_classes`; what the single-label consumers read.
- `binary_<class>` — one per class in `CROP_CLASSES`, multi-hot.
- `farm_processed_class` — what the builder had for the farm, for comparison. The script
  prints the crosstab against `processed_class`, the quickest read on how much the
  relabelling changed.
- `farm_labels` — every class X-marked for that farm on the "Farm labels" sheet.
- `crop_label` / `crop_comments` / `label_workbook` — the workbook's own words, verbatim.

## options

- `--dataset` — the builder's `dataset.csv` to relabel.
- `--labelled` — directory of completed workbooks (default `labelled_sheets/`).
- `--output-dir` — where the csvs land (default: alongside `--dataset`).
- `--nsw-sources` / `--vic-sources` — the reaches making up each region.
- `--drop-labels` — workbook labels dropped from the build entirely (default
  `Ambiguous`). Pass with no values to keep everything.
- `--val-fraction` — share of the NSW farms promoted to val (default 0.20).
- `--imagery {symlink,copy,none}` — link or copy the crops under `--output-dir` to make
  it a standalone dataset (default `none`; only useful with `--output-dir` elsewhere).

# building the master dataset

`gen_dataset_master.py` combines the four current FLIP datasets into one training corpus
with four named test sets, written to `original_master_2026_09_18/`. The earlier
`original_master_2026_09_04/` build — three sources, before the `generalisation_extra`
relabels — is what the 2026-09-04 evaluation, the SAM3 relabel and the post-classifier
were measured on, and is left as it was.

    python gen_dataset_master.py                 # ~38 GB of imagery copied, a few minutes
    python gen_dataset_master.py --dry-run       # csvs + README only, no imagery
    python gen_dataset_master_html.py            # the summary dashboard
    xdg-open original_master_2026_09_18/summary.html

The four sources, in the order the generated README introduces them:

    historical            flip-dataset-processing/output/flip_historical
                          whole-farm .png photographs from the original pipeline
    autocrops             flip-geoimage-dataset-builder/original_new_2026_08_21
                          building crops re-cut from those same photographs, farm-level
                          labels
    generalisation        flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation
                          the case-study subset of those crops, relabelled crop by crop
                          by `gen_dataset_croplevel.py` above
    generalisation_extra  flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation_extra
                          crops cut from the Lot_* parcels on the same reaches plus
                          Casino, Corowa, Hanwood and Redlands, relabelled the same way

They are not independent, which is the whole difficulty. `autocrops` and `historical` are
two different crops of the same source photographs (2,398 names in common), and
`generalisation` is the crop-level relabelling of exactly the imagery `historical` holds
whole in `gen_all_df.csv`. `generalisation_extra` records no source photograph and shares
no crop with `generalisation`, but eight of its farms are also in `autocrops`. Their
upstream splits were also decided on different principles — a curated 2022 FarmFinder
hold-out for `historical` and `autocrops`, a geographic NSW/VIC hold-out for the two
relabelled sources — so the generated README documents each one rather than leaving a
reader to assume a single rule.

## what comes out

    dataset.csv                     26,985 rows / 9,390 groups, every row with its provenance
    train_df.csv                    18,727      val_df.csv                       4,537
    train_overlap.csv                  118      val_overlap.csv                     32
    test_autocrops.csv               1,361      test_autocrop_gen_vic.csv        1,146
    test_gen_original.csv              777      test_gen_original_overlap.csv      151
    test_original.csv                  136
    README.md                       generated: provenance, rules, per-split class tables
    summary.html                    the dashboard, from gen_dataset_master_html.py
    autocrops/ generalisation/ generalisation_extra/ historical/
                                    imagery, each at its original relative path

`generalisation_extra` adds 1,017 rows, all to train/val: its VIC reaches are not
labelled yet, so `test_autocrop_gen_vic.csv` is unchanged from the 2026-09-04 build.
None of its rows were pulled to overlap — its eight farms shared with `autocrops` are
all on the training side there too.

Every input row appears exactly once across those nine csvs — the script asserts it, along
with no group in two splits and no train/val row reaching a test set, before writing
anything.

## how overlap is resolved

**Test wins.** A train/val row whose `farm_uid` *or* source-image stem appears in any of
the four test sets is pulled out of train/val. **Nothing is deleted** — pulled rows go to
`train_overlap.csv` / `val_overlap.csv` with an `overlap_reason`, and their imagery is
copied like any other row. The pull is by whole group, so a group is never split between
a split and its overlap.

The one exception is `gen_all`, divided by `generalisation`'s split *before* the test pool
is built. A literal test-wins would send 100% of `generalisation`'s train/val to overlap,
because all of it sits inside `gen_all`; instead the 151 `gen_all` images whose crops are
in `generalisation` train/val go to `test_gen_original_overlap.csv` and the remaining 777
form `test_gen_original.csv`.

`test_gen_original` and `test_autocrop_gen_vic` share 242 source images on purpose — both
are test sets, and they measure whole-image old labels against crop-level new labels.
**Never pool scores across the two.**

## grouping

A group is **dataset x identifier x imagery source** — the unit training draws on, so it
follows the level the labels were actually assigned at:

    historical            image path x collection      3,536 groups   1 row each
    autocrops             farm_uid x ecw_stem          3,208 groups   mean 6.5, max 10
    generalisation        crop path x ecw_stem         1,629 groups   1 row each
    generalisation_extra  crop path x ecw_stem         1,017 groups   1 row each

`autocrops` groups by farm because one farm-level label covers every crop of it; the
others group by image because each image carries its own label. `generalisation` in
particular sits on the same farms as `autocrops` but was labelled crop by crop, so the
crop is the training example — its split integrity comes from its own farm-grouped
geographic split upstream, not from `group_id`.

The **capture** is in the key because a farm is often flown more than once: 675 of the
2,429 `autocrops` farms appear in two or more ECW captures, and crops from two flights are
two photographs of the farm rather than one. Keying on the farm alone merged them into
groups of up to 37; with the capture in the key every group is capped at the builder's ten
building clusters per farm per capture — asserted at build time, not imposed here.

The **dataset** is in the key because `autocrops` and `generalisation` share 24
`farm_uid`s, and a bare identifier would merge groups across two independently split
sources. Groups never span datasets and never span splits; the raw `farm_uid`,
`group_identifier` and `group_imagery` are all kept as columns.

## options

`gen_dataset_master.py`:

- `--autocrops` / `--generalisation` / `--generalisation-extra` / `--historical` — the
  four source directories.
- `--output-dir` — where the master dataset is written (default
  `original_master_2026_09_18/`).
- `--imagery {copy,symlink,none}` — copy the imagery in (default), symlink it, or write
  only the csvs.
- `--val-fraction` — share of the `hpai` rows given to val (default 0.20). `hpai_df.csv`
  is disjoint from `flip_historical`'s own train/val and carries no split of its own, so
  one is assigned here, stratified on `processed_class`.
- `--random-state` — seed for that split (default 42).
- `--dry-run` — csvs and README, no imagery.

`gen_dataset_master_html.py`:

- `--dataset` — the master `dataset.csv` to summarise.
- `--file` — where to write the html (default `summary.html` beside the dataset).

The dashboard counts every breakdown three ways — images, groups and farms — because they
do not move together: a split can be large by image and small by farm, and a class can
look well represented while resting on a handful of farms. Only `autocrops` and
`generalisation` carry a `farm_uid` at all.

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
    output_eval/gen_evaluation_by_region.csv  the same, per reach: per class and per reach macro
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
- **Over regions.** Nothing is pooled across reaches for the headline numbers — the
  region breakdown is a separate pass that scores each reach on its own units, against
  the classes that reach actually contains, so its macro is over a different class set
  per reach. Compare the two models *within* a reach; comparing reaches to each other
  confounds difficulty with class mix, which is why each row carries its own prevalence.
- **Uncertainty.** 95 % percentile intervals from resampling the evaluation units — crops
  or farms — and rescoring the ensemble. The old and new models see the *same* resamples,
  so the `delta AP` interval is paired. A regional macro interval is reported only when
  at least half the resamples kept every class in that reach; where a reach rests on a
  class with one or two positives, most draws lose it and the interval is left blank
  (`[--]` in the terminal, no whisker in the dashboard) rather than quietly re-averaged
  over whichever classes survived.

## options

`gen_evaluation.py`:

- `--aggregation {max,mean}` — how a farm's per-crop scores become one farm score
  (default `max`).
- `--min-confidence {High,Medium,Low}` — drop farm-level X marks the labeller was less
  sure of, turning them into negatives. A sensitivity check, not the headline.
- `--region-classes` — print every class within every reach, not just each reach's macro.
  The per-class rows are written to `gen_evaluation_by_region.csv` either way.
- `--bootstrap` — resamples per interval (default 2000).
- `--seed` — bootstrap seed (default 0).
- `--output` — where the CSVs and PNGs land (default `output_eval/`).

`gen_evaluation_dashboard.py`:

- `--output` — the directory to read the CSVs from (default `output_eval/`).
- `--file` — where to write the HTML (default `<output>/gen_evaluation_dashboard.html`).


## the dated successors

`gen_evaluation.py` scores the original ComFe generalisation runs against the workbooks.
Two later scripts score later sweeps against the crop-level relabelled VIC hold-out and
write CSVs *and* the dashboard themselves in one pass, so there is no separate dashboard
step:

- `gen_evaluation_2026_08_21.py` — the two multiclass families against the 2026-08-21
  multilabel pair (DINOv2 linear probe, ComFe), on `relabelled_test_df.csv`. Writes
  `output_eval_2026_08_21/`.
- `gen_evaluation_2026_09_04.py` — the same four plus the 2026-09-04 pair retrained on
  `original_master_2026_09_04/train_df.csv`, on `test_autocrop_gen_vic.csv` (the same
  1,146 crops row for row, checked at load time). Writes `output_eval_2026_09_04/`.
- `gen_evaluation_2026_09_04_backbones.py` — the 2026-09-11 backbone sweep under
  `view/flip_2026_09_04/other_backbones/`: the master-build linear probe retrained on
  DINOv2 ViT-L/14 with registers, DINOv3 ViT-L/16 (web and satellite weights) and
  C-RADIOv4, beside the 2026-09-04 ViT-S probe and ComFe re-read as `lin_s` and `comfe_l`.
  Imports every scoring and chart routine from the 2026-09-04 script, drops the cascades,
  and adds `macro_pairs.csv`: the well-sampled macro bootstrapped and differenced pair by
  pair, which is the one number to read a backbone off. Runs with no saved prediction
  whose log has been quiet for a day are reported as dead rather than pending. Writes
  `output_eval_2026_09_04_backbones/`.

      .venv/bin/python gen_evaluation_2026_09_04.py
      xdg-open output_eval_2026_09_04/evaluation_dashboard.html
      .venv/bin/python gen_evaluation_2026_09_04_backbones.py
      xdg-open output_eval_2026_09_04_backbones/evaluation_dashboard.html

The 2026-09-04 page adds two things. The per-class table carries a training count per
multilabel generation (397 relabelled NSW crops against 17,927 master crops, of which
only the 397 are crop-labelled), and a class is well sampled only if it clears the bar in
both. And it ends with a gallery of real hold-out farms — the whole-farm image beside
every building crop, each crop's annotation and every model's top class — chosen per
annotated class as the newest ComFe's highest-, median- and lowest-scoring farm, so each
group shows a miss beside a hit. `examples.csv` records which farms and why. The
imagery is inlined as small JPEGs, so the page is a few megabytes and opens anywhere.

The 2026-09-04 runs save their prediction index as the test split's `group_id` rather
than a row number (the data module was given `test_csv_group`); the loader maps either
form back onto rows. Both scripts take `--aggregation {max,mean,top2}`, `--bootstrap`
(default 200), `--seed`, `--output` and `--file`. The 2026-09-04 script also takes
`--farm-threshold`: the farm-level confusion panel is set against set, and this says how
a farm's scores become predicted classes — `prevalence` (default) predicts each class for
as many farms as are annotated with it, so precision equals recall and the diagonal reads
as recall; a number predicts every class scoring above it. The operating points land in
`operating_points_farm.csv`. A second farm-level panel repeats this with beef + dairy
merged into `cattle` and the three pig classes into `pigs` (`CLASS_GROUPS` in the
script), writing `confusion_farm_merged.csv` and `operating_points_farm_merged.csv`.


# adding SAM3 structure detections

The dataset builder's `sam3_pipeline.py` ran SAM3, text-prompted instance segmentation,
over every building crop of the two builder releases the master draws on and left one row
per crop in `flip-geoimage-dataset-builder/sam3_results/{training,generalisation}/
sam3_detections.csv`: per prompt, how many objects it found, their mean and total area in
square metres and the strongest detection score. Five prompts are a building gate (`roof`,
`building`, `shed`, `shelter`, `house`), five are class features that all but never fire
on a paddock (`water tank`, `silo`, `vehicle`, `cattle yard`, `pond`). Two scripts bring
that into this repository.

## `gen_sam3_postprocess_relabel.py` — the `sam3_*.csv` files

    .venv/bin/python gen_sam3_postprocess_relabel.py

For every csv in `original_master_2026_09_04/` this writes a `sam3_`-prefixed copy beside
it — `sam3_train_df.csv` is `train_df.csv` plus SAM3 columns, same rows in the same order
— and a `sam3_README.md` tallying the join. The originals are not touched. The join is on
the crop's own relative path, so every `autocrops` and `generalisation` row lands and the
`historical` whole-farm photographs, which were never cut into crops, come through with
`sam3_available = False` and empty SAM3 columns.

It also relabels. 98 % of the master's training labels are farm-level, so a dairy's nine
paddocks are all "dairy", which the 2026-09-04 evaluation names as the mechanism by which
the master models lose paddock. Checked on the 1,629 human-labelled crops, "no gate
detection" picks out paddock with precision 0.96 and recall 0.83, so every *farm-level*
crop with no building detected is relabelled `paddock` — 7,761 of the 17,927 training
crops, 1,849 of the 4,320 validation crops, 592 of `test_autocrops` — and every
human-labelled crop is left as it was. The original label stays on the row in
`crop_classes_pre_sam3` / `processed_class_pre_sam3` / `n_classes_pre_sam3`,
`sam3_relabel` marks what changed and `binary_*` is recomputed, so it can be undone column
by column. `--no-relabel` attaches the features and changes nothing; `--gate-min-score`
(default 0.5, the pipeline's own detection threshold) moves the bar.

Columns, all prefixed `sam3_`: the pipeline's `{prompt}_{count,mean_area_m2,
total_area_m2,max_score}`, a `{prompt}_area_frac` normalised by the crop's area so crops of
different sizes and resolutions compare, the crop's size and resolution, and the gate
summaries `gate_score`, `gate_count`, `gate_area_frac` and the boolean `building`.

## `gen_post_classifier.py` — post-classifiers on top of the scored models

    .venv/bin/python gen_post_classifier.py
    .venv/bin/python gen_post_classifier.py --base comfe_m lin_m cascade_p --cv-repeats 5

Reads the `sam3_*.csv` files and the per-crop scores `gen_evaluation_2026_09_04.py`
wrote to `output_eval_2026_09_04/scores_crop.csv`, and derives three models per `--base`
(default the master ComFe and the linear-probe-gated cascade) without a GPU:

- `gate_<base>` — no fitting. Livestock scores times SAM3's building score, paddock one
  minus it. The SAM3 twin of the evaluation's cascades, which gate on a 2026-08-21 model.
- `sam3_prior` — SAM3 features only, one gradient-boosted classifier per class fitted on
  `sam3_train_df.csv` (never VIC). What the detections alone can say.
- `post_<base>` / `postcv_<base>` — a logistic stacker per class over the base's scores
  for every class, the SAM3 features and the prior. `postcv_` is fitted on the hold-out
  under farm-grouped stratified 5-fold cross-validation, repeated `--cv-repeats` times as
  seeds, and scored out of fold; `post_` is fitted on the validation split, which is out
  of region for VIC, and needs per-crop scores on `val_df.csv` that the training runs did
  not save.
- `xgb_<base>` / `xgbcv_<base>` — the same stacker with gradient-boosted trees (xgboost,
  shallow, positives up-weighted) in place of the logistic regression, so an interaction
  between a model score and a detection can be learned. `--stacker {logistic,xgboost,both}`
  chooses; the default is both, side by side. Produce them with the builder's `*_predict_percrop.yaml` config pointed at
  `val_df.csv` (`group_pool: null`, so the saved index is the csv's row order), drop the
  runs under `comfe-run-flip/view/flip_2026_09_04_postclassifier_test/` (or `--split-runs`), and
  `--fit-on auto` picks them up; `--fit-on both` reports both fits side by side.

Everything is scored with the evaluation script's own functions — same hold-out, same
bootstrap, crop and farm level — and written to `output_post_classifier_2026_09_04/` in
the same layouts (`evaluation.csv`, `evaluation_pairs.csv`, `evaluation_by_region.csv`,
`agreement.csv`, `scores_{crop,farm}.csv`, `models.csv`) plus `stacker_coefficients.csv`,
which says what each stacker leans on (signed standardised coefficients for the logistic
stacker, gain importances for xgboost), and `stacker_fits.csv`. Classes with fewer than
`--min-positives` positives in the fitting data pass the base through unchanged. The
console table prints, per class, the base, its derivatives and the paired bootstrap
difference of the stacker against the base.

First pass, cross-validated on the hold-out, crop level: the gate alone lifts paddock AP
from 0.56 (master ComFe) to 0.96, above the linear probe's 0.95; the stacker on the master
ComFe is clear of zero on beef (+0.10), residential (+0.25), other_industrial (+0.10) and
paddock (+0.41), and takes the well-sampled macro from 0.25 to 0.32. The xgboost stacker
lands in the same place (macro 0.32) with a larger residential gain (+0.34) and
other_industrial (+0.19) but no beef gain to speak of, and is weaker at farm level (macro
0.41 against the logistic stacker's 0.43). The coefficients read sensibly — houses and
vehicles argue for residential, sheds against — but the CV folds are small and
in-region, so treat it as a ceiling until the val-fitted stacker exists.

# Notes from next

Out of distribution stuff

Can we pull in ALL imagery...

Train model to include Paddock, residential, other/industrial

Then look at inference on a whole region from the generalisation data... What would it look like the government for interpretation?
PIC data, land use data, FLIP outputs... Do we improve rather than comprimise quality of labels?


Train on all the NSW stuff... Test on Victoria...
Generate data for Balliana - pull everything I haven't already given Hisanthe to label. I'm going to need to move away from PFI's - and go to unique generated identifiers.

