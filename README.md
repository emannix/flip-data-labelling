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

## options

- `--input` — `dataset.csv`, or one of the original flat `.xlsx` relabelling sheets.
- `--sources` — reaches to keep, matched case-insensitively against the start of the
  `source` shapefile name. Pass with no values to keep everything.
- `--exclude-labelled` — directory of completed workbooks whose farms are already done
  (default `labelled_sheets/`). Pass `""` to keep everything.

# building the crop-level dataset

`gen_dataset_croplevel.py` turns the completed workbooks in `labelled_sheets/` back into
a dataset the models can train on. The builder's `dataset.csv` labels every crop with its
*farm's* `Farm_type`, which is wrong for most of them — a farm's ten crops are usually
one shed and nine paddocks — so this reads the per-crop `Label` from each workbook's
"Image labels" sheet and writes `dataset_relabelled.csv` keyed on that instead.

    python gen_dataset_croplevel.py

Defaults to the same `original_new_2026_08_21_generalisation/dataset.csv` as
`make_spreadsheet.py`, and writes into that same directory — beside the build's own
`dataset.csv`, never over it, and with the relative `image_path` values still resolving
so no imagery is moved or copied. Of the build's 2,913 crops, the 1,705 that have been
labelled come through less the 76 marked `Ambiguous`, for 1,629. The rest (bacchusmarsh
and gisborne) are waiting on their workbooks.

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

`gen_dataset_master.py` combines the three current FLIP datasets into one training corpus
with four named test sets, written to `original_master_2026_09_04/`.

    python gen_dataset_master.py                 # ~38 GB of imagery copied, a few minutes
    python gen_dataset_master.py --dry-run       # csvs + README only, no imagery
    python gen_dataset_master_html.py            # the summary dashboard
    xdg-open original_master_2026_09_04/summary.html

The three sources, in the order the generated README introduces them:

    historical       flip-dataset-processing/output/flip_historical
                     whole-farm .png photographs from the original pipeline
    autocrops        flip-geoimage-dataset-builder/original_new_2026_08_21
                     building crops re-cut from those same photographs, farm-level labels
    generalisation   flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation
                     the case-study subset of those crops, relabelled crop by crop
                     by `gen_dataset_croplevel.py` above

They are not independent, which is the whole difficulty. `autocrops` and `historical` are
two different crops of the same source photographs (2,398 names in common), and
`generalisation` is the crop-level relabelling of exactly the imagery `historical` holds
whole in `gen_all_df.csv`. Their upstream splits were also decided on three different
principles — a curated 2022 FarmFinder hold-out for `historical` and `autocrops`, a
geographic NSW/VIC hold-out for `generalisation` — so the generated README documents each
one rather than leaving a reader to assume a single rule.

## what comes out

    dataset.csv                     25,968 rows / 7,594 groups, every row with its provenance
    train_df.csv                    17,927      val_df.csv                       4,320
    train_overlap.csv                  118      val_overlap.csv                     32
    test_autocrops.csv               1,361      test_autocrop_gen_vic.csv        1,146
    test_gen_original.csv              777      test_gen_original_overlap.csv      151
    test_original.csv                  136
    README.md                       generated: provenance, rules, per-split class tables
    summary.html                    the dashboard, from gen_dataset_master_html.py
    autocrops/ generalisation/ historical/     imagery, each at its original relative path

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

    historical       image path x collection      3,536 groups   1 row each
    autocrops        farm_uid x ecw_stem          3,208 groups   mean 6.5, max 10
    generalisation   crop path x ecw_stem         1,629 groups   1 row each

`autocrops` groups by farm because one farm-level label covers every crop of it; the other
two group by image because each image carries its own label. `generalisation` in
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

- `--autocrops` / `--generalisation` / `--historical` — the three source directories.
- `--output-dir` — where the master dataset is written (default
  `original_master_2026_09_04/`).
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


# Notes from next

Out of distribution stuff

Can we pull in ALL imagery...

Train model to include Paddock, residential, other/industrial

Then look at inference on a whole region from the generalisation data... What would it look like the government for interpretation?
PIC data, land use data, FLIP outputs... Do we improve rather than comprimise quality of labels?


Train on all the NSW stuff... Test on Victoria...
Generate data for Balliana - pull everything I haven't already given Hisanthe to label. I'm going to need to move away from PFI's - and go to unique generated identifiers.

