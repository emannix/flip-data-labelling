# 2026-09-04 — master dataset (`gen_dataset_master.py`)

Combine the three current FLIP datasets into one training corpus with four named test
sets, written to `original_master_2026_09_04/`, every row carrying its provenance, and
nothing from any source dropped on the floor.

## sources

| # | key | path | what it is |
|---|-----|------|------------|
| 1 | `autocrops` | `flip-geoimage-dataset-builder/original_new_2026_08_21` | 21,492 building crops (`.tif`) cut from ECW aerial imagery, keyed on `farm_uid`. Labels are **farm-level** `Farm_type` applied to every crop of the farm. Splits already assigned in `{train,val,test}_df.csv`. |
| 2 | `generalisation` | `flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation` | 1,629 building crops from the nine case-study reaches, **crop-level** labels read from the `labelled_sheets/` workbooks by `gen_dataset_croplevel.py`. Splits in `relabelled_{train,val,test}_df.csv`; geographic (NSW train/val, VIC test). |
| 3 | `historical` | `flip-dataset-processing/output/flip_historical` | 3,536 whole-farm `.png` images from the original pipeline. Farm-level labels, no `farm_uid`. Split across `train/val/test/hpai/gen/gen_all_df.csv`. |

Row counts in use: 20,803 + 1,629 + 3,536 = **25,968**.

### how they relate

These are not three independent collections — (1) and (3) are two different crops of the
same underlying source photographs, and (2) is the crop-level relabelling of the same
case-study imagery that (3)'s `gen_all_df.csv` holds whole.

- (1)`.filename` and (3)`.image_name` are the same source-image names: 2,398 stems in common.
- Every one of (2)'s 396 distinct source stems is in (3)'s `gen_all_df.csv`.
- (1) and (2) share 24 `farm_uid`s.

So overlap has to be resolved explicitly rather than assumed away. `gen_df.csv` is a
strict subset of `gen_all_df.csv` (166 of 928), so only `gen_all` is read; the 166 are
marked with a flag.

## decisions taken

1. **Test wins.** A train/val row whose `farm_uid` *or* source-image stem appears in any
   test file is pulled out of train/val.
2. **Nothing is lost.** Pulled rows are written to `*_overlap.csv` rather than deleted,
   and their imagery is copied like any other row.
3. **`gen_all` is split by (2)'s split first.** A literal "test wins" would send 100% of
   (2)'s train/val to overlap, because all of it is inside `gen_all`. Instead the 151
   `gen_all` images whose crops are in (2)'s train/val go to
   `test_gen_original_overlap.csv`, and the remaining 777 form `test_gen_original.csv`.
   (2)'s 483 train/val rows are then kept, and `test_gen_original` is genuinely held out.
   Test-wins applies unmodified everywhere else.
4. **The 242 `gen_all` images behind `test_gen_vic` stay in `test_gen_original`.** Both
   are test sets, so there is no train leakage; they measure different things (whole-image
   old labels vs crop-level new labels). Each such row is flagged with the other test file
   it appears in. **Never pool scores across the two.**
5. **`hpai_df` is split 80/20 into train/val**, stratified on `processed_class`,
   `random_state=42` — it has no existing train/val assignment.
6. **Class columns are the union of all 13**, missing filled `False`. (1) carries 11,
   (2) 13 (adds `paddock`, `other_industrial`), (3) 10 (no `goats`). The README states
   plainly that `False` here means "the source never assessed this class", not a
   verified negative.
7. **Imagery is really copied**, not symlinked — ~38 GB, against 1.3 TB free.

## output layout

```
original_master_2026_09_04/
├── README.md                            generated; provenance of every source and every rule below
├── dataset.csv                          all 25,968 rows, unified schema, with `split`
├── train_df.csv                  17,927
├── val_df.csv                     4,320
├── train_overlap.csv                118  pulled from train: hits a test file
├── val_overlap.csv                   32  pulled from val: hits a test file
├── test_autocrops.csv             1,361  (1) test_df
├── test_gen_vic.csv               1,146  (2) relabelled_test_df — VIC, crop-level labels
├── test_gen_original.csv            777  (3) gen_all minus (2)'s train/val imagery
├── test_gen_original_overlap.csv    151  the part of gen_all whose crops are in training
├── test_original.csv                136  (3) test_df
├── autocrops/                     31.0 GB   crops + source images, at their original relative paths
├── generalisation/                 2.0 GB
└── historical/                     5.0 GB
```

25,968 rows in, 25,968 rows out across the ten CSVs — the invariant the script asserts
before writing.

## unified schema

Provenance columns, on every file including the standalone ones (requirement 1):

- `source_dataset` — `autocrops` | `generalisation` | `historical`
- `source_dataset_path` — absolute path of the directory it was read from
- `source_file` — the CSV within it (`train_df.csv`, `relabelled_test_df.csv`, `hpai_df.csv`, `gen_all_df.csv`, …)
- `source_split` — that file's own notion of the split, verbatim
- `split` — the new split: `train`, `val`, `train_overlap`, `val_overlap`, `test_autocrops`, `test_gen_vic`, `test_gen_original`, `test_gen_original_overlap`, `test_original`
- `label_level` — `farm` for (1) and (3), `crop` for (2)
- `overlap_with` — comma-separated names of the other splits this row's farm or source image also appears in; `""` when it is unique
- `overlap_reason` — `farm_uid` | `source_image` | `farm_uid,source_image` | `""`

Identity and imagery:

- `image_path` — relative to the master root, e.g. `autocrops/farm-…/buildings/….tif`
- `source_image_path` — relative to the master root, where the source carried one
- `source_image_name` / `source_image_stem` — `filename` for (1)/(2), `image_name` for (3); the stem (lower-cased, extension stripped) is the cross-dataset join key
- `farm_uid` — (1)/(2) only, blank for (3)

Labels:

- `processed_class`, `crop_classes` (single-entry for (1)/(3)), `n_classes`
- `binary_<class>` × 13: aqua, backyardpig, beef, commercialpig, dairy, freerangepig,
  goats, horse, poultry, residential, sheep, paddock, other_industrial
- `label_status` — `labelled`, or `notavailable` / `otherlivestock` for the 762 `gen_all`
  rows that carry a placeholder `processed_class` and no positive binary. They are kept
  (nothing is lost) but a loader should filter on this.

Source-specific columns are carried through where present and blank elsewhere:
`Farm_type`, `PFI`, `source`, `Lat`, `Long`, `crop_*`, `blank_reason`, `region`,
`crop_label`, `crop_comments`, `farm_labels`, `farm_processed_class`, `label_workbook`,
`image_classes`, `in_*` flags.

## build order

1. Read the ten source CSVs; normalise each to the unified schema, stamping provenance.
2. Split `gen_all` on (2)'s train/val stems → `test_gen_original` + its overlap file.
3. Stratify `hpai` 80/20 into train/val.
4. Build the test pool (stems ∪ farm_uids over the four test files) and pull matching
   train/val rows into `train_overlap` / `val_overlap`, recording `overlap_reason`.
5. Compute `overlap_with` across all final splits (including test↔test, for decision 4).
6. Assert: every input row appears exactly once in the output; no row is in two splits;
   no farm straddles train/val and a test file.
7. Copy imagery into the three subfolders at each row's relative path, rewriting
   `image_path`/`source_image_path` to the master-relative form.
8. Write the CSVs, then generate `README.md` from the run's own numbers — per-source
   provenance, the combination rules above, per-split × class tables, and the overlap
   ledger.

## CLI

```
python gen_dataset_master.py [--output-dir original_master_2026_09_04]
                             [--autocrops PATH] [--generalisation PATH] [--historical PATH]
                             [--imagery {copy,symlink,none}]   # default copy
                             [--val-fraction 0.20] [--random-state 42]
                             [--dry-run]                       # CSVs + README, no imagery
```

## risks / notes

- The 38 GB copy is the slow step; `--dry-run` gets the CSVs and README for inspection first.
- (1)'s `source_image_path` PNGs are the same files as (3)'s images (~3.3 GB), so they
  are copied into both subfolders. Deliberate: each subfolder stays self-contained.
- Mixing farm-level labels ((1), (3)) with crop-level labels ((2)) in one training set
  is the known compromise — `label_level` makes it filterable, and the README says so.
- `binary_paddock` / `binary_other_industrial` are only ever true for (2) rows.
