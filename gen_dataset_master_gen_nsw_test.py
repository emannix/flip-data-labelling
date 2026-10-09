"""Fold the NSW crop-labelled case studies into the VIC hold-out: one bigger relabelled test.

`test_autocrop_gen_vic.csv` is the only test set with crop-level labels, and it rests on
two Victorian reaches: 1,146 crops, three of them poultry, five sheep, eight commercialpig.
Every other crop that was relabelled the same way is NSW and sits in train and val: the
`generalisation` source's five case studies (Bega, Caniaba, Freemans, Mangrove and Nowra;
397 and 86 crops, 49 of them poultry) and the `generalisation_extra` parcel crops on those
reaches plus Casino, Corowa, Hanwood and Redlands (800 and 217). This script moves all of
them to the test side and writes a second set of split csvs beside the originals, so a run
can be pointed at either:

    test_autocrop_gen_vic_nsw.csv      test_autocrop_gen_vic + every generalisation and
                                       generalisation_extra train/val row
    train_minus_gen_nsw_df.csv         train_df minus those rows and minus what overlaps them
    val_minus_gen_nsw_df.csv           val_df, likewise
    train_minus_gen_nsw_overlap.csv    the train rows pulled by the test-wins rule, with the reason
    val_minus_gen_nsw_overlap.csv      the val rows pulled
    dataset_gen_nsw.csv                every row of dataset.csv under the new split names, for
                                       gen_dataset_master_html.py (--dataset it for gen_nsw_summary.html)
    gen_nsw_README.md                  counts, the class table per split, what moved and why

Everything else - `test_autocrops`, `test_original`, `test_gen_original` and its overlap,
`train_overlap`, `val_overlap`, `dataset.csv` - is untouched, and the originals are never
rewritten. Rerun `gen_sam3_postprocess_relabel.py --master <dir>` afterwards and the new
csvs get their `sam3_` twins like every other csv in the directory.

*Test wins, again.* The master build's rule applies to the new test set exactly as it did
to the old four: any train/val row whose `farm_uid` or source-image stem is in the test set
is pulled out, by whole group, into the overlap csv with the reason. The `generalisation`
crops were cut on 24 farms that `autocrops` also holds, so the autocrops rows on those
farms, and the eight `generalisation_extra` shares with it, go to overlap (103 rows in the
2026-09-18 build); the `historical` whole-farm images of the same photographs already sit
in `test_gen_original_overlap`, a test-side split, so nothing is pulled by image. With both
relabelled sources on the test side, train and val carry farm-level labels only.

Usage:

    .venv/bin/python gen_dataset_master_gen_nsw_test.py
    .venv/bin/python gen_dataset_master_gen_nsw_test.py --master original_master_2026_09_18
    .venv/bin/python gen_dataset_master_html.py --dataset original_master_2026_09_18/dataset_gen_nsw.csv
    .venv/bin/python gen_sam3_postprocess_relabel.py --master original_master_2026_09_18
"""

from __future__ import annotations

import argparse
import collections
from datetime import date
from pathlib import Path

import pandas as pd

from gen_dataset_master import CLASSES, SOURCE_ORDER, class_table

MASTER = Path("original_master_2026_09_18")

MOVED_SOURCES = ["generalisation", "generalisation_extra"]
NEW_TEST = "test_autocrop_gen_vic_nsw"
OLD_TEST = "test_autocrop_gen_vic"
# split name -> output csv, in the order they are reported.
FILES = {
    NEW_TEST: "test_autocrop_gen_vic_nsw.csv",
    "train_minus_gen_nsw": "train_minus_gen_nsw_df.csv",
    "val_minus_gen_nsw": "val_minus_gen_nsw_df.csv",
    "train_minus_gen_nsw_overlap": "train_minus_gen_nsw_overlap.csv",
    "val_minus_gen_nsw_overlap": "val_minus_gen_nsw_overlap.csv",
}
RENAMED = {"train": "train_minus_gen_nsw", "val": "val_minus_gen_nsw"}
OVERLAP_OF = {"train_minus_gen_nsw": "train_minus_gen_nsw_overlap", "val_minus_gen_nsw": "val_minus_gen_nsw_overlap"}
README = "gen_nsw_README.md"
DATASET = "dataset_gen_nsw.csv"


def count(df: pd.DataFrame, unit: str) -> int:
    """images, groups or farms in a frame, as the html summary counts them."""
    if unit == "images":
        return len(df)
    if unit == "groups":
        return df["group_id"].nunique()
    farms = df["farm_uid"]
    return farms[(farms != "") & (farms != "nan")].nunique()


def read(path: Path) -> pd.DataFrame:
    """Read as text so every value is written back exactly as it was."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def resplit(df: pd.DataFrame) -> pd.DataFrame:
    """The new split for every row: the move, then the test-wins rule against the new test."""
    df = df.copy()
    df["split_original"] = df["split"]
    moved = df["source_dataset"].isin(MOVED_SOURCES) & df["split"].isin(["train", "val"])
    df.loc[moved | (df["split"] == OLD_TEST), "split"] = NEW_TEST
    df["split"] = df["split"].replace(RENAMED)
    df["moved"] = moved

    held_out = df["split"] == NEW_TEST
    test_stems = set(df.loc[held_out, "source_image_stem"]) - {""}
    test_farms = set(df.loc[held_out, "farm_uid"]) - {"", "nan"}
    pool = df["split"].isin(OVERLAP_OF)
    by_image = pool & df["source_image_stem"].isin(test_stems)
    by_farm = pool & df["farm_uid"].isin(test_farms)
    reason = pd.Series("", index=df.index)
    reason[by_farm] = "farm_uid"
    reason[by_image] = "source_image"
    reason[by_farm & by_image] = "farm_uid,source_image"
    pulled_groups = set(df.loc[by_farm | by_image, "group_id"])
    pulled = pool & df["group_id"].isin(pulled_groups)
    reason[pulled & (reason == "")] = "group"
    df.loc[pulled, "overlap_reason"] = "gen_nsw:" + reason[pulled]
    for split, overlap in OVERLAP_OF.items():
        df.loc[pulled & (df["split"] == split), "split"] = overlap
    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """`overlap_with` recomputed over the new splits, as the master build computes it."""
    by_stem = collections.defaultdict(set)
    by_farm = collections.defaultdict(set)
    for split, key in zip(df["split"], df["source_image_stem"]):
        if key:
            by_stem[key].add(split)
    for split, key in zip(df["split"], df["farm_uid"]):
        if key and key != "nan":
            by_farm[key].add(split)
    df["overlap_with"] = [
        ",".join(sorted((by_stem[s] | by_farm[f]) - {split}))
        for split, s, f in zip(df["split"], df["source_image_stem"], df["farm_uid"])
    ]
    return df


def check(df: pd.DataFrame, original: pd.DataFrame) -> list[str]:
    """What must hold before anything is written."""
    problems = []
    touched = original["split"].isin(["train", "val", OLD_TEST])
    now = df["split"].isin(FILES)
    if not touched.equals(now):
        problems.append("the rows in the new csvs are not exactly the old train/val/test_autocrop_gen_vic rows")
    if (df["split"] == NEW_TEST).sum() != (original["split"] == OLD_TEST).sum() + int(df["moved"].sum()):
        problems.append(f"{NEW_TEST} is not the old test plus the moved rows")
    untouched = ~touched
    if not df.loc[untouched, "split"].equals(original.loc[untouched, "split"]):
        problems.append("a row outside train/val/test_autocrop_gen_vic changed split")
    held = df[df["split"] == NEW_TEST]
    pool = df[df["split"].isin(OVERLAP_OF)]
    farms = set(held["farm_uid"]) - {"", "nan"}
    stems = set(held["source_image_stem"]) - {""}
    leak_farm = pool["farm_uid"].isin(farms).sum()
    leak_stem = pool["source_image_stem"].isin(stems).sum()
    if leak_farm or leak_stem:
        problems.append(f"{leak_farm} train/val rows share a farm and {leak_stem} a source image with {NEW_TEST}")
    split_groups = df[df["split"].isin(FILES)].groupby("group_id")["split"].nunique()
    if (split_groups > 1).any():
        problems.append(f"{int((split_groups > 1).sum())} groups are divided between two of the new splits")
    for name in CLASSES:
        if f"binary_{name}" not in df.columns:
            problems.append(f"binary_{name} missing")
    return problems


GROUP_IS = {"farm+capture": "the farm+capture", "image": "the image"}
# What each split in this set is built from, in reporting order; the untouched splits
# repeat the master README's wording.
BUILT_FROM = {
    "train_minus_gen_nsw": "the master's `train` minus every generalisation and generalisation_extra row and minus the autocrops rows on their farms: autocrops train + historical train + ~80% of historical hpai",
    "val_minus_gen_nsw": "the master's `val`, likewise: autocrops val + historical val + ~20% of historical hpai",
    "train_overlap": "unchanged: the master's train rows pulled for reaching one of its four test sets",
    "val_overlap": "unchanged: the master's val rows pulled likewise",
    "train_minus_gen_nsw_overlap": f"train rows pulled for reaching `{NEW_TEST}` (autocrops rows on the generalisation / generalisation_extra farms)",
    "val_minus_gen_nsw_overlap": f"val rows pulled for reaching `{NEW_TEST}`",
    "test_autocrops": "unchanged: autocrops `test_df.csv`",
    NEW_TEST: f"the master's `{OLD_TEST}` (generalisation `relabelled_test_df.csv`, the VIC reaches) + generalisation `relabelled_train_df.csv` / `relabelled_val_df.csv` (the five NSW case studies) + generalisation_extra `relabelled_train_df.csv` / `relabelled_val_df.csv` (the NSW parcel crops): every crop-level label there is",
    "test_gen_original": "unchanged: historical `gen_all_df.csv`, less the images whose crops were in generalisation train/val",
    "test_gen_original_overlap": "unchanged: the part of `gen_all_df.csv` that lost to generalisation's split (see below)",
    "test_original": "unchanged: historical `test_df.csv`",
}


def sources_table(df: pd.DataFrame) -> list[str]:
    rows = []
    for name in SOURCE_ORDER:
        part = df[df["source_dataset"] == name]
        if part.empty:
            continue
        rows.append(
            f"| `{name}` | `{part['source_dataset_path'].iloc[0]}` | {len(part):,} | "
            f"{part['group_id'].nunique():,} | {part['label_level'].iloc[0]}-level | "
            f"{GROUP_IS.get(part['group_level'].iloc[0], part['group_level'].iloc[0])} |"
        )
    return rows


def landing_table(df: pd.DataFrame) -> str:
    """source x split counts, naming the master's split beside a row that moved."""
    where = df["split"].where(
        df["split"] == df["split_original"].replace(RENAMED),
        df["split"] + " (was " + df["split_original"] + ")",
    )
    # the renamed train/val read as unchanged; everything else that differs is a move or pull
    table = pd.crosstab(where, df["source_dataset"]).reindex(columns=[s for s in SOURCE_ORDER if s in set(df["source_dataset"])])
    order = sorted(table.index, key=lambda v: next((i for i, name in enumerate(BUILT_FROM) if v.startswith(name)), 99))
    return table.loc[order].to_string()


def source_file_table(df: pd.DataFrame) -> str:
    counts = df.groupby(["source_dataset", "source_file", "split"], sort=False).size()
    lines = []
    for (source, file, split), n in sorted(
        counts.items(),
        key=lambda item: (SOURCE_ORDER.index(item[0][0]), item[0][1], list(BUILT_FROM).index(item[0][2])),
    ):
        lines.append(f"  {source:<22}{file:<26}{split:<30}{n:>7,} rows")
    return "\n".join(lines)


def formed_table(df: pd.DataFrame) -> list[str]:
    return [
        f"| `{name}` | {int((df['split'] == name).sum()):,} | "
        f"{df.loc[df['split'] == name, 'group_id'].nunique():,} | {built} |"
        for name, built in BUILT_FROM.items()
    ]


def readme(df: pd.DataFrame, master: Path) -> str:
    moved = df[df["moved"]]
    pulled = df[df["split"].isin(OVERLAP_OF.values())]
    old_test = df[df["split_original"] == OLD_TEST]
    new_test = df[df["split"] == NEW_TEST]
    table = df[df["split"].isin(FILES)].copy()
    table["n_classes"] = table["n_classes"].astype(int)
    lines = [
        f"# {NEW_TEST}: the NSW case studies folded into the VIC hold-out",
        "",
        f"Written by `gen_dataset_master_gen_nsw_test.py` on {date.today().isoformat()} from "
        f"`{master.name}/dataset.csv`. The originals are untouched; these csvs are a second "
        "set of splits for the same rows, to be used together.",
        "",
        "## what moved",
        "",
        f"The {len(moved):,} `generalisation` and `generalisation_extra` rows that were in train "
        f"({int((moved['split_original'] == 'train').sum()):,}) and val "
        f"({int((moved['split_original'] == 'val').sum()):,}) - every crop-labelled NSW row, from "
        f"{', '.join(sorted(moved['source'].unique()))} - join the {len(old_test):,} rows of "
        f"`{OLD_TEST}` to make `{NEW_TEST}` ({len(new_test):,} rows, "
        f"{new_test['farm_uid'].nunique():,} farms, {new_test['source'].nunique()} reaches). "
        f"Every row is crop-labelled; `region` and `source` say which reach a row is from, so "
        f"the VIC and NSW halves can be scored apart as well as together.",
        "",
        "Test wins: train/val rows whose farm or source photograph is in the new test set are "
        f"pulled out by whole group, {len(pulled):,} in all "
        f"({int((pulled['split'] == 'train_minus_gen_nsw_overlap').sum()):,} train, "
        f"{int((pulled['split'] == 'val_minus_gen_nsw_overlap').sum()):,} val), with the reason in "
        "`overlap_reason` (`gen_nsw:farm_uid`, `gen_nsw:source_image`, `gen_nsw:group`):",
        "",
        "```",
        pulled.groupby(["source_dataset", "split_original", "overlap_reason"]).size().to_string(),
        "```",
        "",
        "With both relabelled sources on the test side, train and val carry farm-level "
        "labels only (`autocrops`, `historical`).",
        "",
        "## where the data came from",
        "",
        "The same four sources as `dataset.csv`, unchanged; only where their rows land differs.",
        "",
        "| key | source | rows | groups | label level | group is |",
        "|---|---|---|---|---|---|",
        *sources_table(df),
        "",
        "Where each source's rows land in this split set (the master's own split in brackets "
        "where a row moved):",
        "",
        "```",
        landing_table(df),
        "```",
        "",
        "Per source file, as read, and the split it went to:",
        "",
        "```",
        source_file_table(df),
        "```",
        "",
        "## how the splits were formed",
        "",
        "| split | rows | groups | built from |",
        "|---|---|---|---|",
        *formed_table(df),
        "",
        "### test wins",
        "",
        f"As in the master build: a train/val row whose `farm_uid` **or** source-image stem "
        f"appears in `{NEW_TEST}` is pulled out, by whole group, into "
        "`train_minus_gen_nsw_overlap.csv` / `val_minus_gen_nsw_overlap.csv` with "
        "`overlap_reason` saying which key matched. The master's own `train_overlap` / "
        "`val_overlap` (rows that reached one of its four test sets) stay as they were, so a "
        "run on this set should train on `train_minus_gen_nsw_df.csv` alone and leave all four "
        "overlap csvs out.",
        "",
        "### the gen_all exception is now moot, and left alone",
        "",
        "The master divided `historical`'s `gen_all_df.csv` so that the 151 whole-farm images "
        "whose crops were in `generalisation` train/val went to `test_gen_original_overlap` "
        "rather than pulling those crops out of training. With those crops now on the test "
        f"side, both halves of `gen_all` are test-side and the 151 could rejoin "
        "`test_gen_original`; they are kept apart so `test_gen_original` stays the same 777 "
        "rows as the master's and the two are comparable. `test_gen_original` still shares "
        f"source imagery with `{NEW_TEST}` by design: never pool scores across the two.",
        "",
        "## rows per split",
        "",
        "Counted three ways, as the html summary counts them: images (rows), groups "
        "(`group_id`, the unit training draws on) and farms (distinct `farm_uid`; "
        "`historical` carries none).",
        "",
        "```",
        f"  {'file':<36}{'images':>8}{'groups':>9}{'farms':>8}",
        "\n".join(
            f"  {FILES[name]:<36}{count(part, 'images'):>8,}{count(part, 'groups'):>9,}"
            f"{count(part, 'farms'):>8,}"
            for name in FILES
            for part in [df[df["split"] == name]]
        ),
        "```",
        "",
        f"`{DATASET}` holds every row of `dataset.csv` under these split names (the other "
        "splits keep theirs), and is what `gen_dataset_master_html.py --dataset` reads to "
        "write `gen_nsw_summary.html`, the same page as `summary.html` over this split set.",
        "",
        "## rows per split x class",
        "",
        "A multi-label row counts under each of its classes.",
        "",
        "```",
        class_table(table, "split"),
        "```",
        "",
        f"## `{NEW_TEST}` by reach x class",
        "",
        "```",
        class_table(table[table["split"] == NEW_TEST].assign(source=lambda d: d["source"].str.replace("_labels.*", "", regex=True)), "source"),
        "```",
        "",
        "## columns",
        "",
        "The master build's columns, plus `split_original` (the split the row had in "
        "`dataset.csv`) and `moved` (True on the rows this script moved to test). "
        "`overlap_with` is recomputed over the new splits.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=MASTER, help="the master build directory")
    args = parser.parse_args()

    original = read(args.master / "dataset.csv")
    print(f"{args.master / 'dataset.csv'}: {len(original):,} rows")
    df = annotate(resplit(original))

    problems = check(df, original)
    if problems:
        print("\nrefusing to write:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"\n{'split':<36}{'rows':>8}{'groups':>9}")
    for name, filename in FILES.items():
        part = df[df["split"] == name].drop(columns=[])
        part.to_csv(args.master / filename, index=False)
        print(f"  {filename:<34}{len(part):>8,}{part['group_id'].nunique():>9,}")

    table = df[df["split"].isin(FILES)].copy()
    table["n_classes"] = table["n_classes"].astype(int)
    print("\nrows per split x class\n")
    print(class_table(table, "split"))
    pulled = df[df["split"].isin(OVERLAP_OF.values())]
    print(f"\npulled to overlap: {len(pulled):,} rows")
    print(pulled.groupby(["source_dataset", "split_original", "overlap_reason"]).size().to_string())

    df.to_csv(args.master / DATASET, index=False)
    (args.master / README).write_text(readme(df, args.master))
    print(f"\nwrote {DATASET} and {README} in {args.master}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
