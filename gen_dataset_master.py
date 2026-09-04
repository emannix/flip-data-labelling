"""Combine the three FLIP datasets into one master corpus with four named test sets.

    1  historical      flip-dataset-processing/output/flip_historical
                       whole-farm photographs from the original FLIP pipeline
    2  autocrops       flip-geoimage-dataset-builder/original_new_2026_08_21
                       building crops cut from those same photographs, farm-level labels
    3  generalisation  flip-geoimage-dataset-builder/original_new_2026_08_21_generalisation
                       the case-study subset of those crops, relabelled crop by crop

These are not three independent collections. (1) and (3) are two different crops of the
same underlying source photographs — 2,398 source-image names in common — and (2) is the
crop-level relabelling of exactly the imagery (3) holds whole in gen_all_df.csv. Combining
them therefore has to resolve overlap explicitly rather than assume it away.

Test wins: a train/val row whose farm_uid or source-image stem appears in any of the four
test sets is pulled out of train/val. Nothing is deleted — pulled rows go to
train_overlap.csv / val_overlap.csv and their imagery is copied like any other row, so
every input row appears exactly once somewhere in the output.

The one exception is gen_all, which is split by (2)'s split *before* the test pool is
built. A literal test-wins would otherwise send 100% of (2)'s train/val to overlap, since
all of it sits inside gen_all; instead the gen_all images whose crops are in (2)'s
train/val go to test_gen_original_overlap.csv and the rest form test_gen_original.csv.
That keeps (2)'s training rows and leaves test_gen_original genuinely held out.

gen_all also contains the source images behind test_autocrop_gen_vic. Both are test sets,
so there is no train leakage, and both are kept because they measure different things —
whole-image old labels against crop-level new labels. Rows in both carry it in
overlap_with. Never pool scores across the two.

Grouping. A group is dataset x identifier x imagery source, and it is the unit training
draws on, so it follows the level the labels were actually assigned at. For (1), where one
farm-level label covers every crop of a farm in one aerial capture, the group is that
farm-and-capture. For (2) and (3), where each image carries its own label, the group is the
single image.

The capture is in the key because a farm is often flown more than once: 675 of the 2,429
autocrops farms appear in two or more ECW captures, and crops from two flights are two
photographs of the farm rather than one. Keying on the farm alone merged them into groups
of up to 37; with the capture in the key every group is capped at the builder's ten
building clusters per farm per capture, which the build asserts rather than assumes.

group_id is prefixed with the dataset because (1) and (2) share 24 farm_uids and would
otherwise merge across two independently split sources. The raw farm_uid and ecw_stem stay
on every row, so regrouping (2) by farm and capture later remains available.

Output goes to original_master_2026_09_04/ — the csvs and README at the top, each source's
imagery copied into its own subfolder at its original relative path.
"""

import argparse
import collections
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_AUTOCROPS = Path(
    "/home/mannixe/FLIP/flip-geoimage-dataset-builder/original_new_2026_08_21"
)
DEFAULT_GENERALISATION = Path(
    "/home/mannixe/FLIP/flip-geoimage-dataset-builder/"
    "original_new_2026_08_21_generalisation"
)
DEFAULT_HISTORICAL = Path(
    "/home/mannixe/FLIP/flip-dataset-processing/output/flip_historical"
)
DEFAULT_OUTPUT = Path("original_master_2026_09_04")

# Each source's imagery is copied into its own subfolder, so the two sources that share
# source photographs cannot collide and each subfolder stays self-contained.
SUBDIR = {
    "autocrops": "autocrops",
    "generalisation": "generalisation",
    "historical": "historical",
}

# The union of all three sources' classes. (1) carries 11, (2) adds paddock and
# other_industrial, (3) has 10 (no goats). A class its source never assessed is written
# False, which means "not assessed", not "verified absent" — see the README.
CLASSES = [
    "aqua", "backyardpig", "beef", "commercialpig", "dairy", "freerangepig",
    "goats", "horse", "poultry", "residential", "sheep",
    "paddock", "other_industrial",
]
BINARY = ["binary_" + cls for cls in CLASSES]

# split name -> output csv. Order is the order they are reported in.
SPLIT_FILES = {
    "train": "train_df.csv",
    "val": "val_df.csv",
    "train_overlap": "train_overlap.csv",
    "val_overlap": "val_overlap.csv",
    "test_autocrops": "test_autocrops.csv",
    "test_autocrop_gen_vic": "test_autocrop_gen_vic.csv",
    "test_gen_original": "test_gen_original.csv",
    "test_gen_original_overlap": "test_gen_original_overlap.csv",
    "test_original": "test_original.csv",
}
# The splits whose farms and source images the train/val rows are held out from.
# test_gen_original_overlap is deliberately not among them: it is the part of gen_all
# that lost to (2)'s split, so holding training out from it would defeat the exception.
TEST_SPLITS = [
    "test_autocrops", "test_autocrop_gen_vic", "test_gen_original", "test_original",
]
TRAIN_SPLITS = ["train", "val"]
OVERLAP_OF = {"train": "train_overlap", "val": "val_overlap"}

VAL_FRACTION = 0.20
RANDOM_STATE = 42

# A group is dataset x identifier x imagery source. The imagery source matters because a
# farm is often flown more than once: 675 of the 2,429 autocrops farms appear in two or
# more ECW captures, and crops from different captures are different photographs of the
# farm, not the same one. Keying on the farm alone merged them and produced groups of up
# to 37; adding the capture caps every group at the builder's ten building clusters per
# farm per capture, which is asserted rather than assumed.
MAX_GROUP = 10

# How each source groups, declared once and read by both the loaders and the README.
# The group is the unit training draws on, so it follows the level the labels were
# actually assigned at: a whole farm-and-capture where one label covers every crop of it,
# and the single image where each image was labelled on its own.
GROUP_SPEC = {
    "historical": {
        "identifier": "image_files_out_rel", "imagery": "collection", "level": "image",
        "describe": ("the image path", "the collection directory"),
    },
    "autocrops": {
        "identifier": "farm_uid", "imagery": "ecw_stem", "level": "farm+capture",
        "describe": ("`farm_uid`", "`ecw_stem` (the capture)"),
    },
    "generalisation": {
        "identifier": "image_path", "imagery": "ecw_stem", "level": "image",
        "describe": ("the crop's own path", "`ecw_stem` (the capture)"),
    },
}

# The order the three sources are introduced in: the original pipeline first, then the
# two builds derived from its imagery, so each one can be explained in terms of the last.
SOURCE_ORDER = ["historical", "autocrops", "generalisation"]

# This repository — the one that builds the dataset the README ships inside. Named at
# the top of the generated README so someone who receives only the dataset directory can
# find the code that produced it and the workbooks the crop-level labels came from.
THIS_REPO = "https://github.com/emannix/flip-data-labelling"
THIS_REPO_NAME = "emannix/flip-data-labelling"

# Why the three sources cannot be treated as independent samples. Shared by the generated
# README and the summary dashboard, so the two never drift apart.
NOT_INDEPENDENT = (
    "`autocrops` and `historical` are two different crops of the same underlying source "
    "photographs, and `generalisation` is the crop-level relabelling of exactly the "
    "imagery `historical/gen_all_df.csv` holds whole. Overlap between them is therefore "
    "resolved explicitly, not assumed away.\n\n"
    "Two consequences worth holding onto:\n\n"
    "- **`test_autocrops` and `test_original` are the same hold-out.** Both are defined "
    "by `farmfinder_test_2022.xlsx` — one at crop level, one at whole-image level. They "
    "share 125 source photographs. Treat them as two views of one test set, not as two "
    "independent ones.\n"
    "- **`autocrops` and `generalisation` overlap by farm** (24 `farm_uid`s), even "
    "though they never share a crop file. That is why the test-wins rule below checks "
    "`farm_uid` as well as the source photograph."
)

# Where each source is built and how its own splits were decided. Written into the
# generated README, because the three were split on three different principles and a
# reader who assumes one rule for all of them will misread the test numbers.
SOURCE_NOTES = {
    "autocrops": {
        "repo": "https://github.com/emannix/flip-geoimage-dataset-builder",
        "repo_name": "emannix/flip-geoimage-dataset-builder",
        "built_by": "`extract_imagery_aerial_csv.py`",
        "rows": (
            "One building crop per detected building cluster, cut as `.tif` from ECW "
            "aerial imagery and filed under a generated `farm_uid`. Largely the **same "
            "farms and the same source photographs as `historical` above**, re-cut: "
            "where `historical` has one image of the whole farm, this has one image per "
            "building on it. The label is the "
            "farm's `Farm_type`, read from that farm's metadata JSON and applied to "
            "**every** crop of the farm — so a farm's shed and its nine paddocks all "
            "carry the farm's class."
        ),
        "split": (
            "**Not geographic, and not random.** The hold-out is a curated list "
            "inherited from the 2022 FarmFinder work, so it is the same hold-out the "
            "original FLIP models were measured against:\n\n"
            "- `farmfinder_test_2022.xlsx` — a crop whose source photograph is named in "
            "this spreadsheet becomes **test**.\n"
            "- `farmfinder_train_2022.xlsx` — the `train_exceptions` list. A non-test "
            "crop named here is **excluded from every split**: it stays in the build's "
            "`dataset.csv` tagged `split=\"excluded\"` but is never written to a "
            "per-split csv, so it never reaches this dataset.\n"
            "- Everything else is **train**, with 20% of farms promoted to **val** — "
            "farm-grouped so a farm's crops never straddle the two, stratified on the "
            "farm's class, `random_state=42`.\n\n"
            "Test therefore spans the same regions as train; it is a held-out *set of "
            "farms*, not a held-out landscape."
        ),
    },
    "generalisation": {
        "repo": "https://github.com/emannix/flip-geoimage-dataset-builder",
        "repo_name": "emannix/flip-geoimage-dataset-builder",
        "built_by": (
            "`extract_imagery_aerial_csv.py` in that repository, then relabelled "
            "crop-by-crop by `gen_dataset_croplevel.py` in **this** one"
        ),
        "rows": (
            "The same kind of building crops as `autocrops`, cut from the case-study "
            "reaches — which are the same images `historical` carries in "
            "`gen_all_df.csv`. The "
            "difference is the label: each crop was labelled **individually** by a human "
            "in the workbooks under `labelled_sheets/`, rather than inheriting its "
            "farm's class. This is the only source whose labels are crop-level, and the "
            "only one that can say `paddock` or `other_industrial`."
        ),
        "split": (
            "**Geographic — a whole-region hold-out.** The point is that the held-out "
            "set is a different landscape, not a different farm in the same one:\n\n"
            "- **train / val — NSW**: Bega, Caniaba, Freemans, Mangrove, Nowra.\n"
            "- **test — VIC**: Balliang and Wyuna. `gen_dataset_croplevel.py` also names "
            "Bacchus Marsh and Gisborne as VIC test reaches, but their workbooks are not "
            "labelled yet, so they contribute no rows.\n\n"
            "Within NSW the train/val cut is farm-grouped and stratified on the farm's "
            "most common non-background crop class, 20% of farms to val, "
            "`random_state=42`.\n\n"
            "Because the split is regional, the class mix differs sharply between the "
            "two sides — read the per-class test numbers with the class tables below in "
            "hand rather than assuming train and test are comparable populations."
        ),
    },
    "historical": {
        "repo": (
            "https://gitlab.unimelb.edu.au/farm-location-image-processing/"
            "flip-dataset-processing"
        ),
        "repo_name": "farm-location-image-processing/flip-dataset-processing "
                     "(UniMelb GitLab, not GitHub)",
        "built_by": "`historical_processing.py`",
        "rows": (
            "Whole-farm `.png` photographs from the original FLIP pipeline — one image "
            "*is* one farm, so there are no crops and no `farm_uid`. Drawn from "
            "`2024_farms/`, `nsw_farms_data/`, `vic_farms_data/`, "
            "`vic_farms_data_extra/`, `vic_hpai_data/` and `case study images/`."
        ),
        "split": (
            "**By membership list, not geography.** Each image is tested against four "
            "lists in turn, and whatever is left over is the training pool:\n\n"
            "- `in_test_data` — named in `farmfinder_test_2022.xlsx` → `test_df.csv`. "
            "The same spreadsheet defines the `autocrops` test set below, which is why "
            "those two are two views of one hold-out rather than two independent ones.\n"
            "- `in_train_exceptions` — named in `farmfinder_train_2022.xlsx` → excluded "
            "from every split.\n"
            "- `in_gen_data` — in the case-study list → `gen_all_df.csv` (everything) "
            "and `gen_df.csv` (the same rows filtered to the ten accepted classes). "
            "`gen_df` is a strict subset, so only `gen_all` is read here and its "
            "labelled members are flagged `in_gen_labelled`.\n"
            "- `in_hpai_data` — path contains `hpai` → `hpai_df.csv`.\n"
            "- Everything else → `train_df.csv` / `val_df.csv`, an 80/20 split "
            "stratified on `processed_class`, `random_state=42`.\n\n"
            "That last split is **row-level, not farm-grouped** — unlike `autocrops`, "
            "which groups by farm. This source carries no farm identity at all, so if "
            "two photographs of one farm exist, nothing prevented them landing on "
            "opposite sides of train/val, and nothing here can detect it."
        ),
    },
}

PROVENANCE_COLUMNS = [
    "source_dataset", "source_dataset_path", "source_file", "source_split",
    "label_level", "label_status",
]
GROUP_COLUMNS = [
    "group_id", "group_level", "group_size", "group_identifier", "group_imagery",
]
IDENTITY_COLUMNS = [
    "image_path", "source_relpath", "source_image_path", "source_image_relpath",
    "source_image_name", "source_image_stem", "farm_uid",
]
LABEL_COLUMNS = ["processed_class", "crop_classes", "n_classes"]
OVERLAP_COLUMNS = ["overlap_with", "overlap_reason"]
# Carried through where the source has them, blank elsewhere.
EXTRA_COLUMNS = [
    "Farm_type", "PFI", "source", "region", "Lat", "Long", "ecw_stem",
    "building_cluster", "crop_width", "crop_height", "crop_res", "crop_std",
    "crop_fill_frac", "blank_reason", "crop_label", "crop_comments", "farm_labels",
    "farm_processed_class", "label_workbook", "image_classes", "in_gen_labelled",
    "in_train_exceptions", "in_test_data", "in_gen_data", "in_hpai_data",
    "in_train_data",
]
COLUMNS = (
    PROVENANCE_COLUMNS + GROUP_COLUMNS + IDENTITY_COLUMNS + LABEL_COLUMNS
    + OVERLAP_COLUMNS + BINARY + EXTRA_COLUMNS + ["split"]
)


def stem(name):
    """Source-image name -> the cross-dataset join key.

    (1) and (2) record it as `filename`, (3) as `image_name`, and the same photograph can
    appear as .png in one and .PNG in another, so the key is lower-cased and stripped of
    its extension.
    """
    return os.path.splitext(str(name))[0].lower()


def read_csv(path):
    return pd.read_csv(path, dtype={"building_cluster": str, "PFI": str})


def normalise(df, dataset, root, source_file, source_split, label_level,
              image_path, source_image_path, source_image_name,
              group_identifier, group_imagery, group_level):
    """One source's rows in the unified schema, with provenance stamped on.

    The caller passes the source's own column names for the four things every source
    names differently — where the image is, where its source photograph is, what that
    photograph is called, and what identifies the group — and everything else is either
    carried through under its own name or derived from the binary_ columns.
    """
    out = pd.DataFrame(index=df.index)
    out["source_dataset"] = dataset
    out["source_dataset_path"] = str(root)
    out["source_file"] = source_file
    out["source_split"] = source_split
    out["label_level"] = label_level

    subdir = SUBDIR[dataset]
    out["source_relpath"] = df[image_path].astype(str)
    out["image_path"] = subdir + "/" + out["source_relpath"]
    if source_image_path is None:
        out["source_image_relpath"] = ""
        out["source_image_path"] = ""
    else:
        out["source_image_relpath"] = df[source_image_path].fillna("").astype(str)
        out["source_image_path"] = out["source_image_relpath"].map(
            lambda p: f"{subdir}/{p}" if p else ""
        )
    out["source_image_name"] = df[source_image_name].fillna("").astype(str)
    out["source_image_stem"] = out["source_image_name"].map(stem)
    out["farm_uid"] = df["farm_uid"].astype(str) if "farm_uid" in df else ""

    # dataset x identifier x imagery source. The identifier is the farm where there is
    # one and the image itself otherwise; the imagery source is the capture the crop came
    # from, so two flights over one farm stay two groups. Prefixed with the dataset so a
    # farm_uid shared between two independently split sources cannot merge their groups.
    out["group_level"] = group_level
    out["group_identifier"] = df[group_identifier].astype(str)
    out["group_imagery"] = df[group_imagery].astype(str)
    out["group_id"] = (
        dataset + ":" + out["group_identifier"] + ":" + out["group_imagery"]
    )

    # every source already carries its labels as binary_ columns; the class list is their
    # union in canonical order, so it is derived the same way for all three
    for cls, col in zip(CLASSES, BINARY):
        out[col] = df[col].fillna(False).astype(bool) if col in df else False
    positives = out[BINARY].to_numpy()
    out["crop_classes"] = [
        ",".join(cls for cls, hit in zip(CLASSES, row) if hit) for row in positives
    ]
    out["n_classes"] = positives.sum(axis=1)
    out["processed_class"] = df["processed_class"].fillna("").astype(str)
    # (3)'s gen_all carries 762 rows whose processed_class is a placeholder rather than a
    # class and whose binary_ columns are all False. They are kept, but a loader has to
    # be able to filter them out of training and scoring.
    out["label_status"] = out["n_classes"].map(lambda n: "labelled" if n else "unlabelled")
    out.loc[out["n_classes"] == 0, "label_status"] = out.loc[
        out["n_classes"] == 0, "processed_class"
    ].replace("", "unlabelled")

    for column in EXTRA_COLUMNS:
        if column in df.columns:
            out[column] = df[column]
    return out


def group_args(dataset):
    """The grouping keyword arguments `normalise` takes for one source."""
    spec = GROUP_SPEC[dataset]
    return {
        "group_identifier": spec["identifier"],
        "group_imagery": spec["imagery"],
        "group_level": spec["level"],
    }


def load_autocrops(root):
    """(1) — one farm-level label covers every crop of a farm in one aerial capture, so
    that farm-and-capture is the group."""
    frames = []
    for split in ("train", "val", "test"):
        name = f"{split}_df.csv"
        frames.append(normalise(
            read_csv(root / name), "autocrops", root, name, split, "farm",
            image_path="image_path", source_image_path="source_image_path",
            source_image_name="filename", **group_args("autocrops"),
        ))
    return pd.concat(frames, ignore_index=True)


def load_generalisation(root):
    """(2) — crop-level labels from the workbooks, so each crop is its own group.

    Its crops sit on the same farms and captures as (1)'s, but every one of them was
    labelled individually, so the crop is what training draws on and the group is the
    single image. Split integrity does not rest on that: this source's own split is
    geographic and farm-grouped upstream, so no farm straddles train, val and test even
    though group_id no longer says so. farm_uid and ecw_stem stay on every row for anyone
    who wants to regroup by farm and capture.
    """
    frames = []
    for split in ("train", "val", "test"):
        name = f"relabelled_{split}_df.csv"
        frames.append(normalise(
            read_csv(root / name), "generalisation", root, name, split, "crop",
            image_path="image_path", source_image_path="source_image_path",
            source_image_name="filename", **group_args("generalisation"),
        ))
    return pd.concat(frames, ignore_index=True)


def load_historical(root):
    """(3) — one whole-farm image per row, so the group is that image.

    There is no farm identity and no capture to key on, so the identifier is the image's
    own path and the imagery source is the collection directory it was filed under.

    gen_df.csv is a strict subset of gen_all_df.csv (166 of 928), so only gen_all is read
    and the 166 are flagged with in_gen_labelled rather than counted twice.
    """
    labelled = {stem(x) for x in pd.read_csv(root / "gen_df.csv")["image_name"]}
    frames = []
    for split, name in [("train", "train_df.csv"), ("val", "val_df.csv"),
                        ("test", "test_df.csv"), ("hpai", "hpai_df.csv"),
                        ("gen_all", "gen_all_df.csv")]:
        raw = pd.read_csv(root / name)
        raw["in_gen_labelled"] = raw["image_name"].map(stem).isin(labelled)
        raw["collection"] = raw["image_files_out_rel"].astype(str).str.split("/").str[0]
        frames.append(normalise(
            raw, "historical", root, name, split, "farm",
            image_path="image_files_out_rel", source_image_path=None,
            source_image_name="image_name", **group_args("historical"),
        ))
    return pd.concat(frames, ignore_index=True)


def split_hpai(df, val_fraction, random_state):
    """Give ~val_fraction of the hpai rows to val, stratified on processed_class.

    hpai_df.csv is disjoint from flip_historical's own train_df and val_df and carries no
    split of its own, so one has to be assigned. Classes with fewer than two rows cannot
    be stratified and stay in train.
    """
    hpai = df["source_file"] == "hpai_df.csv"
    if not hpai.any():
        return df
    pool = df[hpai]
    counts = pool["processed_class"].value_counts()
    stratifiable = pool[pool["processed_class"].isin(counts[counts >= 2].index)]
    df.loc[hpai, "source_split"] = "hpai/train"
    if len(stratifiable) >= 2:
        _, val = train_test_split(
            stratifiable, test_size=val_fraction,
            stratify=stratifiable["processed_class"], random_state=random_state,
        )
        df.loc[val.index, "source_split"] = "hpai/val"
    return df


def assign_splits(df, val_fraction, random_state):
    """The new split for every row, before the overlap rule is applied.

    gen_all is divided here rather than later: the images whose crops are in (2)'s
    train/val go to test_gen_original_overlap so that (2)'s training rows survive the
    test-wins rule that follows.
    """
    df = split_hpai(df, val_fraction, random_state)

    generalisation_train = set(
        df.loc[
            (df["source_dataset"] == "generalisation")
            & df["source_split"].isin(["train", "val"]),
            "source_image_stem",
        ]
    )
    gen_all = df["source_file"] == "gen_all_df.csv"
    superseded = gen_all & df["source_image_stem"].isin(generalisation_train)

    df["split"] = ""
    for dataset, source_split, new in [
        ("autocrops", "train", "train"), ("autocrops", "val", "val"),
        ("autocrops", "test", "test_autocrops"),
        ("generalisation", "train", "train"), ("generalisation", "val", "val"),
        ("generalisation", "test", "test_autocrop_gen_vic"),
        ("historical", "train", "train"), ("historical", "val", "val"),
        ("historical", "hpai/train", "train"), ("historical", "hpai/val", "val"),
        ("historical", "test", "test_original"),
    ]:
        df.loc[
            (df["source_dataset"] == dataset) & (df["source_split"] == source_split),
            "split",
        ] = new
    df.loc[gen_all & ~superseded, "split"] = "test_gen_original"
    df.loc[superseded, "split"] = "test_gen_original_overlap"
    return df


def apply_test_wins(df):
    """Pull train/val rows that reach into a test set out into the *_overlap splits.

    A row is pulled when its farm or its source photograph is in one of the four test
    sets. The pull is by whole group: a group is the unit a label was asserted over, so
    half of one in train and half in overlap would be meaningless. In practice it never
    has to divide one — every autocrops farm maps to exactly one source image — and the
    assertions check that stays true.
    """
    held_out = df["split"].isin(TEST_SPLITS)
    test_stems = set(df.loc[held_out, "source_image_stem"]) - {""}
    test_farms = set(df.loc[held_out, "farm_uid"]) - {"", "nan"}

    pool = df["split"].isin(TRAIN_SPLITS)
    by_image = pool & df["source_image_stem"].isin(test_stems)
    by_farm = pool & df["farm_uid"].isin(test_farms)
    df["overlap_reason"] = ""
    df.loc[by_farm, "overlap_reason"] = "farm_uid"
    df.loc[by_image, "overlap_reason"] = "source_image"
    df.loc[by_farm & by_image, "overlap_reason"] = "farm_uid,source_image"

    # promote to whole groups, so a group is never divided between a split and its overlap
    pulled_groups = set(df.loc[by_farm | by_image, "group_id"])
    pulled = pool & df["group_id"].isin(pulled_groups)
    df.loc[pulled & (df["overlap_reason"] == ""), "overlap_reason"] = "group"
    for split, overlap in OVERLAP_OF.items():
        df.loc[pulled & (df["split"] == split), "split"] = overlap
    return df


def annotate(df):
    """group_size, and the other splits each row's farm or source image also appears in."""
    df["group_size"] = df.groupby("group_id")["group_id"].transform("size")

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


def check(df, totals):
    """Everything that must hold before a single file is written."""
    problems = []
    if len(df) != sum(totals.values()):
        problems.append(f"row count {len(df)} != {sum(totals.values())} read in")
    unknown = set(df["split"]) - set(SPLIT_FILES)
    if unknown:
        problems.append(f"rows landed in unknown splits: {sorted(unknown)}")

    straddling = df.groupby("group_id")["split"].nunique()
    straddling = straddling[straddling > 1]
    if not straddling.empty:
        problems.append(
            f"{len(straddling)} group_ids appear in more than one split, e.g. "
            f"{sorted(straddling.index)[:3]}"
        )

    oversized = df.groupby("group_id").size()
    oversized = oversized[oversized > MAX_GROUP]
    if not oversized.empty:
        problems.append(
            f"{len(oversized)} groups hold more than {MAX_GROUP} rows, e.g. "
            f"{oversized.sort_values(ascending=False).head(3).to_dict()}"
        )

    duplicated = df["image_path"].duplicated()
    if duplicated.any():
        problems.append(f"{duplicated.sum()} image_path values appear more than once")

    # the point of the whole exercise: no training row may reach into a test set
    held_out = df["split"].isin(TEST_SPLITS)
    test_stems = set(df.loc[held_out, "source_image_stem"]) - {""}
    test_farms = set(df.loc[held_out, "farm_uid"]) - {"", "nan"}
    training = df[df["split"].isin(TRAIN_SPLITS)]
    leaked = (
        training["source_image_stem"].isin(test_stems)
        | training["farm_uid"].isin(test_farms)
    )
    if leaked.any():
        problems.append(f"{leaked.sum()} train/val rows still reach a test split")
    return problems


def copy_imagery(df, output_dir, mode, workers=8):
    """Put every row's imagery under output_dir at the path image_path already names.

    Each source keeps its own subfolder, so the two that share source photographs each get
    their own copy and neither subfolder depends on the other.
    """
    if mode == "none":
        return 0, []
    jobs = {}
    for row in df.itertuples():
        root = Path(row.source_dataset_path)
        jobs[row.image_path] = root / row.source_relpath
        if row.source_image_path:
            jobs[row.source_image_path] = root / row.source_image_relpath

    missing, copied = [], 0
    def one(item):
        relative, source = item
        target = output_dir / relative
        if target.exists() or target.is_symlink():
            return None
        if not source.exists():
            return relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(source, target)
        else:
            target.symlink_to(os.path.relpath(source.resolve(), target.parent))
        return ""

    total = len(jobs)
    print(f"{mode}ing {total} image files into {output_dir}/ ...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(one, jobs.items()), 1):
            if result:
                missing.append(result)
            elif result == "":
                copied += 1
            if done % 2000 == 0:
                print(f"  {done}/{total}", flush=True)
    return copied, missing


def class_table(df, index):
    """crops per <index> x class, counting a multi-label row under each of its classes."""
    exploded = (
        df[df["n_classes"] > 0]
        .assign(cls=lambda d: d["crop_classes"].str.split(","))
        .explode("cls")
        .reset_index(drop=True)
    )
    if exploded.empty:
        return "(no labelled rows)"
    return pd.crosstab(exploded["cls"], exploded[index]).to_string()


def readme(df, totals, args, missing):
    """The provenance document, written from this run's own numbers."""
    counts = {name: int((df["split"] == name).sum()) for name in SPLIT_FILES}
    groups = df.groupby("split")["group_id"].nunique()
    sources = [
        (name, df[df["source_dataset"] == name])
        for name in SOURCE_ORDER
        if (df["source_dataset"] == name).any()
    ]

    lines = [
        "# FLIP master dataset — original_master_2026_09_04",
        "",
        f"Built by `gen_dataset_master.py` on {pd.Timestamp.today():%Y-%m-%d} from three "
        "existing datasets. Every row records where it came from and how it was placed.",
        "",
        f"**Built from:** [{THIS_REPO_NAME}]({THIS_REPO}) — `gen_dataset_master.py` "
        "assembles this directory, `gen_dataset_croplevel.py` produces the crop-level "
        "labels one of the three sources depends on, and `labelled_sheets/` holds the "
        "labelling workbooks those came from. Start there to rebuild or extend this "
        "dataset.",
        "",
        f"**{len(df):,} rows** over **{df['group_id'].nunique():,} groups**, split across "
        f"{len(SPLIT_FILES)} csvs and repeated whole in `dataset.csv`. Every row of every "
        "source appears exactly once: nothing was dropped.",
        "",
        "## where the data came from",
        "",
        "| key | source | rows | groups | label level | group is |",
        "|---|---|---|---|---|---|",
    ]
    for name, group in sources:
        lines.append(
            f"| `{name}` | `{group['source_dataset_path'].iloc[0]}` | {len(group):,} | "
            f"{group['group_id'].nunique():,} | {group['label_level'].iloc[0]}-level | "
            f"the {group['group_level'].iloc[0]} |"
        )
    lines += [
        "",
        "Per source file, as read:",
        "",
        "```",
    ]
    ordered = sorted(totals.items(), key=lambda kv: SOURCE_ORDER.index(kv[0][0]))
    for (dataset, source_file), n in ordered:
        lines.append(f"  {dataset:<15} {source_file:<24} {n:>6,} rows")
    lines += ["```", ""]

    for number, name in enumerate(SOURCE_ORDER, 1):
        note = SOURCE_NOTES[name]
        group = df[df["source_dataset"] == name]
        splits = group.groupby("split").size().sort_values(ascending=False)
        lines += [
            f"### {number}. `{name}` — `{Path(group['source_dataset_path'].iloc[0]).name}`",
            "",
            f"Repository: [{note['repo_name']}]({note['repo']})",
            "",
            f"**Built by.** {note['built_by']}.",
            "",
            f"**What a row is.** {note['rows']}",
            "",
            f"**How it was split, upstream.** {note['split']}",
            "",
            "Where its rows landed here:",
            "",
            "```",
        ]
        for split, n in splits.items():
            lines.append(f"  {split:<28} {n:>6,} rows")
        lines += ["```", ""]

    lines += [
        "### these sources are not independent",
        "",
        NOT_INDEPENDENT,
        "",
        "## how the splits were formed",
        "",
        "| split | rows | groups | built from |",
        "|---|---|---|---|",
        f"| `train` | {counts['train']:,} | {groups.get('train', 0):,} | autocrops train + "
        "generalisation train + historical train + ~80% of historical hpai |",
        f"| `val` | {counts['val']:,} | {groups.get('val', 0):,} | autocrops val + "
        "generalisation val + historical val + ~20% of historical hpai |",
        f"| `train_overlap` | {counts['train_overlap']:,} | "
        f"{groups.get('train_overlap', 0):,} | train rows pulled for reaching a test set |",
        f"| `val_overlap` | {counts['val_overlap']:,} | {groups.get('val_overlap', 0):,} | "
        "val rows pulled for reaching a test set |",
        f"| `test_autocrops` | {counts['test_autocrops']:,} | "
        f"{groups.get('test_autocrops', 0):,} | autocrops `test_df.csv` |",
        f"| `test_autocrop_gen_vic` | {counts['test_autocrop_gen_vic']:,} | "
        f"{groups.get('test_autocrop_gen_vic', 0):,} | generalisation "
        "`relabelled_test_df.csv` — VIC reaches, crop-level labels |",
        f"| `test_gen_original` | {counts['test_gen_original']:,} | "
        f"{groups.get('test_gen_original', 0):,} | historical `gen_all_df.csv`, less the "
        "images whose crops are in generalisation train/val |",
        f"| `test_gen_original_overlap` | {counts['test_gen_original_overlap']:,} | "
        f"{groups.get('test_gen_original_overlap', 0):,} | the part of `gen_all_df.csv` "
        "that lost to generalisation's split |",
        f"| `test_original` | {counts['test_original']:,} | "
        f"{groups.get('test_original', 0):,} | historical `test_df.csv` |",
        "",
        "### test wins",
        "",
        "A train/val row whose `farm_uid` **or** source-image stem appears in any of the "
        f"four test sets ({', '.join('`' + s + '`' for s in TEST_SPLITS)}) was pulled out "
        "of train/val. Pulled rows were **not deleted** — they are in "
        "`train_overlap.csv` and `val_overlap.csv`, with `overlap_reason` saying which "
        "key matched, and their imagery is copied like any other row. The pull is applied "
        "to whole groups so a group is never divided between a split and its overlap.",
        "",
        "### the gen_all exception",
        "",
        "`gen_all_df.csv` is the whole-image view of exactly the imagery the "
        "`generalisation` dataset relabelled at crop level — every one of "
        "generalisation's train/val source stems is inside it. A literal test-wins would "
        "therefore have sent **all** of generalisation's train/val to overlap and left "
        "the relabelling contributing nothing to training. Instead `gen_all` is divided "
        "first: the "
        f"{counts['test_gen_original_overlap']:,} images whose crops are in "
        "generalisation train/val go to `test_gen_original_overlap.csv`, and the "
        f"remaining {counts['test_gen_original']:,} form `test_gen_original.csv`. "
        "Generalisation's training rows survive and `test_gen_original` stays genuinely "
        "held out. Test-wins applies unmodified everywhere else.",
        "",
        "### the two case-study test sets overlap on purpose",
        "",
        "`test_gen_original` also contains the source images behind "
        "`test_autocrop_gen_vic`. Both are test sets, so there is no training leakage, "
        "and both are kept because they measure different things: whole-image labels from "
        "the original pipeline against crop-level labels from the relabelling workbooks. "
        "Affected rows name the other split in `overlap_with`. **Never pool scores across "
        "the two** — it would count the same imagery twice.",
        "",
        "## grouping",
        "",
        "A group is **dataset x identifier x imagery source**, and it is the unit "
        "training draws on — so it follows the level the labels were actually assigned "
        "at. Where one farm-level label covers every crop of a farm in one aerial "
        "capture, the group is that farm-and-capture. Where each image carries its own "
        "label, the group is the single image.",
        "",
        "| source | identifier | imagery source | groups | rows per group |",
        "|---|---|---|---|---|",
    ]
    for name, group in sources:
        sizes = group.groupby("group_id").size()
        identifier, imagery = GROUP_SPEC[name]["describe"]
        lines.append(
            f"| `{name}` | {identifier} | {imagery} | {sizes.size:,} | "
            f"mean {sizes.mean():.1f}, max {sizes.max()} |"
        )
    lines += [
        "",
        "- `group_id` — `<source_dataset>:<group_identifier>:<group_imagery>`",
        "- `group_identifier` / `group_imagery` — the two parts, kept as their own "
        "columns so the grouping can be inspected or rebuilt",
        "- `group_level` — `farm+capture` or `image`",
        "- `group_size` — rows in this group",
        "",
        f"**Every group holds at most {MAX_GROUP} rows**, asserted at build time. That "
        "ceiling is the builder's limit of ten building clusters per farm per capture, "
        "not a cap applied here.",
        "",
        "**Why the capture is in the key.** A farm is often flown more than once — 675 "
        "of the 2,429 `autocrops` farms appear in two or more ECW captures. Crops from "
        "two flights are two photographs of the farm, not one, so keying on the farm "
        "alone merged them and produced groups of up to 37. Splitting on the capture as "
        "well keeps each group to a single photograph of a single place.",
        "",
        "**`generalisation` groups by image, not by farm.** Its crops sit on the same "
        "farms and captures as `autocrops`, but every one of them was labelled "
        "individually in the workbooks, so the crop is what training draws on. Split "
        "integrity does not rest on that: this source's own split is geographic and "
        "farm-grouped upstream, so no farm straddles its train, val and test even though "
        "`group_id` no longer says so. `farm_uid` and `ecw_stem` are on every row for "
        "anyone who wants to regroup it by farm and capture.",
        "",
        "**Why the dataset is in the key.** `autocrops` and `generalisation` share 24 "
        "`farm_uid`s, and a bare identifier would merge groups across two sources whose "
        "splits were assigned independently. **Groups never span datasets.** The raw "
        "`farm_uid` is still on every row, so merging them later stays a deliberate act.",
        "",
        "No `group_id` appears in more than one split — also asserted at build time.",
        "",
        "`group_level` is about how many images share a label; `label_level` is about "
        "what the label describes. They differ for `historical`: its groups are single "
        "images, but each image is a whole farm, so its labels are farm-level.",
        "",
        "Group sizes are uneven — `autocrops` averages several rows per group while the "
        "other two are one row per group — so sampling by group and sampling by row give "
        "quite different class balances. That is a training-side decision this dataset "
        "does not make for you; `group_size` is here so either is available.",
        "",
        "## labels",
        "",
        f"{len(BINARY)} `binary_<class>` columns, multi-hot: "
        f"{', '.join('`' + c + '`' for c in CLASSES)}.",
        "",
        "Sources disagree on which classes they carry — `autocrops` has 11, "
        "`generalisation` 13 (it adds `paddock` and `other_industrial`), `historical` 10 "
        "(no `goats`). A class a source never assessed is written **`False`**. That means "
        "*not assessed*, not *verified absent*. `binary_paddock` and "
        "`binary_other_industrial` are only ever `True` on `generalisation` rows, so "
        "treating their `False` values as negatives will train against the other two "
        "sources rather than with them.",
        "",
        f"`label_status` is `labelled` for {int((df['label_status'] == 'labelled').sum()):,} "
        "rows. The rest carry a placeholder class from `gen_all_df.csv` and no positive "
        "label at all:",
        "",
        "```",
    ]
    for status, n in df.loc[df["label_status"] != "labelled", "label_status"].value_counts().items():
        lines.append(f"  {status:<16} {n:>6,}")
    lines += [
        "```",
        "",
        "They are kept so nothing is lost, but filter them out before training or "
        "scoring.",
        "",
        "## columns",
        "",
        "**Provenance** (on every file, standalone and combined alike)",
        "",
        "- `source_dataset` — `autocrops` | `generalisation` | `historical`",
        "- `source_dataset_path` — the directory it was read from",
        "- `source_file` — the csv within it",
        "- `source_split` — that file's own notion of the split, verbatim",
        "- `split` — the split in this dataset",
        "- `label_level` — `farm` or `crop`; what the label describes",
        "- `label_status` — `labelled`, or the placeholder class for unlabelled rows",
        "",
        "**Grouping** — `group_id`, `group_identifier`, `group_imagery`, "
        "`group_level`, `group_size`",
        "",
        "**Overlap**",
        "",
        "- `overlap_with` — other splits this row's farm or source image also appears in",
        "- `overlap_reason` — why a row was pulled to `*_overlap`: `farm_uid`, "
        "`source_image`, both, or `group` (pulled to keep its group whole)",
        "",
        "**Identity and imagery**",
        "",
        "- `image_path` — relative to this directory, e.g. "
        "`autocrops/farm-…/buildings/….tif`",
        "- `source_relpath` — the same file relative to `source_dataset_path`, so a row "
        "can always be traced back to the build it came from",
        "- `source_image_path` / `source_image_relpath` — the source photograph the crop "
        "was cut from, where the source recorded one",
        "- `source_image_name` / `source_image_stem` — the photograph's name; the stem "
        "(lower-cased, extension stripped) is the key all three sources join on",
        "- `farm_uid` — `autocrops` and `generalisation` only; `historical` has none",
        "",
        "**Labels** — `processed_class` (the primary class), `crop_classes` "
        "(comma-separated, canonical order), `n_classes`, and the 13 `binary_` columns.",
        "",
        "Source-specific columns are carried through under their own names and left blank "
        "where a source did not have them.",
        "",
        "## imagery",
        "",
        f"Each source's files are under its own subfolder "
        f"({', '.join('`' + d + '/`' for d in SUBDIR.values())}) at the relative path "
        "`image_path` names. `autocrops` and `historical` share source photographs; each "
        f"gets its own copy so no subfolder depends on another. Mode for this build: "
        f"`{args.imagery}`.",
        "",
        "## rows per split x class",
        "",
        "A multi-label row counts under each of its classes.",
        "",
        "```",
        class_table(df, "split"),
        "```",
        "",
        "## rows per source x class",
        "",
        "```",
        class_table(df, "source_dataset"),
        "```",
        "",
    ]
    if missing:
        lines += [
            "## warnings",
            "",
            f"{len(missing)} image files named in the csvs were not found in their source "
            f"directory, e.g. `{missing[0]}`.",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--autocrops", type=Path, default=DEFAULT_AUTOCROPS)
    parser.add_argument("--generalisation", type=Path, default=DEFAULT_GENERALISATION)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"where the master dataset is written (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--imagery", choices=["copy", "symlink", "none"], default="copy",
                        help="copy the imagery into the output (default), symlink it, or "
                             "write only the csvs")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                        help=f"share of the hpai rows given to val (default: {VAL_FRACTION})")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--dry-run", action="store_true",
                        help="write the csvs and README but no imagery")
    args = parser.parse_args()
    if args.dry_run:
        args.imagery = "none"

    df = pd.concat([
        load_autocrops(args.autocrops),
        load_generalisation(args.generalisation),
        load_historical(args.historical),
    ], ignore_index=True)
    totals = df.groupby(["source_dataset", "source_file"], sort=False).size().to_dict()
    print(f"read {len(df):,} rows from {len(totals)} source files")

    df = assign_splits(df, args.val_fraction, args.random_state)
    df = apply_test_wins(df)
    df = annotate(df)

    problems = check(df, totals)
    if problems:
        print("\nrefusing to write:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    df = df.reindex(columns=COLUMNS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "dataset.csv", index=False)
    print(f"\n{'split':<28}{'rows':>8}{'groups':>9}")
    for name, filename in SPLIT_FILES.items():
        part = df[df["split"] == name]
        part.to_csv(args.output_dir / filename, index=False)
        print(f"  {filename:<26}{len(part):>8,}{part['group_id'].nunique():>9,}")
    print(f"  {'dataset.csv':<26}{len(df):>8,}{df['group_id'].nunique():>9,}")

    copied, missing = copy_imagery(df, args.output_dir, args.imagery)
    if args.imagery != "none":
        print(f"{copied:,} image files {args.imagery}ed")
        if missing:
            print(f"  warning: {len(missing)} files not found, e.g. {missing[:3]}")

    (args.output_dir / "README.md").write_text(readme(df, totals, args, missing))
    print(f"\nwrote {args.output_dir}/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
