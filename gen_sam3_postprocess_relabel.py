"""Attach the SAM3 structure detections to the 2026-09-04 master build, and relabel.

For every csv in `original_master_2026_09_04/` this writes a `sam3_*.csv` beside it that
carries the same rows in the same order with the SAM3 detections joined on, so
`sam3_train_df.csv` is `train_df.csv` plus SAM3 columns and so on. The prefix keeps the
SAM3 files together in a listing. The originals are not touched.

*What SAM3 contributes.* `sam3_pipeline.py` in the dataset builder ran SAM3 once per text
prompt over every building crop of the two builder releases the master draws on, and
recorded per prompt how many objects it found, their mean and total area in square metres
and the strongest detection score. Five prompts are a building gate (`roof`, `building`,
`shed`, `shelter`, `house`); five are class features that all but never fire on a paddock
(`water tank`, `silo`, `vehicle`, `cattle yard`, `pond`). The join is on the crop's own
relative path, so it is exact: every `autocrops` and `generalisation` row lands, and the
`historical` rows - whole-farm photographs that were never cut into crops - get nothing,
flagged `sam3_available = False`.

*The relabel.* 98 % of the master's training labels are farm-level: a farm's ten crops all
carry the farm's class, so a dairy's nine paddocks are "dairy". The 2026-09-04 evaluation
names that as the mechanism by which the master models gain on livestock and lose on
paddock. SAM3's building gate is checked here against the 1,629 human-labelled crops:
"no gate detection" picks out paddock with precision 0.96 and recall 0.83 at the
pipeline's own 0.5 detection threshold. So every *farm-level* crop with no building
detected is relabelled `paddock`, and every human-labelled crop is left exactly as it was.
The original label travels with the row in `*_pre_sam3` columns and `sam3_relabel` says
what happened, so nothing is lost and a consumer can undo it column by column.
`--no-relabel` attaches the features and changes no label.

*Columns added*, all prefixed `sam3_`:

    sam3_available, sam3_extract_type, sam3_model, sam3_threshold, sam3_mask_threshold
    sam3_width, sam3_height, sam3_res_x_m, sam3_res_y_m, sam3_crop_area_m2
    sam3_{prompt}_{count,mean_area_m2,total_area_m2,max_score}   the pipeline's own
    sam3_{prompt}_area_frac       total_area_m2 / crop_area_m2, so crops of different
                                  sizes and resolutions compare
    sam3_gate_score               max of the gate prompts' max_score
    sam3_gate_count               max of the gate prompts' count
    sam3_gate_area_frac           max of the gate prompts' area_frac
    sam3_building                 sam3_gate_score >= --gate-min-score
    sam3_relabel                  "paddock" where the row was relabelled, else ""
    crop_classes_pre_sam3, processed_class_pre_sam3, n_classes_pre_sam3

`gen_post_classifier.py` reads these files back and builds classifiers on the SAM3
columns. A summary of the join and the relabel is written to
`original_master_2026_09_04/sam3_README.md`.

Usage:

    .venv/bin/python gen_sam3_postprocess_relabel.py
    .venv/bin/python gen_sam3_postprocess_relabel.py --gate-min-score 0.7
    .venv/bin/python gen_sam3_postprocess_relabel.py --no-relabel
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------

MASTER = Path("/home/mannixe/FLIP/flip-data-labelling/original_master_2026_09_04")
SAM3 = Path("/home/mannixe/FLIP/flip-geoimage-dataset-builder/sam3_results")
PREFIX = "sam3_"

# Which SAM3 run scored which master source. The master's `source_dataset` names the
# builder release its crops came from; `sam3_pipeline.py --extract-type` names the run.
# `historical` has no entry: its rows are whole-farm photographs, never cut into crops.
SOURCE_TO_EXTRACT = {
    "autocrops": "training",
    "generalisation": "generalisation",
    "generalisation_extra": "generalisation_extra",
    "generalisation_extra_test": "generalisation_extra_test",
}

# The prompts whose absence says "paddock" - `DEFAULT_PROMPTS` in sam3_pipeline.py, in
# the pipeline's column spelling (spaces to underscores).
GATE_PROMPTS = ["roof", "building", "shed", "shelter", "house"]
STATS = ["count", "mean_area_m2", "total_area_m2", "max_score"]
META = ["width", "height", "res_x_m", "res_y_m", "extract_type", "threshold", "mask_threshold", "model"]

# The label columns the relabel rewrites, and the order the build writes `binary_*` in.
LABEL_COLUMNS = ["crop_classes", "processed_class", "n_classes"]
RELABEL_TO = "paddock"


def binary_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("binary_")]


# --------------------------------------------------------------------------------------
# detections
# --------------------------------------------------------------------------------------


def load_detections(root: Path) -> dict[str, pd.DataFrame]:
    """One wide table per SAM3 run, indexed by the crop's relative path."""
    found = {}
    for extract in SOURCE_TO_EXTRACT.values():
        path = root / extract / "sam3_detections.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if table["image_path"].duplicated().any():
            raise SystemExit(f"{path}: image_path is not unique")
        found[extract] = table.set_index("image_path")
        print(f"{path.relative_to(root.parent)}: {len(table):,} crops")
    if not found:
        raise SystemExit(f"no sam3_detections.csv under {root}")
    return found


def prompts_of(table: pd.DataFrame) -> list[str]:
    """The prompts a run was scored with, read off its `{prompt}_count` columns."""
    return [c.removesuffix("_count") for c in table.columns if c.endswith("_count")]


def features(table: pd.DataFrame, gate_min_score: float) -> pd.DataFrame:
    """The `sam3_*` columns for one run: the pipeline's own numbers plus the derived ones."""
    prompts = prompts_of(table)
    missing = [p for p in GATE_PROMPTS if p not in prompts]
    if missing:
        raise SystemExit(f"gate prompts {missing} were not scored; found {prompts}")

    out = pd.DataFrame(index=table.index)
    out["sam3_available"] = True
    for name in META:
        out[f"sam3_{name}"] = table[name]
    area = table["width"] * table["res_x_m"] * table["height"] * table["res_y_m"]
    out["sam3_crop_area_m2"] = area.round(1)
    for prompt in prompts:
        for stat in STATS:
            out[f"sam3_{prompt}_{stat}"] = table[f"{prompt}_{stat}"]
        out[f"sam3_{prompt}_area_frac"] = (table[f"{prompt}_total_area_m2"] / area).round(5)
    # Named `gate_*`, not `building_*`: "building" is also one of the prompts, with its
    # own sam3_building_{count,...} columns above.
    out["sam3_gate_score"] = table[[f"{p}_max_score" for p in GATE_PROMPTS]].max(axis=1)
    out["sam3_gate_count"] = table[[f"{p}_count" for p in GATE_PROMPTS]].max(axis=1)
    out["sam3_gate_area_frac"] = out[[f"sam3_{p}_area_frac" for p in GATE_PROMPTS]].max(axis=1)
    out["sam3_building"] = out["sam3_gate_score"] >= gate_min_score
    return out


# --------------------------------------------------------------------------------------
# join and relabel
# --------------------------------------------------------------------------------------


def attach(frame: pd.DataFrame, detections: dict[str, pd.DataFrame], gate_min_score: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`frame` with the SAM3 columns joined on, plus a per-source tally of the join."""
    pieces, tally = [], []
    for source, rows in frame.groupby("source_dataset", sort=False, dropna=False):
        extract = SOURCE_TO_EXTRACT.get(source)
        table = detections.get(extract) if extract else None
        if table is None:
            joined = pd.DataFrame(index=rows.index)
            joined["sam3_available"] = False
            matched = 0
        else:
            joined = features(table, gate_min_score).reindex(rows["source_relpath"].to_numpy())
            joined.index = rows.index
            joined["sam3_available"] = joined["sam3_available"].fillna(False).astype(bool)
            matched = int(joined["sam3_available"].sum())
        pieces.append(joined)
        tally.append(
            {
                "source_dataset": source,
                "sam3_run": extract or "",
                "rows": len(rows),
                "with_sam3": matched,
                "without_sam3": len(rows) - matched,
            }
        )
    joined = pd.concat(pieces).reindex(frame.index)
    # The same column set in every file, whatever mix of sources it holds - a file of
    # nothing but historical rows still carries every sam3_* column, all empty.
    columns = list(dict.fromkeys(
        ["sam3_available"]
        + [c for table in detections.values() for c in features(table.head(1), gate_min_score).columns]
    ))
    joined = joined.reindex(columns=columns)
    joined["sam3_available"] = joined["sam3_available"].fillna(False).astype(bool)
    joined["sam3_building"] = joined["sam3_building"].map(
        lambda v: "" if pd.isna(v) else str(bool(v))
    )
    return pd.concat([frame, joined], axis=1), pd.DataFrame(tally)


def relabel(frame: pd.DataFrame) -> pd.DataFrame:
    """Farm-level crops with no building detected become paddock. Returns the tally.

    Only rows whose label came from the farm are touched: a human looked at every
    crop-level row and its label stands. Rows without SAM3 are left alone too.
    """
    frame["sam3_relabel"] = ""
    for name in LABEL_COLUMNS:
        frame[f"{name}_pre_sam3"] = frame[name]
    eligible = (
        (frame["label_level"] == "farm")
        & (frame["sam3_available"].astype(str) == "True")
        & (frame["sam3_building"] == "False")
        & (frame["processed_class"] != RELABEL_TO)
    )
    tally = (
        frame.loc[eligible]
        .groupby(["source_dataset", "processed_class"], sort=True)
        .size()
        .rename("relabelled_to_paddock")
        .reset_index()
        .rename(columns={"processed_class": "from_class"})
    )
    binary = binary_columns(frame)
    if f"binary_{RELABEL_TO}" not in binary:
        raise SystemExit(f"no binary_{RELABEL_TO} column to relabel into")
    frame.loc[eligible, "sam3_relabel"] = RELABEL_TO
    frame.loc[eligible, "crop_classes"] = RELABEL_TO
    frame.loc[eligible, "processed_class"] = RELABEL_TO
    frame.loc[eligible, "n_classes"] = "1"
    for column in binary:
        frame.loc[eligible, column] = str(column == f"binary_{RELABEL_TO}")
    return tally


def gate_check(frame: pd.DataFrame) -> dict[str, float] | None:
    """How the gate does on the crops a human labelled: the only rows it can be checked on."""
    rows = frame[
        (frame["label_level"] == "crop") & (frame["sam3_available"].astype(str) == "True")
    ]
    if rows.empty:
        return None
    paddock = rows["processed_class_pre_sam3"] == RELABEL_TO
    no_building = rows["sam3_building"] == "False"
    return {
        "crops": len(rows),
        "paddock": int(paddock.sum()),
        "paddock_with_no_building": float(no_building[paddock].mean()),
        "building_with_no_building": float(no_building[~paddock].mean()),
        "rule_precision": float((no_building & paddock).sum() / max(no_building.sum(), 1)),
        "rule_recall": float((no_building & paddock).sum() / max(paddock.sum(), 1)),
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def write_readme(
    path: Path,
    written: list[dict],
    relabels: pd.DataFrame,
    check: dict | None,
    gate_min_score: float,
    do_relabel: bool,
) -> None:
    lines = [
        f"# SAM3 columns for {MASTER.name}",
        "",
        f"Written by `gen_sam3_postprocess_relabel.py` on {date.today().isoformat()} from "
        f"`{SAM3}`. Every `{PREFIX}*.csv` is the csv of the same name with the SAM3 detections "
        "joined on, same rows in the same order.",
        "",
        "## the join",
        "",
        "| file | rows | with SAM3 | without | relabelled to paddock |",
        "|---|---|---|---|---|",
    ]
    for row in written:
        lines.append(
            f"| `{row['file']}` | {row['rows']:,} | {row['with_sam3']:,} | "
            f"{row['without_sam3']:,} | {row['relabelled']:,} |"
        )
    lines += [
        "",
        "Rows without SAM3 are the `historical` whole-farm photographs, which were never cut "
        "into crops; every `autocrops` and `generalisation` crop is matched on its relative path.",
        "",
        "## the relabel",
        "",
    ]
    if not do_relabel:
        lines.append("Run with `--no-relabel`: no label was changed.")
    else:
        lines += [
            f"A farm-level crop (`label_level = farm`) whose strongest building-gate detection "
            f"(`sam3_gate_score`, the max over {', '.join(GATE_PROMPTS)}) is below "
            f"{gate_min_score} is relabelled `{RELABEL_TO}`. Crop-level rows, which a human "
            "labelled one by one, are never touched. The original label is kept in "
            "`crop_classes_pre_sam3`, `processed_class_pre_sam3` and `n_classes_pre_sam3`; "
            "`sam3_relabel` marks the rows that changed; `binary_*` is recomputed.",
            "",
        ]
        if check:
            lines += [
                "Checked on the human-labelled crops in `dataset.csv`:",
                "",
                f"- {check['crops']:,} crops, {check['paddock']:,} of them paddock",
                f"- no building detected on {check['paddock_with_no_building']:.1%} of paddock "
                f"crops and {check['building_with_no_building']:.1%} of building crops",
                f"- the rule *no building means paddock* has precision "
                f"{check['rule_precision']:.3f} and recall {check['rule_recall']:.3f}",
                "",
            ]
        if not relabels.empty:
            lines += ["Relabelled in `dataset.csv`, by source and original class:", "",
                      "| source | from | rows |", "|---|---|---|"]
            for _, row in relabels.iterrows():
                lines.append(
                    f"| {row['source_dataset']} | {row['from_class']} | "
                    f"{row['relabelled_to_paddock']:,} |"
                )
            lines.append("")
    lines += [
        "## columns",
        "",
        "```",
        "sam3_available                 True where a SAM3 row was joined",
        "sam3_extract_type              which sam3_pipeline.py run scored the crop",
        "sam3_model, sam3_threshold, sam3_mask_threshold",
        "sam3_width, sam3_height, sam3_res_x_m, sam3_res_y_m, sam3_crop_area_m2",
        "sam3_{prompt}_count            objects found for the prompt",
        "sam3_{prompt}_mean_area_m2     mean mask area",
        "sam3_{prompt}_total_area_m2    total mask area",
        "sam3_{prompt}_max_score        strongest detection score, 0 when none",
        "sam3_{prompt}_area_frac        total_area_m2 / crop_area_m2",
        "sam3_gate_score                max over the gate prompts of max_score",
        "sam3_gate_count                max over the gate prompts of count",
        "sam3_gate_area_frac            max over the gate prompts of area_frac",
        "sam3_building                  sam3_gate_score >= gate-min-score",
        "sam3_relabel                   'paddock' where relabelled, else empty",
        "crop_classes_pre_sam3, processed_class_pre_sam3, n_classes_pre_sam3",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--master", type=Path, default=MASTER, help="the master build directory")
    parser.add_argument("--sam3", type=Path, default=SAM3, help="the dataset builder's sam3_results/")
    parser.add_argument(
        "--gate-min-score",
        type=float,
        default=0.5,
        help="a crop is a building if any gate prompt scores at least this (default 0.5, "
        "the pipeline's own detection threshold, so this is 'any detection at all')",
    )
    parser.add_argument("--no-relabel", action="store_true", help="attach the columns, change no label")
    args = parser.parse_args()

    detections = load_detections(args.sam3)
    for extract, table in detections.items():
        print(f"  {extract}: prompts {prompts_of(table)}")

    written, relabels, check = [], pd.DataFrame(), None
    for path in sorted(args.master.glob("*.csv")):
        if path.name.startswith(PREFIX):
            continue
        # Read as text so every existing value is written back exactly as it was.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        if "source_dataset" not in frame.columns or "source_relpath" not in frame.columns:
            print(f"{path.name}: no source_dataset / source_relpath, skipped")
            continue
        frame, tally = attach(frame, detections, args.gate_min_score)
        relabelled = 0
        if "processed_class" in frame.columns:
            if args.no_relabel:
                frame["sam3_relabel"] = ""
                for name in LABEL_COLUMNS:
                    frame[f"{name}_pre_sam3"] = frame[name]
            else:
                changed = relabel(frame)
                relabelled = int(changed["relabelled_to_paddock"].sum())
                if path.stem == "dataset":
                    relabels = changed
            if path.stem == "dataset":
                check = gate_check(frame)
        out = path.with_name(f"{PREFIX}{path.name}")
        frame.to_csv(out, index=False, float_format="%.6g")
        written.append(
            {
                "file": out.name,
                "rows": len(frame),
                "with_sam3": int(tally["with_sam3"].sum()),
                "without_sam3": int(tally["without_sam3"].sum()),
                "relabelled": relabelled,
            }
        )
        print(
            f"{out.name}: {len(frame):,} rows, {written[-1]['with_sam3']:,} with SAM3, "
            f"{relabelled:,} relabelled to {RELABEL_TO}"
        )
        for _, row in tally.iterrows():
            print(f"    {row['source_dataset']:<16} {row['with_sam3']:>6,} / {row['rows']:<6,} via {row['sam3_run'] or '-'}")

    if check:
        print(
            f"\ngate check on {check['crops']:,} human-labelled crops ({check['paddock']:,} paddock): "
            f"no building on {check['paddock_with_no_building']:.1%} of paddock and "
            f"{check['building_with_no_building']:.1%} of building crops; "
            f"rule 'no building -> paddock' precision {check['rule_precision']:.3f} "
            f"recall {check['rule_recall']:.3f}"
        )
    if not relabels.empty:
        print("\nrelabelled in dataset.csv:")
        for _, row in relabels.iterrows():
            print(f"  {row['source_dataset']:<16} {row['from_class']:<18} {row['relabelled_to_paddock']:>6,}")

    readme = args.master / f"{PREFIX}README.md"
    write_readme(readme, written, relabels, check, args.gate_min_score, not args.no_relabel)
    print(f"\nwrote {len(written)} csvs and {readme.name} in {args.master}")


if __name__ == "__main__":
    main()
