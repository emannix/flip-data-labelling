"""Score the ComFe generalisation runs against the hand labels in `labelled_sheets/`.

The generalisation pass is prediction-only: `test_csv_label_override` forces every row
of `dataset.csv` to a single class so the datamodule can map the split onto the training
vocabulary, which means the labels saved next to the predictions are meaningless (see
the comment in `goo/datasets/base_csv_dataset_image.py`). The real annotations are the
relabelling workbooks, and this script joins the two back together.

Two levels of evaluation, because the workbooks label at two levels:

* image  - "Image labels" gives one class per building crop, joined to `dataset.csv` on
           `image_path`. Positives for a class are the crops carrying that label; every
           other labelled crop (including Paddock / Other/Industrial) is a negative.
           `Ambiguous` and `Multiple Classes` crops are dropped from every class.
* farm   - "Farm labels" gives a multi-label X per farm, joined on `PFI`. A farm's score
           for a class is the maximum (default) over its crops, so this asks "would the
           class be surfaced for this farm at all", which is how the model is used.

Scores are `average_precision_score` per class - the area under precision-recall - which
is what you want here: the classes are rare, the operating threshold is not fixed, and
the model emits an unnormalised per-class score rather than a decision. Each AP is
reported next to the prevalence of the class, since a random ranker scores the
prevalence, and confidence intervals come from a bootstrap over the evaluation units.
The old and new models are compared on the *same* bootstrap resamples so the paired
difference carries its own interval.

Alongside the AP tables the script draws confusion matrices of the annotated label
against the model's argmax class. AP never fixes an operating point, so it says nothing
about *where* a class leaks; the confusion matrices do, and they include the two
annotated non-classes (Paddock, Other/Industrial) that the models have no output for and
must therefore mis-assign somewhere.

Everything lands in `output_eval/`.

Usage:

    .venv/bin/python gen_evaluation.py
    .venv/bin/python gen_evaluation.py --aggregation mean --min-confidence Medium
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message="Data Validation extension is not supported")

DATASET = Path(
    "/home/mannixe/FLIP/flip-geoimage-dataset-builder/"
    "original_new_2026_07_24_generalisation/dataset.csv"
)
LABELLED = Path("labelled_sheets")
OUTPUT = Path("output_eval")

# Each model is a set of seed runs sharing a checkpoint sweep. `train_csv` is the
# training split whose sorted class vocabulary fixes the column order of the saved
# y_hat, i.e. the vocabulary the checkpoint's head was fitted against.
MODELS = {
    "old (2026-03-06, flip_historical)": {
        "key": "old",
        "runs": Path(
            "/home/mannixe/FLIP/comfe-run-flip/view/flip_orig_comfe/"
            "flip_comfe_multiclass2_gen"
        ),
        "train_csv": Path(
            "/home/mannixe/FLIP/flip-dataset-processing/output/flip_historical/train_df.csv"
        ),
    },
    "new (2026-08-07, original_new_2026_07_24)": {
        "key": "new",
        "runs": Path(
            "/home/mannixe/FLIP/comfe-run-flip/view/flip_2026_07_24/comfe_gen"
        ),
        "train_csv": Path(
            "/home/mannixe/FLIP/flip-geoimage-dataset-builder/"
            "original_new_2026_07_24/train_df.csv"
        ),
    },
}

# Column order of the saved y_hat: sorted unique `processed_class` of the training split.
CLASSES = [
    "aqua",
    "backyardpig",
    "beef",
    "commercialpig",
    "dairy",
    "freerangepig",
    "horse",
    "poultry",
    "residential",
    "sheep",
]

# The workbooks carry a `goat` column and a `goats` processed_class; neither model has a
# goat output, so the class is reported as unevaluable rather than silently dropped.
WORKBOOK_ONLY_CLASSES = ["goat"]

# Image labels that make no claim about any single class.
EXCLUDED_IMAGE_LABELS = {"Ambiguous", "Multiple Classes"}

CONFIDENCE_ORDER = {"High": 3, "Medium": 2, "Low": 1}

FARM_SHEET = "Farm labels"
IMAGE_SHEET = "Image labels"


# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------


def check_class_order(train_csv: Path) -> None:
    """The head's column order is the sorted training vocabulary - confirm it is CLASSES."""
    if not train_csv.exists():
        print(f"  ! {train_csv} missing, assuming the default class order")
        return
    classes = sorted(set(pd.read_csv(train_csv, keep_default_na=False)["processed_class"]))
    if classes != CLASSES:
        raise SystemExit(
            f"{train_csv} has class order {classes}, expected {CLASSES}; the saved y_hat "
            "columns would not line up with the class names."
        )


def load_run(run_dir: Path, n_rows: int) -> pd.DataFrame:
    """One seed run's per-class scores, reindexed onto `dataset.csv` row order."""
    predictions = run_dir / "tensorboard" / "version_0" / "predictions"
    y_hat_file = next(predictions.glob("*_y_hat_predictions*.csv"))
    index_file = next(predictions.glob("*_index_predictions*.csv"))

    y_hat = pd.read_csv(y_hat_file).to_numpy(dtype=float)
    index = pd.read_csv(index_file)["index"].to_numpy()
    if y_hat.shape != (n_rows, len(CLASSES)):
        raise SystemExit(f"{y_hat_file} is {y_hat.shape}, expected {(n_rows, len(CLASSES))}")
    if not np.array_equal(np.sort(index), np.arange(n_rows)):
        raise SystemExit(f"{index_file} is not a permutation of the {n_rows} dataset rows")

    ordered = np.empty_like(y_hat)
    ordered[index] = y_hat
    return pd.DataFrame(ordered, columns=CLASSES)


def load_model(name: str, spec: dict, n_rows: int) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Every seed of a model, plus the mean-probability ensemble over those seeds."""
    check_class_order(spec["train_csv"])
    run_dirs = sorted(d for d in spec["runs"].iterdir() if d.is_dir())
    seeds = [load_run(d, n_rows) for d in run_dirs]
    ensemble = sum(seeds) / len(seeds)
    print(f"  {name}: {len(seeds)} seed run(s)")
    return seeds, ensemble


def load_workbooks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The image-level and farm-level annotations, concatenated over the workbooks."""
    images, farms = [], []
    for path in sorted(LABELLED.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        book = pd.ExcelFile(path)
        image = book.parse(IMAGE_SHEET)
        farm = book.parse(FARM_SHEET)
        image["workbook"] = path.name
        farm["workbook"] = path.name
        images.append(image)
        farms.append(farm)

    image = pd.concat(images, ignore_index=True)
    farm = pd.concat(farms, ignore_index=True)
    # Trailing blank rows survive the Excel read; a workbook row without a PFI is one.
    farm = farm[farm["Farm PFI"].notna()].copy()
    farm["PFI"] = farm["Farm PFI"].astype(float).astype(int)
    image = image[image["image_path"].notna()].copy()
    return image, farm


# --------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------


def image_truth(
    image: pd.DataFrame, dataset: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Per-crop one-hot truth, matching `dataset.csv` row positions, and the raw label."""
    row_of = pd.Series(np.arange(len(dataset)), index=dataset["image_path"])
    missing = ~image["image_path"].isin(row_of.index)
    if missing.any():
        raise SystemExit(f"{missing.sum()} labelled crops are not in {DATASET}")

    label = image["Label"].astype("string").str.strip()
    keep = label.notna() & ~label.isin(EXCLUDED_IMAGE_LABELS)
    image = image[keep]
    label = label[keep]

    truth = pd.DataFrame(
        {name: (label == name).to_numpy() for name in CLASSES + WORKBOOK_ONLY_CLASSES},
        index=image.index,
    )
    return truth, row_of[image["image_path"]].to_numpy(), label


def farm_truth(farm: pd.DataFrame, min_confidence: str | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-farm multi-label truth (an X in the class column) and the farm PFIs.

    `min_confidence` drops X marks the labeller was less sure of, turning them into
    negatives - a sensitivity check, not the headline, since a hedged X is still a
    sighting of the class.
    """
    floor = CONFIDENCE_ORDER[min_confidence] if min_confidence else 0
    columns = {}
    for name in CLASSES + WORKBOOK_ONLY_CLASSES:
        if name not in farm.columns:
            columns[name] = np.zeros(len(farm), dtype=bool)
            continue
        marked = farm[name].astype("string").str.strip().str.upper().eq("X").fillna(False)
        if floor:
            confidence = farm[f"{name} confidence"].astype("string").str.strip()
            marked &= confidence.map(CONFIDENCE_ORDER).fillna(0).ge(floor)
        columns[name] = marked.to_numpy()
    return pd.DataFrame(columns, index=farm.index), farm["PFI"].to_numpy()


def farm_scores(
    scores: pd.DataFrame, dataset: pd.DataFrame, pfis: np.ndarray, aggregation: str
) -> np.ndarray:
    """Aggregate per-crop scores up to one score per farm, in `pfis` order."""
    per_farm = scores.groupby(dataset["PFI"].astype(int).to_numpy()).agg(aggregation)
    return per_farm.reindex(pfis).to_numpy()


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------


def bootstrap_indices(n: int, reps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def ap_ci(y_true: np.ndarray, y_score: np.ndarray, resamples: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap interval for AP, over resampled evaluation units."""
    values = []
    for draw in resamples:
        truth = y_true[draw]
        if truth.any() and not truth.all():
            values.append(average_precision_score(truth, y_score[draw]))
    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def paired_delta(
    y_true: np.ndarray, a: np.ndarray, b: np.ndarray, resamples: np.ndarray
) -> tuple[float, float, float]:
    """AP(b) - AP(a) with a paired bootstrap interval: both models see the same resample."""
    deltas = []
    for draw in resamples:
        truth = y_true[draw]
        if truth.any() and not truth.all():
            deltas.append(
                average_precision_score(truth, b[draw]) - average_precision_score(truth, a[draw])
            )
    if not deltas:
        return np.nan, np.nan, np.nan
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(low), float(high)


def evaluate(
    level: str,
    truth: pd.DataFrame,
    model_scores: dict[str, np.ndarray],
    model_seeds: dict[str, list[np.ndarray]],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Per-class AP for every model at one evaluation level."""
    n = len(truth)
    resamples = bootstrap_indices(n, reps, seed)
    names = list(model_scores)
    rows = []

    for position, name in enumerate(CLASSES + WORKBOOK_ONLY_CLASSES):
        y_true = truth[name].to_numpy()
        positives = int(y_true.sum())
        row = {
            "level": level,
            "class": name,
            "n": n,
            "positives": positives,
            "prevalence": positives / n if n else np.nan,
        }
        evaluable = name in CLASSES and 0 < positives < n
        for model in names:
            if not evaluable:
                row[f"{model} | AP"] = np.nan
                continue
            y_score = model_scores[model][:, position]
            row[f"{model} | AP"] = average_precision_score(y_true, y_score)
            row[f"{model} | AUC"] = roc_auc_score(y_true, y_score)
            low, high = ap_ci(y_true, y_score, resamples)
            row[f"{model} | AP lo"] = low
            row[f"{model} | AP hi"] = high
            seeds = [average_precision_score(y_true, s[:, position]) for s in model_seeds[model]]
            row[f"{model} | AP seed mean"] = float(np.mean(seeds))
            row[f"{model} | AP seed sd"] = float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0
        if evaluable and len(names) == 2:
            delta, low, high = paired_delta(
                y_true,
                model_scores[names[0]][:, position],
                model_scores[names[1]][:, position],
                resamples,
            )
            row["delta AP (new - old)"] = delta
            row["delta lo"] = low
            row["delta hi"] = high
        rows.append(row)

    table = pd.DataFrame(rows)
    macro = {"level": level, "class": "MACRO (evaluable)", "n": n}
    evaluable = table[table[f"{names[0]} | AP"].notna()]
    macro["positives"] = int(evaluable["positives"].sum())
    macro["prevalence"] = float(evaluable["prevalence"].mean())
    for model in names:
        macro[f"{model} | AP"] = float(evaluable[f"{model} | AP"].mean())
        macro[f"{model} | AUC"] = float(evaluable[f"{model} | AUC"].mean())
    if len(names) == 2:
        macro["delta AP (new - old)"] = float(evaluable["delta AP (new - old)"].mean())
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)


# --------------------------------------------------------------------------------------
# confusion matrices
# --------------------------------------------------------------------------------------


def confusion(label: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    """Annotated label (rows) against the model's argmax class (columns), as counts.

    Paddock and Other/Industrial are kept as their own rows: neither is a model output,
    so every one of those crops is forced onto some class, and which one it lands on is
    the whole story behind a low AP.
    """
    predicted = np.array(CLASSES)[scores.argmax(axis=1)]
    rows = [name for name in CLASSES if (label == name).any()]
    rows += [name for name in ("Paddock", "Other/Industrial") if (label == name).any()]
    rows += sorted(set(label) - set(rows))
    table = pd.crosstab(pd.Series(label.to_numpy(), name="annotated"), pd.Series(predicted, name="predicted"))
    return table.reindex(index=rows, columns=CLASSES, fill_value=0)


def farm_confusion(truth: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Farm-level analogue: each annotated class (rows) against the farm's top class.

    Farm labels are multi-label, so a farm carrying two X marks contributes a row to
    each; farms with no X at all become a `no class marked` row, which is the farm-level
    equivalent of the Paddock row above.
    """
    predicted = np.array(CLASSES)[scores.argmax(axis=1)]
    marks = truth[CLASSES + WORKBOOK_ONLY_CLASSES].to_numpy()
    annotated, top = [], []
    for row in range(len(truth)):
        present = [
            name for name, flag in zip(CLASSES + WORKBOOK_ONLY_CLASSES, marks[row]) if flag
        ] or ["no class marked"]
        annotated.extend(present)
        top.extend([predicted[row]] * len(present))

    rows = [name for name in CLASSES + WORKBOOK_ONLY_CLASSES if name in set(annotated)]
    rows.append("no class marked")
    table = pd.crosstab(pd.Series(annotated, name="annotated"), pd.Series(top, name="predicted"))
    return table.reindex(index=rows, columns=CLASSES, fill_value=0)


def plot_confusions(matrices: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    """Row-normalised confusion matrices, one panel per model, counts printed in-cell."""
    figure, axes = plt.subplots(
        1, len(matrices), figsize=(6.4 * len(matrices), 5.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, (name, table) in zip(axes, matrices.items()):
        counts = table.to_numpy()
        totals = counts.sum(axis=1, keepdims=True)
        fraction = np.divide(counts, totals, out=np.zeros_like(counts, float), where=totals > 0)
        axis.imshow(fraction, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        axis.set_xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
        axis.set_yticks(
            range(len(table.index)),
            [f"{row}  (n={total})" for row, total in zip(table.index, totals.ravel())],
        )
        axis.set_xlabel("model argmax class")
        axis.set_ylabel("annotated label")
        axis.set_title(name, fontsize=10)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                if counts[i, j]:
                    axis.text(
                        j,
                        i,
                        counts[i, j],
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if fraction[i, j] > 0.5 else "#333333",
                    )
        axis.set_xticks(np.arange(-0.5, len(table.columns)), minor=True)
        axis.set_yticks(np.arange(-0.5, len(table.index)), minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", length=0)

    figure.suptitle(title, fontsize=12)
    figure.savefig(path, dpi=200)
    plt.close(figure)


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def report(table: pd.DataFrame, models: list[str], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    header = f"{'class':<14}{'pos':>5}{'prev':>7}"
    for model in models:
        header += f"{model.split(' ')[0]:>28}"
    header += f"{'delta AP':>22}"
    print(header)
    for _, row in table.iterrows():
        line = f"{row['class']:<14}{int(row['positives']):>5}{row['prevalence']:>7.3f}"
        for model in models:
            ap = row.get(f"{model} | AP")
            if pd.isna(ap):
                line += f"{'-':>28}"
            else:
                low, high = row.get(f"{model} | AP lo"), row.get(f"{model} | AP hi")
                interval = "" if pd.isna(low) else f" [{low:.2f},{high:.2f}]"
                line += f"{f'{ap:.3f}{interval}':>28}"
        delta = row.get("delta AP (new - old)")
        if pd.isna(delta):
            line += f"{'-':>22}"
        else:
            low, high = row.get("delta lo"), row.get("delta hi")
            interval = "" if pd.isna(low) else f" [{low:+.2f},{high:+.2f}]"
            line += f"{f'{delta:+.3f}{interval}':>22}"
        print(line)
    print("prev = prevalence, the AP a random ranker scores. Intervals are 95% bootstrap.")


def score_columns(
    ensembles: dict[str, np.ndarray],
    seeds: dict[str, list[np.ndarray]],
    keys: dict[str, str],
) -> dict[str, np.ndarray]:
    """Score columns for the processed CSVs, ensemble and per seed.

    `{key}|{class}` is the seed-mean ensemble, `{key}#{n}|{class}` the nth seed run, so
    the dashboard can show the spread across training seeds without re-reading the runs.
    """
    columns = {}
    for model, values in ensembles.items():
        sources = [(keys[model], values)]
        sources += [(f"{keys[model]}#{n}", s) for n, s in enumerate(seeds[model], start=1)]
        for prefix, array in sources:
            for position, name in enumerate(CLASSES):
                columns[f"{prefix}|{name}"] = array[:, position]
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation",
        default="max",
        choices=["max", "mean"],
        help="how a farm's per-crop scores become one farm score (default max)",
    )
    parser.add_argument(
        "--min-confidence",
        default=None,
        choices=list(CONFIDENCE_ORDER),
        help="drop farm-level X marks below this labeller confidence",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    dataset = pd.read_csv(DATASET, keep_default_na=False)
    print(f"{DATASET.name}: {len(dataset)} crops, {dataset['PFI'].nunique()} farms")

    seeds_by_model, ensembles = {}, {}
    for name, spec in MODELS.items():
        seeds_by_model[name], ensembles[name] = load_model(name, spec, len(dataset))
    models = list(MODELS)

    image, farm = load_workbooks()
    print(
        f"{LABELLED}: {len(image)} labelled crops over {farm['PFI'].nunique()} farms "
        f"from {image['source'].nunique()} reaches"
    )

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- image level -------------------------------------------------------------
    truth, rows, label = image_truth(image, dataset)
    dropped = len(image) - len(truth)
    print(f"image level: {len(truth)} crops ({dropped} Ambiguous/Multiple Classes dropped)")
    image_ensembles = {m: ensembles[m].to_numpy()[rows] for m in models}
    image_seeds = {m: [s.to_numpy()[rows] for s in seeds_by_model[m]] for m in models}
    image_table = evaluate(
        "image", truth, image_ensembles, image_seeds, args.bootstrap, args.seed
    )

    # ---- farm level --------------------------------------------------------------
    truth, pfis = farm_truth(farm, args.min_confidence)
    print(
        f"farm level: {len(truth)} farms, scores aggregated with {args.aggregation}"
        + (f", X marks below {args.min_confidence} confidence dropped" if args.min_confidence else "")
    )
    farm_ensembles = {m: farm_scores(ensembles[m], dataset, pfis, args.aggregation) for m in models}
    farm_seeds = {
        m: [farm_scores(s, dataset, pfis, args.aggregation) for s in seeds_by_model[m]]
        for m in models
    }
    farm_table = evaluate("farm", truth, farm_ensembles, farm_seeds, args.bootstrap, args.seed)

    report(image_table, models, "Image level - average precision per class")
    report(farm_table, models, f"Farm level ({args.aggregation} over crops) - average precision")

    # ---- confusion matrices ------------------------------------------------------
    image_confusions = {m: confusion(label, image_ensembles[m]) for m in models}
    farm_confusions = {m: farm_confusion(truth, farm_ensembles[m]) for m in models}
    plot_confusions(
        image_confusions,
        args.output / "confusion_image.png",
        "Image level: annotated crop label vs model argmax (shading = row fraction)",
    )
    plot_confusions(
        farm_confusions,
        args.output / "confusion_farm.png",
        f"Farm level ({args.aggregation} over crops): annotated class vs farm's top class",
    )

    # ---- processed CSVs ----------------------------------------------------------
    # The joined scores are the intermediate the dashboard reads, so it never has to
    # touch the run directories or the workbooks again.
    keys = {name: spec["key"] for name, spec in MODELS.items()}
    labelled = image.loc[label.index]
    pd.DataFrame(
        {
            "source": labelled["source"].to_numpy(),
            "PFI": labelled["Farm PFI"].astype(float).astype(int).to_numpy(),
            "image_path": labelled["image_path"].to_numpy(),
            "label": label.to_numpy(),
            **score_columns(image_ensembles, image_seeds, keys),
        }
    ).to_csv(args.output / "scores_image.csv", index=False)

    pd.DataFrame(
        {
            "source": farm["source"].to_numpy(),
            "PFI": pfis,
            "n_images": farm["n_images"].to_numpy(),
            **{f"true|{name}": truth[name].to_numpy() for name in CLASSES + WORKBOOK_ONLY_CLASSES},
            **score_columns(farm_ensembles, farm_seeds, keys),
        }
    ).to_csv(args.output / "scores_farm.csv", index=False)

    out = args.output / "gen_evaluation.csv"
    pd.concat([image_table, farm_table], ignore_index=True).to_csv(out, index=False)
    for level, confusions in (("image", image_confusions), ("farm", farm_confusions)):
        frame = pd.concat({keys[m]: c for m, c in confusions.items()}, names=["model", "annotated"])
        frame.to_csv(args.output / f"confusion_{level}.csv")
    pd.DataFrame(
        [
            {
                "key": spec["key"],
                "model": name,
                "aggregation": args.aggregation,
                "min_confidence": args.min_confidence or "",
                "seeds": len(seeds_by_model[name]),
                "runs": str(spec["runs"]),
            }
            for name, spec in MODELS.items()
        ]
    ).to_csv(args.output / "models.csv", index=False)

    print(f"\nwrote {out}, scores_{{image,farm}}.csv, confusion_{{image,farm}}.{{csv,png}}, "
          f"models.csv in {args.output}")


if __name__ == "__main__":
    main()
