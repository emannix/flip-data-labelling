"""Score the 2026-08-21 multilabel runs against the relabelled crop-level test split.

This is the successor to `gen_evaluation.py`, and it differs from it in three ways that
are worth stating up front, because they change what the numbers mean.

*Ground truth comes from the dataset, not the workbooks.* `gen_croplevel_dataset.py` has
already folded the relabelling workbooks back into
`original_new_2026_08_21_generalisation/relabelled_*.csv`, so the truth is the `binary_*`
columns sitting next to the predictions rather than an Excel join done here. The 17
genuinely multi-label crops carry both their classes, and `Ambiguous` is already gone.

*The evaluation set is the VIC hold-out.* The split is geographic - NSW trains, VIC tests
- so the 1,146 test crops over 242 farms (balliang and wyuna) are the only rows every
model has a prediction for. The older checkpoints predicted over the whole
`original_new_2026_07_24_generalisation/dataset.csv`; they are reindexed onto the test
crops on `(ecw_stem, building_cluster, PFI)`, since `image_path` no longer matches across
builds - the current build is keyed on `farm_uid` where the older one was keyed on PFI.

*The models no longer share a vocabulary.* The 2026-08-21 runs are multilabel (sigmoid,
BCE) over eleven classes including `paddock` and `other_industrial`; the older runs are
multiclass (softmax) over ten including `aqua`. AP is rank-based, so a class either model
emits is comparable between them, but `paddock` and `other_industrial` are scored for the
new models only - the old ones have no output to score - and that is reported as blank
rather than as zero. Half this collection is paddock, so this is the single biggest
difference between the two generations and the confusion panels are where it shows.

Because there are now four model families rather than two, there is no single baseline:
every pair of models is differenced on the same bootstrap resamples and the per-class
pairwise table carries all of them.

*Half the vocabulary is too thinly sampled to rank on.* Five of the nine shared classes
were trained on 14 crops or fewer, and three carry single-digit test positives -
`commercialpig` has four training crops and eight positives, `sheep` four and five. Their
AP is real but its bootstrap interval spans most of the unit range, and an unweighted
macro gives them the same weight as `residential`'s 102 positives, which is enough for
three uninformative numbers to invert the headline. So the per-class table carries the
training count beside the positives, dims any class short of either
MIN_TRAIN_EXAMPLES or MIN_POSITIVES, marks any AP whose own interval is at least
WIDE_INTERVAL wide, and reports a third macro over the classes that clear both bars. The
starved classes are dimmed rather than dropped: not doing `commercialpig` on four
examples is a genuine weakness, it just is not a *ranking*.

Runs without a saved prediction are skipped with a note and the family is reported as
pending; rerun once they land and it fills in.

Writes the CSVs *and* the self-contained HTML dashboard into `output_eval_2026_08_21/`.

Usage:

    .venv/bin/python gen_evaluation_2026_08_21.py
    .venv/bin/python gen_evaluation_2026_08_21.py --aggregation top2 --bootstrap 4000
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------

BUILDER = Path("/home/mannixe/FLIP/flip-geoimage-dataset-builder")
VIEW = Path("/home/mannixe/FLIP/comfe-run-flip/view")

# The evaluation set: the VIC hold-out of the relabelled crop-level build.
TEST_CSV = BUILDER / "original_new_2026_08_21_generalisation" / "relabelled_test_df.csv"

# What the older checkpoints predicted over, and the key that maps it onto TEST_CSV.
LEGACY_DATASET = BUILDER / "original_new_2026_07_24_generalisation" / "dataset.csv"
JOIN_KEY = ["ecw_stem", "building_cluster", "PFI"]

OUTPUT = Path("output_eval_2026_08_21")

# Column order of each generation's saved y_hat.
#
# Legacy: the sorted `processed_class` of the multiclass training split.
LEGACY_CLASSES = [
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
# 2026-08-21: `test_csv_label` from the run config, verbatim and in order. Read back off
# each run's own `.hydra/config.yaml` before scoring rather than trusted from here.
MULTILABEL_CLASSES = [
    "backyardpig",
    "beef",
    "commercialpig",
    "dairy",
    "freerangepig",
    "horse",
    "other_industrial",
    "paddock",
    "poultry",
    "residential",
    "sheep",
]

# Every class any model can emit, in a stable display order: the nine shared livestock
# classes first, then the two the new generation adds, then the two nothing can score.
SHARED_CLASSES = [
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
NEW_ONLY_CLASSES = ["other_industrial", "paddock"]
LEGACY_ONLY_CLASSES = ["aqua"]
# Carried in the build's `binary_*` columns but emitted by nothing, so unevaluable.
UNMODELLED_CLASSES = ["goats"]

ALL_CLASSES = SHARED_CLASSES + NEW_ONLY_CLASSES + LEGACY_ONLY_CLASSES + UNMODELLED_CLASSES

MACRO = "MACRO (evaluable)"
MACRO_SHARED = "MACRO (shared 9)"
MACRO_SAMPLED = "MACRO (well sampled)"
LEVELS = {"crop": "crop", "farm": "farm"}

# What counts as enough data to say anything about a class.
#
# Half this vocabulary is trained on single-digit examples, and a class with four training
# crops and five test positives produces an AP whose bootstrap interval spans most of the
# unit range - `old` scores 0.29 on sheep with a 95 % interval of [0.01, 0.70]. Those
# numbers carry no information, but an unweighted macro gives them the same weight as
# residential's 102 positives, so three of them are enough to invert the headline. The
# macro is therefore reported a third way, over the classes that clear both bars, and the
# rule is deliberately a property of the *data* rather than of any model's scores so that
# the subset cannot be chosen to flatter a model.
MIN_TRAIN_EXAMPLES = 6
MIN_POSITIVES = 10
# A per-class interval at least this wide is flagged in the table: the estimate is too
# loose to rank models with, whatever its point value.
WIDE_INTERVAL = 0.40

# The training split the 2026-08-21 models were fitted on, read only for its per-class
# counts so the table can show what each class was actually trained on.
TRAIN_CSV = BUILDER / "original_new_2026_08_21_generalisation" / "relabelled_train_df.csv"

# The four model families, in the categorical-slot order the dashboard paints them.
# `runs` is either an explicit directory of seed runs (legacy) or a (directory, substring)
# pair discovered under the 2026-08-21 view.
MODELS = [
    {
        "key": "old",
        "slot": 1,
        "label": "ComFe multiclass (2026-03-06)",
        "detail": "trained on flip_historical, farm-level labels",
        "classes": LEGACY_CLASSES,
        "space": "legacy",
        "runs": VIEW / "flip_orig_comfe" / "flip_comfe_multiclass2_gen",
        "match": None,
    },
    {
        "key": "jul",
        "slot": 2,
        "label": "ComFe multiclass (2026-08-07)",
        "detail": "trained on original_new_2026_07_24, farm-level labels",
        "classes": LEGACY_CLASSES,
        "space": "legacy",
        "runs": VIEW / "flip_2026_07_24" / "comfe_gen",
        "match": None,
    },
    {
        "key": "lin",
        "slot": 3,
        "label": "DINOv2 linear probe (2026-08-21)",
        "detail": "multilabel, trained on the relabelled crop-level NSW split",
        "classes": MULTILABEL_CLASSES,
        "space": "test",
        "runs": VIEW / "flip_2026_08_21",
        "match": "linear_finetune",
    },
    {
        "key": "comfe",
        "slot": 4,
        "label": "ComFe multilabel (2026-08-21)",
        "detail": "multilabel, trained on the relabelled crop-level NSW split",
        "classes": MULTILABEL_CLASSES,
        "space": "test",
        "runs": VIEW / "flip_2026_08_21",
        "match": "comfe_dinov2",
    },
]


def region_name(source: str) -> str:
    """The reach a row belongs to, read off its shapefile name."""
    stem = str(source).split("_labels")[0].removesuffix(".shp")
    return stem.replace("_", " ").strip().title()


# --------------------------------------------------------------------------------------
# loading runs
# --------------------------------------------------------------------------------------


def run_seed(run_dir: Path) -> str:
    """The run's seed, read off its hydra overrides, tagged with its launch batch.

    The 2026-08-21 ComFe runs came off two launches (`…_1-5_*` and `…_1-2_1`) whose
    configs are identical bar `num_workers`, and both include a seed 1. They are treated
    as one family, so the batch has to stay in the label or the two seed-1 runs collide.
    """
    seed = "?"
    overrides = run_dir / ".hydra" / "overrides.yaml"
    if overrides.exists():
        found = re.search(r"seed=(\S+)", overrides.read_text())
        if found:
            seed = found.group(1)
    batch = re.search(r"_(\d+-\d+)_\d+$", run_dir.name)
    return f"{batch.group(1)}/{seed}" if batch else seed


def config_classes(run_dir: Path) -> list[str] | None:
    """`test_csv_label` from the run's own config, stripped of the `binary_` prefix."""
    config = run_dir / ".hydra" / "config.yaml"
    if not config.exists():
        return None
    block = re.search(
        r"^    test_csv_label:\n((?:    - \S+\n)+)", config.read_text(), re.MULTILINE
    )
    if not block:
        return None
    return [
        line.strip().removeprefix("- ").removeprefix("binary_")
        for line in block.group(1).splitlines()
    ]


def predictions_dir(run_dir: Path) -> Path | None:
    """The run's saved predictions, or None while it is still training."""
    path = run_dir / "tensorboard" / "version_0" / "predictions"
    if not path.is_dir():
        return None
    if not list(path.glob("*_y_hat_predictions*.csv")):
        return None
    return path


def load_run(run_dir: Path, n_rows: int, classes: list[str]) -> np.ndarray:
    """One seed run's per-class scores, reindexed onto its dataset's row order."""
    path = predictions_dir(run_dir)
    y_hat = pd.read_csv(next(path.glob("*_y_hat_predictions*.csv"))).to_numpy(dtype=float)
    index = pd.read_csv(next(path.glob("*_index_predictions*.csv")))["index"].to_numpy()
    if y_hat.shape != (n_rows, len(classes)):
        raise SystemExit(
            f"{run_dir.name}: y_hat is {y_hat.shape}, expected {(n_rows, len(classes))}"
        )
    if not np.array_equal(np.sort(index), np.arange(n_rows)):
        raise SystemExit(f"{run_dir.name}: index is not a permutation of {n_rows} rows")
    ordered = np.empty_like(y_hat)
    ordered[index] = y_hat
    return ordered


def discover(spec: dict) -> list[Path]:
    """The seed runs belonging to a model family, in a stable order."""
    if not spec["runs"].is_dir():
        return []
    dirs = [d for d in spec["runs"].iterdir() if d.is_dir()]
    if spec["match"]:
        dirs = [d for d in dirs if spec["match"] in d.name]
    return sorted(dirs)


def load_model(spec: dict, rows: np.ndarray, n_legacy: int, n_test: int) -> dict:
    """Every seed of one family, aligned to the test crops, plus the seed-mean ensemble.

    Runs still training have no saved prediction and are counted as pending rather than
    treated as an error - the ComFe sweep is expected to land after the first pass here.
    """
    found = discover(spec)
    ready = [d for d in found if predictions_dir(d)]
    pending = [d for d in found if not predictions_dir(d)]

    seeds, labels = [], []
    for run_dir in ready:
        classes = config_classes(run_dir)
        if classes and classes != spec["classes"]:
            raise SystemExit(
                f"{run_dir.name}: config class order {classes} != expected {spec['classes']}"
            )
        n_rows = n_legacy if spec["space"] == "legacy" else n_test
        scores = load_run(run_dir, n_rows, spec["classes"])
        seeds.append(scores[rows] if spec["space"] == "legacy" else scores)
        labels.append(run_seed(run_dir))

    return {
        **spec,
        "seeds": seeds,
        "seed_labels": labels,
        "ensemble": sum(seeds) / len(seeds) if seeds else None,
        "n_found": len(found),
        "n_pending": len(pending),
        "pending_names": [d.name for d in pending],
    }


def column_of(model: dict, name: str) -> int | None:
    """Where a class sits in a model's output, or None if it has no such output."""
    return model["classes"].index(name) if name in model["classes"] else None


def scores_for(model: dict, name: str, matrix: np.ndarray | None = None) -> np.ndarray | None:
    position = column_of(model, name)
    if position is None:
        return None
    source = model["ensemble"] if matrix is None else matrix
    return source[:, position]


# --------------------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------------------


def crop_truth(test: pd.DataFrame) -> pd.DataFrame:
    """Per-crop multi-hot truth, straight off the build's `binary_*` columns."""
    columns = {}
    for name in ALL_CLASSES:
        column = f"binary_{name}"
        if column in test.columns:
            columns[name] = test[column].to_numpy(dtype=float).astype(bool)
        else:
            columns[name] = np.zeros(len(test), dtype=bool)
    return pd.DataFrame(columns)


def farm_frame(test: pd.DataFrame) -> pd.DataFrame:
    """One row per farm, carrying the farm's X-marked classes and its reach."""
    farms = test.drop_duplicates("farm_uid")[
        ["farm_uid", "PFI", "source", "farm_labels"]
    ].reset_index(drop=True)
    farms["n_crops"] = farms["farm_uid"].map(test["farm_uid"].value_counts())
    return farms


def farm_truth(farms: pd.DataFrame) -> pd.DataFrame:
    """Per-farm multi-label truth, parsed from the workbook's `farm_labels` string.

    Farm labels are about what livestock the farm runs, so `paddock` and
    `other_industrial` never appear there - they are crop-level classes only, and come
    out of this as all-negative, hence unevaluable at the farm level.
    """
    marked = farms["farm_labels"].fillna("").astype(str).apply(
        lambda value: {part.strip() for part in value.split(",") if part.strip()}
    )
    return pd.DataFrame(
        {name: marked.apply(lambda names, n=name: n in names).to_numpy() for name in ALL_CLASSES}
    )


def farm_scores(scores: np.ndarray, keys: np.ndarray, order: np.ndarray, how: str) -> np.ndarray:
    """Aggregate per-crop scores up to one score per farm, in `order`.

    `max` is the natural reading - a farm runs dairy if any one of its buildings is a
    dairy - but it is also the noisiest, since a single over-confident crop sets the
    farm's score outright. `top2` averages the two highest-scoring crops instead, which
    costs nothing and is steadier at the ~4.7 crops per farm this build carries.
    """
    grouped = pd.DataFrame(scores).groupby(keys)
    if how == "top2":
        frame = grouped.apply(
            lambda block: block.apply(lambda column: column.nlargest(2).mean())
        )
    else:
        frame = grouped.agg(how)
    return frame.reindex(order).to_numpy()


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

# Below this share of usable bootstrap draws the percentile interval is not reported:
# too much of the resample distribution is missing for the quantiles to mean anything.
MIN_DRAW_SHARE = 0.5


def bootstrap_indices(n: int, reps: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def bootstrap_ap(y_true: np.ndarray, y_score: np.ndarray, resamples: np.ndarray) -> np.ndarray:
    """AP on every resample, NaN where a draw happens to contain no positives."""
    values = np.full(len(resamples), np.nan)
    for position, draw in enumerate(resamples):
        truth = y_true[draw]
        if truth.any() and not truth.all():
            values[position] = average_precision_score(truth, y_score[draw])
    return values


def interval_of(draws: np.ndarray) -> tuple[float, float]:
    """95 % percentile interval, or NaN when too few draws were usable."""
    usable = np.isfinite(draws)
    if not len(draws) or usable.mean() < MIN_DRAW_SHARE:
        return np.nan, np.nan
    low, high = np.percentile(draws[usable], [2.5, 97.5])
    return float(low), float(high)


def train_counts(train: pd.DataFrame) -> dict[str, int]:
    """How many training crops carry each class, off the split's `binary_*` columns.

    Used at both levels: the models are fitted on crops either way, so the crop count is
    what a farm-level class was trained on too.
    """
    return {
        name: int(train[f"binary_{name}"].to_numpy(dtype=float).sum())
        if f"binary_{name}" in train.columns
        else 0
        for name in ALL_CLASSES
    }


def evaluate(
    level: str,
    truth: pd.DataFrame,
    models: list[dict],
    reps: int,
    seed: int,
    trained: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class AP for every model, plus the all-pairs paired differences.

    Both tables come out of one set of bootstrap draws per class, so the pairwise
    differences are paired by construction - every model sees the same resample - and the
    difference interval is consistent with the two AP intervals beside it.
    """
    n = len(truth)
    resamples = bootstrap_indices(n, reps, seed)
    scored = [m for m in models if m["ensemble"] is not None]
    rows, pairs = [], []

    for name in ALL_CLASSES:
        y_true = truth[name].to_numpy()
        positives = int(y_true.sum())
        evaluable = 0 < positives < n
        n_train = trained.get(name, 0)
        row = {
            "level": level,
            "class": name,
            "n": n,
            "n_train": n_train,
            "positives": positives,
            "prevalence": positives / n if n else np.nan,
            # Model-independent: it asks whether the data can support a comparison at all,
            # not whether any particular model did well.
            "well_sampled": bool(
                n_train >= MIN_TRAIN_EXAMPLES and positives >= MIN_POSITIVES
            ),
        }

        draws: dict[str, np.ndarray] = {}
        for model in scored:
            y_score = scores_for(model, name)
            key = model["key"]
            if y_score is None:
                # The model has no output for this class at all - blank, never zero.
                row[f"{key}|AP"] = np.nan
                row[f"{key}|status"] = "no output"
                continue
            if not evaluable:
                row[f"{key}|AP"] = np.nan
                row[f"{key}|status"] = "no positives"
                continue
            row[f"{key}|status"] = "scored"
            row[f"{key}|AP"] = float(average_precision_score(y_true, y_score))
            row[f"{key}|AUC"] = float(roc_auc_score(y_true, y_score))
            draws[key] = bootstrap_ap(y_true, y_score, resamples)
            low, high = interval_of(draws[key])
            row[f"{key}|AP_lo"], row[f"{key}|AP_hi"] = low, high
            row[f"{key}|AP_ci"] = float(high - low) if np.isfinite(low) and np.isfinite(high) else np.nan
            per_seed = [
                float(average_precision_score(y_true, scores_for(model, name, s)))
                for s in model["seeds"]
            ]
            row[f"{key}|AP_seed_mean"] = float(np.mean(per_seed))
            row[f"{key}|AP_seed_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
            row[f"{key}|n_seeds"] = len(per_seed)

        # Every ordered pair, in slot order, so "b - a" always reads later-minus-earlier.
        for i, a in enumerate(scored):
            for b in scored[i + 1 :]:
                if a["key"] not in draws or b["key"] not in draws:
                    continue
                delta = draws[b["key"]] - draws[a["key"]]
                low, high = interval_of(delta)
                pairs.append(
                    {
                        "level": level,
                        "class": name,
                        "a": a["key"],
                        "b": b["key"],
                        "pair": f"{b['key']}-{a['key']}",
                        "positives": positives,
                        "delta_AP": float(row[f"{b['key']}|AP"] - row[f"{a['key']}|AP"]),
                        "delta_lo": low,
                        "delta_hi": high,
                        "clear_of_zero": bool(
                            np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
                        ),
                    }
                )
        rows.append(row)

    table = pd.DataFrame(rows)

    # The macro is the unweighted mean over the classes a model can actually score, so it
    # is NOT a like-for-like number across the two generations: the new models average in
    # paddock and other_industrial, which the old ones have no output for. Three macros are
    # reported:
    #
    #   MACRO (evaluable)    each model over its own scoreable classes - not comparable
    #                        across generations, but it is what each model actually does.
    #   MACRO (shared 9)     the nine classes every model emits - like for like on
    #                        vocabulary, but five of those nine have <= 14 training crops
    #                        and three have single-digit test positives, so it is dominated
    #                        by estimates whose intervals span most of the unit range.
    #   MACRO (well sampled) the shared classes that clear MIN_TRAIN_EXAMPLES and
    #                        MIN_POSITIVES - like for like on vocabulary *and* carrying
    #                        enough data to rank models with. This is the honest headline.
    #
    # All three are kept rather than one replaced, because dropping the starved classes
    # hides a real weakness: the new models genuinely cannot do commercialpig or sheep on
    # four training examples. The point is only that those classes cannot be *ranked*.
    macro_rows = []
    well = [name for name in SHARED_CLASSES if table.loc[table["class"] == name, "well_sampled"].any()]
    for label, subset in (
        (MACRO, None),
        (MACRO_SHARED, SHARED_CLASSES),
        (MACRO_SAMPLED, well),
    ):
        macro = {"level": level, "class": label, "n": n}
        for model in scored:
            key = model["key"]
            usable = table[table[f"{key}|AP"].notna()]
            if subset is not None:
                usable = usable[usable["class"].isin(subset)]
            if usable.empty:
                continue
            macro[f"{key}|AP"] = float(usable[f"{key}|AP"].mean())
            macro[f"{key}|AUC"] = float(usable[f"{key}|AUC"].mean())
            macro[f"{key}|n_classes"] = int(len(usable))
            macro[f"{key}|status"] = "scored"
            macro["positives"] = int(usable["positives"].sum())
            macro["prevalence"] = float(usable["prevalence"].mean())
            macro["n_train"] = int(usable["n_train"].sum())
        macro_rows.append(macro)

    return pd.concat([table, pd.DataFrame(macro_rows)], ignore_index=True), pd.DataFrame(pairs)


def region_summary(
    level: str,
    truth: pd.DataFrame,
    regions: np.ndarray,
    models: list[dict],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """AP per class per reach, and the macro over each reach's scoreable classes.

    Two reaches only - balliang and wyuna - but they are different landscapes with very
    different class mixes, so every row carries the prevalence a random ranker would
    score. Without it the reaches are not comparable to each other, only the models
    within a reach are.
    """
    scored = [m for m in models if m["ensemble"] is not None]
    rows = []

    for region in sorted(set(regions)):
        mask = regions == region
        n = int(mask.sum())
        subset = truth[mask]
        names = [c for c in ALL_CLASSES if 0 < subset[c].sum() < n]
        if not names:
            continue
        resamples = bootstrap_indices(n, reps, seed)

        point: dict[tuple[str, str], float] = {}
        draws: dict[tuple[str, str], np.ndarray] = {}
        per_seed: dict[tuple[str, str], list[float]] = {}
        for model in scored:
            key = model["key"]
            mine = [name for name in names if column_of(model, name) is not None]
            if not mine:
                continue
            for name in mine:
                y_true = subset[name].to_numpy()
                y_score = scores_for(model, name)[mask]
                point[(key, name)] = float(average_precision_score(y_true, y_score))
                draws[(key, name)] = bootstrap_ap(y_true, y_score, resamples)
                per_seed[(key, name)] = [
                    float(average_precision_score(y_true, scores_for(model, name, s)[mask]))
                    for s in model["seeds"]
                ]
            # Derived from the class results rather than resampled separately, so the
            # macro interval stays consistent with the class intervals underneath it.
            # `mean` not `nanmean` on purpose: a draw that loses a rare class's only
            # positive leaves the macro undefined, and averaging over whichever classes
            # survive would quietly change the estimand draw to draw and bias the
            # interval upward. Those draws drop out; `draws_used` records how many stayed.
            point[(key, MACRO)] = float(np.mean([point[(key, c)] for c in mine]))
            draws[(key, MACRO)] = np.mean([draws[(key, c)] for c in mine], axis=0)
            per_seed[(key, MACRO)] = list(np.mean([per_seed[(key, c)] for c in mine], axis=0))

        counts = {name: int(subset[name].sum()) for name in names}
        counts[MACRO] = int(subset[names].to_numpy().sum())
        shares = {name: float(subset[name].mean()) for name in names}
        shares[MACRO] = float(np.mean(list(shares.values())))

        for name in names + [MACRO]:
            for model in scored:
                key = model["key"]
                if (key, name) not in point:
                    continue
                common = {
                    "level": level,
                    "region": region,
                    "class": name,
                    "model": key,
                    "n_units": n,
                    "positives": counts[name],
                    "prevalence": shares[name],
                }
                low, high = interval_of(draws[(key, name)])
                rows.append(
                    {
                        **common,
                        "seed": "ensemble",
                        "n_classes": sum(1 for c in names if (key, c) in point),
                        "AP": point[(key, name)],
                        "AP_lo": low,
                        "AP_hi": high,
                        "draws_used": int(np.isfinite(draws[(key, name)]).sum()),
                    }
                )
                for label, value in zip(model["seed_labels"], per_seed[(key, name)]):
                    rows.append({**common, "seed": label, "AP": value})

    return pd.DataFrame(rows)


def top_k_agreement(
    truth: pd.DataFrame, models: list[dict], depths=(1, 2, 3)
) -> pd.DataFrame:
    """Share of units whose annotation is among the model's k highest-scoring classes.

    Scored per model over *its own* vocabulary and over the units carrying at least one
    class that model can emit - a paddock crop is a right answer the new models can give
    and the old ones cannot, so pooling them onto a common denominator would score the
    old models on a question they were never asked. `n_units` differs between the
    generations for exactly that reason and is reported beside every number.
    """
    rows = []
    for model in [m for m in models if m["ensemble"] is not None]:
        mine = [name for name in model["classes"] if name in truth.columns]
        matrix = truth[mine].to_numpy()
        keep = matrix.any(axis=1)
        if not keep.any():
            continue
        matrix = matrix[keep]
        values = np.column_stack([scores_for(model, name) for name in mine])[keep]
        for depth in depths:
            ranked = np.argsort(-values, axis=1)[:, :depth]
            hit = np.take_along_axis(matrix, ranked, axis=1).any(axis=1)
            rows.append(
                {
                    "model": model["key"],
                    "depth": depth,
                    "agreement": float(hit.mean()),
                    "n_units": int(keep.sum()),
                    "n_classes": len(mine),
                }
            )
    return pd.DataFrame(rows)


def confusion(truth: pd.DataFrame, model: dict, extra_row: str) -> pd.DataFrame:
    """Annotated class (rows) against the model's top-scoring class (columns), as counts.

    Multi-label units contribute one row per annotated class, each paired with the same
    single top class - so the diagonal means "this class is *the* top class for its
    unit", which a unit carrying two classes can satisfy for only one of them. Row totals
    therefore count annotated (unit, class) pairs, not units.

    Rows are every annotated class, including ones the model cannot emit: those are the
    interesting ones, since every such unit is forced onto some other column and where it
    lands is the whole story behind that column's precision.
    """
    predicted = np.array(model["classes"])[model["ensemble"].argmax(axis=1)]
    present = [name for name in ALL_CLASSES if name in truth.columns and truth[name].any()]
    marks = truth[present].to_numpy()

    annotated, top = [], []
    for row in range(len(truth)):
        names = [name for name, flag in zip(present, marks[row]) if flag] or [extra_row]
        annotated.extend(names)
        top.extend([predicted[row]] * len(names))

    order = [name for name in present if name in set(annotated)]
    if extra_row in set(annotated):
        order.append(extra_row)
    table = pd.crosstab(
        pd.Series(annotated, name="annotated"), pd.Series(top, name="predicted")
    )
    return table.reindex(index=order, columns=model["classes"], fill_value=0)


# --------------------------------------------------------------------------------------
# dashboard: palette and svg helpers
# --------------------------------------------------------------------------------------

# Categorical slots 1-4 of the validated reference palette, used unchanged; light value
# first, dark second. Four series clear every hard gate on the *adjacent* pairlist that
# grouped bars and dot rows sit on (worst CVD dE 9.1 light / 8.4 dark against a target of
# 8; worst normal-vision dE 22.9 / 19.8 against a floor of 15). They do NOT clear the
# all-pairs pairlist - orange and yellow collide once every series can touch every other
# - so anywhere all four marks overlay in one frame, the precision-recall panels, the
# chart is faceted one model per panel instead of superimposed.
#
# Slots 3 and 4 sit below 3:1 against the light surface, so the relief rule applies: every
# chart here carries direct value labels and a full table underneath it, never colour alone.
SERIES = {
    "old": ("#2a78d6", "#3987e5"),
    "jul": ("#eb6834", "#d95926"),
    "lin": ("#1baf7a", "#199e70"),
    "comfe": ("#eda100", "#c98500"),
}
# Diverging blue-red with a neutral grey midpoint, for the paired differences.
DIVERGING = {"up": ("#2a78d6", "#3987e5"), "down": ("#d03b3b", "#e66767")}

# Blue sequential ramp: light mode runs light->dark from the surface, dark mode runs
# dark->light, so in both cases "more" is further from the page.
HEAT_BREAKS = [0.03, 0.08, 0.16, 0.30, 0.50, 0.75]
HEAT_FLIP = 4


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg(width: float, height: float, body: str, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {width:g} {height:g}" width="100%" height="{height:g}" '
        f'role="img" aria-label="{escape(label)}" preserveAspectRatio="xMidYMid meet">'
        f"{body}</svg>"
    )


def text(x: float, y: float, value: object, cls: str = "tick", anchor: str = "start") -> str:
    return (
        f'<text x="{x:g}" y="{y:g}" class="{cls}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def tip(content: str) -> str:
    return f"<title>{escape(content)}</title>"


def rounded_right(x: float, y: float, width: float, height: float, radius: float = 4) -> str:
    """A bar anchored at the baseline with only its data end rounded."""
    radius = max(0.0, min(radius, width, height / 2))
    return (
        f"M{x:g},{y:g} H{x + width - radius:g} "
        f"Q{x + width:g},{y:g} {x + width:g},{y + radius:g} "
        f"V{y + height - radius:g} "
        f"Q{x + width:g},{y + height:g} {x + width - radius:g},{y + height:g} "
        f"H{x:g} Z"
    )


def heat_index(fraction: float) -> int:
    return int(np.searchsorted(HEAT_BREAKS, fraction, side="right"))


# --------------------------------------------------------------------------------------
# dashboard: charts
# --------------------------------------------------------------------------------------


def ap_chart(table: pd.DataFrame, models: list[dict], level: str) -> str:
    """Per-class AP for every model, with the prevalence baseline marked on each row.

    The bar is the seed-mean ensemble, whiskered with its bootstrap interval; the open
    circles are the individual seed runs, so the spread from the training seed sits next
    to the spread from the labels. A class a model cannot emit gets a hatched stub and an
    explicit "no output" note rather than an empty gap that would read as a zero.
    """
    keys = [m["key"] for m in models]
    rows = table[
        table["class"].isin(ALL_CLASSES)
        & table[[f"{k}|AP" for k in keys]].notna().any(axis=1)
    ]

    left, right, top, gap = 132, 118, 40, 4
    band = 16
    row_height = len(keys) * band + 2 * gap + 10
    width = 860
    height = top + len(rows) * row_height + 40
    plot = width - left - right

    def x_of(value: float) -> float:
        return left + value * plot

    parts = []
    for value in np.arange(0, 1.01, 0.25):
        parts.append(
            f'<line class="grid" x1="{x_of(value):g}" y1="{top - 14:g}" '
            f'x2="{x_of(value):g}" y2="{top + len(rows) * row_height:g}"/>'
        )
        parts.append(text(x_of(value), top - 20, f"{value:.2f}", "tick", "middle"))

    for index, (_, row) in enumerate(rows.iterrows()):
        y0 = top + index * row_height + gap
        parts.append(
            text(left - 14, y0 + row_height / 2 - 4, row["class"], "rowlabel", "end")
        )
        for offset, model in enumerate(models):
            key = model["key"]
            y = y0 + offset * band
            middle = y + (band - 3) / 2
            value = row.get(f"{key}|AP")
            if pd.isna(value):
                why = row.get(f"{key}|status", "")
                parts.append(
                    f'<rect class="absent" x="{left:g}" y="{y:g}" width="26" '
                    f'height="{band - 3:g}"/>'
                )
                parts.append(text(left + 32, middle + 4, why, "absent-note"))
                continue
            parts.append(
                f'<g class="mark"><path d="'
                f'{rounded_right(left, y, max(float(value) * plot, 1.5), band - 3)}" '
                f'fill="var(--series-{key})"/>'
                + tip(
                    f"{model['label']}: AP {value:.3f} "
                    f"[{row[f'{key}|AP_lo']:.3f}, {row[f'{key}|AP_hi']:.3f}] "
                    f"- {row['class']}, {int(row['positives'])} positives, "
                    f"{int(row.get(f'{key}|n_seeds', 0))} seeds"
                )
                + "</g>"
            )
            low, high = row.get(f"{key}|AP_lo"), row.get(f"{key}|AP_hi")
            if pd.notna(low) and pd.notna(high):
                parts.append(
                    f'<line class="whisker" x1="{x_of(low):g}" y1="{middle:g}" '
                    f'x2="{x_of(high):g}" y2="{middle:g}" stroke="var(--series-{key})"/>'
                )
            # Direct value label: slots 3 and 4 fall under the light-surface relief rule,
            # so the number is never carried by the bar colour alone.
            parts.append(
                text(width - right + 8, middle + 4, f"{value:.3f}", "value")
            )
        prevalence = float(row["prevalence"])
        parts.append(
            f'<line class="baseline-mark" x1="{x_of(prevalence):g}" y1="{y0 - 2:g}" '
            f'x2="{x_of(prevalence):g}" y2="{y0 + len(keys) * band:g}"/>'
        )
        parts.append(
            text(
                width - right + 62,
                y0 + row_height / 2 - 4,
                f"n={int(row['positives'])}",
                "note",
            )
        )

    parts.append(
        f'<line class="axis" x1="{left:g}" y1="{top + len(rows) * row_height:g}" '
        f'x2="{width - right:g}" y2="{top + len(rows) * row_height:g}"/>'
    )
    parts.append(text(left + plot / 2, height - 8, "average precision", "axistitle", "middle"))
    return svg(width, height, "".join(parts), f"Average precision per class, {level} level")


def pair_table(pairs: pd.DataFrame, models: list[dict], level: str) -> str:
    """Every pairwise AP difference per class, shaded by direction and strength.

    No model is the baseline, so this is the whole comparison matrix rather than a
    reference column: each cell is AP(b) - AP(a) on the same bootstrap resamples, and a
    cell whose interval clears zero is marked. Everything else is within sampling noise.
    """
    rows = pairs[pairs["level"] == level]
    if rows.empty:
        return '<p class="lede">No pair of models can be compared at this level yet.</p>'
    labels = {m["key"]: m["label"] for m in models}
    order = list(dict.fromkeys(rows["pair"]))
    classes = [c for c in ALL_CLASSES if c in set(rows["class"])]

    head = "".join(
        f'<th><span class="pairhead">'
        f'<i class="key-line" style="background:var(--series-{p.split("-")[0]})"></i>'
        f'&minus;<i class="key-line" style="background:var(--series-{p.split("-")[1]})"></i>'
        f'<br>{escape(p.split("-")[0])} &minus; {escape(p.split("-")[1])}</span></th>'
        for p in order
    )
    body = []
    for name in classes:
        cells = []
        for pair in order:
            entry = rows[(rows["class"] == name) & (rows["pair"] == pair)]
            if entry.empty:
                cells.append('<td class="muted na">&mdash;</td>')
                continue
            value = entry.iloc[0]
            delta = float(value["delta_AP"])
            strength = min(abs(delta) / 0.30, 1.0)
            pole = "up" if delta >= 0 else "down"
            bounds = (
                ""
                if pd.isna(value["delta_lo"])
                else f'<span class="muted">[{value["delta_lo"]:+.2f}, {value["delta_hi"]:+.2f}]</span>'
            )
            mark = ' <span class="clear">&#9679;</span>' if value["clear_of_zero"] else ""
            caption = (
                f"{name}: {labels[value['b']]} minus {labels[value['a']]} "
                f"= {delta:+.3f} AP"
            )
            cells.append(
                f'<td class="delta-cell" style="--tint:{strength:.3f}" '
                f'data-pole="{pole}" title="{escape(caption)}">'
                f'<span class="delta-value">{delta:+.3f}{mark}</span><br>{bounds}</td>'
            )
        first = rows[rows["class"] == name].iloc[0]
        body.append(
            f'<tr><th scope="row">{escape(name)}'
            f'<span class="rowcount">n={int(first["positives"])}</span></th>'
            + "".join(cells)
            + "</tr>"
        )
    return (
        f'<div class="scroll"><table class="data pairs">'
        f'<thead><tr><th scope="col">class</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def pr_panels(truth: pd.DataFrame, models: list[dict], level: str) -> str:
    """Precision-recall behind each class's AP, faceted one panel per model.

    Faceted rather than overlaid on purpose: four categorical hues in one frame do not
    clear the all-pairs colour-separation gate, and a precision-recall panel is exactly
    the case where every curve can touch every other. One curve per panel means identity
    comes from the panel's own heading, not from telling four line colours apart.
    """
    scored = [m for m in models if m["ensemble"] is not None]
    classes = [
        c
        for c in ALL_CLASSES
        if c in truth.columns and truth[c].any() and any(column_of(m, c) is not None for m in scored)
    ]
    size, gap_x, gap_y = 128, 18, 40
    pad_left, pad_top = 116, 44
    width = pad_left + len(scored) * (size + gap_x)
    height = pad_top + len(classes) * (size + gap_y) + 16

    parts = []
    for column, model in enumerate(scored):
        x0 = pad_left + column * (size + gap_x)
        parts.append(
            f'<rect x="{x0:g}" y="{pad_top - 26:g}" width="10" height="10" rx="2" '
            f'fill="var(--series-{model["key"]})"/>'
        )
        parts.append(text(x0 + 15, pad_top - 17, model["key"], "panelhead"))

    for row, name in enumerate(classes):
        y0 = pad_top + row * (size + gap_y)
        y_true = truth[name].to_numpy()
        parts.append(text(pad_left - 14, y0 + size / 2 - 4, name, "rowlabel", "end"))
        parts.append(
            text(pad_left - 14, y0 + size / 2 + 11, f"n={int(y_true.sum())}", "note", "end")
        )
        for column, model in enumerate(scored):
            x0 = pad_left + column * (size + gap_x)
            parts.append(
                f'<rect x="{x0:g}" y="{y0:g}" width="{size:g}" height="{size:g}" class="panel"/>'
            )
            y_score = scores_for(model, name)
            if y_score is None:
                parts.append(
                    text(x0 + size / 2, y0 + size / 2, "no output", "absent-note", "middle")
                )
                continue
            for fraction in (0.25, 0.5, 0.75):
                parts.append(
                    f'<line class="grid" x1="{x0:g}" y1="{y0 + fraction * size:g}" '
                    f'x2="{x0 + size:g}" y2="{y0 + fraction * size:g}"/>'
                )
            prevalence = float(y_true.mean())
            parts.append(
                f'<line class="baseline-mark" x1="{x0:g}" '
                f'y1="{y0 + (1 - prevalence) * size:g}" x2="{x0 + size:g}" '
                f'y2="{y0 + (1 - prevalence) * size:g}"/>'
            )
            for seed in model["seeds"]:
                points = curve_points(y_true, scores_for(model, name, seed), x0, y0, size)
                parts.append(
                    f'<polyline class="curve curve-seed" points="{points}" '
                    f'stroke="var(--series-{model["key"]})"/>'
                )
            points = curve_points(y_true, y_score, x0, y0, size)
            ap = average_precision_score(y_true, y_score)
            parts.append(
                f'<g class="mark"><polyline class="curve" points="{points}" '
                f'stroke="var(--series-{model["key"]})"/>'
                + tip(f"{model['label']} - {name}: AP {ap:.3f}, prevalence {prevalence:.3f}")
                + "</g>"
            )
            parts.append(text(x0 + 4, y0 + size + 14, f"AP {ap:.3f}", "value"))

    parts.append(
        f'<text transform="translate(14,{pad_top + size / 2:g}) rotate(-90)" '
        f'class="axistitle" text-anchor="middle">precision</text>'
    )
    return svg(width, height, "".join(parts), f"Precision-recall curves, {level} level")


def region_chart(regions: pd.DataFrame, models: list[dict], level: str) -> str:
    """Macro AP per reach, every model, with each reach's own prevalence baseline."""
    rows = regions[
        (regions["level"] == level)
        & (regions["class"] == MACRO)
        & (regions["seed"] == "ensemble")
    ]
    if rows.empty:
        return ""
    order = rows.groupby("region")["n_units"].first().sort_values(ascending=False).index.tolist()
    scored = [m for m in models if m["ensemble"] is not None]

    left, right, top, gap = 148, 118, 40, 6
    band = 16
    row_height = len(scored) * band + 2 * gap + 12
    width = 860
    height = top + len(order) * row_height + 40
    plot = width - left - right

    def x_of(value: float) -> float:
        return left + value * plot

    parts = []
    for value in np.arange(0, 1.01, 0.25):
        parts.append(
            f'<line class="grid" x1="{x_of(value):g}" y1="{top - 14:g}" '
            f'x2="{x_of(value):g}" y2="{top + len(order) * row_height:g}"/>'
        )
        parts.append(text(x_of(value), top - 20, f"{value:.2f}", "tick", "middle"))

    for index, region in enumerate(order):
        group = rows[rows["region"] == region]
        y0 = top + index * row_height + gap
        first = group.iloc[0]
        parts.append(text(left - 14, y0 + row_height / 2 - 10, region, "rowlabel", "end"))
        parts.append(
            text(left - 14, y0 + row_height / 2 + 5, f"{int(first['n_units'])} units", "note", "end")
        )
        for offset, model in enumerate(scored):
            key = model["key"]
            entry = group[group["model"] == key]
            y = y0 + offset * band
            middle = y + (band - 3) / 2
            if entry.empty:
                continue
            value = float(entry.iloc[0]["AP"])
            parts.append(
                f'<g class="mark"><path d="'
                f'{rounded_right(left, y, max(value * plot, 1.5), band - 3)}" '
                f'fill="var(--series-{key})"/>'
                + tip(
                    f"{model['label']}: macro AP {value:.3f} over "
                    f"{int(entry.iloc[0]['n_classes'])} classes - {region}"
                )
                + "</g>"
            )
            low, high = entry.iloc[0]["AP_lo"], entry.iloc[0]["AP_hi"]
            if pd.notna(low) and pd.notna(high):
                parts.append(
                    f'<line class="whisker" x1="{x_of(low):g}" y1="{middle:g}" '
                    f'x2="{x_of(high):g}" y2="{middle:g}" stroke="var(--series-{key})"/>'
                )
            seeded = regions[
                (regions["level"] == level)
                & (regions["class"] == MACRO)
                & (regions["region"] == region)
                & (regions["model"] == key)
                & (regions["seed"] != "ensemble")
            ]
            for _, run in seeded.iterrows():
                parts.append(
                    f'<g class="mark"><circle class="seed" cx="{x_of(run["AP"]):g}" '
                    f'cy="{middle:g}" r="3.2" stroke="var(--series-{key})"/>'
                    + tip(f"{key} seed {run['seed']}: macro AP {run['AP']:.3f} - {region}")
                    + "</g>"
                )
            parts.append(
                text(width - right + 8, middle + 4, f"{value:.3f}  ({key})", "value")
            )
        prevalence = float(first["prevalence"])
        parts.append(
            f'<line class="baseline-mark" x1="{x_of(prevalence):g}" y1="{y0 - 2:g}" '
            f'x2="{x_of(prevalence):g}" y2="{y0 + len(scored) * band:g}"/>'
        )

    parts.append(
        f'<line class="axis" x1="{left:g}" y1="{top + len(order) * row_height:g}" '
        f'x2="{width - right:g}" y2="{top + len(order) * row_height:g}"/>'
    )
    parts.append(
        text(left + plot / 2, height - 8, "macro average precision", "axistitle", "middle")
    )
    return svg(width, height, "".join(parts), f"Macro AP by region, {level} level")


def curve_points(y_true: np.ndarray, y_score: np.ndarray, x0: float, y0: float, size: float,
                 budget: int = 220) -> str:
    """A precision-recall polyline, thinned to `budget` vertices.

    `precision_recall_curve` returns one vertex per distinct threshold, which is a vertex
    per unit on a set this size. At the panel sizes here that detail is well under a pixel
    and only inflates the page, so the curve is sampled on an even index grid with the two
    endpoints pinned - the shape survives, the byte count does not.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    if len(recall) > budget:
        keep = np.unique(
            np.concatenate(
                [np.linspace(0, len(recall) - 1, budget).astype(int), [0, len(recall) - 1]]
            )
        )
        precision, recall = precision[keep], recall[keep]
    return " ".join(
        f"{x0 + r * size:.1f},{y0 + (1 - p) * size:.1f}" for r, p in zip(recall, precision)
    )


def confusion_table(counts: pd.DataFrame, model: dict) -> str:
    """Row-normalised confusion heatmap as a real table, so it reads without the colour."""
    totals = counts.sum(axis=1)
    head = "".join(f"<th><span>{escape(column)}</span></th>" for column in counts.columns)
    body = []
    for name, row in counts.iterrows():
        total = totals[name]
        cells = []
        for column, value in row.items():
            fraction = value / total if total else 0.0
            step = heat_index(fraction)
            classes = "cell" + (" cell-flip" if step >= HEAT_FLIP else "")
            classes += " diag" if column == name else ""
            cells.append(
                f'<td class="{classes}" style="--step:{step}" '
                f'title="{escape(f"{name} -> {column}: {value} of {total} ({fraction:.0%})")}">'
                f'{value if value else ""}</td>'
            )
        body.append(
            f'<tr><th scope="row">{escape(name)}<span class="rowcount">{total}</span></th>'
            + "".join(cells)
            + "</tr>"
        )
    return (
        f'<figure class="confusion"><figcaption>'
        f'<i class="key-line" style="background:var(--series-{model["key"]})"></i>'
        f'{escape(model["label"])}</figcaption>'
        f'<div class="scroll"><table class="heat">'
        f'<thead><tr><th scope="col" class="corner">annotated \\ top class</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></figure>"
    )


# --------------------------------------------------------------------------------------
# dashboard: page
# --------------------------------------------------------------------------------------


def series_tokens(index: int) -> str:
    return "".join(
        f"  --series-{key}: {values[index]};\n" for key, values in SERIES.items()
    )


STYLE_LIGHT = """
:root {
  color-scheme: light;
  --plane: #f7f7f4;
  --surface: #fcfcfb;
  --ink: #17171a;
  --ink-2: #52514e;
  --muted: #86857f;
  --grid: #e3e2db;
  --axis: #c6c5bd;
  --rule: rgba(17,17,20,0.10);
  --absent: #dcdbd4;
  --pole-up: #2a78d6;
  --pole-down: #d03b3b;
  --neutral: #f0efec;
  --good: #006300;
  --heat-0: #f4f7fb; --heat-1: #cde2fb; --heat-2: #9ec5f4; --heat-3: #6da7ec;
  --heat-4: #3987e5; --heat-5: #256abf; --heat-6: #0d366b;
  --heat-ink: #17171a;
  --heat-ink-flip: #ffffff;
"""

STYLE_DARK = """
  color-scheme: dark;
  --plane: #0e0e0d;
  --surface: #1a1a19;
  --ink: #f5f5f0;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #3d3d3a;
  --rule: rgba(255,255,255,0.12);
  --absent: #33322f;
  --pole-up: #3987e5;
  --pole-down: #e66767;
  --neutral: #383835;
  --good: #0ca30c;
  --heat-0: #1f2429; --heat-1: #0d366b; --heat-2: #184f95; --heat-3: #256abf;
  --heat-4: #3987e5; --heat-5: #6da7ec; --heat-6: #9ec5f4;
  --heat-ink: #f5f5f0;
  --heat-ink-flip: #0b0b0b;
"""

STYLE_BODY = """
body {
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.55;
  margin: 0;
  padding: 40px 24px 88px;
}
.page { max-width: 1120px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }

.masthead { display: flex; flex-direction: column; gap: 10px; }
.eyebrow {
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0;
}
h1 { font-size: 30px; line-height: 1.15; margin: 0; text-wrap: balance; font-weight: 620; }
.standfirst { margin: 0; color: var(--ink-2); max-width: 68ch; }

.banner {
  background: var(--surface); border: 1px solid var(--rule); border-left: 3px solid var(--pole-down);
  border-radius: 6px; padding: 12px 16px; color: var(--ink-2); font-size: 13.5px;
}
.banner strong { color: var(--ink); }

.roster { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; }
.model {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 14px 16px; display: flex; flex-direction: column; gap: 4px;
}
.model-name { display: flex; align-items: center; gap: 8px; font-weight: 600; font-size: 14px; }
.model-key {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px;
  color: var(--muted); background: var(--plane); border-radius: 3px; padding: 0 5px;
}
.model-note { font-size: 12.5px; color: var(--ink-2); margin: 0; }
.model-runs { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.pending { color: var(--pole-down); font-weight: 600; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.tile {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 16px 18px; display: flex; flex-direction: column; gap: 6px;
}
.tile-label { margin: 0; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.tile-note { margin: 0; font-size: 12.5px; color: var(--ink-2); }
.tile-wide { grid-column: span 2; }
@media (max-width: 760px) { .tile-wide { grid-column: span 1; } }
.mini { width: 100%; }
.mini th, .mini td { text-align: right; padding: 4px 0 4px 14px; white-space: nowrap; }
.mini thead th {
  font-size: 11px; font-weight: 500; color: var(--muted);
  border-bottom: 1px solid var(--grid); padding-bottom: 6px;
}
.mini thead th .key-line { margin-right: 5px; }
.mini tbody th { text-align: left; padding-left: 0; font-weight: 500; color: var(--ink-2); font-size: 13px; }
.mini tbody td { font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }

section { display: flex; flex-direction: column; gap: 14px; }
h2 {
  font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; margin: 0;
  padding-bottom: 8px; border-bottom: 1px solid var(--rule); color: var(--ink);
}
h3 { font-size: 15px; margin: 0; font-weight: 600; }
.lede { margin: 0; color: var(--ink-2); max-width: 74ch; }

.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 20px 22px 14px; display: flex; flex-direction: column; gap: 12px;
}
.card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px 20px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; margin-left: auto; }
.legend span { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-2); }
.key-line { width: 14px; height: 3px; border-radius: 2px; display: inline-block; flex: none; }
.key-tick { width: 2px; height: 12px; background: var(--muted); display: inline-block; }
.key-seed {
  width: 9px; height: 9px; border-radius: 50%; display: inline-block;
  background: var(--surface); border: 1.5px solid var(--ink-2);
}
.scroll { overflow-x: auto; }

.mark { cursor: default; }
.mark:hover path, .mark:hover circle, .mark:hover polyline { opacity: 0.75; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.baseline-mark { stroke: var(--muted); stroke-width: 1.5; }
.whisker { stroke-width: 1.5; opacity: 0.55; }
.curve { fill: none; stroke-width: 2; stroke-linejoin: round; }
.curve-seed { stroke-width: 1; opacity: 0.35; }
.seed { fill: var(--surface); stroke-width: 1.5; }
.panel { fill: none; stroke: var(--grid); stroke-width: 1; }
.absent { fill: var(--absent); }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.rowlabel { font-size: 12.5px; fill: var(--ink-2); }
.panelhead { font-size: 12px; fill: var(--ink); font-weight: 600; }
.note { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.value { font-size: 11px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
.absent-note { font-size: 10.5px; fill: var(--muted); font-style: italic; }
.axistitle { font-size: 11.5px; fill: var(--muted); }

table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
.data { width: 100%; }
.data th, .data td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
.data thead th { color: var(--muted); font-weight: 500; font-size: 11.5px; white-space: nowrap; }
.data tbody th { text-align: left; font-weight: 500; white-space: nowrap; }
.data tr.total th, .data tr.total td { font-weight: 650; border-top: 1px solid var(--axis); }
.data tr.headline th, .data tr.headline td { background: var(--tint-row, color-mix(in srgb, var(--ink) 5%, transparent)); }
/* Not enough training crops or test positives for the AP to separate models. Dimmed
   rather than dropped: the weakness is real, it just cannot be ranked. */
.data tr.starved th, .data tr.starved td { color: var(--muted); }
.data tr.starved th { font-style: italic; }
/* A single estimate whose own interval is too wide to read a ranking off. */
.data td.loose { position: relative; }
.data td.loose::after {
  content: "~"; position: absolute; left: 2px; top: 50%; transform: translateY(-50%);
  color: var(--muted); font-size: 10px;
}
.table-note { font-size: 12px; color: var(--ink-2); margin: 10px 0 0; max-width: 78ch; line-height: 1.5; }
.muted { color: var(--muted); font-weight: 400; font-size: 11px; }
.rowcount { color: var(--muted); font-size: 11px; margin-left: 8px; font-weight: 400; }

.pairs td { text-align: center; line-height: 1.3; }
.pairs .pairhead { display: inline-block; font-size: 11px; line-height: 1.4; }
.pairs .pairhead .key-line { vertical-align: middle; margin: 0 2px; }
.delta-cell { background: color-mix(in srgb, var(--tint-color) calc(var(--tint) * 42%), transparent); }
.delta-cell[data-pole="up"] { --tint-color: var(--pole-up); }
.delta-cell[data-pole="down"] { --tint-color: var(--pole-down); }
.delta-value { font-weight: 600; font-size: 12.5px; }
.clear { color: var(--ink); font-size: 9px; vertical-align: middle; }
.na { text-align: center; }

.confusions { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 18px; }
.confusion { margin: 0; display: flex; flex-direction: column; gap: 10px; }
.confusion figcaption { font-size: 13px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
.heat th, .heat td { padding: 0; }
.heat thead th { font-size: 10.5px; color: var(--muted); font-weight: 500; padding: 0 0 6px; }
.heat thead th span { display: block; writing-mode: vertical-rl; transform: rotate(180deg); }
.heat .corner { writing-mode: horizontal-tb; text-align: right; padding-right: 10px; white-space: nowrap; }
.heat tbody th {
  text-align: right; font-size: 12px; font-weight: 500; color: var(--ink-2);
  padding-right: 10px; white-space: nowrap;
}
.heat .cell {
  width: 34px; height: 26px; text-align: center; font-size: 11px; color: var(--heat-ink);
  background: var(--heat-0); border: 2px solid var(--surface); border-radius: 3px;
}
.heat .cell-flip { color: var(--heat-ink-flip); }
.heat .diag { outline: 1.5px solid var(--axis); outline-offset: -2px; }
.heat .cell[style*="--step:1"] { background: var(--heat-1); }
.heat .cell[style*="--step:2"] { background: var(--heat-2); }
.heat .cell[style*="--step:3"] { background: var(--heat-3); }
.heat .cell[style*="--step:4"] { background: var(--heat-4); }
.heat .cell[style*="--step:5"] { background: var(--heat-5); }
.heat .cell[style*="--step:6"] { background: var(--heat-6); }

.ramp { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--muted); }
.ramp i { width: 22px; height: 10px; border-radius: 2px; display: inline-block; }

.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px 32px; }
.notes p { margin: 0; color: var(--ink-2); font-size: 13.5px; }
.notes strong { color: var(--ink); font-weight: 600; }
code {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.9em;
  background: var(--plane); border: 1px solid var(--rule); border-radius: 3px; padding: 0 4px;
}
a { color: var(--pole-up); }
:focus-visible { outline: 2px solid var(--pole-up); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def stylesheet() -> str:
    """Light tokens on bare :root, dark redefined under both the media query and the
    explicit theme stamp, so a viewer toggle wins in either direction."""
    return (
        STYLE_LIGHT
        + series_tokens(0)
        + "}\n"
        + '@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"]) {\n'
        + STYLE_DARK
        + series_tokens(1)
        + "  }\n}\n"
        + ':root[data-theme="dark"] {\n'
        + STYLE_DARK
        + series_tokens(1)
        + "}\n"
        + STYLE_BODY
    )


def legend(models: list[dict], seeds: bool = False) -> str:
    entries = "".join(
        f'<span><i class="key-line" style="background:var(--series-{m["key"]})"></i>'
        f'{escape(m["key"])}</span>'
        for m in models
        if m["ensemble"] is not None
    )
    seed_key = '<span><i class="key-seed"></i>seed run</span>' if seeds else ""
    return (
        f'<div class="legend">{entries}{seed_key}'
        '<span><i class="key-tick"></i>prevalence</span></div>'
    )


def roster_card(models: list[dict]) -> str:
    cards = []
    for model in models:
        n_ready = len(model["seeds"])
        if model["n_pending"]:
            state = (
                f'<span class="pending">{model["n_pending"]} run(s) still training</span>'
            )
        elif n_ready == 0:
            state = '<span class="pending">no runs found</span>'
        else:
            state = ""
        cards.append(
            f'<div class="model"><p class="model-name">'
            f'<i class="key-line" style="background:var(--series-{model["key"]})"></i>'
            f'{escape(model["label"])}<span class="model-key">{escape(model["key"])}</span></p>'
            f'<p class="model-note">{escape(model["detail"])}</p>'
            f'<p class="model-runs">{n_ready} seed run(s) scored, '
            f'{len(model["classes"])} output classes {state}</p></div>'
        )
    return f'<div class="roster">{"".join(cards)}</div>'


def macro_tiles(metrics: pd.DataFrame, models: list[dict]) -> str:
    """The headline numbers: the shared-9 macro beside the well-sampled one.

    The shared-9 macro is like for like on *vocabulary* - a macro over each model's own
    classes would hand the new generation paddock, 57 % of the crops and by far the
    easiest class here, for nothing. But it is not like for like on *evidence*: five of
    those nine classes were trained on <= 14 crops and three carry single-digit test
    positives, and an unweighted mean gives a four-example class the same say as a
    102-positive one. Both are shown, with the well-sampled macro marked as the one to
    rank on and the gap between them left visible rather than reconciled.
    """
    scored = [m for m in models if m["ensemble"] is not None]
    tiles = []
    for level in LEVELS:
        rows = {
            label: metrics[(metrics["level"] == level) & (metrics["class"] == label)]
            for label in (MACRO_SHARED, MACRO_SAMPLED)
        }
        if rows[MACRO_SHARED].empty:
            continue
        shared, sampled = rows[MACRO_SHARED].iloc[0], (
            rows[MACRO_SAMPLED].iloc[0] if not rows[MACRO_SAMPLED].empty else None
        )
        body = []
        for m in scored:
            key = m["key"]
            if pd.isna(shared.get(f"{key}|AP")):
                continue
            best = sampled.get(f"{key}|AP") if sampled is not None else np.nan
            body.append(
                f'<tr><th scope="row">'
                f'<i class="key-line" style="background:var(--series-{key})"></i>'
                f"{escape(key)}</th>"
                f'<td class="muted">{shared[f"{key}|AP"]:.3f}</td>'
                f"<td>{'-' if pd.isna(best) else f'{best:.3f}'}</td></tr>"
            )
        n_classes = (
            int(sampled.get(f"{scored[0]['key']}|n_classes", 0))
            if sampled is not None
            else 0
        )
        tiles.append(
            f'<div class="tile"><p class="tile-label">macro AP, {escape(level)} level</p>'
            f'<table class="mini"><thead><tr><th></th>'
            f'<th class="muted">shared 9</th><th>well sampled</th></tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table>"
            f'<p class="tile-note">both are unweighted means over classes every model can '
            f'emit. <b>well sampled</b> keeps only the {n_classes} of those nine with at '
            f'least {MIN_TRAIN_EXAMPLES} training crops and {MIN_POSITIVES} test '
            f'positives - the rest are estimated too loosely to rank on. Rank on the '
            f'right-hand column.</p></div>'
        )
    return "".join(tiles)


def agreement_card(agreement: pd.DataFrame, models: list[dict]) -> str:
    if agreement.empty:
        return ""
    scored = [m for m in models if m["ensemble"] is not None]
    levels = list(dict.fromkeys(agreement["level"]))
    head = "".join(
        f'<th><i class="key-line" style="background:var(--series-{m["key"]})"></i>'
        f"{escape(level)} {escape(m['key'])}</th>"
        for level in levels
        for m in scored
    )
    body = []
    for depth in sorted(set(agreement["depth"])):
        cells = []
        for level in levels:
            for model in scored:
                entry = agreement[
                    (agreement["level"] == level)
                    & (agreement["depth"] == depth)
                    & (agreement["model"] == model["key"])
                ]
                cells.append(
                    f"<td>{entry.iloc[0]['agreement']:.1%}</td>" if not entry.empty else "<td>-</td>"
                )
        body.append(f'<tr><th scope="row">top-{depth}</th>{"".join(cells)}</tr>')
    counts = "; ".join(
        f"{level}: "
        + ", ".join(
            f"{m['key']} {int(agreement[(agreement['level'] == level) & (agreement['model'] == m['key'])].iloc[0]['n_units'])}"
            for m in scored
            if not agreement[
                (agreement["level"] == level) & (agreement["model"] == m["key"])
            ].empty
        )
        for level in levels
    )
    return (
        '<div class="tile tile-wide"><p class="tile-label">agreement with the annotation</p>'
        f'<div class="scroll"><table class="mini"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'<p class="tile-note">an annotated class is among the model\'s k highest-scoring '
        f'outputs. Each model is scored over its own vocabulary and over the units carrying '
        f'a class it can emit, so the denominators differ &mdash; units scored ({escape(counts)})'
        f'</p></div>'
    )


def metrics_table(metrics: pd.DataFrame, models: list[dict], level: str) -> str:
    rows = metrics[metrics["level"] == level]
    scored = [m for m in models if m["ensemble"] is not None]
    head = (
        "<tr><th>class</th><th>train</th><th>positives</th><th>prevalence</th>"
        + "".join(
            f'<th><i class="key-line" style="background:var(--series-{m["key"]})"></i>'
            f"AP {escape(m['key'])}</th>"
            for m in scored
        )
        + "".join(f"<th>AUC {escape(m['key'])}</th>" for m in scored)
        + "</tr>"
    )
    body = []
    for _, row in rows.iterrows():
        cells = []
        for model in scored:
            value = row.get(f"{model['key']}|AP")
            if pd.isna(value):
                why = row.get(f"{model['key']}|status") or "-"
                cells.append(f'<td class="muted">{escape(why)}</td>')
            else:
                low, high = row.get(f"{model['key']}|AP_lo"), row.get(f"{model['key']}|AP_hi")
                bounds = (
                    ""
                    if pd.isna(low)
                    else f' <span class="muted">[{low:.2f}, {high:.2f}]</span>'
                )
                width = row.get(f"{model['key']}|AP_ci")
                loose = ' class="loose"' if pd.notna(width) and width >= WIDE_INTERVAL else ""
                cells.append(f"<td{loose}>{value:.3f}{bounds}</td>")
        for model in scored:
            value = row.get(f"{model['key']}|AUC")
            cells.append("<td>-</td>" if pd.isna(value) else f"<td>{value:.3f}</td>")
        label = str(row["class"])
        classes = []
        if label.startswith("MACRO"):
            classes.append("total")
            if label == MACRO_SAMPLED:
                classes.append("headline")
        elif not row.get("well_sampled", True):
            # Too few training examples or too few test positives to rank models on.
            classes.append("starved")
        emphasis = f' class="{" ".join(classes)}"' if classes else ""
        positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
        prevalence = "" if pd.isna(row.get("prevalence")) else f"{row['prevalence']:.3f}"
        trained = "" if pd.isna(row.get("n_train")) else int(row["n_train"])
        body.append(
            f'<tr{emphasis}><th scope="row">{escape(label)}</th>'
            f"<td>{trained}</td><td>{positives}</td><td>{prevalence}</td>{''.join(cells)}</tr>"
        )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
        f'<p class="table-note"><b>train</b> is the number of crops carrying that class in the '
        f'2026-08-21 training split, which is what the multilabel models were fitted on. '
        f'Dimmed rows fall short of either bar - {MIN_TRAIN_EXAMPLES} training crops or '
        f'{MIN_POSITIVES} test positives - so their AP cannot separate models; an AP cell '
        f'is marked when its own 95 % interval is at least {WIDE_INTERVAL:.2f} wide.</p>'
    )


def region_table(regions: pd.DataFrame, models: list[dict], level: str) -> str:
    rows = regions[(regions["level"] == level) & (regions["seed"] == "ensemble")]
    if rows.empty:
        return ""
    scored = [m for m in models if m["ensemble"] is not None]
    order = rows.groupby("region")["n_units"].first().sort_values(ascending=False).index.tolist()
    head = (
        "<tr><th>region</th><th>class</th><th>positives</th><th>prevalence</th>"
        + "".join(
            f'<th><i class="key-line" style="background:var(--series-{m["key"]})"></i>'
            f"{escape(m['key'])}</th>"
            for m in scored
        )
        + "</tr>"
    )
    body = []
    for region in order:
        group = rows[rows["region"] == region]
        names = [MACRO] + [c for c in ALL_CLASSES if c in set(group["class"])]
        for position, name in enumerate(names):
            entry = group[group["class"] == name]
            if entry.empty:
                continue
            first = entry.iloc[0]
            cells = []
            for model in scored:
                mine = entry[entry["model"] == model["key"]]
                cells.append(
                    f"<td>{mine.iloc[0]['AP']:.3f}</td>"
                    if not mine.empty
                    else '<td class="muted">no output</td>'
                )
            label = (
                f'{escape(region)} <span class="muted">{int(first["n_units"])} units</span>'
                if position == 0
                else ""
            )
            emphasis = ' class="total"' if position == 0 else ""
            body.append(
                f'<tr{emphasis}><th scope="row">{label}</th>'
                f"<td>{escape('macro' if name == MACRO else name)}</td>"
                f"<td>{int(first['positives'])}</td><td>{first['prevalence']:.3f}</td>"
                f"{''.join(cells)}</tr>"
            )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def build_page(
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    regions: pd.DataFrame,
    agreement: pd.DataFrame,
    confusions: dict[str, dict[str, pd.DataFrame]],
    truths: dict[str, pd.DataFrame],
    models: list[dict],
    n_crops: int,
    n_farms: int,
    aggregation: str,
) -> str:
    scored = [m for m in models if m["ensemble"] is not None]
    waiting = [m for m in models if m["ensemble"] is None or m["n_pending"]]
    ramp = "".join(f'<i style="background:var(--heat-{step})"></i>' for step in range(7))

    banner = ""
    if waiting:
        detail = "; ".join(
            f"{escape(m['label'])} ({m['n_pending']} run(s) still training)"
            if m["n_pending"]
            else f"{escape(m['label'])} (nothing saved yet)"
            for m in waiting
        )
        banner = (
            f'<div class="banner"><strong>Incomplete sweep.</strong> {detail}. '
            f"Everything below is scored on the runs that have landed; rerun "
            f"<code>gen_evaluation_2026_08_21.py</code> once the rest finish and the "
            f"charts fill in.</div>"
        )

    sections = [
        f"""
<header class="masthead">
  <p class="eyebrow">Crop-level relabelled build &middot; VIC hold-out &middot;
  {escape(n_crops)} crops &middot; {escape(n_farms)} farms</p>
  <h1>Does the multilabel retrain read a paddock as a paddock?</h1>
  <p class="standfirst">Four checkpoints scored against the hand-relabelled crop labels in
  <code>relabelled_test_df.csv</code>. The split is geographic &mdash; NSW trains, VIC tests
  &mdash; so every number here is out-of-region. The 2026-08-21 models are multilabel and can
  emit <code>paddock</code> and <code>other_industrial</code>; the two older multiclass models
  cannot, and have to force those crops onto a livestock class. Average precision by class,
  because the classes are rare and every model emits a ranking rather than a decision.</p>
</header>
""",
        banner,
        roster_card(models),
        f'<div class="tiles">{macro_tiles(metrics, models)}'
        f"{agreement_card(agreement, models)}</div>",
        f"""
<section>
  <h2>Average precision by class</h2>
  <p class="lede">A random ranker scores the prevalence, marked on each row &mdash; a bar that
  stops near the tick has found nothing. Whiskers are 95&nbsp;% bootstrap intervals over the
  evaluation units. A grey stub means the model has no output for that class at all, which is
  not the same as scoring zero on it.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level &mdash; one annotated crop per unit</h3>
    {legend(models)}</div>
    <div class="scroll">{ap_chart(metrics[metrics["level"] == "crop"], scored, "crop")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; {escape(aggregation)} over each farm's crops</h3>
    {legend(models)}</div>
    <p class="lede">Farm labels come from the workbook's &ldquo;Farm labels&rdquo; sheet, which
    records livestock only &mdash; <code>paddock</code> and <code>other_industrial</code> are
    crop-level classes and have no farm-level truth to score against.</p>
    <div class="scroll">{ap_chart(metrics[metrics["level"] == "farm"], scored, "farm")}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Every pairwise difference</h2>
  <p class="lede">No model is the baseline here, so this is the full comparison matrix rather
  than a reference column. Each cell is AP(left) &minus; AP(right) with both models scored on
  the <em>same</em> bootstrap resamples, so the difference carries its own interval. A
  &#9679; marks an interval clear of zero &mdash; a change these {escape(n_crops)} crops can
  actually support. Everything else is within sampling noise, however large the point
  estimate looks.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3></div>
    {pair_table(pairs, models, "crop")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level</h3></div>
    {pair_table(pairs, models, "farm")}
  </div>
</section>
""",
        f"""
<section>
  <h2>By reach</h2>
  <p class="lede">The VIC hold-out is two reaches, and they are different landscapes with very
  different class mixes, so each carries its own prevalence baseline. Read the models against
  each other <em>within</em> a reach rather than reading one reach against the other. A missing
  whisker means too few bootstrap draws kept every class in that reach.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level &mdash; macro AP over each model's scoreable classes</h3>
    {legend(models, seeds=True)}</div>
    <div class="scroll">{region_chart(regions, scored, "crop")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level</h3>{legend(models, seeds=True)}</div>
    <div class="scroll">{region_chart(regions, scored, "farm")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Every class within each reach, crop level</h3></div>
    {region_table(regions, models, "crop")}
  </div>
</section>
""",
        f"""
<section>
  <h2>The curves behind the numbers</h2>
  <p class="lede">Precision against recall at every threshold, crop level, one panel per model
  so the curves are never asked to be told apart by colour alone. The thin curves are the
  individual seed runs behind each ensemble; the horizontal tick is the prevalence, so a curve
  hugging it is no better than picking at random.</p>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level</h3>
      <div class="legend"><span><i class="key-seed"></i>thin line = one seed run</span>
      <span><i class="key-tick"></i>prevalence</span></div>
    </div>
    <div class="scroll">{pr_panels(truths["crop"], scored, "crop")}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Where the classes leak</h2>
  <p class="lede">Average precision never fixes an operating point, so it cannot say
  <em>where</em> a class goes wrong. These are the annotated class against the model's
  top-scoring class, shaded by row fraction, each model over its own vocabulary. This is the
  panel the whole retrain is about: {escape(int(truths["crop"]["paddock"].sum()))} of the
  {escape(n_crops)} crops are paddock, and the two multiclass models have no column to put them
  in &mdash; whichever livestock class absorbs them is what drags that class's precision down.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3>
      <div class="ramp">less{ramp}more of the row</div>
    </div>
    <div class="confusions">
      {"".join(confusion_table(confusions["crop"][m["key"]], m) for m in scored)}
    </div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; each annotated class against the farm's top class</h3>
      <div class="ramp">less{ramp}more of the row</div>
    </div>
    <p class="lede">Farm labels are multi-label, so a farm carrying two classes appears in both
    rows, each paired with the same single top class &mdash; the diagonal here means
    &ldquo;this class is <em>the</em> top class for its farm&rdquo;, which a farm with two
    classes can satisfy for only one of them. Row totals count annotated (farm, class) pairs,
    not farms. Note that the multilabel models can put a farm's top score on
    <code>paddock</code>, which no farm is ever annotated with.</p>
    <div class="confusions">
      {"".join(confusion_table(confusions["farm"][m["key"]], m) for m in scored)}
    </div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Full results</h2>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3></div>
    {metrics_table(metrics, models, "crop")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level ({escape(aggregation)} over crops)</h3></div>
    {metrics_table(metrics, models, "farm")}
  </div>
</section>
""",
        f"""
<section>
  <h2>How this was scored</h2>
  <div class="notes">
    <p><strong>Ground truth.</strong> The <code>binary_*</code> columns of
    <code>relabelled_test_df.csv</code>, which <code>gen_croplevel_dataset.py</code> built from
    the workbooks in <code>labelled_sheets/</code>. {escape(n_crops)} crops over
    {escape(n_farms)} farms; <code>Ambiguous</code> crops were already dropped at build time,
    and the 17 genuinely multi-label crops carry both their classes.</p>
    <p><strong>Which rows.</strong> The VIC hold-out only. The split is geographic, so these
    are reaches no model trained on &mdash; but it also means the class mix differs sharply
    from NSW: poultry is 49 crops in NSW against 3 here, beef 7 against 49.</p>
    <p><strong>Aligning the older models.</strong> They predicted over
    <code>original_new_2026_07_24_generalisation/dataset.csv</code> and are reindexed onto these
    crops on <code>(ecw_stem, building_cluster, PFI)</code>. <code>image_path</code> cannot be
    used: the current build is keyed on <code>farm_uid</code>, the older one on PFI.</p>
    <p><strong>Class vocabularies differ.</strong> Read back off each run's own
    <code>.hydra/config.yaml</code> before scoring. A class a model cannot emit is reported
    blank, never as zero, and the headline macro is restricted to the nine classes all four
    share so that paddock does not flatter the newer pair.</p>
    <p><strong>Thin classes.</strong> Several classes rest on very few positives here &mdash;
    poultry on 3 crops, sheep on 5, commercialpig on 8. Their APs are extremely unstable and
    the intervals say so; treat any pairwise difference on those rows as unreadable.</p>
    <p><strong>Uncertainty.</strong> Percentile bootstrap over the evaluation units, paired
    across models. An interval is withheld when fewer than half the draws were usable.</p>
  </div>
</section>
""",
    ]

    return (
        "<title>Crop-level multilabel evaluation</title>"
        f"<style>{stylesheet()}</style>"
        f'<div class="page">{"".join(sections)}</div>'
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation",
        default="max",
        choices=["max", "mean", "top2"],
        help="how a farm's per-crop scores become one farm score (default max)",
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--file", type=Path, default=None, help="where to write the dashboard")
    args = parser.parse_args()

    test = pd.read_csv(TEST_CSV, keep_default_na=False)
    legacy = pd.read_csv(LEGACY_DATASET, keep_default_na=False)
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    trained = train_counts(train)
    print(f"{TEST_CSV.name}: {len(test)} crops, {test['farm_uid'].nunique()} farms")
    print(f"{TRAIN_CSV.name}: {len(train)} crops, per class: " + ", ".join(
        f"{name} {trained[name]}" for name in ALL_CLASSES if trained[name]
    ))

    # Map the legacy prediction rows onto the test crops.
    row_of = pd.Series(np.arange(len(legacy)), index=pd.MultiIndex.from_frame(legacy[JOIN_KEY]))
    if row_of.index.duplicated().any():
        raise SystemExit(f"{JOIN_KEY} is not unique in {LEGACY_DATASET}")
    rows = row_of.reindex(pd.MultiIndex.from_frame(test[JOIN_KEY]))
    if rows.isna().any():
        raise SystemExit(f"{int(rows.isna().sum())} test crops are absent from {LEGACY_DATASET}")
    rows = rows.to_numpy()

    models = []
    for spec in MODELS:
        model = load_model(spec, rows, len(legacy), len(test))
        note = f"{len(model['seeds'])} seed run(s)"
        if model["n_pending"]:
            note += f", {model['n_pending']} still training"
        print(f"  {model['key']:>6}: {note}")
        for name in model["pending_names"]:
            print(f"          pending: {name}")
        models.append(model)

    scored = [m for m in models if m["ensemble"] is not None]
    if not scored:
        raise SystemExit("no model has a saved prediction yet")

    args.output.mkdir(parents=True, exist_ok=True)

    # ---- crop level ----------------------------------------------------------------
    crop = crop_truth(test)
    crop_regions = test["source"].map(region_name).to_numpy()

    # ---- farm level ----------------------------------------------------------------
    farms = farm_frame(test)
    farm = farm_truth(farms)
    farm_regions = farms["source"].map(region_name).to_numpy()
    keys = test["farm_uid"].to_numpy()
    order = farms["farm_uid"].to_numpy()
    print(
        f"farm level: {len(farms)} farms, crop scores aggregated with {args.aggregation}"
    )

    farm_models = []
    for model in models:
        if model["ensemble"] is None:
            farm_models.append({**model, "ensemble": None, "seeds": []})
            continue
        farm_models.append(
            {
                **model,
                "ensemble": farm_scores(model["ensemble"], keys, order, args.aggregation),
                "seeds": [
                    farm_scores(s, keys, order, args.aggregation) for s in model["seeds"]
                ],
            }
        )

    truths = {"crop": crop, "farm": farm}
    by_level = {"crop": models, "farm": farm_models}

    metrics, pairs, regions, agreement = [], [], [], []
    for level in LEVELS:
        table, pair = evaluate(
            level, truths[level], by_level[level], args.bootstrap, args.seed, trained
        )
        metrics.append(table)
        pairs.append(pair)
        regions.append(
            region_summary(
                level,
                truths[level],
                crop_regions if level == "crop" else farm_regions,
                by_level[level],
                args.bootstrap,
                args.seed,
            )
        )
        found = top_k_agreement(truths[level], by_level[level])
        found.insert(0, "level", level)
        agreement.append(found)

    metrics = pd.concat(metrics, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    regions = pd.concat(regions, ignore_index=True)
    agreement = pd.concat(agreement, ignore_index=True)

    confusions = {
        "crop": {
            m["key"]: confusion(crop, m, "no class marked")
            for m in models
            if m["ensemble"] is not None
        },
        "farm": {
            m["key"]: confusion(farm, m, "no class marked")
            for m in farm_models
            if m["ensemble"] is not None
        },
    }

    # ---- console summary -------------------------------------------------------------
    for level in LEVELS:
        title = f"{level} level - average precision per class"
        print(f"\n{title}\n" + "-" * len(title))
        header = f"{'class':<20}{'train':>6}{'pos':>5}{'prev':>7}"
        for model in scored:
            header += f"{model['key']:>22}"
        print(header)
        for _, row in metrics[metrics["level"] == level].iterrows():
            positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
            prevalence = "" if pd.isna(row.get("prevalence")) else f"{row['prevalence']:.3f}"
            trained = "" if pd.isna(row.get("n_train")) else int(row["n_train"])
            mark = "" if row.get("well_sampled", True) or str(row["class"]).startswith("MACRO") else " *"
            line = f"{str(row['class']) + mark:<20}{trained:>6}{positives:>5}{prevalence:>7}"
            for model in scored:
                value = row.get(f"{model['key']}|AP")
                if pd.isna(value):
                    why = row.get(f"{model['key']}|status") or "-"
                    line += f"{why:>22}"
                else:
                    low, high = row.get(f"{model['key']}|AP_lo"), row.get(f"{model['key']}|AP_hi")
                    bounds = "" if pd.isna(low) else f" [{low:.2f},{high:.2f}]"
                    line += f"{f'{value:.3f}{bounds}':>22}"
            print(line)
        print(
            "prev = prevalence, the AP a random ranker scores. 95% bootstrap intervals.\n"
            f"* = fewer than {MIN_TRAIN_EXAMPLES} training crops or {MIN_POSITIVES} test "
            f"positives; scored but too loosely estimated to rank models on, and excluded\n"
            f"    from {MACRO_SAMPLED}."
        )

    # ---- outputs ---------------------------------------------------------------------
    metrics.to_csv(args.output / "evaluation.csv", index=False)
    pairs.to_csv(args.output / "evaluation_pairs.csv", index=False)
    regions.to_csv(args.output / "evaluation_by_region.csv", index=False)
    agreement.to_csv(args.output / "agreement.csv", index=False)
    for level, tables in confusions.items():
        pd.concat(tables, names=["model", "annotated"]).to_csv(
            args.output / f"confusion_{level}.csv"
        )

    pd.DataFrame(
        [
            {
                "key": m["key"],
                "model": m["label"],
                "detail": m["detail"],
                "classes": " ".join(m["classes"]),
                "seeds": len(m["seeds"]),
                "seed_labels": " ".join(m["seed_labels"]),
                "pending": m["n_pending"],
                "aggregation": args.aggregation,
                "runs": str(m["runs"]),
            }
            for m in models
        ]
    ).to_csv(args.output / "models.csv", index=False)

    def score_columns(source: list[dict]) -> dict[str, np.ndarray]:
        columns = {}
        for model in source:
            if model["ensemble"] is None:
                continue
            variants = [(model["key"], model["ensemble"])]
            variants += [
                (f"{model['key']}#{label}", array)
                for label, array in zip(model["seed_labels"], model["seeds"])
            ]
            for prefix, array in variants:
                for name in model["classes"]:
                    columns[f"{prefix}|{name}"] = array[:, model["classes"].index(name)]
        return columns

    pd.DataFrame(
        {
            "region": crop_regions,
            "source": test["source"].to_numpy(),
            "farm_uid": test["farm_uid"].to_numpy(),
            "PFI": test["PFI"].to_numpy(),
            "image_path": test["image_path"].to_numpy(),
            "crop_label": test["crop_label"].to_numpy(),
            "crop_classes": test["crop_classes"].to_numpy(),
            **{f"true|{name}": crop[name].to_numpy() for name in ALL_CLASSES},
            **score_columns(models),
        }
    ).to_csv(args.output / "scores_crop.csv", index=False)

    pd.DataFrame(
        {
            "region": farm_regions,
            "source": farms["source"].to_numpy(),
            "farm_uid": order,
            "PFI": farms["PFI"].to_numpy(),
            "n_crops": farms["n_crops"].to_numpy(),
            "farm_labels": farms["farm_labels"].to_numpy(),
            **{f"true|{name}": farm[name].to_numpy() for name in ALL_CLASSES},
            **score_columns(farm_models),
        }
    ).to_csv(args.output / "scores_farm.csv", index=False)

    destination = args.file or args.output / "evaluation_dashboard.html"
    destination.write_text(
        build_page(
            metrics,
            pairs,
            regions,
            agreement,
            confusions,
            truths,
            models,
            len(test),
            len(farms),
            args.aggregation,
        ),
        encoding="utf-8",
    )

    print(
        f"\nwrote {destination} plus evaluation{{,_pairs,_by_region}}.csv, agreement.csv, "
        f"scores_{{crop,farm}}.csv, confusion_{{crop,farm}}.csv, models.csv in {args.output}"
    )


if __name__ == "__main__":
    main()
