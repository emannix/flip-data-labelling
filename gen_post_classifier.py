"""Post-classifiers that fold the SAM3 structure detections into the 2026-09-04 scores.

Takes the `sam3_*.csv` files `gen_sam3_postprocess_relabel.py` writes into the master
build and the per-crop model scores `gen_evaluation_2026_09_04.py` writes into
`output_eval_2026_09_04/`, and derives three new models per base model without touching a
GPU. They are scored with the evaluation script's own machinery - the same VIC hold-out,
the same bootstrap, the same crop and farm levels - so every number is like for like with
the dashboard, and the paired differences against the base say whether SAM3 helped.

*What SAM3 knows.* Per crop: how many roofs, buildings, sheds, shelters and houses it
found and how strongly (the building gate), and how many water tanks, silos, vehicles,
cattle yards and ponds (class features that all but never fire on a paddock). The gate
alone separates paddock from building crops with AP well above 0.9 on the human-labelled
crops; the class features carry weaker but class-specific signal, silos and water tanks
for poultry, cattle yards for beef and sheep, vehicles for residential.

*The three derived models*, for each `--base` (default the master ComFe and the
linear-probe-gated cascade, the two the dashboard ranks highest):

    gate_<base>    no fitting. Livestock and other_industrial scores are the base's,
                   multiplied by SAM3's building score; paddock is one minus that score.
                   The SAM3 twin of the evaluation script's cascades, which gate on a
                   2026-08-21 model's paddock output instead.
    sam3_prior     SAM3 features only, one gradient-boosted classifier per class fitted
                   on the relabelled master training split (NSW and the training
                   collection, never VIC). Shared across bases; it asks what the
                   detections alone can say about a crop.
    post_<base>    a stacker: one logistic regression per class over the base's scores
                   for every class, the SAM3 features and the prior's scores. Fitted in
                   one of two ways, see below.
    xgb_<base>     the same stacker with gradient-boosted trees (xgboost) in place of
                   the logistic regression, so interactions between a model score and
                   a detection can be learned. `--stacker` picks one or both.

*Where the stacker is fitted.* Every saved prediction so far is on the test split, so by
default the stacker is fitted on the hold-out itself under farm-grouped, stratified
K-fold cross-validation and scored on its out-of-fold predictions (`--fit-on cv`). That
is honest - no crop is scored by a stacker that saw its farm - but it is in-region and
the folds are small. The cleaner fit is on the validation split, which is out of region
for VIC (`--fit-on val`): it needs per-crop scores on `val_df.csv` from the base's
checkpoints, which the training runs did not save. The dataset builder's
`*_predict_percrop.yaml` config produces exactly that when its `test_csv_file` is
`val_df.csv` (ungrouped, so the saved index is the csv's row order, which is the
`sam3_val_df.csv` row order too). Point `--split-runs` at the directory those runs land
in; `--fit-on auto` (the default) uses them for every base that has them and falls back
to cross-validation for the rest. `--fit-on both` reports both fits side by side.

Classes with fewer than `--min-positives` positives in the fitting data cannot be fitted
and pass the base's score through unchanged; `models.csv` records which.

Writes into `output_post_classifier_2026_09_04/`: `evaluation.csv`,
`evaluation_pairs.csv`, `evaluation_by_region.csv`, `agreement.csv`, `scores_crop.csv`,
`scores_farm.csv`, `models.csv` in the evaluation script's layouts, plus
`stacker_coefficients.csv` (which features each stacker leans on: standardised
coefficients for the logistic stacker, gain importances for xgboost) and
`stacker_fits.csv`.

Usage:

    .venv/bin/python gen_post_classifier.py
    .venv/bin/python gen_post_classifier.py --base comfe_m lin_m cascade_p --cv-repeats 5
    .venv/bin/python gen_post_classifier.py --fit-on val --split-runs /path/to/percrop/runs
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gen_evaluation_2026_09_04 as ev

# --------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------

MASTER = ev.MASTER
EVAL = ev.OUTPUT
OUTPUT = Path("output_post_classifier_2026_09_04")
PREFIX = "sam3_"

TEST_CSV = MASTER / f"{PREFIX}test_autocrop_gen_vic.csv"
TRAIN_CSV = MASTER / f"{PREFIX}train_df.csv"
VAL_CSV = MASTER / f"{PREFIX}val_df.csv"
SCORES_CSV = EVAL / "scores_crop.csv"
MODELS_CSV = EVAL / "models.csv"

# Where per-crop predictions on the other splits are looked for: one run directory per
# (checkpoint x split), each carrying its `.hydra/config.yaml` and `predictions/`. The
# base a run belongs to is read off its name with the evaluation script's `match`
# substrings, the split off its config's `test_csv_file`.
SPLIT_RUNS = ev.VIEW / "flip_2026_09_04_percrop"
MATCH = {m["key"]: m["match"] for m in ev.MODELS if m["match"]}
# A cascade has no checkpoints of its own; its val scores would have to be built from
# its parents', which is not done here, so it can only be fitted by cross-validation.

DEFAULT_BASES = ["comfe_m", "cascade_p"]
GATE_PROMPTS = ["roof", "building", "shed", "shelter", "house"]
PRIOR_KEY = "sam3_prior"
LIVESTOCK = [c for c in ev.MASTER_CLASSES if c not in ev.NEW_ONLY_CLASSES]
EPS = 1e-5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------
# SAM3 features
# --------------------------------------------------------------------------------------


def sam3_prompts(frame: pd.DataFrame) -> list[str]:
    return [
        c.removeprefix("sam3_").removesuffix("_count")
        for c in frame.columns
        if c.startswith("sam3_") and c.endswith("_count") and not c.startswith("sam3_gate_")
    ]


def sam3_design(frame: pd.DataFrame, prompts: list[str]) -> tuple[np.ndarray, list[str]]:
    """The SAM3 feature matrix for a frame of `sam3_*.csv` rows, and the column names.

    Counts and areas are logged, areas also expressed as a fraction of the crop so that
    crops of different sizes and resolutions compare. Rows without SAM3 get zeros and a
    flag; on the hold-out there are none.
    """
    number = lambda name: pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    available = frame["sam3_available"].astype(str).str.lower().eq("true").to_numpy()
    columns, names = [], []

    def add(name: str, values: np.ndarray) -> None:
        values = np.where(np.isfinite(values), values, 0.0)
        columns.append(np.where(available, values, 0.0))
        names.append(name)

    for prompt in prompts:
        add(f"{prompt}:log_count", np.log1p(number(f"sam3_{prompt}_count")))
        add(f"{prompt}:max_score", number(f"sam3_{prompt}_max_score"))
        add(f"{prompt}:log_area", np.log1p(number(f"sam3_{prompt}_total_area_m2")))
        add(f"{prompt}:area_frac", number(f"sam3_{prompt}_area_frac"))
    add("gate:score", number("sam3_gate_score"))
    add("gate:log_count", np.log1p(number("sam3_gate_count")))
    add("gate:area_frac", number("sam3_gate_area_frac"))
    add("crop:log_res_m", np.log(np.maximum(number("sam3_res_x_m"), 1e-3)))
    add("crop:log_area_m2", np.log(np.maximum(number("sam3_crop_area_m2"), 1.0)))
    columns.append(available.astype(float))
    names.append("sam3_available")
    return np.column_stack(columns), names


def building_score(frame: pd.DataFrame, floor: float) -> np.ndarray:
    """SAM3's belief that the crop holds a building, in [floor, 1]."""
    score = pd.to_numeric(frame["sam3_gate_score"], errors="coerce").fillna(0.0).to_numpy()
    return floor + (1.0 - floor) * np.clip(score, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# base models, from the evaluation's saved scores
# --------------------------------------------------------------------------------------


def load_bases(scores: pd.DataFrame, roster: pd.DataFrame) -> list[dict]:
    """Every model the evaluation scored, rebuilt from `scores_crop.csv` and `models.csv`."""
    models = []
    for slot, row in enumerate(roster.itertuples(index=False), start=1):
        classes = str(row.classes).split()
        columns = [f"{row.key}|{c}" for c in classes]
        if not all(c in scores.columns for c in columns):
            print(f"  {row.key}: no saved scores, skipped")
            continue
        ensemble = scores[columns].to_numpy(dtype=float)
        seeds, labels = [], []
        for label in dict.fromkeys(str(row.seed_labels).split()):
            seed_columns = [f"{row.key}#{label}|{c}" for c in classes]
            if all(c in scores.columns for c in seed_columns):
                seeds.append(scores[seed_columns].to_numpy(dtype=float))
                labels.append(label)
        models.append(
            {
                "key": row.key,
                "slot": slot,
                "label": row.model,
                "detail": row.detail,
                "classes": classes,
                "space": "test",
                "runs": row.runs,
                "match": MATCH.get(row.key),
                "seeds": seeds or [ensemble],
                "seed_labels": labels or ["ensemble"],
                "ensemble": ensemble,
                "n_found": len(seeds),
                "n_pending": int(row.pending),
                "pending_names": [],
                "fit": "",
                "passthrough": "",
            }
        )
    return models


def run_split(run_dir: Path) -> str | None:
    """Which split a saved run scored, off its config's `test_csv_file`."""
    config = run_dir / ".hydra" / "config.yaml"
    if not config.exists():
        return None
    found = re.findall(r"^\s+test_csv_file:\s*'?([^\s']+)'?\s*$", config.read_text(), re.MULTILINE)
    return found[-1] if found else None


def run_grouped(run_dir: Path) -> bool:
    """True if the run pooled its scores over groups, which per-crop fitting cannot use."""
    config = run_dir / ".hydra" / "config.yaml"
    text = config.read_text() if config.exists() else ""
    found = re.findall(r"^\s+group_pool:\s*(\S+)\s*$", text, re.MULTILINE)
    return bool(found) and found[-1].lower() not in ("null", "none", "~")


def load_split_scores(
    base: dict, runs_dir: Path, split_csv: str, frame: pd.DataFrame
) -> dict | None:
    """Per-crop scores of `base` on another split, or None if no run has produced them."""
    if not base.get("match") or not runs_dir.is_dir():
        return None
    candidates = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and base["match"] in d.name and run_split(d) == split_csv
    )
    ready = [d for d in candidates if ev.predictions_dir(d)]
    if not ready:
        return None
    seeds, labels = [], []
    for run_dir in ready:
        if run_grouped(run_dir):
            print(f"    {run_dir.name}: grouped scores, skipped (needs group_pool: null)")
            continue
        classes = ev.config_classes(run_dir)
        if classes and classes != base["classes"]:
            raise SystemExit(f"{run_dir.name}: class order {classes} != {base['classes']}")
        seeds.append(ev.load_run(run_dir, len(frame), base["classes"], frame["group_id"].to_numpy()))
        labels.append(ev.run_seed(run_dir))
    if not seeds:
        return None
    return {"seeds": seeds, "seed_labels": labels, "ensemble": sum(seeds) / len(seeds)}


# --------------------------------------------------------------------------------------
# derived models
# --------------------------------------------------------------------------------------


def derived(base: dict, key: str, label: str, detail: str, slot: int, **extra) -> dict:
    return {
        **base,
        "key": key,
        "slot": slot,
        "label": label,
        "detail": detail,
        "runs": f"derived from {base['key']}",
        "match": None,
        "n_pending": 0,
        "pending_names": [],
        **extra,
    }


def gate_model(base: dict, building: np.ndarray, slot: int) -> dict:
    """The base gated by SAM3: livestock x building, paddock = 1 - building."""

    def combine(matrix: np.ndarray) -> np.ndarray:
        out = matrix.copy()
        for j, name in enumerate(base["classes"]):
            if name == "paddock":
                out[:, j] = 1.0 - building
            elif name != "other_industrial":
                out[:, j] = matrix[:, j] * building
        return out

    return derived(
        base,
        f"gate_{base['key']}",
        f"SAM3 gate on {base['label']}",
        f"{base['key']} livestock x SAM3 building score; paddock = 1 - building; "
        f"other_industrial from {base['key']}",
        slot,
        seeds=[combine(s) for s in base["seeds"]],
        seed_labels=list(base["seed_labels"]),
        ensemble=combine(base["ensemble"]),
        n_found=len(base["seeds"]),
    )


def truth_matrix(frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return np.column_stack(
        [ev.flags(frame[f"binary_{c}"]) if f"binary_{c}" in frame.columns else np.zeros(len(frame), bool)
         for c in classes]
    )


def fit_prior(
    train: pd.DataFrame, targets: list[pd.DataFrame], prompts: list[str], min_positives: int, seed: int
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """One boosted classifier per class on SAM3 features alone, applied to each target frame.

    Fitted on the training split only, so its scores on val and test are both out of
    sample and the val ones can feed a stacker without leaking.
    """
    rows = train[train["sam3_available"].astype(str).str.lower() == "true"]
    x_train, _ = sam3_design(rows, prompts)
    y_train = truth_matrix(rows, ev.MASTER_CLASSES)
    designs = [sam3_design(t, prompts)[0] for t in targets]
    fitted, skipped = [], []
    outputs = [[] for _ in targets]
    for j, name in enumerate(ev.MASTER_CLASSES):
        positives = int(y_train[:, j].sum())
        if positives < min_positives or positives == len(y_train):
            skipped.append(f"{name} ({positives} positives)")
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ).fit(x_train, y_train[:, j])
        for output, design in zip(outputs, designs):
            output.append(model.predict_proba(design)[:, 1])
        fitted.append(name)
    return [np.column_stack(o) for o in outputs], fitted, skipped


def stack_features(
    base_scores: np.ndarray, sam3: np.ndarray, prior_scores: np.ndarray | None
) -> np.ndarray:
    parts = [logit(base_scores), sam3]
    if prior_scores is not None:
        parts.append(logit(prior_scores))
    return np.column_stack(parts)


def stack_names(base: dict, sam3_names: list[str], prior_classes: list[str] | None) -> list[str]:
    names = [f"{base['key']}:{c}" for c in base["classes"]] + sam3_names
    if prior_classes:
        names += [f"prior:{c}" for c in prior_classes]
    return names


STACK_C = 0.3
STACK_SEED = 0
STACKERS = {
    "logistic": ("post", "logistic stacker", "coef"),
    "xgboost": ("xgb", "xgboost stacker", "gain"),
}


def stacker(kind: str, positives: int, negatives: int) -> object:
    """A fresh per-class learner. Both weight the positives up, since AP is a ranking."""
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=STACK_C, class_weight="balanced", max_iter=5000),
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=max(negatives / max(positives, 1), 1.0),
            tree_method="hist",
            random_state=STACK_SEED,
            n_jobs=4,
            verbosity=0,
        )
    raise SystemExit(f"unknown stacker {kind!r}")


def fit_apply(
    kind: str, x_fit: np.ndarray, y_fit: np.ndarray, x_apply: np.ndarray, coefficients: list
) -> np.ndarray | None:
    """One class's stacker: fitted on (x_fit, y_fit), applied to x_apply. None if unfittable."""
    positives = int(y_fit.sum())
    if positives == 0 or positives == len(y_fit):
        return None
    model = stacker(kind, positives, len(y_fit) - positives).fit(x_fit, y_fit)
    if kind == "logistic":
        coefficients.append(model[-1].coef_[0])
    else:
        coefficients.append(model.feature_importances_)
    return model.predict_proba(x_apply)[:, 1]


def post_model_val(
    kind: str,
    base: dict,
    val_scores: dict,
    val_frame: pd.DataFrame,
    sam3_val: np.ndarray,
    sam3_test: np.ndarray,
    prior_val: np.ndarray | None,
    prior_test: np.ndarray | None,
    min_positives: int,
    slot: int,
    feature_names: list[str],
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """The stacker fitted on the validation split and applied to the hold-out."""
    y_val = truth_matrix(val_frame, base["classes"])
    coefficients: dict[str, list] = {c: [] for c in base["classes"]}
    passthrough = []

    def fit_one(val_matrix: np.ndarray, test_matrix: np.ndarray) -> np.ndarray:
        x_fit = stack_features(val_matrix, sam3_val, prior_val)
        x_apply = stack_features(test_matrix, sam3_test, prior_test)
        out = test_matrix.copy()
        for j, name in enumerate(base["classes"]):
            if int(y_val[:, j].sum()) < min_positives:
                if name not in passthrough:
                    passthrough.append(name)
                continue
            scores = fit_apply(kind, x_fit, y_val[:, j], x_apply, coefficients[name])
            if scores is not None:
                out[:, j] = scores
        return out

    prefix, what, measure = STACKERS[kind]
    ensemble = fit_one(val_scores["ensemble"], base["ensemble"])
    paired = len(val_scores["seeds"]) == len(base["seeds"])
    seeds = (
        [fit_one(v, t) for v, t in zip(val_scores["seeds"], base["seeds"])] if paired else [ensemble]
    )
    labels = (
        [f"{v}x{t}" for v, t in zip(val_scores["seed_labels"], base["seed_labels"])]
        if paired else ["ensemble"]
    )
    model = derived(
        base,
        f"{prefix}_{base['key']}",
        f"SAM3 {what} on {base['label']}, fitted on val",
        f"{what} per class over {base['key']} scores, SAM3 features and the "
        f"SAM3 prior; fitted on {VAL_CSV.name} ({len(val_frame):,} crops), applied to VIC",
        slot,
        seeds=seeds,
        seed_labels=labels,
        ensemble=ensemble,
        n_found=len(seeds),
        fit="val",
        passthrough=" ".join(passthrough),
    )
    fits = [
        {"model": model["key"], "class": c, "stacker": kind, "fit": "val",
         "positives_fit": int(y_val[:, j].sum()), "fitted": c not in passthrough}
        for j, c in enumerate(base["classes"])
    ]
    return model, coefficient_table(model["key"], coefficients, feature_names, measure), fits


def post_model_cv(
    kind: str,
    base: dict,
    test_frame: pd.DataFrame,
    truth: pd.DataFrame,
    sam3_test: np.ndarray,
    prior_test: np.ndarray | None,
    folds: int,
    repeats: int,
    min_positives: int,
    seed: int,
    slot: int,
    feature_names: list[str],
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """The stacker under farm-grouped stratified K-fold on the hold-out, out of fold.

    Folds are drawn once per repeat on the crop's first class so the rare classes are
    spread across folds, and grouped on `farm_uid` so no stacker is scored on a crop
    whose farm it was fitted on. Each repeat is a seed; the ensemble is their mean.
    """
    y = np.column_stack([truth[c].to_numpy() for c in base["classes"]])
    groups = test_frame["farm_uid"].to_numpy()
    strata = test_frame["processed_class"].astype(str).to_numpy()
    x_all = stack_features(base["ensemble"], sam3_test, prior_test)
    coefficients: dict[str, list] = {c: [] for c in base["classes"]}
    passthrough = [c for j, c in enumerate(base["classes"]) if int(y[:, j].sum()) < min_positives]

    seeds = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # classes with fewer members than folds
        for repeat in range(repeats):
            out = base["ensemble"].copy()
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            for fit_idx, apply_idx in splitter.split(x_all, strata, groups):
                for j, name in enumerate(base["classes"]):
                    if name in passthrough:
                        continue
                    scores = fit_apply(
                        kind, x_all[fit_idx], y[fit_idx, j], x_all[apply_idx], coefficients[name]
                    )
                    if scores is not None:
                        out[apply_idx, j] = scores
            seeds.append(out)

    prefix, what, measure = STACKERS[kind]
    model = derived(
        base,
        f"{prefix}cv_{base['key']}",
        f"SAM3 {what} on {base['label']}, cross-validated",
        f"{what} per class over {base['key']} scores, SAM3 features and the "
        f"SAM3 prior; {folds}-fold farm-grouped CV on the hold-out x {repeats} repeats, "
        f"scored out of fold",
        slot,
        seeds=seeds,
        seed_labels=[f"cv{r}" for r in range(repeats)],
        ensemble=sum(seeds) / len(seeds),
        n_found=len(seeds),
        fit="cv",
        passthrough=" ".join(passthrough),
    )
    fits = [
        {"model": model["key"], "class": c, "stacker": kind, "fit": "cv",
         "positives_fit": int(y[:, j].sum()), "fitted": c not in passthrough}
        for j, c in enumerate(base["classes"])
    ]
    return model, coefficient_table(model["key"], coefficients, feature_names, measure), fits


def coefficient_table(
    key: str, coefficients: dict[str, list], names: list[str], measure: str
) -> pd.DataFrame:
    """Per class, what the stacker leans on: `measure` says whether `coef` is a
    standardised logistic coefficient (signed) or an xgboost gain importance (unsigned)."""
    rows = []
    for name, fits in coefficients.items():
        if not fits:
            continue
        mean = np.mean(fits, axis=0)
        spread = np.std(fits, axis=0) if len(fits) > 1 else np.zeros_like(mean)
        for feature, value, sd in zip(names, mean, spread):
            rows.append(
                {"model": key, "class": name, "feature": feature, "measure": measure,
                 "coef": float(value), "coef_sd": float(sd), "n_fits": len(fits)}
            )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["rank"] = (
            table["coef"].abs().groupby([table["model"], table["class"]]).rank(ascending=False, method="first").astype(int)
        )
    return table


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def score_columns(models: list[dict]) -> dict[str, np.ndarray]:
    columns = {}
    for model in models:
        if model["ensemble"] is None:
            continue
        variants = [(model["key"], model["ensemble"])]
        variants += [(f"{model['key']}#{l}", a) for l, a in zip(model["seed_labels"], model["seeds"])]
        for prefix, array in variants:
            for name in model["classes"]:
                columns[f"{prefix}|{name}"] = array[:, model["classes"].index(name)]
    return columns


def print_table(metrics: pd.DataFrame, pairs: pd.DataFrame, level: str, base: dict, family: list[dict]) -> None:
    """Per class: the base, its derived models, and the paired post - base difference."""
    keys = [base["key"]] + [m["key"] for m in family]
    stacked = [m["key"] for m in family if m.get("fit")]
    title = f"{level} level - AP, {base['key']} and its SAM3 derivatives"
    print(f"\n{title}\n" + "-" * len(title))
    header = f"{'class':<20}{'pos':>5}" + "".join(f"{k[:16]:>18}" for k in keys)
    header += "".join(f"{k.split('_')[0] + ' - base':>26}" for k in stacked)
    print(header)
    for _, row in metrics[metrics["level"] == level].iterrows():
        positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
        mark = "" if row.get("well_sampled", True) or str(row["class"]).startswith("MACRO") else " *"
        line = f"{str(row['class']) + mark:<20}{positives:>5}"
        for key in keys:
            value = row.get(f"{key}|AP")
            line += f"{'-' if pd.isna(value) else f'{value:.3f}':>18}"
        for key in stacked:
            delta = ""
            hit = pairs[(pairs["level"] == level) & (pairs["class"] == row["class"])
                        & (pairs["a"] == base["key"]) & (pairs["b"] == key)]
            if not hit.empty:
                d = hit.iloc[0]
                flag = " +" if d["clear_of_zero"] and d["delta_AP"] > 0 else (" -" if d["clear_of_zero"] else "")
                bounds = "" if pd.isna(d["delta_lo"]) else f" [{d['delta_lo']:+.2f},{d['delta_hi']:+.2f}]"
                delta = f"{d['delta_AP']:+.3f}{bounds}{flag}"
            line += f"{delta:>26}"
        print(line)


def main() -> None:
    global STACK_C, STACK_SEED
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", nargs="+", default=DEFAULT_BASES, help="base model keys from models.csv")
    parser.add_argument("--fit-on", default="auto", choices=["auto", "val", "cv", "both"],
                        help="where the stacker is fitted (default auto: val where per-crop val "
                        "scores exist, else cross-validation on the hold-out)")
    parser.add_argument("--split-runs", type=Path, default=SPLIT_RUNS,
                        help="directory of per-crop prediction runs on the other splits")
    parser.add_argument("--stacker", default="both", choices=["logistic", "xgboost", "both"],
                        help="which learner the stacker uses (default both, reported side by side)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3, help="repeats of the CV fit, each a seed")
    parser.add_argument("--min-positives", type=int, default=5,
                        help="a class with fewer positives in the fitting data passes the base through")
    parser.add_argument("--C", type=float, default=STACK_C, help="stacker inverse regularisation")
    parser.add_argument("--gate-floor", type=float, default=0.05,
                        help="the building score a crop keeps when SAM3 finds nothing")
    parser.add_argument("--no-prior", action="store_true", help="skip the SAM3-only prior")
    parser.add_argument("--aggregation", default="max", choices=["max", "mean", "top2"])
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    STACK_C = args.C
    STACK_SEED = args.seed
    kinds = ["logistic", "xgboost"] if args.stacker == "both" else [args.stacker]

    for path in (TEST_CSV, TRAIN_CSV, VAL_CSV, SCORES_CSV, MODELS_CSV):
        if not path.exists():
            raise SystemExit(f"missing {path}: run gen_sam3_postprocess_relabel.py and "
                             f"gen_evaluation_2026_09_04.py first")

    # ---- inputs ------------------------------------------------------------------------
    scores = pd.read_csv(SCORES_CSV, keep_default_na=False, low_memory=False)
    roster = pd.read_csv(MODELS_CSV, keep_default_na=False)
    test = pd.read_csv(TEST_CSV, keep_default_na=False, low_memory=False)
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False, low_memory=False)
    val = pd.read_csv(VAL_CSV, keep_default_na=False, low_memory=False)
    if len(test) != len(scores) or not (test["image_path"].to_numpy() == scores["image_path"].to_numpy()).all():
        raise SystemExit(f"{TEST_CSV.name} and {SCORES_CSV} are not the same crops in the same order")
    prompts = sam3_prompts(test)
    print(f"{TEST_CSV.name}: {len(test)} crops, {test['farm_uid'].nunique()} farms, "
          f"SAM3 prompts {prompts}")
    for name, frame in (("train", train), ("val", val)):
        with_sam3 = int(frame["sam3_available"].astype(str).str.lower().eq("true").sum())
        relabelled = int(frame["sam3_relabel"].astype(str).eq("paddock").sum()) if "sam3_relabel" in frame else 0
        print(f"{PREFIX}{name}_df.csv: {len(frame):,} crops, {with_sam3:,} with SAM3, "
              f"{relabelled:,} relabelled to paddock")

    models = load_bases(scores, roster)
    by_key = {m["key"]: m for m in models}
    bases = []
    for key in args.base:
        if key not in by_key:
            raise SystemExit(f"--base {key}: not in {MODELS_CSV} or has no saved scores; "
                             f"have {sorted(by_key)}")
        bases.append(by_key[key])
    print("bases: " + ", ".join(f"{b['key']} ({len(b['seeds'])} seeds)" for b in bases))

    crop_truth = pd.DataFrame({c: ev.flags(scores[f"true|{c}"]) for c in ev.ALL_CLASSES})
    sam3_test, sam3_names = sam3_design(test, prompts)
    sam3_val, _ = sam3_design(val, prompts)
    building = building_score(test, args.gate_floor)

    # ---- the SAM3-only prior -----------------------------------------------------------
    prior_test = prior_val = None
    prior_classes = None
    slot = len(models) + 1
    if not args.no_prior:
        (prior_val, prior_test), fitted, skipped = fit_prior(
            train, [val, test], prompts, args.min_positives, args.seed
        )
        prior_classes = fitted
        print(f"{PRIOR_KEY}: fitted on {len(fitted)} classes"
              + (f", skipped {', '.join(skipped)}" if skipped else ""))
        models.append(
            {
                "key": PRIOR_KEY,
                "slot": slot,
                "label": "SAM3 prior (derived)",
                "detail": f"gradient-boosted classifier per class on SAM3 features alone, "
                          f"fitted on {TRAIN_CSV.name}",
                "classes": fitted,
                "space": "test",
                "runs": f"derived from {TRAIN_CSV.name}",
                "match": None,
                "seeds": [prior_test],
                "seed_labels": ["train"],
                "ensemble": prior_test,
                "n_found": 1,
                "n_pending": 0,
                "pending_names": [],
                "fit": "train",
                "passthrough": " ".join(s.split(" ")[0] for s in skipped),
            }
        )
        slot += 1

    # ---- per base: gate and stacker ----------------------------------------------------
    families: dict[str, list[dict]] = {}
    coefficients, fits = [], []
    for base in bases:
        family = [gate_model(base, building, slot)]
        slot += 1
        names = stack_names(base, sam3_names, prior_classes)

        val_scores = load_split_scores(base, args.split_runs, "val_df.csv", val)
        want_val = args.fit_on in ("val", "both") or (args.fit_on == "auto" and val_scores is not None)
        want_cv = args.fit_on in ("cv", "both") or (args.fit_on == "auto" and val_scores is None)
        if want_val and val_scores is None:
            print(f"  {base['key']}: no per-crop val scores under {args.split_runs}; "
                  f"run the *_predict_percrop config with test_csv_file: val_df.csv "
                  f"(group_pool: null) and copy the runs there. Falling back to CV.")
            want_val, want_cv = False, True
        for kind in kinds:
            if want_val:
                model, coef, fit = post_model_val(
                    kind, base, val_scores, val, sam3_val, sam3_test, prior_val, prior_test,
                    args.min_positives, slot, names,
                )
                family.append(model); coefficients.append(coef); fits += fit; slot += 1
                print(f"  {base['key']}: {kind} stacker fitted on val, "
                      f"{len(val_scores['seeds'])} val seed run(s)")
            if want_cv:
                model, coef, fit = post_model_cv(
                    kind, base, test, crop_truth, sam3_test, prior_test, args.folds,
                    args.cv_repeats, args.min_positives, args.seed, slot, names,
                )
                family.append(model); coefficients.append(coef); fits += fit; slot += 1
                print(f"  {base['key']}: {kind} stacker cross-validated on the hold-out, "
                      f"{args.folds} folds x {args.cv_repeats} repeats")
        for model in family:
            if model["passthrough"]:
                print(f"    {model['key']}: passing {base['key']} through for {model['passthrough']}")
        families[base["key"]] = family
        models += family

    # ---- scoring, exactly as the evaluation does ---------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)
    trained = ev.train_counts(pd.read_csv(ev.GEN_TRAIN_CSV, keep_default_na=False))
    trained_master = ev.train_counts(
        pd.read_csv(ev.MASTER_TRAIN_CSV, keep_default_na=False, low_memory=False)
    )
    crop_regions = scores["region"].to_numpy()

    farms = ev.farm_frame(test)
    farm_truth = ev.farm_truth(farms)
    farm_regions = farms["source"].map(ev.region_name).to_numpy()
    keys, order = test["farm_uid"].to_numpy(), farms["farm_uid"].to_numpy()
    farm_models = [
        {
            **m,
            "ensemble": ev.farm_scores(m["ensemble"], keys, order, args.aggregation),
            "seeds": [ev.farm_scores(s, keys, order, args.aggregation) for s in m["seeds"]],
        }
        for m in models
    ]
    truths = {"crop": crop_truth, "farm": farm_truth}
    by_level = {"crop": models, "farm": farm_models}

    metrics, pairs, regions, agreement = [], [], [], []
    for level in ev.LEVELS:
        table, pair = ev.evaluate(
            level, truths[level], by_level[level], args.bootstrap, args.seed, trained, trained_master
        )
        metrics.append(table)
        pairs.append(pair)
        regions.append(ev.region_summary(
            level, truths[level], crop_regions if level == "crop" else farm_regions,
            by_level[level], args.bootstrap, args.seed,
        ))
        found = ev.top_k_agreement(truths[level], by_level[level])
        found.insert(0, "level", level)
        agreement.append(found)
    metrics = pd.concat(metrics, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    regions = pd.concat(regions, ignore_index=True)
    agreement = pd.concat(agreement, ignore_index=True)

    for level in ev.LEVELS:
        for base in bases:
            print_table(metrics, pairs, level, base, families[base["key"]])
    print(
        "\n* = too few training crops or test positives to rank models on. "
        "post - base / xgb - base are the paired bootstrap differences of the logistic and "
        "xgboost stackers against the base at the ensemble; + / - marks an interval clear of zero."
    )

    # ---- outputs -----------------------------------------------------------------------
    metrics.to_csv(args.output / "evaluation.csv", index=False)
    pairs.to_csv(args.output / "evaluation_pairs.csv", index=False)
    regions.to_csv(args.output / "evaluation_by_region.csv", index=False)
    agreement.to_csv(args.output / "agreement.csv", index=False)
    pd.DataFrame(
        {
            **{c: scores[c].to_numpy() for c in ["region", "source", "farm_uid", "PFI", "image_path", "crop_label", "crop_classes"]},
            **{f"true|{c}": crop_truth[c].to_numpy() for c in ev.ALL_CLASSES},
            "sam3_gate_score": pd.to_numeric(test["sam3_gate_score"], errors="coerce").to_numpy(),
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
            **{f"true|{c}": farm_truth[c].to_numpy() for c in ev.ALL_CLASSES},
            **score_columns(farm_models),
        }
    ).to_csv(args.output / "scores_farm.csv", index=False)
    pd.DataFrame(
        [
            {
                "key": m["key"], "model": m["label"], "detail": m["detail"],
                "classes": " ".join(m["classes"]), "seeds": len(m["seeds"]),
                "seed_labels": " ".join(m["seed_labels"]), "pending": m["n_pending"],
                "aggregation": args.aggregation, "runs": str(m["runs"]),
                "fit": m.get("fit", ""), "passthrough": m.get("passthrough", ""),
            }
            for m in models
        ]
    ).to_csv(args.output / "models.csv", index=False)
    if coefficients:
        pd.concat(coefficients, ignore_index=True).to_csv(
            args.output / "stacker_coefficients.csv", index=False
        )
    pd.DataFrame(fits).to_csv(args.output / "stacker_fits.csv", index=False)
    print(f"\nwrote evaluation{{,_pairs,_by_region}}.csv, agreement.csv, scores_{{crop,farm}}.csv, "
          f"models.csv, stacker_coefficients.csv, stacker_fits.csv in {args.output}")


if __name__ == "__main__":
    main()
