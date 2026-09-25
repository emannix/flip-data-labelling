"""The Victorian PIC register against our farm labels, and the models against both.

A PIC (property identification code) record says what stock a property is registered
to run. Every VIC farm the builder cut carries the PIC records that matched its polygon
in its `farm-*_metadata.json`, so the register can be read as one more classifier and
scored against our crop-by-crop relabelling on exactly the farms and truth the model
evaluation uses. That answers two questions: how good is the register as a farm-type
label, and how much do the deep-learning pipeline and its SAM3 post-classifier gain
(or lose) against it.

*Scope.* Only the relabelled VIC reaches, i.e. generalisation's Balliang and Wyuna, the
242 farms in the post-classifier's `scores_farm.csv`. The other VIC workbooks are not
labelled yet; `generalisation_extra`'s VIC farm jsons carry no PIC records at all
(empty `farm_meta`), so when those labels land they will need joining to the register
first.

*What the register says.* `build_labelled_farms` in the builder's
`building_shapefile_datasets.py` joins each case-study label polygon (`balliang_labels.shp`,
`wyuna_labels.shp`, ...) to the property (PFI) holding its centroid, and separately to the
2019 register layer (`Vic_Active_PICs_Producers_20191111`): every register polygon holding
the label's centroid, or failing that the nearest within 100 m (`join_farms`), lands in the
farm json's `farm_meta`, one entry per PIC, its stock counts in `<species>_left`. When
several label polygons fall on one property, the first is kept at the top level and the
rest go to `duplicate_entries`, each with its own `farm_meta`. The main reading, `pic`, is
the per-species max over every PIC matched from any of the farm's label rows, each PIC
counted once.

The label shapefiles also carry a stock value of their own (`sheep_1`, `beef`, `dairy`,
`pigs`, `poultry`, `horses_1`, `goats_1`; repeated in `farm_meta` as `<species>_right`),
which lands at the top level of the json and of each duplicate. It usually equals one of
the matched PICs; where a label's centroid sits in two PICs it tends to hold only one of
them. Its per-species max over the top level and duplicates is kept as `pic_labelshp`, a
check on the main reading.

*The register's labels come from the stock counts alone* (`CandType` is not used): a farm
runs a species when the register counts more than `--min-stock` (default 5) of it. Each
species is judged on its own, so one farm can run several; a coarse class (`cattle`,
`livestock`) is present when any of its species is. A farm with no value runs nothing.

*Models.* Farm scores come from the post-classifier's `scores_farm.csv`, plus the
2026-09-18 ComFe and ViT-L linear probe trained on the SAM3 relabel
(`view/flip_comfe_2026_09_18/`), which have no post-classifier pass and are read from
their seed runs here: seed-mean per crop, max over the farm's crops, as for the others.
Their test csv holds the same 1,146 crops, so they are scored on the same farms and truth.

*Classes.* Farm truth is the evaluation's `true|<class>`, parsed from the workbooks'
farm sheet. The register has no pig grade, so the three pig classes are compared as one
`pig`, and the models' pig score is the max over the three. `cattle` (beef or dairy) and
`livestock` (any of them) are added because the labellers' own beef/dairy confusions
and the register's mixed holdings both argue for a coarser view alongside the fine one.

*How the models are put on the register's footing.* The register makes hard calls; the
models make scores. Each model is scored three ways per class:

    AP          threshold-free, over the seed-mean ensemble score. The register's AP
                uses the stock count itself as the score.
    matched     the model flags as many farms as the register does, its top-k by score:
                a like-for-like comparison at the register's own budget.
    prevalence  the model flags as many farms as the truth has positives, the
                operating point the evaluation's confusions use.

Every metric comes with a 95 % farm bootstrap interval, and `pairs.csv` carries the
paired bootstrap difference model minus register, so a difference whose interval
clears zero is one the 242 farms can actually see.

*A caution on reading it.* The truth is what a labeller could see from the air, the
register is what the property declared. Stock that needs no structures (sheep, beef on
pasture) is often in the register and invisible in the imagery, so for those classes a
low register precision is at least as much a statement about the labels as about the
register. `disagreements.csv` lists every farm/class where the two part ways, with the
headline model's score, for review against the imagery.

Writes into `output_pic_comparison_2026_09_25/`: `farm_pic.csv` (per-farm register
features, truth and model scores; the table a register-aware post-classifier would
start from), `comparison.csv`, `pairs.csv`, `crosstab.csv`, `disagreements.csv`, and
`comparison.html`: per model, survey against pipeline AP and F1 (at the survey's budget)
with the paired difference and its interval, one table per model.

Usage:

    .venv/bin/python gen_pic_comparison.py
    .venv/bin/python gen_pic_comparison.py --models postcv_comfe_m comfe_m lin_m --bootstrap 2000
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import gen_evaluation_2026_09_04 as ev

BUILDER = Path("/home/mannixe/FLIP/flip-geoimage-dataset-builder")
BUILDS = [BUILDER / "original_new_2026_08_21_generalisation"]
SCORES = Path("output_post_classifier_2026_09_04/scores_farm.csv")
OUTPUT = Path("output_pic_comparison_2026_09_25")

# Runs scored here directly rather than through gen_post_classifier.py: the 2026-09-18
# pair, trained on the SAM3 relabel of original_master_2026_09_18. Their test csv is the
# same 1,146 crops on the same 242 farms as the 2026-09-04 one, so their farm scores are
# built the same way (seed-mean per crop, max over the farm's crops) and join on
# farm_uid against the same truth.
RUN_TEST_CSV = Path("original_master_2026_09_18/sam3_test_autocrop_gen_vic.csv")
RUNS = ev.VIEW / "flip_comfe_2026_09_18"
EXTRA_MODELS = [
    {
        "key": "comfe_s3",
        "label": "ComFe multilabel (2026-09-18, SAM3 relabel)",
        "detail": "DINOv2 ViT-L/14 w/ reg., trained on the SAM3 relabel of the 2026-09-18 master build",
        "match": "comfe_dinov2",
    },
    {
        "key": "lin_s3",
        "label": "DINOv2 ViT-L linear probe (2026-09-18, SAM3 relabel)",
        "detail": "DINOv2 ViT-L/14 w/ reg., trained on the SAM3 relabel of the 2026-09-18 master build",
        "match": "linear_finetune",
    },
]

# Only models never fitted on VIC labels are a fair comparison: the ComFe master and its
# fixed-rule SAM3 gate, and the 2026-09-18 SAM3-relabel pair. postcv_comfe_m, the SAM3
# logistic stacker, is cross-validated on this very hold-out and is carried as a
# reference, not a contender.
DEFAULT_MODELS = ["comfe_m", "gate_comfe_m", "comfe_s3", "lin_s3", "postcv_comfe_m"]

# The farm-level survey value's top-level key in the farm json, per register species.
TOP_LEVEL = {
    "sheep": "sheep_1",
    "beef": "beef",
    "dairy": "dairy",
    "pigs": "pigs",
    "poultry": "poultry",
    "horses": "horses_1",
    "goats": "goats_1",
    "alpaca": "alpaca_1",
    "buffalo": "buffalo_1",
}
# Register species -> our class. alpaca and buffalo have no class and are only carried.
SPECIES = {
    "sheep": "sheep",
    "beef": "beef",
    "dairy": "dairy",
    "pigs": "pig",
    "poultry": "poultry",
    "horses": "horse",
    "goats": "goats",
    "alpaca": None,
    "buffalo": None,
}
PIGS = ["backyardpig", "commercialpig", "freerangepig"]
# Compared classes, and the model/truth classes each one is the union of.
CLASSES = {
    "pig": PIGS,
    "beef": ["beef"],
    "dairy": ["dairy"],
    "sheep": ["sheep"],
    "horse": ["horse"],
    "poultry": ["poultry"],
    "goats": ["goats"],
    "cattle": ["beef", "dairy"],
    "livestock": PIGS + ["beef", "dairy", "sheep", "horse", "poultry", "goats"],
}
# The register's own classes behind each compared class.
PIC_PARTS = {
    "cattle": ["beef", "dairy"],
    "livestock": ["pig", "beef", "dairy", "sheep", "horse", "poultry", "goats"],
}
MIN_DRAW_SHARE = 0.9


def number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_pic(builds: list[Path], farms: set[str]) -> pd.DataFrame:
    """One row per farm: the max over its matched PICs, and the label shapefiles' own value, per species."""
    rows = []
    for build in builds:
        for path in sorted(build.glob("farm-*/farm-*_metadata.json")):
            meta = json.loads(path.read_text())
            if meta["farm_uid"] not in farms:
                continue
            surveys = [meta] + list(meta.get("duplicate_entries") or [])
            # Every label row on the farm, duplicates included, carries its own register
            # join; the same register polygon reached from two rows is counted once.
            records = {}
            for survey in surveys:
                fm = survey.get("farm_meta") or {}
                for k, v in fm.get("PicType", {}).items():
                    if v not in ("", None):
                        records.setdefault(fm.get("farm_index", {}).get(k, k), (fm, k))
            row = {
                "farm_uid": meta["farm_uid"],
                "pic_build": build.name,
                "pic_n_records": len(records),
                "pic_ids": ";".join(str(int(i)) if isinstance(i, float) else str(i) for i in sorted(records, key=str)),
                "pic_types": ";".join(sorted({fm["PicType"][k] for fm, k in records.values()})),
                "pic_addresses": " | ".join(sorted({str(fm.get("PrimPropAd", {}).get(k, "")) for fm, k in records.values()})),
            }
            row["pic_has_value"] = bool(records)
            row["pic_labelshp_has_value"] = any(s.get(k) not in ("", None) for s in surveys for k in TOP_LEVEL.values())
            row["pic_n_duplicates"] = len(surveys) - 1
            row["pic_candtype"] = ";".join(sorted({str(s.get("CandType", "")) for s in surveys} - {"", "None"}))
            for species in SPECIES:
                survey = [number(s.get(TOP_LEVEL[species])) for s in surveys]
                left = [number(fm.get(f"{species}_left", {}).get(k)) for fm, k in records.values()]
                row[f"pic|{species}"] = np.nanmax(left) if np.isfinite(left).any() else 0.0
                row[f"pic_labelshp|{species}"] = np.nanmax(survey) if np.isfinite(survey).any() else 0.0
            rows.append(row)
    pic = pd.DataFrame(rows)
    missing = farms - set(pic["farm_uid"])
    if missing:
        raise SystemExit(f"{len(missing)} scored farms have no metadata json, e.g. {sorted(missing)[:3]}")
    return pic


def run_scores(farms: pd.Series, test_csv: Path, runs: Path, wanted: list[str]) -> tuple[pd.DataFrame, dict]:
    """Farm scores for the EXTRA_MODELS in `wanted`, read straight off their seed runs.

    Returns `<key>|<class>` columns in `farms` order and each key's display name. A family
    with no finished seed is left out with a note, as the evaluation scripts do.
    """
    test = pd.read_csv(test_csv, low_memory=False)
    missing = set(farms) - set(test["farm_uid"])
    if missing:
        raise SystemExit(f"{len(missing)} scored farms are not in {test_csv}, e.g. {sorted(missing)[:3]}")
    columns, names = {"farm_uid": farms.to_numpy()}, {}
    for spec in EXTRA_MODELS:
        if spec["key"] not in wanted:
            continue
        model = ev.load_model(
            {**spec, "runs": runs, "classes": ev.MASTER_CLASSES, "space": "test"},
            None, 0, len(test), test["group_id"].to_numpy(),
        )
        if model["ensemble"] is None:
            print(f"{spec['key']}: no finished run under {runs} matching {spec['match']!r}, skipped")
            continue
        farm = ev.farm_scores(model["ensemble"], test["farm_uid"].to_numpy(), farms.to_numpy(), "max")
        for i, name in enumerate(ev.MASTER_CLASSES):
            columns[f"{spec['key']}|{name}"] = farm[:, i]
        names[spec["key"]] = f"{spec['label']}, {len(model['seeds'])} seeds"
    return pd.DataFrame(columns), names


def register_counts(pic: pd.DataFrame, reading: str, name: str) -> np.ndarray:
    """Per-species stock counts behind a compared class, one column per species."""
    parts = PIC_PARTS.get(name, [name])
    return np.column_stack(
        [pic[f"{reading}|{species}"].to_numpy() for species, ours in SPECIES.items() if ours in parts]
    )


def register_flags(pic: pd.DataFrame, reading: str, name: str, min_stock: float) -> np.ndarray:
    """The register's call for a compared class: any species behind it above min_stock."""
    return (register_counts(pic, reading, name) > min_stock).any(axis=1)


def register_score(pic: pd.DataFrame, reading: str, name: str) -> np.ndarray:
    """The register's ranking score for AP: total head of the species behind the class."""
    return register_counts(pic, reading, name).sum(axis=1)


def model_score(scores: pd.DataFrame, key: str, name: str) -> np.ndarray | None:
    columns = [f"{key}|{c}" for c in CLASSES[name] if f"{key}|{c}" in scores.columns]
    return scores[columns].max(axis=1).to_numpy() if columns else None


def truth_of(scores: pd.DataFrame, name: str) -> np.ndarray:
    columns = [f"true|{c}" for c in CLASSES[name] if f"true|{c}" in scores.columns]
    return scores[columns].astype(bool).any(axis=1).to_numpy()


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    flagged = np.zeros(len(score), dtype=bool)
    if k:
        flagged[np.argsort(-score, kind="stable")[:k]] = True
    return flagged


def prf(truth: np.ndarray, flagged: np.ndarray) -> tuple[float, float, float]:
    hits = (truth & flagged).sum()
    precision = hits / flagged.sum() if flagged.sum() else np.nan
    recall = hits / truth.sum() if truth.sum() else np.nan
    f1 = 2 * hits / (truth.sum() + flagged.sum()) if truth.sum() + flagged.sum() else np.nan
    return precision, recall, f1


def ap(truth: np.ndarray, score: np.ndarray) -> float:
    return average_precision_score(truth, score) if truth.any() and not truth.all() else np.nan


def interval(draws: np.ndarray) -> tuple[float, float]:
    usable = np.isfinite(draws)
    if not len(draws) or usable.mean() < MIN_DRAW_SHARE:
        return np.nan, np.nan
    low, high = np.percentile(draws[usable], [2.5, 97.5])
    return float(low), float(high)


def metrics(truth: np.ndarray, flagged: np.ndarray | None, score: np.ndarray | None) -> dict:
    """Point metrics for one predictor; flagged may be None for a threshold-free entry."""
    out = {"AP": ap(truth, score) if score is not None else np.nan}
    if flagged is not None:
        out["predicted"] = int(flagged.sum())
        out["true_positives"] = int((truth & flagged).sum())
        out["precision"], out["recall"], out["F1"] = prf(truth, flagged)
    return out


def predictors(scores: pd.DataFrame, pic: pd.DataFrame, models: list[str], name: str, min_stock: float) -> dict:
    """Every (predictor, operating point) for one class, as a function of a farm draw.

    Each value is `draw -> (flagged, score)`; model operating points re-rank inside the
    draw so the matched and prevalence budgets are those of the resample, not the full set.
    """
    truth = truth_of(scores, name)
    out = {}
    for reading in ["pic", "pic_labelshp"]:
        flag = register_flags(pic, reading, name, min_stock)
        count = register_score(pic, reading, name)
        out[(reading, "register")] = lambda d, f=flag, c=count: (f[d], c[d])
    pic_flag = register_flags(pic, "pic", name, min_stock)
    for key in models:
        score = model_score(scores, key, name)
        if score is None:
            continue
        out[(key, "matched")] = lambda d, s=score: (top_k(s[d], int(pic_flag[d].sum())), s[d])
        out[(key, "prevalence")] = lambda d, s=score: (top_k(s[d], int(truth[d].sum())), s[d])
    return out


def compare(scores: pd.DataFrame, pic: pd.DataFrame, models: list[str], reps: int, seed: int, min_stock: float):
    n = len(scores)
    draws = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    everyone = np.arange(n)
    rows, pairs = [], []
    for name in CLASSES:
        truth = truth_of(scores, name)
        entries = predictors(scores, pic, models, name, min_stock)
        point = {k: metrics(truth, *f(everyone)) for k, f in entries.items()}
        boot = {
            k: pd.DataFrame([metrics(truth[d], *f(d)) for d in draws]) for k, f in entries.items()
        }
        for (who, how), values in point.items():
            row = {"class": name, "farms": n, "positives": int(truth.sum()), "predictor": who, "operating_point": how}
            for metric, value in values.items():
                row[metric] = value
                if metric in ("AP", "precision", "recall", "F1"):
                    row[f"{metric}_lo"], row[f"{metric}_hi"] = interval(boot[(who, how)][metric].to_numpy())
            rows.append(row)
        register = boot[("pic", "register")]
        for (who, how), frame in boot.items():
            if who in ("pic", "pic_labelshp"):
                continue
            for metric in ["AP", "precision", "recall", "F1"]:
                if how == "prevalence" and metric == "AP":
                    continue  # AP does not depend on the operating point
                diff = frame[metric].to_numpy() - register[metric].to_numpy()
                low, high = interval(diff)
                pairs.append(
                    {
                        "class": name,
                        "model": who,
                        "operating_point": how,
                        "metric": metric,
                        "model_value": point[(who, how)][metric],
                        "register_value": point[("pic", "register")][metric],
                        "difference": point[(who, how)][metric] - point[("pic", "register")][metric],
                        "lo": low,
                        "hi": high,
                        "clear_of_zero": bool(np.isfinite(low) and (low > 0 or high < 0)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def crosstab(scores: pd.DataFrame, pic: pd.DataFrame, min_stock: float) -> pd.DataFrame:
    rows = []
    has_value = pic["pic_has_value"].to_numpy()
    for name in CLASSES:
        truth = truth_of(scores, name)
        flag = register_flags(pic, "pic", name, min_stock)
        rows.append(
            {
                "class": name,
                "labelled_and_registered": int((truth & flag).sum()),
                "labelled_not_registered": int((truth & ~flag & has_value).sum()),
                "labelled_no_pic_value": int((truth & ~has_value).sum()),
                "registered_not_labelled": int((~truth & flag).sum()),
                "neither": int((~truth & ~flag).sum()),
            }
        )
    return pd.DataFrame(rows)


def disagreements(scores: pd.DataFrame, pic: pd.DataFrame, headline: str, min_stock: float) -> pd.DataFrame:
    rows = []
    for name in CLASSES:
        if name == "livestock":
            continue
        truth = truth_of(scores, name)
        flag = register_flags(pic, "pic", name, min_stock)
        count = register_score(pic, "pic", name)
        score = model_score(scores, headline, name)
        for i in np.flatnonzero(truth != flag):
            rows.append(
                {
                    "class": name,
                    "farm_uid": scores["farm_uid"].iat[i],
                    "region": scores["region"].iat[i],
                    "PFI": scores["PFI"].iat[i],
                    "farm_labels": scores["farm_labels"].iat[i],
                    "labelled": bool(truth[i]),
                    "register_count": count[i],
                    "pic_has_value": pic["pic_has_value"].iat[i],
                    "pic_candtype": pic["pic_candtype"].iat[i],
                    "pic_n_duplicates": pic["pic_n_duplicates"].iat[i],
                    "pic_n_records": pic["pic_n_records"].iat[i],
                    "pic_types": pic["pic_types"].iat[i],
                    f"{headline}_score": score[i] if score is not None else np.nan,
                    f"{headline}_rank": int((score > score[i]).sum()) + 1 if score is not None else np.nan,
                    "pic_addresses": pic["pic_addresses"].iat[i],
                }
            )
    return pd.DataFrame(rows).sort_values(["class", "labelled", f"{headline}_rank"])


# Row order of the html tables: coarse classes first, then the fine ones by how many
# farms carry them; the sparse ones are kept but marked.
HTML_ORDER = ["livestock", "cattle", "dairy", "horse", "pig", "beef", "sheep", "poultry", "goats"]
HTML_NAMES = {"livestock": "any livestock"}
MIN_HTML_POSITIVES = 5


def model_names(scores_path: Path) -> dict[str, str]:
    models = scores_path.parent / "models.csv"
    if not models.exists():
        return {}
    return dict(pd.read_csv(models)[["key", "model"]].itertuples(index=False))


def html_report(table: pd.DataFrame, pairs: pd.DataFrame, models: list[str], names: dict[str, str],
                n_farms: int, n_pic: int, args) -> str:
    """Survey vs pipeline, AP and F1 at the survey's budget, one table per model."""
    positives = table.drop_duplicates("class").set_index("class")["positives"]
    matched = pairs[pairs["operating_point"] == "matched"].set_index(["model", "class", "metric"])

    def value(v: float) -> str:
        return "&ndash;" if pd.isna(v) else f"{v:.2f}"

    def diff(row) -> str:
        if pd.isna(row["difference"]):
            return "&ndash;"
        text = f"{row['difference']:+.2f}".replace("-", "&minus;")
        if pd.notna(row["lo"]):
            text += f" <span class=ci>[{row['lo']:+.2f}, {row['hi']:+.2f}]</span>".replace("-", "&minus;")
        return f"<strong>{text}</strong>" if row["clear_of_zero"] else text

    sections = []
    for key in models:
        rows = []
        for name in HTML_ORDER:
            if (key, name, "AP") not in matched.index:
                continue
            a, f = matched.loc[(key, name, "AP")], matched.loc[(key, name, "F1")]
            sparse = positives[name] < MIN_HTML_POSITIVES
            rows.append(
                f"<tr{' class=sparse' if sparse else ''}><th>{HTML_NAMES.get(name, name)}</th>"
                f"<td>{positives[name]}</td>"
                f"<td>{value(a['register_value'])}</td><td>{value(a['model_value'])}</td><td>{diff(a)}</td>"
                f"<td>{value(f['register_value'])}</td><td>{value(f['model_value'])}</td><td>{diff(f)}</td></tr>"
            )
        title = html.escape(names.get(key, key))
        sections.append(
            f"<h2>{title} <code>{html.escape(key)}</code></h2>"
            "<div class=scroll><table><thead><tr><th>class</th><th>labelled farms</th>"
            "<th>survey AP</th><th>pipeline AP</th><th>difference</th>"
            "<th>survey F1</th><th>pipeline F1</th><th>difference</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>PIC survey vs pipeline</title>
<style>
:root {{ --bg: #fbfaf8; --fg: #1f1e1c; --muted: #6b6760; --rule: #dedbd4; --accent: #2f5d8a; color-scheme: light; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #1b1a18; --fg: #ecebe7; --muted: #a09c94; --rule: #3a3833; --accent: #8fb6de; color-scheme: dark; }}
}}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 2rem 1rem; }}
main {{ max-width: 60rem; margin: 0 auto; }}
h1 {{ font-size: 1.4rem; margin: 0 0 .5rem; }}
h2 {{ font-size: 1.05rem; margin: 2rem 0 .5rem; }}
code {{ color: var(--muted); font-size: .85em; }}
p {{ color: var(--muted); max-width: 48rem; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: .4rem .6rem; border-bottom: 1px solid var(--rule); text-align: right; white-space: nowrap; }}
th:first-child {{ text-align: left; }}
thead th {{ font-weight: 600; color: var(--muted); font-size: .85rem; }}
td strong {{ color: var(--accent); }}
.ci {{ color: var(--muted); font-size: .85em; }}
tr.sparse {{ color: var(--muted); }}
</style></head><body><main>
<h1>PIC survey vs pipeline, against our farm labels</h1>
<p>{n_farms} relabelled VIC farms (Balliang, Wyuna); {n_pic} have at least one matched PIC.
The survey calls a species present when its largest count across the farm's matched PICs
is above {args.min_stock:g}; each species is judged on its own. AP ranks farms by head count
(survey) or score (pipeline). F1 is at the survey's budget: the pipeline flags as many farms
as the survey does. Differences are pipeline minus survey with a 95&nbsp;% paired farm
bootstrap interval ({args.bootstrap} draws); <strong>bold</strong> clears zero. Greyed rows
have fewer than {MIN_HTML_POSITIVES} labelled farms.</p>
{''.join(sections)}
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=SCORES, help="farm scores from gen_post_classifier.py")
    parser.add_argument("--builds", type=Path, nargs="+", default=BUILDS, help="builder directories holding the farm jsons")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model keys; the first is the headline")
    parser.add_argument("--min-stock", type=float, default=5,
                        help="the register calls a species present above this head count")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=Path, default=RUNS,
                        help="seed runs of the EXTRA_MODELS keys, scored here directly")
    parser.add_argument("--run-test-csv", type=Path, default=RUN_TEST_CSV,
                        help="the test csv those runs predicted on, in its row order")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, low_memory=False)
    extra, extra_names = run_scores(scores["farm_uid"], args.run_test_csv, args.runs, args.models)
    scores = scores.merge(extra, on="farm_uid", how="left", validate="one_to_one")
    pic = read_pic(args.builds, set(scores["farm_uid"]))
    pic = pic.set_index("farm_uid").loc[scores["farm_uid"]].reset_index()

    table, pairs = compare(scores, pic, args.models, args.bootstrap, args.seed, args.min_stock)
    args.output.mkdir(exist_ok=True)

    keep = ["region", "source", "farm_uid", "PFI", "n_crops", "farm_labels"]
    keep += [c for c in scores.columns if c.startswith("true|")]
    keep += [c for c in scores.columns if c.split("|")[0] in args.models]
    scores[keep].merge(pic, on="farm_uid").to_csv(args.output / "farm_pic.csv", index=False)
    table.to_csv(args.output / "comparison.csv", index=False)
    pairs.to_csv(args.output / "pairs.csv", index=False)
    crosstab(scores, pic, args.min_stock).to_csv(args.output / "crosstab.csv", index=False)
    disagreements(scores, pic, args.models[0], args.min_stock).to_csv(args.output / "disagreements.csv", index=False)
    (args.output / "comparison.html").write_text(
        html_report(table, pairs, args.models, {**model_names(args.scores), **extra_names}, len(scores),
                    int(pic["pic_has_value"].sum()), args)
    )

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"{len(scores)} farms, {int(pic['pic_has_value'].sum())} with a matched PIC, "
              f"{int(pic['pic_labelshp_has_value'].sum())} with a label-shapefile stock value")
        print(crosstab(scores, pic, args.min_stock).to_string(index=False))
        view = table[table["operating_point"].isin(["register", "matched"])]
        print(
            view.pivot_table(index="class", columns="predictor", values="F1", sort=False)
            .round(2)
            .to_string()
        )
    print(f"wrote {args.output}/")


if __name__ == "__main__":
    main()
