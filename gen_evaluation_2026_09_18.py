"""Score the 2026-09-18 master-build runs on the VIC hold-out: ComFe v1 against the linear probe.

Third of the dated successors to `gen_evaluation.py`. Every scoring and charting routine is
imported from `gen_evaluation_2026_09_04.py`, the paired macro and the run-state loader from
`gen_evaluation_2026_09_04_backbones.py`; what differs is the roster and the question.

*The question.* `original_master_2026_09_18/` adds 1,017 human-labelled NSW parcel crops
(`generalisation_extra`) to the 2026-09-04 master build, and its SAM3 relabel turns every
farm-labelled crop with no building detection into `paddock`. Three sweeps have been run on
it, each pairing a ComFe v1 head with a linear probe on the same frozen backbone:

    master_grouped/   DINOv2 ViT-L/14 +reg   plain csvs, farm labels pooled per group (the 09_04 recipe)
    (top level)       DINOv2 ViT-L/14 +reg   sam3_*.csv, per-image labels, no grouping
    wave7b/           DINOv2 ViT-B/14        plain csvs, grouped; the comfev2 wave 7b arms v1, v1_0904, linear
    wave7d/           DINOv2 ViT-B/14 +reg   the same arms on the +registers checkpoint

With the 2026-09-04 ViT-L pair re-read beside them, the page can hold two of three things
fixed and read the third:

    head      ComFe v1 against the linear probe, on each (backbone, data) in turn
    data      09_04 master -> 09_18 master -> 09_18 SAM3 relabel, on ViT-L +reg, both heads
    backbone  ViT-L +reg -> ViT-B no-reg -> ViT-B +reg, on the grouped 09_18 master, both heads

*The roster.* Keys name the head and the (backbone, data) cell. `l` is ViT-L/14 with
registers, `b` ViT-B/14 without, `breg` ViT-B/14 with; `m18` the grouped 09_18 master,
`s3` its SAM3 relabel.

    lin_l       linear probe   ViT-L +reg   09_04 master, grouped        the backbones page's lin_l, re-read
    comfe_l     ComFe v1       ViT-L +reg   09_04 master, grouped        the 09_04 page's comfe_m, re-read
    lin_m18     linear probe   ViT-L +reg   09_18 master, grouped
    comfe_m18   ComFe v1       ViT-L +reg   09_18 master, grouped
    lin_s3      linear probe   ViT-L +reg   09_18 SAM3 relabel, per image
    comfe_s3    ComFe v1       ViT-L +reg   09_18 SAM3 relabel, per image
    lin_b       linear probe   ViT-B        09_18 master, grouped        wave 7b `linear`
    comfe_b     ComFe v1       ViT-B        09_18 master, grouped        wave 7b `v1`, the paper's 36 + 36 head
    comfe_b04   ComFe v1       ViT-B        09_18 master, grouped        wave 7b `v1_0904`, the production 100 + 100 head
    lin_breg    linear probe   ViT-B +reg   09_18 master, grouped        wave 7d `linear`
    comfe_breg  ComFe v1       ViT-B +reg   09_18 master, grouped        wave 7d `v1`

The production ViT-L ComFe heads are the `v1_0904` settings (100 class + 100 background
prototypes, class contrast on, 5 image prototypes); the wave 7 `v1` arm is the ComFe paper's
head rule (3 per class, 36 + 36). Both are ComFe v1 - none of the comfev2 recipe arms (F1 to
F3, the block, the wave 8 readouts) are scored here; they sit in the same view folders and
the wave 7 / 8 results documents under `comfe-run-flip/plans/` carry their log mAP.

*Same truth, same subset.* `test_autocrop_gen_vic.csv` is the same 1,146 crops in every
build - the 2026-09-18 copies, plain and `sam3_`, are checked against the 2026-09-04 one on
`(ecw_stem, building_cluster, PFI)` and on every `binary_*` column at load time - and the
well-sampled class subset is still defined on the 2026-08-21 and 2026-09-04 training splits,
so the headline macro is directly comparable with the three earlier dashboards. The 2026-09-18
training counts are shown in their own table.

*One caution on the wave 7 numbers.* The wave 7 results document quotes macro mAP over all
twelve classes at the best validation checkpoint, per seed; this page's headline is the
seed-ensemble AP over the six well-sampled livestock classes, as on every earlier page. The
`MACRO (evaluable)` row's seed mean is the number closest to the log figure.

Writes the CSVs *and* the self-contained HTML dashboard into `output_eval_2026_09_18/`.

Usage:

    .venv/bin/python gen_evaluation_2026_09_18.py
    .venv/bin/python gen_evaluation_2026_09_18.py --bootstrap 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import gen_evaluation_2026_09_04 as base
import gen_evaluation_2026_09_04_backbones as sweep
from gen_evaluation_2026_09_04 import (
    ALL_CLASSES,
    EXAMPLES_CONFUSED,
    GEN_TEST_CSV,
    GEN_TRAIN_CSV,
    JOIN_KEY,
    LEVELS,
    LIGHTBOX,
    MACRO,
    MACRO_SAMPLED,
    MACRO_SHARED,
    MASTER_CLASSES,
    MASTER_TRAIN_CSV,
    MIN_POSITIVES,
    MIN_TRAIN_EXAMPLES,
    SHARED_CLASSES,
    TEST_CSV,
    VIEW,
    WIDE_INTERVAL,
    agreement_card,
    ap_chart,
    choose_examples,
    confusion,
    confusion_sets,
    confusion_table,
    crop_truth,
    decisions,
    escape,
    evaluate,
    farm_frame,
    farm_gallery,
    farm_scores,
    farm_truth,
    legend,
    macro_tiles,
    merge_model,
    merge_truth,
    merged_order,
    operating_table,
    pair_table,
    pr_panels,
    region_chart,
    region_name,
    region_summary,
    region_table,
    stylesheet,
    top_k_agreement,
    train_counts,
)
from gen_evaluation_2026_09_04_backbones import (
    STALE_HOURS,
    load_model,
    macro_pair_card,
    macro_pairs,
    metrics_table,
)

OUTPUT = Path("output_eval_2026_09_18")
MASTER_18 = Path("/home/mannixe/FLIP/flip-data-labelling/original_master_2026_09_18")

# The 2026-09-18 copies of the hold-out. Scoring and the gallery read the 2026-09-04 copy
# (`TEST_CSV`), whose imagery paths the gallery already resolves; these two are only checked
# against it, so a saved prediction in either file's row order is in TEST_CSV's row order.
TEST_CSV_18 = MASTER_18 / "test_autocrop_gen_vic.csv"
TEST_CSV_S3 = MASTER_18 / "sam3_test_autocrop_gen_vic.csv"
MASTER_18_TRAIN_CSV = MASTER_18 / "train_df.csv"
SAM3_18_TRAIN_CSV = MASTER_18 / "sam3_train_df.csv"

SWEEP_04 = VIEW / "flip_2026_09_04"
SWEEP_18 = VIEW / "flip_comfe_2026_09_18"

VITL = "DINOv2 ViT-L/14 w/ reg."
VITB = "DINOv2 ViT-B/14"
VITB_REG = "DINOv2 ViT-B/14 w/ reg."
GROUPED_04 = "09_04 master, grouped"
GROUPED_18 = "09_18 master, grouped"
SAM3_18 = "09_18 SAM3 relabel, per image"

# The eleven families. `head`, `backbone` and `data` are the three axes the page reads
# along; `runs` and `match` locate the seed runs as on the earlier pages. `match` carries
# the launch-batch hash wherever the view folder holds an abandoned earlier batch of the
# same name (the 2026-09-25 12:48 and 16:49 launches of the sam3 pair left no prediction).
MODELS = [
    {
        "key": "lin_l",
        "label": "Linear probe, ViT-L +reg, 09_04 master",
        "head": "linear",
        "backbone": VITL,
        "data": GROUPED_04,
        "wave": "2026-09-11 backbone sweep",
        "alias": "lin_l on the backbones page",
        "detail": "the backbones page's lin_l re-read: one linear layer on frozen facebook/dinov2-with-registers-large, trained on original_master_2026_09_04 with PFI group batches",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_04 / "other_backbones",
        "match": "dinov2_linear_finetune_vitlwr_1-5",
    },
    {
        "key": "comfe_l",
        "label": "ComFe v1, ViT-L +reg, 09_04 master",
        "head": "comfe",
        "backbone": VITL,
        "data": GROUPED_04,
        "wave": "2026-09-04 master sweep",
        "alias": "comfe_m on the 09_04 page, comfe_l on the backbones page",
        "detail": "the 2026-09-04 comfe_m re-read: the production head (100 + 100 prototypes, class contrast) on the same backbone, trained on original_master_2026_09_04 grouped",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_04,
        "match": "comfe_dinov2_vitlwr_master_1-5",
    },
    {
        "key": "lin_m18",
        "label": "Linear probe, ViT-L +reg, 09_18 master",
        "head": "linear",
        "backbone": VITL,
        "data": GROUPED_18,
        "wave": "2026-09-26 launch ef0d3",
        "alias": None,
        "detail": "lin_l's recipe on original_master_2026_09_18: the same farm-level labels pooled per group, plus the 1,017 generalisation_extra crops",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "master_grouped",
        "match": "ef0d3_flip_2026_09_18_dinov2_linear_finetune_vitlwr_master_1-5",
    },
    {
        "key": "comfe_m18",
        "label": "ComFe v1, ViT-L +reg, 09_18 master",
        "head": "comfe",
        "backbone": VITL,
        "data": GROUPED_18,
        "wave": "2026-09-26 launch 268df",
        "alias": None,
        "detail": "comfe_l's recipe on original_master_2026_09_18, grouped; isolates the new generalisation_extra data from the SAM3 relabel",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "master_grouped",
        "match": "268df_flip_2026_09_18_comfe_dinov2_vitlwr_master_1-5",
    },
    {
        "key": "lin_s3",
        "label": "Linear probe, ViT-L +reg, 09_18 SAM3 relabel",
        "head": "linear",
        "backbone": VITL,
        "data": SAM3_18,
        "wave": "2026-09-25 launch 7b60e",
        "alias": "lin_s3 on the PIC comparison",
        "detail": "trained on sam3_train_df.csv: farm-labelled crops with no SAM3 building detection relabelled paddock, every label per image, no grouping, ImbalanceUpSampler over rows",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18,
        "match": "7b60e_flip_2026_09_18_dinov2_linear_finetune_vitlwr_sam3_1-5",
    },
    {
        "key": "comfe_s3",
        "label": "ComFe v1, ViT-L +reg, 09_18 SAM3 relabel",
        "head": "comfe",
        "backbone": VITL,
        "data": SAM3_18,
        "wave": "2026-09-25 launch 3ec60",
        "alias": "comfe_s3 on the PIC comparison",
        "detail": "the production head on sam3_train_df.csv, per-image labels, no grouping",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18,
        "match": "3ec60_flip_2026_09_18_comfe_dinov2_vitlwr_sam3_1-5",
    },
    {
        "key": "lin_b",
        "label": "Linear probe, ViT-B, 09_18 master",
        "head": "linear",
        "backbone": VITB,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7b `linear`",
        "alias": None,
        "detail": "facebook/dinov2-base without registers, 86M parameters; the grouped 09_18 probe with the DINOv2 paper's 13-lr x avgpool x blocks head sweep, early stopping on val macro mAP",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "wave7b",
        "match": "flip_comfe_v2w7b_linear_1-5",
    },
    {
        "key": "comfe_b",
        "label": "ComFe v1, ViT-B, 09_18 master (paper head)",
        "head": "comfe",
        "backbone": VITB,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7b `v1`",
        "alias": None,
        "detail": "ComFe v1 settings on the v2 code, the paper's head rule: 3 class prototypes per class (36 + 36 background), class contrast off, 5 image prototypes",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "wave7b",
        "match": "flip_comfe_v2w7b_v1_1-5",
    },
    {
        "key": "comfe_b04",
        "label": "ComFe v1, ViT-B, 09_18 master (production head)",
        "head": "comfe",
        "backbone": VITB,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7b `v1_0904`",
        "alias": None,
        "detail": "the production head settings (100 + 100 prototypes, class contrast on) on the v2 code and ViT-B; comfe_m18 with the backbone swapped",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "wave7b",
        "match": "flip_comfe_v2w7b_v1_0904_1-5",
    },
    {
        "key": "lin_breg",
        "label": "Linear probe, ViT-B +reg, 09_18 master",
        "head": "linear",
        "backbone": VITB_REG,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7d `linear`",
        "alias": None,
        "detail": "lin_b on facebook/dinov2-with-registers-base, otherwise identical",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "wave7d",
        "match": "flip_comfe_v2w7d_linear_1-5",
    },
    {
        "key": "comfe_breg",
        "label": "ComFe v1, ViT-B +reg, 09_18 master (paper head)",
        "head": "comfe",
        "backbone": VITB_REG,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7d `v1`",
        "alias": None,
        "detail": "comfe_b on facebook/dinov2-with-registers-base, otherwise identical",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP_18 / "wave7d",
        "match": "flip_comfe_v2w7d_v1_1-5",
    },
]

# The named contrasts the page leads with, each a (left minus right) pair read off the
# paired macro. Grouped by the axis they vary; everything else in the pair is held fixed.
CONTRASTS = [
    ("Head: ComFe v1 minus the linear probe", [
        ("comfe_l", "lin_l", "ViT-L +reg, 09_04 master"),
        ("comfe_m18", "lin_m18", "ViT-L +reg, 09_18 master"),
        ("comfe_s3", "lin_s3", "ViT-L +reg, 09_18 SAM3 relabel"),
        ("comfe_b", "lin_b", "ViT-B, 09_18 master"),
        ("comfe_b04", "lin_b", "ViT-B, 09_18 master, production head"),
        ("comfe_breg", "lin_breg", "ViT-B +reg, 09_18 master"),
    ]),
    ("Data: what the 2026-09-18 build changed, ViT-L +reg", [
        ("comfe_m18", "comfe_l", "ComFe: + generalisation_extra, still grouped"),
        ("comfe_s3", "comfe_m18", "ComFe: + SAM3 relabel, per-image labels"),
        ("comfe_s3", "comfe_l", "ComFe: 09_18 SAM3 against 09_04"),
        ("lin_m18", "lin_l", "probe: + generalisation_extra, still grouped"),
        ("lin_s3", "lin_m18", "probe: + SAM3 relabel, per-image labels"),
        ("lin_s3", "lin_l", "probe: 09_18 SAM3 against 09_04"),
    ]),
    ("Backbone: on the grouped 09_18 master", [
        ("comfe_b04", "comfe_m18", "ComFe, production head: ViT-B against ViT-L +reg"),
        ("comfe_b", "comfe_b04", "ComFe on ViT-B: paper head against production head"),
        ("comfe_breg", "comfe_b", "ComFe on ViT-B: + registers"),
        ("lin_b", "lin_m18", "probe: ViT-B against ViT-L +reg"),
        ("lin_breg", "lin_b", "probe on ViT-B: + registers"),
    ]),
]

# The gallery is picked on the 2026-09-04 ComFe, as on both earlier pages, so the same
# farms come up and the new models are read on them.
REFERENCE_KEYS = ["comfe_l", "comfe_m18", "comfe_s3", "lin_l"]

# Colours: lin_l and comfe_l keep the backbones page's blue and green. Each (backbone, data)
# cell has a hue family and the two heads take its lighter and darker member where the
# palette allows, so a probe and its ComFe sit together in the charts.
SERIES = {
    "lin_l": ("#2a78d6", "#3987e5"),
    "comfe_l": ("#008300", "#2aa32a"),
    "lin_m18": ("#1baf7a", "#199e70"),
    "comfe_m18": ("#7a8a00", "#a3b520"),
    "lin_s3": ("#e87ba4", "#d55181"),
    "comfe_s3": ("#4a3aa7", "#9085e9"),
    "lin_b": ("#eb6834", "#d95926"),
    "comfe_b": ("#8a5a2b", "#b07a45"),
    "comfe_b04": ("#eda100", "#c98500"),
    "lin_breg": ("#e34948", "#e66767"),
    "comfe_breg": ("#5e6b7a", "#9aa7b5"),
}
base.SERIES = SERIES
sweep.SERIES = SERIES


# --------------------------------------------------------------------------------------
# page pieces particular to this roster
# --------------------------------------------------------------------------------------


def roster_card(models: list[dict]) -> str:
    """One card per family, naming the head, backbone and data cell it fills."""
    cards = []
    for model in models:
        n_ready = len(model["seeds"])
        states = []
        if model["n_pending"]:
            states.append(f'<span class="pending">{model["n_pending"]} run(s) still training</span>')
        if model["n_dead"]:
            states.append(
                f'<span class="pending">{model["n_dead"]} run(s) dead: nothing saved, '
                f"log quiet for over {STALE_HOURS} h</span>"
            )
        if not n_ready and not states:
            states.append('<span class="pending">no runs found</span>')
        epochs = [e for e in model["epochs"] if e is not None]
        hours = [h for h in model["hours"] if np.isfinite(h)]
        runs = f"{n_ready} seed run(s) scored, {len(model['classes'])} output classes"
        if epochs:
            runs += f"; saved at epoch {'/'.join(str(e) for e in epochs)}"
        if hours:
            runs += f"; {np.mean(hours):.1f} h per run to the start of testing"
        alias = f'<span class="model-key">{escape(model["alias"])}</span>' if model["alias"] else ""
        head = "ComFe v1" if model["head"] == "comfe" else "linear probe"
        cards.append(
            f'<div class="model"><p class="model-name">'
            f'<i class="key-line" style="background:var(--series-{model["key"]})"></i>'
            f'{escape(model["label"])}<span class="model-key">{escape(model["key"])}</span>{alias}</p>'
            f'<p class="model-note"><b>{escape(head)}</b> on <b>{escape(model["backbone"])}</b>, '
            f'<b>{escape(model["data"])}</b> &mdash; {escape(model["wave"])}. {escape(model["detail"])}</p>'
            f'<p class="model-runs">{runs} {" ".join(states)}</p></div>'
        )
    return f'<div class="roster">{"".join(cards)}</div>'


def contrast_lookup(macro: pd.DataFrame, label: str, level: str, b: str, a: str) -> dict | None:
    """The paired macro row for b minus a at one level, whichever way round it was stored."""
    rows = macro[(macro["macro"] == label) & (macro["level"] == level)]
    hit = rows[(rows["b"] == b) & (rows["a"] == a)]
    sign = 1.0
    if hit.empty:
        hit = rows[(rows["b"] == a) & (rows["a"] == b)]
        sign = -1.0
    if hit.empty:
        return None
    row = hit.iloc[0]
    low, high = row["delta_lo"], row["delta_hi"]
    if sign < 0 and pd.notna(low):
        low, high = -high, -low
    return {
        "delta": sign * float(row["delta_AP"]),
        "lo": low,
        "hi": high,
        "clear": bool(row["clear_of_zero"]),
        "AP_b": float(row["AP_b"] if sign > 0 else row["AP_a"]),
        "AP_a": float(row["AP_a"] if sign > 0 else row["AP_b"]),
    }


def seed_macro(level: str, truth: pd.DataFrame, models: list[dict], subsets: dict[str, list[str]]) -> pd.DataFrame:
    """Each seed run's macro AP over a class subset, and the mean and sd over seeds.

    The per-class table reports the seed-ensemble AP and, per class, the seed mean; the
    macro rows carry only the ensemble. The wave 7 results documents quote the mean of
    per-seed macro mAP, so this is the footing to read those figures on. A class is
    counted when it has both positives and negatives in the truth and the model emits it.
    """
    n = len(truth)
    rows = []
    for model in models:
        if model["ensemble"] is None:
            continue
        for label, classes in subsets.items():
            usable = [
                c for c in classes
                if 0 < truth[c].sum() < n and base.column_of(model, c) is not None
            ]
            if not usable:
                continue
            per_seed = [
                float(np.mean([
                    base.average_precision_score(truth[c].to_numpy(), base.scores_for(model, c, seed))
                    for c in usable
                ]))
                for seed in model["seeds"]
            ]
            rows.append({
                "level": level,
                "macro": label,
                "model": model["key"],
                "n_classes": len(usable),
                "n_seeds": len(per_seed),
                "AP_ensemble": float(np.mean([
                    base.average_precision_score(truth[c].to_numpy(), base.scores_for(model, c))
                    for c in usable
                ])),
                "AP_seed_mean": float(np.mean(per_seed)),
                "AP_seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan,
                "AP_seeds": " ".join(f"{v:.4f}" for v in per_seed),
            })
    return pd.DataFrame(rows)


def contrast_card(macro: pd.DataFrame, scored: list[dict], label: str) -> str:
    """The named contrasts, crop and farm side by side, read off the paired macro."""
    present = {m["key"] for m in scored}
    levels = [level for level in LEVELS if level in set(macro[macro["macro"] == label]["level"])]
    head = "".join(f"<th>{escape(level)}</th>" for level in levels)
    blocks = []
    for title, pairs in CONTRASTS:
        body = []
        for b, a, note in pairs:
            if b not in present or a not in present:
                continue
            cells = []
            for level in levels:
                entry = contrast_lookup(macro, label, level, b, a)
                if entry is None:
                    cells.append("<td>-</td>")
                    continue
                strength = min(abs(entry["delta"]) / 0.10, 1.0)
                pole = "up" if entry["delta"] >= 0 else "down"
                mark = ' <span class="clear">&#9679;</span>' if entry["clear"] else ""
                bounds = (
                    ""
                    if pd.isna(entry["lo"])
                    else f'<span class="muted">[{entry["lo"]:+.3f}, {entry["hi"]:+.3f}]</span>'
                )
                caption = (
                    f"{label}, {level}: {b} {entry['AP_b']:.3f} minus {a} {entry['AP_a']:.3f} "
                    f"= {entry['delta']:+.3f} AP"
                )
                cells.append(
                    f'<td class="delta-cell" style="--tint:{strength:.3f}" data-pole="{pole}" '
                    f'title="{escape(caption)}"><span class="delta-value">{entry["delta"]:+.3f}{mark}</span>'
                    f"<br>{bounds}</td>"
                )
            body.append(
                f'<tr><th scope="row"><span class="pairhead">'
                f'<i class="key-line" style="background:var(--series-{b})"></i>'
                f'&minus;<i class="key-line" style="background:var(--series-{a})"></i> '
                f"{escape(b)} &minus; {escape(a)}</span><br>"
                f'<span class="muted">{escape(note)}</span></th>{"".join(cells)}</tr>'
            )
        if body:
            blocks.append(
                f'<tr class="group-row"><th colspan="{len(levels) + 1}">{escape(title)}</th></tr>'
                + "".join(body)
            )
    if not blocks:
        return ""
    return (
        f'<div class="scroll"><table class="data pairs"><thead><tr><th>pair</th>{head}</tr></thead>'
        f"<tbody>{''.join(blocks)}</tbody></table></div>"
    )


def training_card(counts: dict[str, dict[str, int]], sizes: dict[str, int]) -> str:
    """Crops carrying each class in the three master training splits."""
    builds = list(counts)
    head = "".join(
        f"<th>{escape(build)}<br><span class=\"muted\">{sizes[build]:,} crops</span></th>"
        for build in builds
    )
    rows = []
    for name in ALL_CLASSES:
        values = [counts[build].get(name, 0) for build in builds]
        if not any(values):
            continue
        cells = "".join(f"<td>{value:,}</td>" for value in values)
        rows.append(f'<tr><th scope="row">{escape(name)}</th>{cells}</tr>')
    return (
        f'<div class="scroll"><table class="data"><thead><tr><th>class</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def build_page(
    metrics: pd.DataFrame,
    pairs: pd.DataFrame,
    macro: pd.DataFrame,
    regions: pd.DataFrame,
    agreement: pd.DataFrame,
    confusions: dict[str, dict[str, pd.DataFrame]],
    truths: dict[str, pd.DataFrame],
    models: list[dict],
    n_crops: int,
    n_farms: int,
    aggregation: str,
    denominators: dict[str, dict[str, pd.Series]],
    operating: pd.DataFrame,
    operating_merged: pd.DataFrame,
    farm_rule: str,
    gallery: str,
    reference: dict | None,
    train_sizes: dict[str, int],
    train_by_class: dict[str, dict[str, int]],
) -> str:
    scored = [m for m in models if m["ensemble"] is not None]
    waiting = [m for m in models if m["ensemble"] is None or m["n_pending"] or m["n_dead"]]
    if farm_rule == "prevalence":
        rule_note = (
            "A class is predicted for a farm when the farm is among the <em>k</em> "
            "highest-scoring farms for that class, with <em>k</em> the number of farms "
            "annotated with it &mdash; the operating point at prevalence, where predicted and "
            "annotated counts agree and precision equals recall. Rerun with "
            "<code>--farm-threshold 0.5</code> to see a fixed score cut instead."
        )
    else:
        rule_note = (
            f"A class is predicted for a farm when its score exceeds {escape(farm_rule)}, "
            f"whatever the class or model. The models&rsquo; scores are not on a common "
            f"scale, so read the operating-point table underneath before comparing panels."
        )
    merged_caption = "; ".join(
        f"{group} = {' + '.join(members)}" for group, members in base.CLASS_GROUPS.items()
    )
    ramp = "".join(f'<i style="background:var(--heat-{step})"></i>' for step in range(7))
    diag_key = '<div class="ramp diag-key"><i></i>annotated class is the top class</div>'

    banner = ""
    if waiting:
        parts = []
        for m in waiting:
            if m["n_pending"]:
                parts.append(f"{escape(m['label'])} ({m['n_pending']} run(s) still training)")
            elif m["n_dead"]:
                parts.append(f"{escape(m['label'])} ({m['n_dead']} dead run(s) left out)")
            else:
                parts.append(f"{escape(m['label'])} (nothing saved yet)")
        banner = (
            f'<div class="banner"><strong>Incomplete sweep.</strong> {"; ".join(parts)}. '
            f"Everything below is scored on the runs that have landed.</div>"
        )

    n_comfe = sum(m["head"] == "comfe" for m in scored)
    n_probe = sum(m["head"] == "linear" for m in scored)

    sections = [
        f"""
<header class="masthead">
  <p class="eyebrow">2026-09-18 master build &middot; VIC hold-out &middot;
  {escape(n_crops)} crops &middot; {escape(n_farms)} farms</p>
  <h1>ComFe v1 against the linear probe on the 2026-09-18 build</h1>
  <p class="standfirst">{n_comfe} ComFe v1 heads and {n_probe} linear probes, scored against
  the hand-relabelled crop labels of the VIC hold-out. The 2026-09-18 build adds the
  {escape(f"{train_sizes['09_18 master'] - train_sizes['09_04 master']:,}")} human-labelled
  NSW parcel crops of <code>generalisation_extra</code>, and its SAM3 relabel turns every
  farm-labelled crop with no detected building into <code>paddock</code> &mdash; the
  mechanism the 2026-09-04 page named for the master models losing paddock. Each sweep on it
  pairs a ComFe head with a linear probe on the same frozen backbone, so three things can be
  read with the other two held fixed: the <b>head</b> (ComFe against the probe, on each
  backbone and data recipe), the <b>data</b> (09_04 master &rarr; 09_18 master &rarr; 09_18
  SAM3 relabel, on ViT-L with registers), and the <b>backbone</b> (ViT-L with registers
  &rarr; ViT-B without &rarr; ViT-B with, on the grouped 09_18 master). The 2026-09-04 ViT-L
  pair is re-read from its own run directories, so its numbers repeat the earlier pages
  exactly. Average precision by class, as before; the headline is the well-sampled macro
  defined on the same training splits as every earlier page.</p>
</header>
""",
        banner,
        roster_card(models),
        f'<div class="tiles">{macro_tiles(metrics, models)}'
        f"{agreement_card(agreement, models)}</div>",
        f"""
<section id="contrasts">
  <h2>The three questions, paired</h2>
  <p class="lede">Each cell is the well-sampled macro AP of the left model minus the
  right model&rsquo;s, both computed on the <em>same</em> bootstrap resamples, so the
  difference carries its own 95&nbsp;% interval; a &#9679; marks an interval clear of zero.
  Within each block everything but the named thing is held fixed. The full all-pairs matrix
  is further down.</p>
  <div class="card">
    <div class="card-head"><h3>{escape(MACRO_SAMPLED)}</h3></div>
    {contrast_card(macro, scored, MACRO_SAMPLED)}
  </div>
  <div class="card">
    <div class="card-head"><h3>{escape(MACRO_SHARED)}</h3></div>
    <p class="lede">The same over all nine livestock classes, including the ones too thin to
    rank on. Wider intervals for the same reason.</p>
    {contrast_card(macro, scored, MACRO_SHARED)}
  </div>
</section>
""",
        f"""
<section id="training">
  <h2>What each build trained on</h2>
  <p class="lede">Crops carrying each class in the training split of the three builds the
  models here were fitted on. The 09_18 master adds <code>generalisation_extra</code>, which
  is where the extra <code>residential</code>, <code>paddock</code> and
  <code>other_industrial</code> come from; the SAM3 relabel then moves
  {escape(f"{train_by_class['09_18 SAM3']['paddock'] - train_by_class['09_18 master']['paddock']:,}")}
  farm-labelled crops to <code>paddock</code>, roughly halving every livestock class. The
  well-sampled subset used for the headline is still defined on the 2026-08-21 and 2026-09-04
  splits, so it is the same six classes as on every earlier page.</p>
  <div class="card">
    {training_card(train_by_class, train_sizes)}
  </div>
</section>
""",
        f"""
<section id="macro-pairs">
  <h2>Every pair, on the macro</h2>
  <p class="lede">The same paired macro for every pair of models, in roster order.</p>
  <div class="card">
    <div class="card-head"><h3>{escape(MACRO_SAMPLED)}</h3></div>
    {macro_pair_card(macro, scored, MACRO_SAMPLED)}
  </div>
</section>
""",
        f"""
<section>
  <h2>Average precision by class</h2>
  <p class="lede">A random ranker scores the prevalence, marked on each row &mdash; a bar that
  stops near the tick has found nothing. Whiskers are 95&nbsp;% bootstrap intervals over the
  evaluation units. Open circles are the individual seed runs, so the spread from the training
  seed sits next to the spread from the labels. The <code>paddock</code> row is where the SAM3
  relabel should show first.</p>
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
  <h2>Every pairwise difference, by class</h2>
  <p class="lede">Each cell is AP(left) &minus; AP(right) with both models scored on the
  <em>same</em> bootstrap resamples. A &#9679; marks an interval clear of zero &mdash; a change
  these {escape(n_crops)} crops can actually support. Everything else is within sampling
  noise, however large the point estimate looks.</p>
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
  <p class="lede">The VIC hold-out is two reaches with very different class mixes, so each
  carries its own prevalence baseline. Read the models against each other <em>within</em> a
  reach. A missing whisker means too few bootstrap draws kept every class in that reach.</p>
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
  <p class="lede">Precision against recall at every threshold, crop level, one panel per model.
  The thin curves are the individual seed runs behind each ensemble; the horizontal tick is
  the prevalence.</p>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level</h3>
      <div class="legend"><span><i class="key-seed"></i>thin line = one seed run</span>
      <span><i class="key-tick"></i>prevalence</span></div>
    </div>
    <div class="scroll">{pr_panels(truths["crop"], scored, "crop")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level, common classes merged &mdash; {merged_caption}</h3>
      <div class="legend"><span><i class="key-seed"></i>thin line = one seed run</span>
      <span><i class="key-tick"></i>prevalence</span></div>
    </div>
    <div class="scroll">{pr_panels(merge_truth(truths["crop"]), [merge_model(m) for m in scored], "crop", merged_order())}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Where the classes leak</h2>
  <p class="lede">The annotated class against the model's top-scoring class, shaded by row
  fraction. {escape(int(truths["crop"]["paddock"].sum()))} of the {escape(n_crops)} crops are
  paddock; the grouped models trained on crops where nearly every paddock carried its
  farm&rsquo;s livestock class, the SAM3-relabel models did not, so the paddock row is where
  the two recipes should part.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <div class="confusions">
      {"".join(confusion_table(confusions["crop"][m["key"]], m) for m in scored)}
    </div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; each annotated class against every class the model predicts</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <p class="lede">Farm labels are multi-label, so here the model&rsquo;s answer is a set
    too. {rule_note} The row count is the number of farms annotated with that class, so the
    diagonal&rsquo;s row fraction is that class&rsquo;s <em>recall</em>.</p>
    <div class="confusions">
      {"".join(confusion_table(confusions["farm"][m["key"]], m, denominators["farm"][m["key"]]) for m in scored)}
    </div>
    <div class="card-head"><h3>Operating points behind the farm-level panel</h3></div>
    {operating_table(operating, scored)}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level, common classes merged &mdash; {merged_caption}</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <div class="confusions">
      {"".join(confusion_table(confusions["farm_merged"][m["key"]], m, denominators["farm_merged"][m["key"]]) for m in scored)}
    </div>
    <div class="card-head"><h3>Operating points, merged classes</h3></div>
    {operating_table(operating_merged, scored, merged_order())}
  </div>
</section>
""",
        f"""
<section id="farms">
  <h2>Real farms from the hold-out</h2>
  <p class="lede">Every building crop of a farm, its hand annotation in bold, and each
  model&rsquo;s top-scoring class with its score. Farms are picked per annotated class to
  span {escape(reference["key"] if reference else "the reference model")}&rsquo;s confidence
  &mdash; its highest-, median- and lowest-scoring farm for that class &mdash; and because
  the reference is the same 2026-09-04 ComFe as on the earlier pages, these are the same
  farms. Each group then adds the {escape(EXAMPLES_CONFUSED)} farms <em>not</em> annotated
  with that class that the models together rank highest for it. Click any image to enlarge
  it; the arrow keys step through the gallery, Escape closes it, and a group heading
  collapses its group.</p>
  <p class="gallery-key"><span><b>&#10003;</b> top class is annotated</span>
  <span><b>&#10007;</b> top class is not annotated; the annotated class&rsquo;s own score
  follows</span><span><b>&#8709;</b> the model has no output for any annotated class, so its
  top class is forced</span><span><b>&middot;</b> no annotation to compare with</span></p>
  {gallery}
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
    <code>test_autocrop_gen_vic.csv</code>. The 2026-09-18 build's copy and its
    <code>sam3_</code> copy are the 2026-09-04 one row for row on
    <code>(ecw_stem, building_cluster, PFI)</code> and on every label column, checked at load
    time, so a prediction saved in any of their row orders is in the same order.
    {escape(n_crops)} crops over {escape(n_farms)} farms.</p>
    <p><strong>Three data recipes.</strong> <em>09_04 master, grouped</em>:
    <code>original_master_2026_09_04/train_df.csv</code>, farm-level labels pooled per
    <code>group_id</code> with PFI group batches. <em>09_18 master, grouped</em>: the same
    recipe on <code>original_master_2026_09_18/train_df.csv</code>, which adds the 1,017
    <code>generalisation_extra</code> parcel crops. <em>09_18 SAM3 relabel, per image</em>:
    <code>sam3_train_df.csv</code>, where every farm-labelled crop with no SAM3 building
    detection is <code>paddock</code>, with grouping dropped (<code>group_pool: null</code>,
    ImbalanceUpSampler over rows). The runs on the plain csvs save their prediction index as
    the test split's <code>group_id</code>, the SAM3 runs as the row number; the loader maps
    either back onto rows.</p>
    <p><strong>Two heads.</strong> Every linear probe is one linear layer on frozen features
    with sigmoid outputs and BCE, at most 60 epochs with early stopping on validation, four
    seeds; the ViT-B probes add the DINOv2 paper's head sweep (13 learning rates &times;
    average pooling &times; last blocks) and stop on validation macro mAP. Every ComFe is
    ComFe v1 &mdash; the original head, not the comfev2 recipe &mdash; 50 epochs, four
    seeds. The ViT-L ComFe heads are the production settings (100 class + 100 background
    prototypes, class contrast on, 5 image prototypes); the wave 7 <code>v1</code> arm is the
    ComFe paper's head rule (36 + 36, class contrast off) and <code>v1_0904</code> the
    production settings on the v2 code, so <code>comfe_b04</code> against
    <code>comfe_m18</code> is the backbone alone.</p>
    <p><strong>Three backbones.</strong> <code>facebook/dinov2-with-registers-large</code>
    (300M), <code>facebook/dinov2-base</code> (86M, no registers) and
    <code>facebook/dinov2-with-registers-base</code>. The wave 7 results document reads the
    no-registers ViT-B as the better FLIP backbone on log macro mAP over all twelve classes
    (v1 25.0 against 21.1 with registers; probe 22.0 against 20.3); this page scores the same
    saved predictions on the headline subset of every earlier page.</p>
    <p><strong>Two families are re-read.</strong> <code>lin_l</code> is the backbones page's
    <code>lin_l</code> and <code>comfe_l</code> the 2026-09-04 <code>comfe_m</code>, from the
    same run directories, so their numbers repeat those pages exactly. <code>comfe_s3</code>
    and <code>lin_s3</code> are the pair the PIC-register comparison of 2026-09-25 scored,
    read the same way. The sam3 pair's two earlier launches on 2026-09-25 (12:48 and 16:49)
    left no prediction and are matched out by batch hash rather than reported as dead.</p>
    <p><strong>Dead runs.</strong> A run with no saved prediction is <em>training</em> if
    its log or a tensorboard event file changed in the last {STALE_HOURS} hours and
    <em>dead</em> otherwise.</p>
    <p><strong>Class vocabularies.</strong> Every model emits the same twelve classes, read
    back off each run's own <code>.hydra/config.yaml</code> before scoring. The headline
    macro is restricted to the nine livestock classes and then to the well-sampled subset,
    defined on the 2026-08-21 and 2026-09-04 training splits, so it is the same six classes
    the earlier pages ranked on. <code>aqua</code> has no VIC positives.</p>
    <p><strong>The paired macro.</strong> For each pair of models, the macro AP is computed
    on every bootstrap resample &mdash; the same resamples as the per-class table &mdash;
    and differenced. A draw that loses every positive of one class is dropped for both
    models; the interval is withheld if fewer than half survive. Written to
    <code>macro_pairs.csv</code>.</p>
    <p><strong>Thin classes.</strong> Poultry rests on 3 crops, sheep on 5, commercialpig on
    8. Their APs are extremely unstable and the intervals say so; treat any pairwise
    difference on those rows as unreadable.</p>
    <p><strong>Uncertainty.</strong> Percentile bootstrap over the evaluation units, paired
    across models. An interval is withheld when fewer than half the draws were usable.</p>
  </div>
</section>
""",
    ]

    return (
        "<title>2026-09-18 evaluation</title>"
        f"<style>{stylesheet()}</style>"
        f'<div class="page">{"".join(sections)}</div>'
        + LIGHTBOX
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def same_rows(reference: pd.DataFrame, other: pd.DataFrame) -> bool:
    """True when `other` is `reference` row for row on the join key and every label."""
    labels = [c for c in reference.columns if c.startswith("binary_")]
    return len(reference) == len(other) and all(
        reference[k].astype(str).tolist() == other[k].astype(str).tolist()
        for k in JOIN_KEY + labels
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation",
        default="max",
        choices=["max", "mean", "top2"],
        help="how a farm's per-crop scores become one farm score (default max)",
    )
    parser.add_argument(
        "--farm-threshold",
        default="prevalence",
        help="how a farm's scores become predicted classes for the farm-level confusion: "
        "'prevalence' predicts each class for as many farms as are annotated with it "
        "(default), or a number predicts every class scoring above it",
    )
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--file", type=Path, default=None, help="where to write the dashboard")
    args = parser.parse_args()

    farm_rule = args.farm_threshold
    if farm_rule != "prevalence":
        try:
            float(farm_rule)
        except ValueError:
            raise SystemExit(f"--farm-threshold must be 'prevalence' or a number, not {farm_rule!r}")

    test = pd.read_csv(TEST_CSV, keep_default_na=False, low_memory=False)
    train = pd.read_csv(GEN_TRAIN_CSV, keep_default_na=False)
    master_train = pd.read_csv(MASTER_TRAIN_CSV, keep_default_na=False, low_memory=False)
    trained = train_counts(train)
    trained_master = train_counts(master_train)
    train_sizes = {"09_04 master": len(master_train)}
    train_by_class = {"09_04 master": trained_master}
    for label, path in (("09_18 master", MASTER_18_TRAIN_CSV), ("09_18 SAM3", SAM3_18_TRAIN_CSV)):
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        train_sizes[label] = len(frame)
        train_by_class[label] = train_counts(frame)
    print(f"{TEST_CSV.name}: {len(test)} crops, {test['farm_uid'].nunique()} farms")
    for label in train_sizes:
        counts = train_by_class[label]
        print(f"{label}: {train_sizes[label]} crops, per class: " + ", ".join(
            f"{name} {counts[name]}" for name in ALL_CLASSES if counts[name]
        ))

    # The same guard as the earlier pages, and then the two 2026-09-18 copies against the
    # 2026-09-04 one on the labels as well: the SAM3 relabel must not have touched them.
    if GEN_TEST_CSV.exists():
        gen_test = pd.read_csv(GEN_TEST_CSV, keep_default_na=False)
        same = len(gen_test) == len(test) and all(
            gen_test[k].astype(str).tolist() == test[k].astype(str).tolist() for k in JOIN_KEY
        )
        if not same:
            raise SystemExit(f"{TEST_CSV} is not {GEN_TEST_CSV} row for row")
        print(f"{TEST_CSV.name} matches {GEN_TEST_CSV.name} row for row")
    for path in (TEST_CSV_18, TEST_CSV_S3):
        other = pd.read_csv(path, keep_default_na=False, low_memory=False)
        if not same_rows(test, other):
            raise SystemExit(f"{path} is not {TEST_CSV} row for row on the key and labels")
        if not test["group_id"].astype(str).equals(other["group_id"].astype(str)):
            raise SystemExit(f"{path}: group_id differs from {TEST_CSV}")
        print(f"{path.parent.name}/{path.name} matches {TEST_CSV.name} row for row, labels included")

    models = []
    for spec in MODELS:
        model = load_model(spec, len(test), test["group_id"].to_numpy())
        note = f"{len(model['seeds'])} seed run(s)"
        if model["n_pending"]:
            note += f", {model['n_pending']} still training"
        if model["n_dead"]:
            note += f", {model['n_dead']} dead"
        if model["epochs"]:
            note += "; saved at epoch " + "/".join(str(e) for e in model["epochs"])
        print(f"  {model['key']:>10}: {note}")
        for name in model["pending_names"]:
            print(f"              pending: {name}")
        for name in model["dead_names"]:
            print(f"              dead: {name}")
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
    print(f"farm level: {len(farms)} farms, crop scores aggregated with {args.aggregation}")

    farm_models = []
    for model in models:
        if model["ensemble"] is None:
            farm_models.append({**model, "ensemble": None, "seeds": []})
            continue
        farm_models.append(
            {
                **model,
                "ensemble": farm_scores(model["ensemble"], keys, order, args.aggregation),
                "seeds": [farm_scores(s, keys, order, args.aggregation) for s in model["seeds"]],
            }
        )

    truths = {"crop": crop, "farm": farm}
    by_level = {"crop": models, "farm": farm_models}

    metrics, pairs, macro, regions, agreement, seeds = [], [], [], [], [], []
    for level in LEVELS:
        table, pair = evaluate(
            level, truths[level], by_level[level], args.bootstrap, args.seed, trained, trained_master
        )
        metrics.append(table)
        pairs.append(pair)
        well = [
            name for name in SHARED_CLASSES
            if table.loc[table["class"] == name, "well_sampled"].any()
        ]
        seeds.append(seed_macro(
            level, truths[level], by_level[level],
            {MACRO: ALL_CLASSES, MACRO_SHARED: SHARED_CLASSES, MACRO_SAMPLED: well},
        ))
        for label, subset in ((MACRO_SAMPLED, well), (MACRO_SHARED, SHARED_CLASSES)):
            macro.append(
                macro_pairs(
                    level, truths[level], by_level[level], subset, label, args.bootstrap, args.seed
                )
            )
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
    macro = pd.concat(macro, ignore_index=True)
    regions = pd.concat(regions, ignore_index=True)
    agreement = pd.concat(agreement, ignore_index=True)
    seeds = pd.concat(seeds, ignore_index=True)

    # Crop level stays against the single top class; farm level is set against set.
    confusions = {"crop": {}, "farm": {}, "farm_merged": {}}
    denominators = {"crop": {}, "farm": {}, "farm_merged": {}}
    operating, operating_merged = [], []
    farm_merged = merge_truth(farm)
    for m in models:
        if m["ensemble"] is not None:
            confusions["crop"][m["key"]] = confusion(crop, m, "no class marked")
    for m in farm_models:
        if m["ensemble"] is None:
            continue
        predicted, points = decisions(m, farm, farm_rule)
        operating.append(points)
        confusions["farm"][m["key"]], denominators["farm"][m["key"]] = confusion_sets(
            farm, m, predicted, "no class marked"
        )
        coarse = merge_model(m)
        predicted, points = decisions(coarse, farm_merged, farm_rule)
        operating_merged.append(points)
        confusions["farm_merged"][m["key"]], denominators["farm_merged"][m["key"]] = confusion_sets(
            farm_merged, coarse, predicted, "no class marked", merged_order()
        )
    operating = pd.concat(operating, ignore_index=True)
    operating.insert(0, "rule", farm_rule)
    operating_merged = pd.concat(operating_merged, ignore_index=True)
    operating_merged.insert(0, "rule", farm_rule)

    # ---- console summary -------------------------------------------------------------
    for level in LEVELS:
        title = f"{level} level - average precision per class"
        print(f"\n{title}\n" + "-" * len(title))
        header = f"{'class':<20}{'pos':>5}{'prev':>7}"
        for model in scored:
            header += f"{model['key']:>12}"
        print(header)
        for _, row in metrics[metrics["level"] == level].iterrows():
            positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
            prevalence = "" if pd.isna(row.get("prevalence")) else f"{row['prevalence']:.3f}"
            mark = "" if row.get("well_sampled", True) or str(row["class"]).startswith("MACRO") else " *"
            line = f"{str(row['class']) + mark:<20}{positives:>5}{prevalence:>7}"
            for model in scored:
                value = row.get(f"{model['key']}|AP")
                line += f"{'-':>12}" if pd.isna(value) else f"{value:>12.3f}"
            print(line)
        print(
            "prev = prevalence, the AP a random ranker scores.\n"
            f"* = fewer than {MIN_TRAIN_EXAMPLES} training crops (either split) or "
            f"{MIN_POSITIVES} test positives; scored but excluded from {MACRO_SAMPLED}."
        )
        for label in (MACRO, MACRO_SAMPLED):
            rows = seeds[(seeds["level"] == level) & (seeds["macro"] == label)]
            if rows.empty:
                continue
            print(f"{label}, mean ± sd of per-seed AP (the wave 7 log figures' footing):")
            print("  " + "  ".join(
                f"{row['model']} {row['AP_seed_mean']:.3f}±{row['AP_seed_sd']:.3f}"
                for _, row in rows.iterrows()
            ))

    title = f"{MACRO_SAMPLED} - the named contrasts"
    print(f"\n{title}\n" + "-" * len(title))
    present = {m["key"] for m in scored}
    for heading, contrasts in CONTRASTS:
        print(f"{heading}")
        for b, a, note in contrasts:
            if b not in present or a not in present:
                continue
            for level in LEVELS:
                entry = contrast_lookup(macro, MACRO_SAMPLED, level, b, a)
                if entry is None:
                    continue
                mark = " *" if entry["clear"] else ""
                bounds = "" if pd.isna(entry["lo"]) else f" [{entry['lo']:+.3f}, {entry['hi']:+.3f}]"
                print(
                    f"  {level:<5} {b + ' - ' + a:<24} {entry['delta']:+.3f}{bounds}{mark}"
                    f"   ({b} {entry['AP_b']:.3f}, {a} {entry['AP_a']:.3f})  {note}"
                )
    print("* = interval clear of zero")

    # ---- outputs ---------------------------------------------------------------------
    metrics.to_csv(args.output / "evaluation.csv", index=False)
    pairs.to_csv(args.output / "evaluation_pairs.csv", index=False)
    macro.to_csv(args.output / "macro_pairs.csv", index=False)
    seeds.to_csv(args.output / "seed_macro.csv", index=False)
    regions.to_csv(args.output / "evaluation_by_region.csv", index=False)
    agreement.to_csv(args.output / "agreement.csv", index=False)
    for level, tables in confusions.items():
        pd.concat(tables, names=["model", "annotated"]).to_csv(
            args.output / f"confusion_{level}.csv"
        )
    operating.to_csv(args.output / "operating_points_farm.csv", index=False)
    operating_merged.to_csv(args.output / "operating_points_farm_merged.csv", index=False)

    pd.DataFrame(
        [
            {
                "key": m["key"],
                "model": m["label"],
                "head": m["head"],
                "backbone": m["backbone"],
                "data": m["data"],
                "wave": m["wave"],
                "alias": m["alias"] or "",
                "detail": m["detail"],
                "classes": " ".join(m["classes"]),
                "seeds": len(m["seeds"]),
                "seed_labels": " ".join(m["seed_labels"]),
                "saved_epochs": " ".join("" if e is None else str(e) for e in m["epochs"]),
                "hours_per_run": np.nanmean(m["hours"]) if m["hours"] else np.nan,
                "pending": m["n_pending"],
                "dead": m["n_dead"],
                "aggregation": args.aggregation,
                "runs": str(m["runs"]),
                "match": m["match"],
            }
            for m in models
        ]
    ).to_csv(args.output / "models.csv", index=False)

    pd.DataFrame(
        [
            {"build": label, "crops": train_sizes[label], **{name: train_by_class[label].get(name, 0) for name in ALL_CLASSES}}
            for label in train_sizes
        ]
    ).to_csv(args.output / "training_counts.csv", index=False)

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

    # ---- farm gallery ------------------------------------------------------------------
    reference = next(
        (m for key in REFERENCE_KEYS for m in farm_models if m["key"] == key and m["ensemble"] is not None),
        None,
    )
    gallery, groups, reasons = "", [], {}
    if reference is not None:
        groups, reasons = choose_examples(farms, farm, reference, farm_models)
        gallery = farm_gallery(groups, reasons, farms, test, crop, models, farm_models)
        print(
            f"\ngallery: {len(reasons)} farms over {len(groups)} groups, "
            f"selected on {reference['key']}"
        )
    pd.DataFrame(
        [
            {
                "group": group["title"],
                "farm_uid": farms.at[index, "farm_uid"],
                "PFI": farms.at[index, "PFI"],
                "region": region_name(farms.at[index, "source"]),
                "n_crops": farms.at[index, "n_crops"],
                "farm_labels": farms.at[index, "farm_labels"],
                "reference": reference["key"] if reference else "",
                "reason": " | ".join(reasons[index]),
            }
            for group in groups
            for index in group["members"]
        ]
    ).to_csv(args.output / "examples.csv", index=False)

    page = build_page(
        metrics,
        pairs,
        macro,
        regions,
        agreement,
        confusions,
        truths,
        models,
        len(test),
        len(farms),
        args.aggregation,
        denominators,
        operating,
        operating_merged,
        farm_rule,
        gallery,
        reference,
        train_sizes,
        train_by_class,
    )
    target = args.file or args.output / "evaluation_dashboard.html"
    target.write_text(page)
    print(f"\nwrote {target} and the CSVs in {args.output}/")


if __name__ == "__main__":
    main()
