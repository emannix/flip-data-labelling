"""Score the 2026-09-11 backbone sweep on the VIC hold-out, beside the 2026-09-04 master pair.

Companion to `gen_evaluation_2026_09_04.py`, from which every scoring and charting routine
is imported unchanged. What differs is the roster and the question.

*The question.* The 2026-09-04 page compared a DINOv2 linear probe with ComFe, both
trained on the master build, and ComFe won on livestock. But the two did not share a
backbone: the probe sat on DINOv2 ViT-S/14 (21M parameters, `dinov2_vits14` off torch
hub) while ComFe sat on DINOv2 ViT-L/14 with registers (300M). The head and the backbone
were confounded. The 2026-09-11 sweep under `other_backbones/` retrains the *same* linear
probe on four larger backbones, all on the master split with the same recipe (60 epochs,
early stopping on validation accuracy, cosine SGD), so the page can ask two things
separately: what the backbone is worth to a linear probe, and how much of ComFe's lead
survives once its probe rival sits on the same ViT-L features. A second ComFe, launched
2026-09-18 under `other_backbones_comfe/` on DINOv3 ViT-L/16, carries the best probe
backbone under the ComFe head, so the two heads can now be read on two backbones each
rather than one.

*The backbones.* Every model is a linear probe on frozen features unless it says ComFe.

    lin_s      DINOv2 ViT-S/14                      LVD-142M web images        the 2026-09-04 `lin_m`
    lin_l      DINOv2 ViT-L/14 with registers       LVD-142M web images        ComFe's backbone
    lin_v3     DINOv3 ViT-L/16                      LVD-1689M web images
    lin_sat    DINOv3 ViT-L/16, satellite weights   SAT-493M aerial/satellite
    lin_radio  C-RADIOv4 SO400M                     agglomerative distillation  relaunched 2026-09-18
    comfe_l    ComFe on DINOv2 ViT-L/14 w/ reg.     LVD-142M web images        the 2026-09-04 `comfe_m`
    comfe_v3   ComFe on DINOv3 ViT-L/16             LVD-1689M web images       launched 2026-09-18

`lin_s` and `comfe_l` are the 2026-09-04 runs re-read, so their numbers repeat the earlier
dashboard exactly; only the keys are renamed so that the key names the backbone. The
C-RADIOv4 runs of 2026-09-11 all stopped while fetching the checkpoint from Hugging Face
and left no prediction, checkpoint or error; they were relaunched on 2026-09-18 as batch
`384a1` and all four seeds have now saved a test prediction. `lin_radio`'s `match` names
that batch, so the four abandoned directories are not read at all. `comfe_v3` is the only
family launched fresh here: its four seeds started at 14:16 on 2026-09-18 and ComFe takes
about four hours a run, so the first pass over this script will report them as still
training and the page fills in on a rerun.

*Same truth, same subset.* `test_autocrop_gen_vic.csv`, 1,146 crops over the two VIC
reaches, and the same well-sampled class subset as the two earlier pages - a class must
clear the training bar in *both* the 2026-08-21 relabelled split and the master split,
even though every model here trained on the master split alone - so the headline macro
is directly comparable across the three dashboards.

*One addition.* The pairwise table on the earlier pages is per class only. A backbone
comparison wants a single paired number, so this page also bootstraps the well-sampled
macro itself: every model's per-class AP on the same resample, averaged over the
well-sampled classes, differenced pair by pair. Those intervals land in `macro_pairs.csv`
and in a card at the top of the page.

No cascades: they were a construction over two ComFe generations and have nothing to say
about backbones.

Writes the CSVs *and* the self-contained HTML dashboard into
`output_eval_2026_09_04_backbones/`.

Usage:

    .venv/bin/python gen_evaluation_2026_09_04_backbones.py
    .venv/bin/python gen_evaluation_2026_09_04_backbones.py --bootstrap 2000
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gen_evaluation_2026_09_04 as base
from gen_evaluation_2026_09_04 import (
    ALL_CLASSES,
    EXAMPLES_CONFUSED,
    GEN_TEST_CSV,
    GEN_TRAIN_CSV,
    JOIN_KEY,
    LEVELS,
    LIGHTBOX,
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
    bootstrap_ap,
    bootstrap_indices,
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
    interval_of,
    legend,
    macro_tiles,
    merge_model,
    merge_truth,
    merged_order,
    operating_table,
    pair_table,
    pr_panels,
    predictions_dir,
    region_chart,
    region_name,
    region_summary,
    region_table,
    stylesheet,
    top_k_agreement,
    train_counts,
)

OUTPUT = Path("output_eval_2026_09_04_backbones")
SWEEP = VIEW / "flip_2026_09_04"
BACKBONES = SWEEP / "other_backbones"
BACKBONES_COMFE = SWEEP / "other_backbones_comfe"

# A run with nothing saved that has written nothing for this long is dead, not training.
# The probes write to main.log every few minutes, but a ComFe run logs nothing at all
# between "Starting training!" and the checkpoint selection about four hours later, so
# liveness is read off the newest of the log and the tensorboard event files rather than
# off main.log alone. The abandoned 2026-09-11 C-RADIOv4 runs left neither.
STALE_HOURS = 24

# The seven families, in display order. `match` is a substring of the run directory name
# and carries the launch-batch suffix, because `dinov3_linear_finetune_vitl16` is a prefix
# of `dinov3_linear_finetune_vitl16_sat`. `alias` is the key the same runs had on the
# 2026-09-04 dashboard, where there is one.
MODELS = [
    {
        "key": "lin_s",
        "slot": 5,
        "label": "Linear probe, DINOv2 ViT-S/14",
        "detail": "the 2026-09-04 lin_m re-read: dinov2_vits14 off torch hub, 21M parameters, LVD-142M web pretraining",
        "backbone": "DINOv2 ViT-S/14",
        "pretraining": "LVD-142M web images",
        "params": "21M",
        "alias": "lin_m",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP,
        "match": "dinov2_linear_finetune_vit_14_1-5",
    },
    {
        "key": "lin_l",
        "slot": 1,
        "label": "Linear probe, DINOv2 ViT-L/14 with registers",
        "detail": "facebook/dinov2-with-registers-large, 300M parameters, LVD-142M web pretraining; the backbone ComFe sits on",
        "backbone": "DINOv2 ViT-L/14 w/ reg.",
        "pretraining": "LVD-142M web images",
        "params": "300M",
        "alias": None,
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": BACKBONES,
        "match": "dinov2_linear_finetune_vitlwr_1-5",
    },
    {
        "key": "lin_v3",
        "slot": 2,
        "label": "Linear probe, DINOv3 ViT-L/16",
        "detail": "facebook/dinov3-vitl16-pretrain-lvd1689m, 300M parameters, LVD-1689M web pretraining",
        "backbone": "DINOv3 ViT-L/16",
        "pretraining": "LVD-1689M web images",
        "params": "300M",
        "alias": None,
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": BACKBONES,
        "match": "dinov3_linear_finetune_vitl16_1-5",
    },
    {
        "key": "lin_sat",
        "slot": 3,
        "label": "Linear probe, DINOv3 ViT-L/16 satellite",
        "detail": "facebook/dinov3-vitl16-pretrain-sat493m, 300M parameters, SAT-493M aerial and satellite pretraining",
        "backbone": "DINOv3 ViT-L/16 sat",
        "pretraining": "SAT-493M satellite imagery",
        "params": "300M",
        "alias": None,
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": BACKBONES,
        "match": "dinov3_linear_finetune_vitl16_sat_1-5",
    },
    {
        "key": "lin_radio",
        "slot": 4,
        "label": "Linear probe, C-RADIOv4 SO400M",
        "detail": "nvidia/C-RADIOv4-SO400M, agglomerative model distilled from several vision foundation teachers",
        "backbone": "C-RADIOv4 SO400M",
        "pretraining": "agglomerative distillation",
        "params": "400M",
        "alias": None,
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": BACKBONES,
        "match": "384a1_flip_2026_09_04_radiov4_linear_finetune_so400m_1-5",
    },
    {
        "key": "comfe_l",
        "slot": 6,
        "label": "ComFe, DINOv2 ViT-L/14 with registers",
        "detail": "the 2026-09-04 comfe_m re-read: ComFe head on facebook/dinov2-with-registers-large, the same backbone as lin_l",
        "backbone": "DINOv2 ViT-L/14 w/ reg.",
        "pretraining": "LVD-142M web images",
        "params": "300M",
        "alias": "comfe_m",
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": SWEEP,
        "match": "comfe_dinov2_vitlwr_master_1-5",
    },
    {
        "key": "comfe_v3",
        "slot": 7,
        "label": "ComFe, DINOv3 ViT-L/16",
        "detail": "ComFe head on facebook/dinov3-vitl16-pretrain-lvd1689m, the same backbone as lin_v3; launched 2026-09-18, 50 epochs on the master split like comfe_l",
        "backbone": "DINOv3 ViT-L/16",
        "pretraining": "LVD-1689M web images",
        "params": "300M",
        "alias": None,
        "classes": MASTER_CLASSES,
        "space": "test",
        "runs": BACKBONES_COMFE,
        "match": "comfe_dinov3_vitl16_master_1-5",
    },
]

# The gallery is picked on the ComFe, as on the earlier page, so the same farms come up.
REFERENCE_KEYS = ["comfe_l", "lin_l", "lin_s", "lin_v3", "lin_sat", "lin_radio", "comfe_v3"]

# The reference palette's categorical slots, light value first. The two re-read families
# keep the colours they had on the 2026-09-04 page (slots 5 and 6), so pink is still the
# ViT-S probe and dark green is still ComFe across the two dashboards; the four new
# backbones take slots 1 to 4, and the DINOv3 ComFe takes slot 7, the purple the cascades
# had on the 2026-09-04 page - no cascade appears here, so the colour is free and no model
# changes colour between the two dashboards. The base module's chart routines read their
# series colours off its own SERIES table, so it is pointed at this one.
SERIES = {
    "lin_s": ("#e87ba4", "#d55181"),
    "lin_l": ("#2a78d6", "#3987e5"),
    "lin_v3": ("#eb6834", "#d95926"),
    "lin_sat": ("#1baf7a", "#199e70"),
    "lin_radio": ("#eda100", "#c98500"),
    "comfe_l": ("#008300", "#008300"),
    "comfe_v3": ("#4a3aa7", "#9085e9"),
}
base.SERIES = SERIES


# --------------------------------------------------------------------------------------
# loading runs: the base loader, plus what state each run is in
# --------------------------------------------------------------------------------------

STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def log_span(run_dir: Path) -> tuple[float | None, float | None]:
    """First and last log timestamps as epoch seconds, or None where the log is missing."""
    log = run_dir / "main.log"
    if not log.exists():
        return None, None
    first = last = None
    with log.open(errors="replace") as handle:
        for line in handle:
            found = STAMP.match(line)
            if not found:
                continue
            stamp = time.mktime(time.strptime(found.group(1), "%Y-%m-%d %H:%M:%S"))
            first = stamp if first is None else first
            last = stamp
    return first, last


def last_write(run_dir: Path) -> float | None:
    """When the run last wrote anything: its log or its tensorboard event files."""
    written = [run_dir / "main.log", *(run_dir / "tensorboard").rglob("events.out.tfevents.*")]
    stamps = [path.stat().st_mtime for path in written if path.exists()]
    return max(stamps) if stamps else None


def run_state(run_dir: Path) -> str:
    """`scored`, `training` or `dead`: dead is nothing saved and a run gone quiet."""
    if predictions_dir(run_dir):
        return "scored"
    moved = last_write(run_dir)
    if moved is not None and time.time() - moved < STALE_HOURS * 3600:
        return "training"
    return "dead"


def saved_epoch(run_dir: Path) -> int | None:
    """The epoch the saved test prediction came from, off the prediction file name."""
    path = predictions_dir(run_dir)
    if path is None:
        return None
    found = re.search(r"epoch=(\d+)", next(path.glob("*_y_hat_predictions*.csv")).name)
    return int(found.group(1)) if found else None


def load_model(spec: dict, n_test: int, group_ids: np.ndarray) -> dict:
    """The base loader's model, with each run's state, best epoch and wall-clock added."""
    model = base.load_model(spec, np.arange(0), 0, n_test, group_ids)
    found = base.discover(spec)
    ready = [d for d in found if predictions_dir(d)]
    states = {d.name: run_state(d) for d in found}
    hours = []
    for run_dir in ready:
        first, last = log_span(run_dir)
        hours.append((last - first) / 3600 if first is not None and last is not None else np.nan)
    return {
        **model,
        "epochs": [saved_epoch(d) for d in ready],
        "hours": hours,
        "n_pending": sum(state == "training" for state in states.values()),
        "pending_names": [name for name, state in states.items() if state == "training"],
        "n_dead": sum(state == "dead" for state in states.values()),
        "dead_names": [name for name, state in states.items() if state == "dead"],
    }


# --------------------------------------------------------------------------------------
# the macro, bootstrapped and paired
# --------------------------------------------------------------------------------------


def macro_pairs(
    level: str,
    truth: pd.DataFrame,
    models: list[dict],
    classes: list[str],
    label: str,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    """Every model's macro AP over `classes` on the same resamples, differenced pairwise.

    The same `bootstrap_indices(n, reps, seed)` as `evaluate`, so a draw here is the same
    draw as in the per-class table. A draw's macro is the plain mean over the classes, so
    a draw that lost every positive of one class is dropped for every model alike, and
    the interval is withheld when fewer than half the draws survive. A model with no
    output for one of the classes has no macro here at all.
    """
    n = len(truth)
    resamples = bootstrap_indices(n, reps, seed)
    scored = [m for m in models if m["ensemble"] is not None]
    usable = [c for c in classes if 0 < truth[c].sum() < n]
    draws: dict[str, np.ndarray] = {}
    point: dict[str, float] = {}
    for model in scored:
        if any(base.column_of(model, c) is None for c in usable):
            continue
        per_class = np.array(
            [
                bootstrap_ap(truth[c].to_numpy(), base.scores_for(model, c), resamples)
                for c in usable
            ]
        )
        draws[model["key"]] = per_class.mean(axis=0)
        point[model["key"]] = float(
            np.mean(
                [
                    base.average_precision_score(truth[c].to_numpy(), base.scores_for(model, c))
                    for c in usable
                ]
            )
        )
    rows = []
    for i, a in enumerate(scored):
        for b in scored[i + 1 :]:
            if a["key"] not in draws or b["key"] not in draws:
                continue
            delta = draws[b["key"]] - draws[a["key"]]
            low, high = interval_of(delta)
            rows.append(
                {
                    "level": level,
                    "macro": label,
                    "n_classes": len(usable),
                    "classes": " ".join(usable),
                    "a": a["key"],
                    "b": b["key"],
                    "pair": f"{b['key']}-{a['key']}",
                    "AP_a": point[a["key"]],
                    "AP_b": point[b["key"]],
                    "delta_AP": point[b["key"]] - point[a["key"]],
                    "delta_lo": low,
                    "delta_hi": high,
                    "clear_of_zero": bool(
                        np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# dashboard pieces that differ from the base page
# --------------------------------------------------------------------------------------


def roster_card(models: list[dict]) -> str:
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
        alias = (
            f'<span class="model-key">{escape(model["alias"])} on the 2026-09-04 page</span>'
            if model["alias"]
            else ""
        )
        cards.append(
            f'<div class="model"><p class="model-name">'
            f'<i class="key-line" style="background:var(--series-{model["key"]})"></i>'
            f'{escape(model["label"])}<span class="model-key">{escape(model["key"])}</span>{alias}</p>'
            f'<p class="model-note"><b>{escape(model["backbone"])}</b>, {escape(model["params"])}, '
            f'{escape(model["pretraining"])}. {escape(model["detail"])}</p>'
            f'<p class="model-runs">{runs} {" ".join(states)}</p></div>'
        )
    return f'<div class="roster">{"".join(cards)}</div>'


def macro_pair_card(macro: pd.DataFrame, models: list[dict], label: str) -> str:
    """Every pairwise difference in one macro, crop and farm side by side."""
    rows = macro[macro["macro"] == label]
    if rows.empty:
        return ""
    levels = [level for level in LEVELS if level in set(rows["level"])]
    order = list(dict.fromkeys(rows["pair"]))
    n_classes = {
        level: int(rows[rows["level"] == level].iloc[0]["n_classes"]) for level in levels
    }
    head = "".join(
        f"<th>{escape(level)}<br><span class=\"muted\">{n_classes[level]} classes</span></th>"
        for level in levels
    )
    body = []
    for pair in order:
        b_key, a_key = pair.split("-")
        cells = []
        for level in levels:
            entry = rows[(rows["level"] == level) & (rows["pair"] == pair)]
            if entry.empty:
                cells.append("<td>-</td>")
                continue
            entry = entry.iloc[0]
            delta = float(entry["delta_AP"])
            # Shaded like the per-class pair table: tint by size, blue up, red down, and
            # a dot only when the interval clears zero.
            strength = min(abs(delta) / 0.10, 1.0)
            pole = "up" if delta >= 0 else "down"
            mark = ' <span class="clear">&#9679;</span>' if entry["clear_of_zero"] else ""
            bounds = (
                ""
                if pd.isna(entry["delta_lo"])
                else f'<span class="muted">[{entry["delta_lo"]:+.3f}, {entry["delta_hi"]:+.3f}]</span>'
            )
            caption = f"{label}, {level}: {b_key} minus {a_key} = {delta:+.3f} AP"
            cells.append(
                f'<td class="delta-cell" style="--tint:{strength:.3f}" data-pole="{pole}" '
                f'title="{escape(caption)}"><span class="delta-value">{delta:+.3f}{mark}</span>'
                f"<br>{bounds}</td>"
            )
        body.append(
            f'<tr><th scope="row"><span class="pairhead">'
            f'<i class="key-line" style="background:var(--series-{b_key})"></i>'
            f'&minus;<i class="key-line" style="background:var(--series-{a_key})"></i> '
            f"{escape(b_key)} &minus; {escape(a_key)}</span></th>{''.join(cells)}</tr>"
        )
    return (
        f'<div class="scroll"><table class="data pairs"><thead><tr><th>pair</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def metrics_table(metrics: pd.DataFrame, models: list[dict], level: str) -> str:
    """The base table with its training-count note rewritten for a roster that all trained
    on the master split."""
    table = base.metrics_table(metrics, models, level)
    table, _ = table.split('<p class="table-note">', 1)
    return table + (
        f'<p class="table-note"><b>train</b> counts are crops carrying the class in the '
        f"2026-08-21 relabelled NSW split and in the master split. Every model here trained "
        f"on the master split; the 2026-08-21 column is kept because the well-sampled subset "
        f"is defined on both, so it is the same subset the earlier dashboards ranked on. "
        f"Master counts are mostly farm-level labels copied onto every building of the farm. "
        f"Dimmed rows fall short of {MIN_TRAIN_EXAMPLES} training crops in either split or of "
        f"{MIN_POSITIVES} test positives, so their AP cannot separate models; an AP cell is "
        f"marked when its own 95 % interval is at least {WIDE_INTERVAL:.2f} wide.</p>"
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
    n_master_train: int,
) -> str:
    scored = [m for m in models if m["ensemble"] is not None]
    waiting = [m for m in models if m["ensemble"] is None or m["n_pending"] or m["n_dead"]]
    if farm_rule == "prevalence":
        rule_note = (
            "A class is predicted for a farm when the farm is among the <em>k</em> "
            "highest-scoring farms for that class, with <em>k</em> the number of farms "
            "annotated with it &mdash; the operating point at prevalence, where predicted and "
            "annotated counts agree and precision equals recall. It is the only rule that is "
            "fair across models whose scores sit on such different scales, and it means a "
            "class no farm is annotated with, such as <code>paddock</code>, is never "
            "predicted. Rerun with <code>--farm-threshold 0.5</code> to see a fixed score cut "
            "instead."
        )
    else:
        rule_note = (
            f"A class is predicted for a farm when its score exceeds {escape(farm_rule)}, "
            f"whatever the class or model &mdash; what a deployment with a single threshold "
            f"would do. The models&rsquo; scores are not on a common scale, so read the "
            f"operating-point table underneath before comparing panels; the default "
            f"<code>--farm-threshold prevalence</code> predicts each class for as many farms "
            f"as are annotated with it instead."
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
            elif m["n_dead"] and not m["seeds"]:
                parts.append(
                    f"{escape(m['label'])} ({m['n_dead']} run(s) dead: every log stops at the "
                    f"checkpoint download, nothing saved)"
                )
            elif m["n_dead"]:
                parts.append(f"{escape(m['label'])} ({m['n_dead']} dead run(s) left out)")
            else:
                parts.append(f"{escape(m['label'])} (nothing saved yet)")
        banner = (
            f'<div class="banner"><strong>Incomplete sweep.</strong> {"; ".join(parts)}. '
            f"Everything below is scored on the runs that have landed; rerun "
            f"<code>gen_evaluation_2026_09_04_backbones.py</code> once more runs have saved a "
            f"prediction and the charts fill in.</div>"
        )

    probes = [m for m in scored if not m["key"].startswith("comfe")]
    comfes = [m for m in scored if m["key"].startswith("comfe")]
    probe_note = f"{len(probes)} linear probe{'s' if len(probes) != 1 else ''}"
    if comfes:
        probe_note += (
            " and ComFe" if len(comfes) == 1 else f" and {len(comfes)} ComFe backbones"
        )

    sections = [
        f"""
<header class="masthead">
  <p class="eyebrow">Backbone sweep &middot; VIC hold-out &middot;
  {escape(n_crops)} crops &middot; {escape(n_farms)} farms</p>
  <h1>Which backbone should the classifier sit on?</h1>
  <p class="standfirst">{escape(probe_note)}, every one trained on the
  {escape(f"{n_master_train:,}")}-crop master build with the same recipe, scored against the
  hand-relabelled crop labels of the VIC hold-out. On the 2026-09-04 page the linear probe
  lost to ComFe on livestock, but the probe sat on DINOv2 ViT-S/14 and ComFe on ViT-L/14
  with registers, so the head and the backbone were confounded. Here the same probe is
  retrained on ComFe&rsquo;s own ViT-L backbone, on DINOv3 ViT-L/16 with web and with
  satellite weights, and on C-RADIOv4, and ComFe itself is retrained on DINOv3 ViT-L/16,
  so each head is now read on more than one backbone. Two questions, kept apart: what a
  bigger or domain-matched backbone is worth to a linear probe, and how much of
  ComFe&rsquo;s lead is its head once the backbone is held fixed. Average precision by class, as before,
  because the classes are rare and every model emits a ranking rather than a decision.</p>
</header>
""",
        banner,
        roster_card(models),
        f'<div class="tiles">{macro_tiles(metrics, models)}'
        f"{agreement_card(agreement, models)}</div>",
        f"""
<section id="macro-pairs">
  <h2>The headline, paired</h2>
  <p class="lede">Each cell is the well-sampled macro AP of the left model minus the right
  model&rsquo;s, both computed on the <em>same</em> bootstrap resamples, so the difference
  carries its own 95&nbsp;% interval; a &#9679; marks an interval clear of zero. This is
  the one number to read a backbone off. The per-class version of the same table is
  further down.</p>
  <div class="card">
    <div class="card-head"><h3>{escape(MACRO_SAMPLED)}</h3></div>
    {macro_pair_card(macro, scored, MACRO_SAMPLED)}
  </div>
  <div class="card">
    <div class="card-head"><h3>{escape(MACRO_SHARED)}</h3></div>
    <p class="lede">The same over all nine livestock classes, including the ones too thin to
    rank on. Wider intervals for the same reason.</p>
    {macro_pair_card(macro, scored, MACRO_SHARED)}
  </div>
</section>
""",
        f"""
<section>
  <h2>Average precision by class</h2>
  <p class="lede">A random ranker scores the prevalence, marked on each row &mdash; a bar that
  stops near the tick has found nothing. Whiskers are 95&nbsp;% bootstrap intervals over the
  evaluation units. Open circles are the individual seed runs, so the spread from the training
  seed sits next to the spread from the labels.</p>
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
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level, common classes merged &mdash; {merged_caption}</h3>
      <div class="legend"><span><i class="key-seed"></i>thin line = one seed run</span>
      <span><i class="key-tick"></i>prevalence</span></div>
    </div>
    <p class="lede">The same curves over the merged vocabulary: a crop is a positive for a
    merged class if it is annotated with any member, and a model&rsquo;s score for it is the
    highest of the members it emits. The prevalence tick moves up accordingly, so compare each
    curve with its own tick rather than with the panel above.</p>
    <div class="scroll">{pr_panels(merge_truth(truths["crop"]), [merge_model(m) for m in scored], "crop", merged_order())}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Where the classes leak</h2>
  <p class="lede">Average precision never fixes an operating point, so it cannot say
  <em>where</em> a class goes wrong. These are the annotated class against the model's
  top-scoring class, shaded by row fraction.
  {escape(int(truths["crop"]["paddock"].sum()))} of the {escape(n_crops)} crops are paddock,
  and every model here was trained on {escape(f"{n_master_train:,}")} crops in which nearly
  every paddock carried its farm&rsquo;s livestock class, so the paddock row is where a
  backbone that sees the building for what it is should show first.</p>
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
    too. {rule_note} A farm annotated with two classes appears in both rows and a farm
    predicted with two classes in both columns, so the cell (dairy, horse) counts dairy farms
    that were predicted horse, whether or not they were also predicted dairy. The row count is
    the number of farms annotated with that class, so the diagonal&rsquo;s row fraction is
    that class&rsquo;s <em>recall</em>. The last column counts farms annotated with the row
    class for which the model predicted nothing at all.</p>
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
    <p class="lede">The same panel with the classes the labellers themselves confuse folded
    together, to ask whether a model gets the <em>kind</em> of farm right when it misses the
    grade. A farm carries a merged class if it carries any member, a model&rsquo;s score for
    it is the highest of the members it emits, and the operating point is recomputed at the
    merged prevalence. A farm annotated beef and dairy is one cattle farm here, not two rows.</p>
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
  &mdash; its highest-, median- and lowest-scoring farm for that class &mdash; so every group
  carries a miss beside a hit, and because the reference is the same ComFe as on the
  2026-09-04 page, these are the same farms. Each group then adds the
  {escape(EXAMPLES_CONFUSED)} farms <em>not</em> annotated with that class that the models
  together rank highest for it, on the mean rank percentile across every model that emits
  it: the shared false alarms. The last group is farms with no livestock annotation at all,
  ranked on the reference model&rsquo;s loudest livestock score. The farm-level line
  compares each model&rsquo;s top <em>livestock</em> class ({escape(aggregation)} over its
  crops) with the farm labels, since no farm is annotated paddock. Click any image to
  enlarge it with the calls beside it; the arrow keys step through every image in the
  gallery, and Escape closes it. Click a group heading to collapse it.</p>
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
    <code>test_autocrop_gen_vic.csv</code> in the master build, which is the 2026-08-21
    <code>relabelled_test_df.csv</code> carried through row for row (checked on
    <code>(ecw_stem, building_cluster, PFI)</code> at load time). {escape(n_crops)} crops over
    {escape(n_farms)} farms; <code>Ambiguous</code> crops were dropped at build time, and the
    17 genuinely multi-label crops carry both their classes.</p>
    <p><strong>What every model trained on.</strong> <code>train_df.csv</code> of
    <code>original_master_2026_09_04/</code>: the 397 relabelled NSW crops plus 15,561
    autocrops and 1,969 historical whole-farm images, both farm-labelled. <code>paddock</code>
    and <code>other_industrial</code> exist only in the 397. The linear probes share one
    recipe &mdash; frozen backbone, one linear layer on the last block, sigmoid outputs and
    BCE, SGD with a cosine schedule, at most 60 epochs with early stopping on validation
    accuracy, four seeds &mdash; and differ only in the backbone. The two ComFe families
    share their own recipe &mdash; frozen backbone, ComFe head, 50 epochs, four seeds
    &mdash; and likewise differ only in the backbone: <code>comfe_l</code> is the
    2026-09-04 run on DINOv2 ViT-L/14 with registers, <code>comfe_v3</code> the
    2026-09-18 run on DINOv3 ViT-L/16. Each ComFe therefore has a linear probe on its own
    backbone to be read against &mdash; <code>lin_l</code> and <code>lin_v3</code>.</p>
    <p><strong>Two families are re-read.</strong> <code>lin_s</code> is the 2026-09-04
    <code>lin_m</code> and <code>comfe_l</code> its <code>comfe_m</code>, from the same run
    directories, so their numbers repeat the earlier dashboard exactly; they are renamed so
    the key names the backbone. The four new probes are read from
    <code>{escape(BACKBONES.name)}/</code> and <code>comfe_v3</code> from
    <code>{escape(BACKBONES_COMFE.name)}/</code>, under the same view.</p>
    <p><strong>Dead runs.</strong> A run with no saved prediction is <em>training</em> if
    its <code>main.log</code> or a tensorboard event file changed in the last
    {STALE_HOURS} hours and <em>dead</em> otherwise. Both are checked because a ComFe run
    logs nothing between the start of training and the checkpoint selection four hours
    later, while its event file keeps growing. C-RADIOv4's first four runs, launched 2026-09-11, ended at the Hugging Face
    checkpoint download with no error, checkpoint or event file; they were relaunched on
    2026-09-18 and all four seeds of that batch have saved a prediction. This page reads the
    relaunched batch only, so the abandoned directories are left out rather than reported
    as dead.</p>
    <p><strong>Class vocabularies.</strong> Every model emits the same twelve classes, read
    back off each run's own <code>.hydra/config.yaml</code> before scoring. The headline
    macro is still restricted to the nine livestock classes and then to the well-sampled
    subset, defined on both training splits, so it is the same subset the 2026-08-21 and
    2026-09-04 pages ranked on. <code>aqua</code> has no VIC positives and is reported as
    such.</p>
    <p><strong>The paired macro.</strong> For each pair of models, the well-sampled macro
    AP is computed on every bootstrap resample &mdash; the same resamples as the per-class
    table, so a draw is a draw &mdash; and differenced. A draw that loses every positive of
    one class is dropped for both models; the interval is withheld if fewer than half
    survive. Written to <code>macro_pairs.csv</code>.</p>
    <p><strong>The gallery.</strong> Whole-farm images are the build's
    <code>source_image_path</code>, crops its <code>image_path</code>, both inlined as small
    JPEGs. Selection is on the reference model's farm-level score and is spread across its
    range on purpose; the farms chosen and why are in <code>examples.csv</code>.</p>
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
        "<title>Backbone sweep evaluation</title>"
        f"<style>{stylesheet()}</style>"
        f'<div class="page">{"".join(sections)}</div>'
        + LIGHTBOX
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
    print(f"{TEST_CSV.name}: {len(test)} crops, {test['farm_uid'].nunique()} farms")
    print(f"master {MASTER_TRAIN_CSV.name}: {len(master_train)} crops, per class: " + ", ".join(
        f"{name} {trained_master[name]}" for name in ALL_CLASSES if trained_master[name]
    ))

    # The same guard as the 2026-09-04 script: the master test split must be the
    # 2026-08-21 one row for row, or the earlier pages' numbers would not be comparable.
    if GEN_TEST_CSV.exists():
        gen_test = pd.read_csv(GEN_TEST_CSV, keep_default_na=False)
        same = len(gen_test) == len(test) and all(
            gen_test[k].astype(str).tolist() == test[k].astype(str).tolist() for k in JOIN_KEY
        )
        if not same:
            raise SystemExit(f"{TEST_CSV} is not {GEN_TEST_CSV} row for row")
        print(f"{TEST_CSV.name} matches {GEN_TEST_CSV.name} row for row")

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
        print(f"  {model['key']:>9}: {note}")
        for name in model["pending_names"]:
            print(f"             pending: {name}")
        for name in model["dead_names"]:
            print(f"             dead: {name}")
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

    metrics, pairs, macro, regions, agreement = [], [], [], [], []
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
        header = f"{'class':<20}{'master':>7}{'pos':>5}{'prev':>7}"
        for model in scored:
            header += f"{model['key']:>22}"
        print(header)
        for _, row in metrics[metrics["level"] == level].iterrows():
            positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
            prevalence = "" if pd.isna(row.get("prevalence")) else f"{row['prevalence']:.3f}"
            master = "" if pd.isna(row.get("n_train_master")) else int(row["n_train_master"])
            mark = "" if row.get("well_sampled", True) or str(row["class"]).startswith("MACRO") else " *"
            line = f"{str(row['class']) + mark:<20}{master:>7}{positives:>5}{prevalence:>7}"
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
            f"* = fewer than {MIN_TRAIN_EXAMPLES} training crops (either split) or "
            f"{MIN_POSITIVES} test positives; scored but excluded from {MACRO_SAMPLED}."
        )

    title = f"{MACRO_SAMPLED} - paired differences"
    print(f"\n{title}\n" + "-" * len(title))
    for _, row in macro[macro["macro"] == MACRO_SAMPLED].iterrows():
        mark = " *" if row["clear_of_zero"] else ""
        bounds = "" if pd.isna(row["delta_lo"]) else f" [{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}]"
        print(
            f"  {row['level']:<5} {row['pair']:<20} {row['delta_AP']:+.3f}{bounds}{mark}"
            f"   ({row['b']} {row['AP_b']:.3f}, {row['a']} {row['AP_a']:.3f}, "
            f"{int(row['n_classes'])} classes)"
        )
    print("* = interval clear of zero")

    # ---- outputs ---------------------------------------------------------------------
    metrics.to_csv(args.output / "evaluation.csv", index=False)
    pairs.to_csv(args.output / "evaluation_pairs.csv", index=False)
    macro.to_csv(args.output / "macro_pairs.csv", index=False)
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
                "alias_2026_09_04": m["alias"] or "",
                "model": m["label"],
                "backbone": m["backbone"],
                "params": m["params"],
                "pretraining": m["pretraining"],
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

    destination = args.file or args.output / "evaluation_dashboard.html"
    destination.write_text(
        build_page(
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
            len(master_train),
        ),
        encoding="utf-8",
    )

    print(
        f"\nwrote {destination} plus evaluation{{,_pairs,_by_region}}.csv, macro_pairs.csv, "
        f"agreement.csv, scores_{{crop,farm}}.csv, confusion_{{crop,farm,farm_merged}}.csv, "
        f"operating_points_farm{{,_merged}}.csv, models.csv, examples.csv in {args.output}"
    )


if __name__ == "__main__":
    main()
