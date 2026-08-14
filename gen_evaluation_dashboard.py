"""Render the generalisation evaluation as a single self-contained HTML dashboard.

Reads only the processed CSVs `gen_evaluation.py` writes into `output_eval/` - the
metrics table, the confusion counts, the joined per-crop and per-farm scores - so the
dashboard never touches the run directories or the workbooks. Regenerate the CSVs first
if the runs change:

    .venv/bin/python gen_evaluation.py
    .venv/bin/python gen_evaluation_dashboard.py

Charts are inline SVG built here rather than by a plotting library, so the page has no
external dependencies and can be published as-is. Colours come from the validated
default data-viz palette used unchanged (categorical slots 1 and 2 for the two models,
the documented blue-red diverging pair for the paired difference, the blue sequential
ramp for the confusion heatmaps), with light and dark steps declared as tokens.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

OUTPUT = Path("output_eval")

# Categorical slots 1 and 2, and the diverging blue-red poles, from the reference
# palette; light value first, dark second.
SERIES = {"old": ("#2a78d6", "#3987e5"), "new": ("#eb6834", "#d95926")}
DIVERGING = {"up": ("#2a78d6", "#3987e5"), "down": ("#d03b3b", "#e66767")}

# Blue sequential ramp: light mode runs light->dark from the surface, dark mode runs
# dark->light, so in both cases "more" is further from the page.
HEAT_LIGHT = ["#f4f7fb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#0d366b"]
HEAT_DARK = ["#1f2429", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
HEAT_BREAKS = [0.03, 0.08, 0.16, 0.30, 0.50, 0.75]
# Index at which in-cell text flips to the opposite ink.
HEAT_FLIP = {"light": 4, "dark": 4}

LEVELS = {"image": "crop", "farm": "farm"}
MACRO = "MACRO (evaluable)"


# --------------------------------------------------------------------------------------
# svg helpers
# --------------------------------------------------------------------------------------


def escape(text: object) -> str:
    return html.escape(str(text), quote=True)


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


# --------------------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------------------


def seed_columns(frame: pd.DataFrame, key: str, name: str) -> list[str]:
    """The per-seed score columns for one model output, in seed order."""
    return sorted(
        column for column in frame.columns if column.startswith(f"{key}#") and column.endswith(f"|{name}")
    )


def ap_chart(
    table: pd.DataFrame,
    models: dict[str, str],
    level: str,
    scores: pd.DataFrame,
    truth: dict[str, np.ndarray],
) -> str:
    """Per-class AP for both models, with the prevalence baseline marked on each row.

    The bar is the seed-mean ensemble; each seed run's own AP is dotted on top of it, so
    the spread from training seed sits next to the bootstrap interval from the labels.
    """
    rows = table[table["class"] != "MACRO (evaluable)"]
    rows = rows[rows[f"{models['old']} | AP"].notna()]

    left, right, top, row_height, gap = 116, 58, 34, 42, 3
    width = 760
    height = top + len(rows) * row_height + 34
    plot = width - left - right

    def x_of(value: float) -> float:
        return left + value * plot

    parts = []
    for value in np.arange(0, 1.01, 0.25):
        parts.append(
            f'<line class="grid" x1="{x_of(value):g}" y1="{top - 12:g}" '
            f'x2="{x_of(value):g}" y2="{top + len(rows) * row_height:g}"/>'
        )
        parts.append(text(x_of(value), top - 18, f"{value:.2f}", "tick", "middle"))

    bar = (row_height - 2 * gap - 8) / 2
    for index, (_, row) in enumerate(rows.iterrows()):
        y0 = top + index * row_height + gap
        parts.append(text(left - 12, y0 + row_height / 2 - 6, row["class"], "rowlabel", "end"))
        for offset, (key, name) in enumerate(models.items()):
            value = float(row[f"{name} | AP"])
            y = y0 + offset * (bar + 2)
            parts.append(
                f'<g class="mark"><path d="{rounded_right(left, y, max(value * plot, 1.5), bar)}" '
                f'fill="var(--series-{key})"/>'
                + tip(
                    f"{key}: AP {value:.3f} "
                    f"[{row[f'{name} | AP lo']:.3f}, {row[f'{name} | AP hi']:.3f}] "
                    f"- {row['class']}, {int(row['positives'])} positives"
                )
                + "</g>"
            )
            low, high = float(row[f"{name} | AP lo"]), float(row[f"{name} | AP hi"])
            middle = y + bar / 2
            parts.append(
                f'<line class="whisker" x1="{x_of(low):g}" y1="{middle:g}" '
                f'x2="{x_of(high):g}" y2="{middle:g}" stroke="var(--series-{key})"/>'
            )
            y_true = truth[row["class"]]
            for seed, column in enumerate(seed_columns(scores, key, row["class"]), start=1):
                seed_ap = average_precision_score(y_true, scores[column].to_numpy())
                parts.append(
                    f'<g class="mark"><circle class="seed" cx="{x_of(seed_ap):g}" cy="{middle:g}" '
                    f'r="3.2" stroke="var(--series-{key})"/>'
                    + tip(f"{key} seed {seed}: AP {seed_ap:.3f} - {row['class']}")
                    + "</g>"
                )
        prevalence = float(row["prevalence"])
        parts.append(
            f'<line class="baseline-mark" x1="{x_of(prevalence):g}" y1="{y0 - 1:g}" '
            f'x2="{x_of(prevalence):g}" y2="{y0 + 2 * bar + 3:g}"/>'
        )
        parts.append(
            text(width - right + 10, y0 + row_height / 2 - 6, f"n={int(row['positives'])}", "note")
        )

    parts.append(
        f'<line class="axis" x1="{left:g}" y1="{top + len(rows) * row_height:g}" '
        f'x2="{width - right:g}" y2="{top + len(rows) * row_height:g}"/>'
    )
    parts.append(
        text(left + plot / 2, height - 6, "average precision", "axistitle", "middle")
    )
    return svg(width, height, "".join(parts), f"Average precision per class, {level} level")


def delta_chart(table: pd.DataFrame, level: str) -> str:
    """Paired AP difference (new - old) with its bootstrap interval, centred on zero."""
    rows = table[(table["class"] != "MACRO (evaluable)") & table["delta AP (new - old)"].notna()]
    span = float(np.ceil(max(rows["delta hi"].abs().max(), rows["delta lo"].abs().max()) * 10) / 10)

    left, right, top, row_height = 116, 58, 34, 34
    width, height = 760, top + len(rows) * row_height + 34
    plot = width - left - right

    def x_of(value: float) -> float:
        return left + (value + span) / (2 * span) * plot

    parts = []
    for value in np.linspace(-span, span, 5):
        parts.append(
            f'<line class="grid" x1="{x_of(value):g}" y1="{top - 12:g}" '
            f'x2="{x_of(value):g}" y2="{top + len(rows) * row_height:g}"/>'
        )
        parts.append(text(x_of(value), top - 18, f"{value:+.2f}", "tick", "middle"))

    parts.append(
        f'<line class="zero" x1="{x_of(0):g}" y1="{top - 12:g}" '
        f'x2="{x_of(0):g}" y2="{top + len(rows) * row_height:g}"/>'
    )

    for index, (_, row) in enumerate(rows.iterrows()):
        y = top + index * row_height + row_height / 2
        delta, low, high = (
            float(row["delta AP (new - old)"]),
            float(row["delta lo"]),
            float(row["delta hi"]),
        )
        pole = "up" if delta >= 0 else "down"
        crosses = low <= 0 <= high
        parts.append(text(left - 12, y + 4, row["class"], "rowlabel", "end"))
        parts.append(
            f'<g class="mark"><line class="interval" x1="{x_of(low):g}" y1="{y:g}" '
            f'x2="{x_of(high):g}" y2="{y:g}" stroke="var(--pole-{pole})"/>'
            f'<circle cx="{x_of(delta):g}" cy="{y:g}" r="5.5" fill="var(--pole-{pole})" '
            f'stroke="var(--surface)" stroke-width="2"/>'
            + tip(f"{row['class']}: {delta:+.3f} AP [{low:+.3f}, {high:+.3f}]")
            + "</g>"
        )
        parts.append(
            text(
                width - right + 10,
                y + 4,
                "overlaps 0" if crosses else "clear of 0",
                "note-strong" if not crosses else "note",
            )
        )

    parts.append(text(left + plot / 2, height - 6, "AP(new) - AP(old)", "axistitle", "middle"))
    return svg(width, height, "".join(parts), f"Paired AP difference, {level} level")


def pr_panels(scores: pd.DataFrame, truth: dict[str, np.ndarray], level: str) -> str:
    """Small multiples of the precision-recall curve behind each class's AP."""
    classes = [name for name, flags in truth.items() if flags.any()]
    columns, size, gap = 3, 190, 26
    rows = int(np.ceil(len(classes) / columns))
    pad_left, pad_top = 40, 26
    width = pad_left + columns * (size + gap)
    height = pad_top + rows * (size + gap + 18) + 10

    parts = []
    for index, name in enumerate(classes):
        column, row = index % columns, index // columns
        x0 = pad_left + column * (size + gap)
        y0 = pad_top + row * (size + gap + 18)
        y_true = truth[name]
        parts.append(
            f'<rect x="{x0:g}" y="{y0:g}" width="{size:g}" height="{size:g}" class="panel"/>'
        )
        for fraction in (0.25, 0.5, 0.75):
            parts.append(
                f'<line class="grid" x1="{x0:g}" y1="{y0 + fraction * size:g}" '
                f'x2="{x0 + size:g}" y2="{y0 + fraction * size:g}"/>'
            )
        prevalence = y_true.mean()
        parts.append(
            f'<line class="baseline-mark" x1="{x0:g}" y1="{y0 + (1 - prevalence) * size:g}" '
            f'x2="{x0 + size:g}" y2="{y0 + (1 - prevalence) * size:g}"/>'
        )
        def curve(column: str, key: str, cls: str) -> str:
            precision, recall, _ = precision_recall_curve(y_true, scores[column].to_numpy())
            points = " ".join(
                f"{x0 + r * size:g},{y0 + (1 - p) * size:g}" for r, p in zip(recall, precision)
            )
            return f'<polyline class="{cls}" points="{points}" stroke="var(--series-{key})"/>'

        # Seeds first, so the ensemble curve reads on top of its own spread.
        for key in SERIES:
            for column in seed_columns(scores, key, name):
                parts.append(curve(column, key, "curve curve-seed"))
        for key in SERIES:
            parts.append(curve(f"{key}|{name}", key, "curve"))
        parts.append(text(x0, y0 - 8, f"{name}  (n={int(y_true.sum())})", "panellabel"))
        parts.append(text(x0 + size / 2, y0 + size + 14, "recall", "tick", "middle"))

    parts.append(
        f'<text transform="translate(12,{pad_top + size / 2:g}) rotate(-90)" '
        f'class="tick" text-anchor="middle">precision</text>'
    )
    return svg(width, height, "".join(parts), f"Precision-recall curves, {level} level")


def region_chart(regions: pd.DataFrame, level: str) -> str:
    """Macro AP per reach, both models, with each reach's own prevalence baseline.

    Reaches are ordered by size, and every row carries its unit count - a reach of
    sixty-odd crops is a different kind of evidence from one of three hundred.
    """
    rows = regions[
        (regions["level"] == level)
        & (regions["class"] == MACRO)
        & (regions["seed"] == "ensemble")
    ]
    order = (
        rows.groupby("region")["n_units"].first().sort_values(ascending=False).index.tolist()
    )

    left, right, top, row_height, gap = 132, 66, 34, 46, 4
    width = 760
    height = top + len(order) * row_height + 34
    plot = width - left - right

    def x_of(value: float) -> float:
        return left + value * plot

    parts = []
    for value in np.arange(0, 1.01, 0.25):
        parts.append(
            f'<line class="grid" x1="{x_of(value):g}" y1="{top - 12:g}" '
            f'x2="{x_of(value):g}" y2="{top + len(order) * row_height:g}"/>'
        )
        parts.append(text(x_of(value), top - 18, f"{value:.2f}", "tick", "middle"))

    bar = (row_height - 2 * gap - 8) / 2
    for index, region in enumerate(order):
        group = rows[rows["region"] == region]
        y0 = top + index * row_height + gap
        first = group.iloc[0]
        parts.append(text(left - 12, y0 + row_height / 2 - 10, region, "rowlabel", "end"))
        parts.append(
            text(
                left - 12,
                y0 + row_height / 2 + 4,
                f"{int(first['n_units'])} units, {int(first['n_classes'])} classes",
                "note",
                "end",
            )
        )
        for offset, key in enumerate(SERIES):
            entry = group[group["model"] == key]
            if entry.empty:
                continue
            value = float(entry.iloc[0]["AP"])
            y = y0 + offset * (bar + 2)
            parts.append(
                f'<g class="mark"><path d="{rounded_right(left, y, max(value * plot, 1.5), bar)}" '
                f'fill="var(--series-{key})"/>'
                + tip(f"{key}: macro AP {value:.3f} - {region}, {int(first['n_units'])} units")
                + "</g>"
            )
            low, high = entry.iloc[0]["AP_lo"], entry.iloc[0]["AP_hi"]
            middle = y + bar / 2
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
        prevalence = float(first["prevalence"])
        parts.append(
            f'<line class="baseline-mark" x1="{x_of(prevalence):g}" y1="{y0 - 1:g}" '
            f'x2="{x_of(prevalence):g}" y2="{y0 + 2 * bar + 3:g}"/>'
        )

    parts.append(
        f'<line class="axis" x1="{left:g}" y1="{top + len(order) * row_height:g}" '
        f'x2="{width - right:g}" y2="{top + len(order) * row_height:g}"/>'
    )
    parts.append(text(left + plot / 2, height - 6, "macro average precision", "axistitle", "middle"))
    return svg(width, height, "".join(parts), f"Macro AP by region, {level} level")


def region_class_table(regions: pd.DataFrame, level: str) -> str:
    """Every class within every reach: AP for both models and the paired difference."""
    rows = regions[(regions["level"] == level) & (regions["seed"] == "ensemble")]
    order = (
        rows.groupby("region")["n_units"].first().sort_values(ascending=False).index.tolist()
    )
    head = (
        "<tr><th>region</th><th>class</th><th>positives</th><th>prevalence</th>"
        "<th>AP old</th><th>AP new</th><th>&Delta; AP</th></tr>"
    )
    body = []
    for region in order:
        group = rows[rows["region"] == region]
        names = [MACRO] + [name for name in group["class"].unique() if name != MACRO]
        for position, name in enumerate(names):
            entry = group[group["class"] == name]
            first = entry.iloc[0]
            values = {key: entry[entry["model"] == key] for key in SERIES}
            delta = entry[entry["delta_AP"].notna()]
            if delta.empty:
                delta_cell = '<td class="muted">&mdash;</td>'
            else:
                value = delta.iloc[0]
                bounds = (
                    ""
                    if pd.isna(value["delta_lo"])
                    else f' <span class="muted">[{value["delta_lo"]:+.2f}, {value["delta_hi"]:+.2f}]</span>'
                )
                delta_cell = f'<td>{value["delta_AP"]:+.3f}{bounds}</td>'
            label = (
                f'{escape(region)} <span class="muted">{int(first["n_units"])} units</span>'
                if position == 0
                else ""
            )
            classes = ' class="total"' if position == 0 else ""
            body.append(
                f'<tr{classes}><th scope="row">{label}</th>'
                f"<td>{escape('macro' if name == MACRO else name)}</td>"
                f"<td>{int(first['positives'])}</td><td>{first['prevalence']:.3f}</td>"
                + "".join(
                    f"<td>{values[key].iloc[0]['AP']:.3f}</td>" if not values[key].empty else "<td>-</td>"
                    for key in SERIES
                )
                + f"{delta_cell}</tr>"
            )
    return (
        f'<div class="scroll"><table class="data region-table"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def heat_index(fraction: float) -> int:
    return int(np.searchsorted(HEAT_BREAKS, fraction, side="right"))


def confusion_table(counts: pd.DataFrame, key: str) -> str:
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
            classes = "cell" + (" cell-flip" if step >= HEAT_FLIP["light"] else "")
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
        f'<figure class="confusion"><figcaption>{escape(key)}</figcaption>'
        f'<div class="scroll"><table class="heat">'
        f'<thead><tr><th scope="col" class="corner">annotated \\ predicted</th>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></figure>"
    )


# --------------------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------------------


def top_k_agreement(
    scores: pd.DataFrame,
    classes: list[str],
    truth: dict[str, np.ndarray],
    depths=(1, 2, 3),
) -> tuple[dict[int, dict[str, float]], int]:
    """Share of units whose annotation is among the model's k highest-scoring classes.

    Only units annotated with one of the model's own classes count - a Paddock crop, or a
    farm with no X against any of the ten, has no right answer to be in the top k. Farm
    labels are multi-label, so a farm counts as a hit when *any* of its classes is there.
    """
    matrix = np.column_stack([truth[name] for name in classes])
    keep = matrix.any(axis=1)
    matrix = matrix[keep]
    agreement = {}
    for depth in depths:
        agreement[depth] = {}
        for key in SERIES:
            values = scores[[f"{key}|{name}" for name in classes]].to_numpy()[keep]
            ranked = np.argsort(-values, axis=1)[:, :depth]
            hit = np.take_along_axis(matrix, ranked, axis=1).any(axis=1)
            agreement[depth][key] = float(hit.mean())
    return agreement, int(keep.sum())


def agreement_card(agreement: dict[str, tuple[dict[int, dict[str, float]], int]]) -> str:
    """Top-1/2/3 agreement at both levels, as one small table rather than six tiles."""
    head = "".join(
        f'<th><span class="swatch swatch-{key}"></span>{escape(unit)} {escape(key)}</th>'
        for unit in LEVELS.values()
        for key in SERIES
    )
    body = []
    for depth in next(iter(agreement.values()))[0]:
        cells = "".join(
            f"<td>{agreement[level][0][depth][key]:.1%}</td>"
            for level in LEVELS
            for key in SERIES
        )
        body.append(f'<tr><th scope="row">top-{depth}</th>{cells}</tr>')
    counts = ", ".join(
        f"{agreement[level][1]} {unit}s" for level, unit in LEVELS.items()
    )
    return (
        '<div class="tile tile-wide"><p class="tile-label">agreement with the annotation</p>'
        f'<div class="scroll"><table class="mini"><thead><tr><th></th>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'<p class="tile-note">the annotated class is among the model\'s k highest-scoring '
        f"classes, over the {counts} that carry one of the ten classes at all</p></div>"
    )


def stat_tiles(metrics: pd.DataFrame, models: dict[str, str]) -> str:
    tiles = []
    for level, unit in LEVELS.items():
        macro = metrics[(metrics["level"] == level) & (metrics["class"] == "MACRO (evaluable)")]
        row = macro.iloc[0]
        values = {key: float(row[f"{name} | AP"]) for key, name in models.items()}
        delta = values["new"] - values["old"]
        tiles.append(
            f'<div class="tile"><p class="tile-label">macro AP, {unit} level</p>'
            f'<p class="tile-value"><span class="swatch swatch-old"></span>{values["old"]:.3f}'
            f'<span class="tile-sep">/</span>'
            f'<span class="swatch swatch-new"></span>{values["new"]:.3f}</p>'
            f'<p class="tile-note">old / new, mean over the {int(row["positives"])} '
            f'annotated positives in {int(row["n"])} {unit}s '
            f'<span class="delta {"delta-up" if delta >= 0 else "delta-down"}">'
            f'{delta:+.3f}</span></p></div>'
        )
    return "".join(tiles)


def metrics_table(metrics: pd.DataFrame, models: dict[str, str], level: str) -> str:
    rows = metrics[metrics["level"] == level]
    head = (
        "<tr><th>class</th><th>positives</th><th>prevalence</th>"
        "<th>AP old</th><th>AP new</th><th>AUC old</th><th>AUC new</th>"
        "<th>&Delta; AP (new &minus; old)</th></tr>"
    )
    body = []
    for _, row in rows.iterrows():
        if pd.isna(row[f"{models['old']} | AP"]):
            cells = "<td>-</td>" * 5
        else:
            interval = (
                ""
                if pd.isna(row.get("delta lo"))
                else f' <span class="muted">[{row["delta lo"]:+.2f}, {row["delta hi"]:+.2f}]</span>'
            )
            cells = (
                f"<td>{row[f'{models['old']} | AP']:.3f}</td>"
                f"<td>{row[f'{models['new']} | AP']:.3f}</td>"
                f"<td>{row[f'{models['old']} | AUC']:.3f}</td>"
                f"<td>{row[f'{models['new']} | AUC']:.3f}</td>"
                f"<td>{row['delta AP (new - old)']:+.3f}{interval}</td>"
            )
        emphasis = ' class="total"' if row["class"] == "MACRO (evaluable)" else ""
        body.append(
            f"<tr{emphasis}><th scope=\"row\">{escape(row['class'])}</th>"
            f"<td>{int(row['positives'])}</td><td>{row['prevalence']:.3f}</td>{cells}</tr>"
        )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


STYLE = """
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
  --series-old: #2a78d6;
  --series-new: #eb6834;
  --pole-up: #2a78d6;
  --pole-down: #d03b3b;
  --good: #006300;
  --heat-0: #f4f7fb; --heat-1: #cde2fb; --heat-2: #9ec5f4; --heat-3: #6da7ec;
  --heat-4: #3987e5; --heat-5: #256abf; --heat-6: #0d366b;
  --heat-ink: #17171a;
  --heat-ink-flip: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0e0e0d;
    --surface: #1a1a19;
    --ink: #f5f5f0;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #3d3d3a;
    --rule: rgba(255,255,255,0.12);
    --series-old: #3987e5;
    --series-new: #d95926;
    --pole-up: #3987e5;
    --pole-down: #e66767;
    --good: #0ca30c;
    --heat-0: #1f2429; --heat-1: #0d366b; --heat-2: #184f95; --heat-3: #256abf;
    --heat-4: #3987e5; --heat-5: #6da7ec; --heat-6: #9ec5f4;
    --heat-ink: #f5f5f0;
    --heat-ink-flip: #0b0b0b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0e0e0d;
  --surface: #1a1a19;
  --ink: #f5f5f0;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #3d3d3a;
  --rule: rgba(255,255,255,0.12);
  --series-old: #3987e5;
  --series-new: #d95926;
  --pole-up: #3987e5;
  --pole-down: #e66767;
  --good: #0ca30c;
  --heat-0: #1f2429; --heat-1: #0d366b; --heat-2: #184f95; --heat-3: #256abf;
  --heat-4: #3987e5; --heat-5: #6da7ec; --heat-6: #9ec5f4;
  --heat-ink: #f5f5f0;
  --heat-ink-flip: #0b0b0b;
}

body {
  background: var(--plane);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.55;
  margin: 0;
  padding: 40px 24px 88px;
}
.page { max-width: 1080px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }

.masthead { display: flex; flex-direction: column; gap: 10px; }
.eyebrow {
  font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--muted); margin: 0;
}
h1 { font-size: 30px; line-height: 1.15; margin: 0; text-wrap: balance; font-weight: 620; }
.standfirst { margin: 0; color: var(--ink-2); max-width: 66ch; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }
.tile {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 16px 18px; display: flex; flex-direction: column; gap: 6px;
}
.tile-label { margin: 0; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }
.tile-value {
  margin: 0; font-size: 26px; font-weight: 600; display: flex; align-items: center;
  gap: 7px; font-variant-numeric: tabular-nums;
}
.tile-sep { color: var(--muted); font-weight: 400; }
.tile-note { margin: 0; font-size: 12.5px; color: var(--ink-2); }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.swatch-old { background: var(--series-old); }
.swatch-new { background: var(--series-new); }
.tile-wide { grid-column: span 2; }
@media (max-width: 720px) { .tile-wide { grid-column: span 1; } }
.mini { width: 100%; }
.mini th, .mini td { text-align: right; padding: 4px 0 4px 14px; white-space: nowrap; }
.mini thead th {
  font-size: 11px; font-weight: 500; color: var(--muted);
  border-bottom: 1px solid var(--grid); padding-bottom: 6px;
}
.mini thead th .swatch { margin-right: 5px; }
.mini tbody th { text-align: left; padding-left: 0; font-weight: 500; color: var(--ink-2); font-size: 13px; }
.mini tbody td { font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }
.delta { font-variant-numeric: tabular-nums; font-weight: 600; }
.delta-up { color: var(--good); }
.delta-down { color: var(--pole-down); }

section { display: flex; flex-direction: column; gap: 14px; }
h2 {
  font-size: 13px; letter-spacing: 0.1em; text-transform: uppercase; margin: 0;
  padding-bottom: 8px; border-bottom: 1px solid var(--rule); color: var(--ink);
}
h3 { font-size: 15px; margin: 0; font-weight: 600; }
.lede { margin: 0; color: var(--ink-2); max-width: 72ch; }

.card {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 6px;
  padding: 20px 22px 14px; display: flex; flex-direction: column; gap: 12px;
}
.card-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px 20px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-left: auto; }
.legend span { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-2); }
.key-line { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
.key-tick { width: 2px; height: 12px; background: var(--axis); display: inline-block; }
.key-seed {
  width: 9px; height: 9px; border-radius: 50%; display: inline-block;
  background: var(--surface); border: 1.5px solid var(--ink-2);
}
.scroll { overflow-x: auto; }

.mark { cursor: default; }
.mark:hover path, .mark:hover circle { opacity: 0.78; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.zero { stroke: var(--axis); stroke-width: 1.5; }
.baseline-mark { stroke: var(--muted); stroke-width: 1.5; }
.whisker { stroke-width: 1.5; opacity: 0.55; }
.interval { stroke-width: 2; opacity: 0.5; }
.curve { fill: none; stroke-width: 2; stroke-linejoin: round; }
.curve-seed { stroke-width: 1; opacity: 0.38; }
.seed { fill: var(--surface); stroke-width: 1.5; }
.panel { fill: none; stroke: var(--grid); stroke-width: 1; }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.rowlabel { font-size: 12.5px; fill: var(--ink-2); }
.panellabel { font-size: 11.5px; fill: var(--ink-2); }
.note { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.note-strong { font-size: 11px; fill: var(--ink-2); font-weight: 600; }
.axistitle { font-size: 11.5px; fill: var(--muted); }

table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
.data { width: 100%; }
.data th, .data td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
.data thead th { color: var(--muted); font-weight: 500; font-size: 11.5px; white-space: nowrap; }
.data tbody th { text-align: left; font-weight: 500; }
.data tr.total th, .data tr.total td { font-weight: 650; border-top: 1px solid var(--axis); }
.region-table tbody th { white-space: nowrap; }
.region-table tbody td:nth-child(2) { text-align: left; color: var(--ink-2); }
.muted { color: var(--muted); font-weight: 400; }

.confusions { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; }
.confusion { margin: 0; display: flex; flex-direction: column; gap: 10px; }
.confusion figcaption { font-size: 13px; font-weight: 600; }
.heat th, .heat td { padding: 0; }
.heat thead th { font-size: 10.5px; color: var(--muted); font-weight: 500; padding: 0 0 6px; }
.heat thead th span { display: block; writing-mode: vertical-rl; transform: rotate(180deg); }
.heat .corner { writing-mode: horizontal-tb; text-align: right; padding-right: 10px; white-space: nowrap; }
.heat tbody th {
  text-align: right; font-size: 12px; font-weight: 500; color: var(--ink-2);
  padding-right: 10px; white-space: nowrap;
}
.heat .rowcount { color: var(--muted); font-size: 11px; margin-left: 6px; }
.heat .cell {
  width: 34px; height: 26px; text-align: center; font-size: 11px; color: var(--heat-ink);
  background: var(--heat-0); border: 2px solid var(--surface); border-radius: 3px;
}
.heat .cell-flip { color: var(--heat-ink-flip); }
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
  background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 0 4px;
}
a { color: var(--pole-up); }
:focus-visible { outline: 2px solid var(--pole-up); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def legend(seeds: bool = False, prevalence_label: str = "prevalence (random baseline)") -> str:
    seed_key = (
        '<span><i class="key-seed"></i>individual seed run</span>' if seeds else ""
    )
    return (
        '<div class="legend">'
        '<span><i class="key-line" style="background:var(--series-old)"></i>old model</span>'
        '<span><i class="key-line" style="background:var(--series-new)"></i>new model</span>'
        f"{seed_key}"
        f'<span><i class="key-tick"></i>{escape(prevalence_label)}</span>'
        "</div>"
    )


def build(output: Path) -> str:
    metrics = pd.read_csv(output / "gen_evaluation.csv")
    catalogue = pd.read_csv(output / "models.csv")
    models = dict(zip(catalogue["key"], catalogue["model"]))
    seeds = dict(zip(catalogue["key"], catalogue["seeds"]))
    aggregation = catalogue["aggregation"].iloc[0]

    crops = pd.read_csv(output / "scores_image.csv")
    farms = pd.read_csv(output / "scores_farm.csv")
    classes = [column.split("|", 1)[1] for column in crops.columns if column.startswith("old|")]

    crop_truth = {name: (crops["label"] == name).to_numpy() for name in classes}
    farm_truth = {
        name: farms[f"true|{name}"].to_numpy(dtype=bool)
        for name in classes
        if f"true|{name}" in farms.columns
    }

    agreement = {
        "image": top_k_agreement(crops, classes, crop_truth),
        "farm": top_k_agreement(farms, classes, farm_truth),
    }

    image_metrics = metrics[metrics["level"] == "image"]
    farm_metrics = metrics[metrics["level"] == "farm"]

    confusion = {
        level: pd.read_csv(output / f"confusion_{level}.csv").set_index(["model", "annotated"])
        for level in LEVELS
    }
    regions = pd.read_csv(output / "gen_evaluation_by_region.csv")
    reaches = regions["region"].nunique()

    ramp = "".join(f'<i style="background:var(--heat-{step})"></i>' for step in range(7))

    sections = [
        f"""
<header class="masthead">
  <p class="eyebrow">ComFe &middot; generalisation set &middot; {escape(len(crops))} hand-labelled crops</p>
  <h1>Does the retrained classifier generalise better to new reaches?</h1>
  <p class="standfirst">Two ComFe checkpoints scored against the relabelling workbooks in
  <code>labelled_sheets/</code>: the original model trained on the historical FLIP set, and the
  model retrained on <code>original_new_2026_07_24</code>. Average precision by class, because
  the classes are rare and the models emit a ranking rather than a decision. Both are
  probability ensembles over their seed runs ({seeds['old']} and {seeds['new']} seeds).</p>
</header>
""",
        f'<div class="tiles">{stat_tiles(metrics, models)}{agreement_card(agreement)}</div>',
        f"""
<section>
  <h2>Average precision by class</h2>
  <p class="lede">A random ranker scores the prevalence, marked on each row - so a bar that
  stops near the tick has found nothing. Whiskers are 95&nbsp;% bootstrap intervals over the
  evaluation units; the open circles are the individual seed runs behind each ensemble.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level &mdash; one annotated building crop per unit</h3>{legend(seeds=True)}</div>
    <div class="scroll">{ap_chart(image_metrics, models, "image", crops, crop_truth)}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; {escape(aggregation)} over each farm's crops</h3>{legend(seeds=True)}</div>
    <div class="scroll">{ap_chart(farm_metrics, models, "farm", farms, farm_truth)}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>What changed, class by class</h2>
  <p class="lede">The two models are scored on the same bootstrap resamples, so the difference
  carries its own interval. An interval clear of zero is a change the {escape(len(crops))} labelled
  crops can actually support; everything else is within sampling noise.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3>
      <div class="legend"><span><i class="key-line" style="background:var(--pole-up)"></i>new model ahead</span>
      <span><i class="key-line" style="background:var(--pole-down)"></i>old model ahead</span></div>
    </div>
    <div class="scroll">{delta_chart(image_metrics, "image")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level</h3>
      <div class="legend"><span><i class="key-line" style="background:var(--pole-up)"></i>new model ahead</span>
      <span><i class="key-line" style="background:var(--pole-down)"></i>old model ahead</span></div>
    </div>
    <div class="scroll">{delta_chart(farm_metrics, "farm")}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>By region</h2>
  <p class="lede">The workbooks are labelled one reach at a time, so the reach is the region.
  Each reach is scored on the classes it actually contains, and its own prevalence baseline is
  marked on the row &mdash; the class mix differs sharply between reaches, so read the two models
  against each other <em>within</em> a reach rather than reading one reach against another.
  A missing whisker means too few bootstrap draws kept every class in that reach.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level &mdash; macro AP across {escape(reaches)} reaches</h3>{legend(seeds=True)}</div>
    <div class="scroll">{region_chart(regions, "image")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; macro AP across {escape(reaches)} reaches</h3>{legend(seeds=True)}</div>
    <div class="scroll">{region_chart(regions, "farm")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Every class within every reach, crop level</h3></div>
    {region_class_table(regions, "image")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Every class within every reach, farm level</h3></div>
    {region_class_table(regions, "farm")}
  </div>
</section>
""",
        f"""
<section>
  <h2>The curves behind the numbers</h2>
  <p class="lede">Precision against recall at every threshold, crop level. The horizontal tick is
  the prevalence: a curve hugging it is no better than picking at random.</p>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level</h3>{legend()}</div>
    <div class="scroll">{pr_panels(crops, crop_truth, "image")}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Where the classes leak</h2>
  <p class="lede">Average precision never fixes an operating point, so it cannot say <em>where</em>
  a class goes wrong. These are the annotated label against the model's top class, shaded by row
  fraction. Paddock and Other/Industrial are not model outputs at all - every one of those crops
  is forced onto some class, and the column it lands in is what drags that class's precision down.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3>
      <div class="ramp">less{ramp}more of the row</div>
    </div>
    <div class="confusions">
      {confusion_table(confusion['image'].loc['old'], models['old'])}
      {confusion_table(confusion['image'].loc['new'], models['new'])}
    </div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; each annotated class against the farm's top class</h3>
      <div class="ramp">less{ramp}more of the row</div>
    </div>
    <p class="lede">Farm labels are multi-label, so a farm carrying two classes appears in both
    rows, each paired with the same single top class &mdash; the diagonal here means &ldquo;this
    class is <em>the</em> top class for its farm&rdquo;, which a farm with two classes can satisfy
    for only one of them. Row totals therefore count annotated (farm, class) pairs, not farms.
    The top-k agreement above uses the looser rule: a farm counts as a hit when <em>any</em> of
    its classes is in the top k.</p>
    <div class="confusions">
      {confusion_table(confusion['farm'].loc['old'], models['old'])}
      {confusion_table(confusion['farm'].loc['new'], models['new'])}
    </div>
  </div>
</section>
""",
        f"""
<section>
  <h2>Full results</h2>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3></div>
    {metrics_table(metrics, models, "image")}
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
    <p><strong>Ground truth.</strong> The relabelling workbooks in <code>labelled_sheets/</code>:
    {escape(len(crops))} crops carrying a single class, and {escape(len(farms))} farms carrying
    multi-label X marks. Crops marked Ambiguous or Multiple Classes are dropped from every class.</p>
    <p><strong>Predictions.</strong> The generalisation pass forces a single dummy label on every
    row, so the labels saved beside the predictions are meaningless; only the saved
    <code>y_hat</code> is used, reordered onto <code>dataset.csv</code> by the saved index.</p>
    <p><strong>Class order.</strong> The head's output columns are the sorted training vocabulary.
    Both checkpoints were trained on the same ten classes in the same order, checked against each
    model's own <code>train_df.csv</code> before scoring.</p>
    <p><strong>Farm scores.</strong> A farm's score for a class is the {escape(aggregation)} over its
    crops - the question is whether the class would surface for that farm at all.</p>
    <p><strong>Not evaluable.</strong> No annotated crop or farm carries <code>aqua</code>, and
    neither model has a <code>goat</code> output, so both classes are reported blank rather than
    counted as zero.</p>
    <p><strong>Uncertainty.</strong> 2000 bootstrap resamples of the evaluation units, percentile
    intervals. Several classes rest on fewer than twenty positives, and their intervals say so.</p>
  </div>
</section>
""",
    ]

    return (
        "<title>ComFe generalisation evaluation</title>"
        f"<style>{STYLE}</style>"
        f'<div class="page">{"".join(sections)}</div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--file", type=Path, default=None, help="where to write the dashboard")
    args = parser.parse_args()

    destination = args.file or args.output / "gen_evaluation_dashboard.html"
    destination.write_text(build(args.output), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
