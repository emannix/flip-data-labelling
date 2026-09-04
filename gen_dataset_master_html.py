"""Render the master dataset as a single self-contained HTML summary.

Reads only `original_master_2026_09_04/dataset.csv` — the file `gen_dataset_master.py`
writes — so the summary never touches the source datasets or the imagery:

    .venv/bin/python gen_dataset_master.py
    .venv/bin/python gen_dataset_master_html.py
    xdg-open original_master_2026_09_04/summary.html

The question it answers is what is actually in each split, counted three ways, because
the dataset counts three different things and they do not move together:

    images   one row — one crop or one whole-farm photograph
    groups   dataset x identifier x imagery source — the unit training draws on: one
             farm in one aerial capture for autocrops, the single image for the others
    farms    distinct farm_uid — autocrops and generalisation only; the historical
             pipeline carries no farm identity at all

A split with many images can hold few farms, and a class can look well represented by
image while resting on a handful of farms, so every class breakdown is given by image
*and* by group rather than picking one.

Charts are inline SVG and CSS built here rather than by a plotting library, so the page
has no external dependencies. Colours are the validated default data-viz palette used
unchanged: categorical slots 1-3 for the three source datasets (the documented
all-pairs-safe prefix), and the blue sequential ramp for the class matrices. Slot 3 sits
below 3:1 on the light surface, so every chart carries direct labels and a table view
rather than leaning on hue.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pandas as pd

# The provenance prose lives in gen_dataset_master.py, beside the code that acts on it,
# so the dashboard and the dataset's own README cannot drift apart.
from gen_dataset_master import (
    NOT_INDEPENDENT,
    SOURCE_NOTES,
    THIS_REPO,
    THIS_REPO_NAME,
)

DEFAULT_DATASET = Path("original_master_2026_09_04/dataset.csv")

# Splits in reading order, with the role each plays. The overlap splits are not test
# sets — they are what was pulled out of somewhere else — so they are banded separately
# rather than being left to look like four more evaluation sets.
SPLITS = [
    ("train", "training"),
    ("val", "training"),
    ("train_overlap", "pulled from training"),
    ("val_overlap", "pulled from training"),
    ("test_autocrops", "test"),
    ("test_autocrop_gen_vic", "test"),
    ("test_gen_original", "test"),
    ("test_gen_original_overlap", "pulled from test_gen_original"),
    ("test_original", "test"),
]
SPLIT_ORDER = [name for name, _ in SPLITS]
ROLE_OF = dict(SPLITS)

# Categorical slots 1-3 from the reference palette, light value first. This prefix is
# the documented all-pairs-safe one; a fourth source would need folding or faceting.
SOURCES = {
    "historical": ("#1baf7a", "#199e70"),
    "autocrops": ("#2a78d6", "#3987e5"),
    "generalisation": ("#eb6834", "#d95926"),
}
SOURCE_ORDER = list(SOURCES)

# Blue sequential ramp: light mode runs light->dark from the surface, dark mode runs
# dark->light, so "more" is further from the page in both.
HEAT_BREAKS = [0.03, 0.08, 0.16, 0.30, 0.50, 0.75]

CLASSES = [
    "aqua", "backyardpig", "beef", "commercialpig", "dairy", "freerangepig",
    "goats", "horse", "poultry", "residential", "sheep",
    "paddock", "other_industrial",
]

UNITS = {
    "images": "one row — a crop, or a whole-farm photograph",
    "groups": "dataset x identifier x imagery source; at most 10 rows",
    "farms": "distinct farm_uid; historical carries none",
}


# --------------------------------------------------------------------------------------
# svg helpers
# --------------------------------------------------------------------------------------


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
    """A bar anchored at its start with only its data end rounded.

    Returns the finished <path> element, not bare path data: an unwrapped `d` string
    dropped into the svg renders as nothing at all, and since this shape is only used for
    the *last* segment of a row, that failure hides whole bars rather than looking broken.
    """
    radius = max(0.0, min(radius, width, height / 2))
    return (
        f'<path d="M{x:g},{y:g} H{x + width - radius:g} '
        f"Q{x + width:g},{y:g} {x + width:g},{y + radius:g} "
        f"V{y + height - radius:g} "
        f"Q{x + width:g},{y + height:g} {x + width - radius:g},{y + height:g} "
        f'H{x:g} Z" />'
    )


def nice_ticks(top: float, count: int = 4) -> list[float]:
    """Tick values from 0 to a round ceiling that is always >= `top`.

    The ceiling has to clear the data, not stop just short of it: the bars are scaled by
    the last tick, so a ceiling below the maximum silently pushes the widest rows — and
    their labels — off the right-hand edge of the viewBox.
    """
    if top <= 0:
        return [0]
    raw = top / count
    magnitude = 10 ** (len(str(int(raw))) - 1)
    step = next(
        (m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), raw
    )
    ticks, value = [], 0.0
    while value < top:
        ticks.append(value)
        value += step
    ticks.append(value)
    return ticks


def inline_md(source: str) -> str:
    """The inline markdown the provenance notes use: links, bold, code, emphasis.

    Escaping runs first and introduces none of the marker characters, so the patterns
    below cannot match anything the escaping produced.
    """
    out = escape(source)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", out)
    return out


def block_md(source: str) -> str:
    """Blank-line-separated paragraphs and `- ` bullet lists. No other block forms."""
    blocks = []
    for block in source.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if all(line.startswith("- ") for line in lines):
            items = "".join(f"<li>{inline_md(line[2:])}</li>" for line in lines)
            blocks.append(f"<ul>{items}</ul>")
        else:
            blocks.append(f"<p>{inline_md(' '.join(lines))}</p>")
    return "".join(blocks)


def heat_index(fraction: float) -> int:
    for index, edge in enumerate(HEAT_BREAKS):
        if fraction < edge:
            return index
    return len(HEAT_BREAKS)


# --------------------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------------------


def has_farm(df: pd.DataFrame) -> pd.Series:
    return df["farm_uid"].notna() & (df["farm_uid"].astype(str).str.strip() != "")


def count(df: pd.DataFrame, unit: str) -> int:
    """Rows, distinct groups, or distinct farms in a frame."""
    if unit == "images":
        return len(df)
    if unit == "groups":
        return df["group_id"].nunique()
    return df.loc[has_farm(df), "farm_uid"].nunique()


def by_split_source(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """unit counts, splits down the side and source datasets across."""
    table = pd.DataFrame(0, index=SPLIT_ORDER, columns=SOURCE_ORDER, dtype=int)
    for (split, source), group in df.groupby(["split", "source_dataset"]):
        if split in table.index and source in table.columns:
            table.loc[split, source] = count(group, unit)
    return table


def exploded(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (row, class), so a multi-label row counts under each of its classes."""
    labelled = df[df["n_classes"] > 0].copy()
    labelled["cls"] = labelled["crop_classes"].str.split(",")
    return labelled.explode("cls").reset_index(drop=True)


def class_by_split(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """unit counts, classes down the side and splits across."""
    rows = exploded(df)
    table = pd.DataFrame(0, index=CLASSES, columns=SPLIT_ORDER, dtype=int)
    for (cls, split), group in rows.groupby(["cls", "split"]):
        if cls in table.index and split in table.columns:
            table.loc[cls, split] = count(group, unit)
    return table


# --------------------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------------------


# A segment this thin is still drawn at this width, so a small source never vanishes
# from a bar. It overstates the smallest categories by a few pixels — generalisation is
# 2% of train and 86 rows of val — so every stacked chart ships the exact counts in a
# table beside it rather than leaving the bar as the only record.
MIN_SEGMENT = 5.0


def stacked_chart(table: pd.DataFrame, unit: str, *, width: float = 900,
                  label_w: float = 210, pad_r: float = 78, row_h: float = 26,
                  gap: float = 9, axis_title: str = "", label: str = "") -> str:
    """Horizontal stacked bars: one row per index entry, segmented by source dataset.

    Every segment wide enough to hold its own number is labelled in place and the row
    total sits at the end, so the chart reads without the legend — which matters because
    one of the three hues is deliberately low-contrast on the light surface.
    """
    top = 30
    plot_w = width - label_w - pad_r
    height = top + len(table) * (row_h + gap) + 30
    ticks = nice_ticks(max(table.sum(axis=1).max(), 1))
    scale = plot_w / max(ticks[-1], 1)

    parts = []
    for value in ticks:
        x = label_w + value * scale
        parts.append(
            f'<line class="grid" x1="{x:g}" y1="{top - 6:g}" x2="{x:g}" '
            f'y2="{height - 30:g}" />'
        )
        parts.append(text(x, top - 12, f"{value:,.0f}", "tick", "middle"))

    for index, name in enumerate(table.index):
        y = top + index * (row_h + gap)
        total = int(table.loc[name].sum())
        segments = int((table.loc[name] > 0).sum())
        parts.append(text(label_w - 12, y + row_h / 2 + 4, name, "rowlabel", "end"))
        cursor = 0.0
        for position, source in enumerate(SOURCE_ORDER):
            value = int(table.loc[name, source])
            if value <= 0:
                continue
            # 2px surface gap between segments, and a floor so a tiny source stays visible
            drawn = max(value * scale - 2, MIN_SEGMENT)
            last = position == max(
                i for i, s in enumerate(SOURCE_ORDER) if table.loc[name, s] > 0
            )
            shape = (
                rounded_right(label_w + cursor, y, drawn, row_h)
                if last
                else f'<rect x="{label_w + cursor:g}" y="{y:g}" width="{drawn:g}" '
                     f'height="{row_h:g}" />'
            )
            parts.append(
                f'<g class="mark" fill="var(--src-{source})">'
                f'{tip(f"{name} · {source}: {value:,} {unit}")}{shape}</g>'
            )
            if drawn > 46 and segments > 1:
                parts.append(text(
                    label_w + cursor + drawn / 2, y + row_h / 2 + 4,
                    f"{value:,}", "inbar", "middle",
                ))
            cursor += drawn + 2
        parts.append(text(
            label_w + cursor + 7, y + row_h / 2 + 4, f"{total:,}", "note-strong",
        ))
    if axis_title:
        parts.append(text(label_w, height - 8, axis_title, "axistitle"))
    return svg(width, height, "".join(parts), label or f"{unit} per row by source dataset")


def source_table(table: pd.DataFrame, unit: str, index_label: str) -> str:
    """The exact counts behind a stacked chart, since the thinnest segments are floored."""
    head = (
        f"<tr><th>{escape(index_label)}</th>"
        + "".join(f"<th>{escape(s)}</th>" for s in SOURCE_ORDER)
        + "<th>total</th></tr>"
    )
    body = []
    for name in table.index:
        cells = "".join(
            f"<td>{int(table.loc[name, s]):,}</td>" if table.loc[name, s] else
            '<td class="muted">&middot;</td>'
            for s in SOURCE_ORDER
        )
        body.append(
            f'<tr><th scope="row">{escape(name)}</th>{cells}'
            f"<td>{int(table.loc[name].sum()):,}</td></tr>"
        )
    totals = "".join(f"<td>{int(table[s].sum()):,}</td>" for s in SOURCE_ORDER)
    body.append(
        f'<tr class="total"><th scope="row">all</th>{totals}'
        f'<td>{int(table.to_numpy().sum()):,}</td></tr>'
    )
    return (
        f'<details><summary>Exact {escape(unit)} counts</summary>'
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></details>'
    )


def stacked_bars(table: pd.DataFrame, unit: str) -> str:
    return stacked_chart(table, unit, axis_title=unit,
                         label=f"{unit} per split by source dataset")


def class_table(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Totals per class across the whole dataset, by which source supplied them."""
    rows = exploded(df)
    table = pd.DataFrame(0, index=CLASSES, columns=SOURCE_ORDER, dtype=int)
    for (cls, source), group in rows.groupby(["cls", "source_dataset"]):
        if cls in table.index:
            table.loc[cls, source] = count(group, unit)
    return table.loc[table.sum(axis=1).sort_values(ascending=False).index]


def group_size_table(df: pd.DataFrame) -> pd.DataFrame:
    """How many groups hold how many images, by source."""
    sizes = df.groupby(["source_dataset", "group_id"]).size().reset_index(name="size")
    buckets = [(1, 1), (2, 3), (4, 6), (7, 9), (10, 14), (15, 19), (20, 10**9)]
    names = ["1", "2-3", "4-6", "7-9", "10-14", "15-19", "20+"]
    table = pd.DataFrame(0, index=names, columns=SOURCE_ORDER, dtype=int)
    for source, group in sizes.groupby("source_dataset"):
        for name, (low, high) in zip(names, buckets):
            table.loc[name, source] = int(
                ((group["size"] >= low) & (group["size"] <= high)).sum()
            )
    return table

def class_matrix(table: pd.DataFrame, unit: str) -> str:
    """Classes x splits, shaded by each class's share of its split, labelled with counts.

    Shading is column-wise on purpose: the question is what a given split is made of, and
    a raw-count shading would simply redraw the size difference between train and the
    test sets.
    """
    totals = table.sum(axis=0)
    head = ['<th class="corner">class</th>']
    for split in table.columns:
        head.append(f'<th><span>{escape(split)}</span></th>')
    head.append('<th class="corner total-col">total</th>')

    body = []
    for cls in table.index:
        cells = []
        for split in table.columns:
            value = int(table.loc[cls, split])
            share = value / totals[split] if totals[split] else 0.0
            step = heat_index(share)
            flip = " cell-flip" if step >= 4 else ""
            label = f"{value:,}" if value else "·"
            cells.append(
                f'<td class="cell{flip}" style="--step:{step}" '
                f'title="{escape(f"{cls} · {split}: {value:,} {unit} ({share:.1%} of the split)")}">'
                f"{label}</td>"
            )
        row_total = int(table.loc[cls].sum())
        body.append(
            f'<tr><th scope="row">{escape(cls)}</th>{"".join(cells)}'
            f'<td class="total-col">{row_total:,}</td></tr>'
        )
    footer = "".join(f'<td class="total-col">{int(totals[s]):,}</td>' for s in table.columns)
    body.append(
        f'<tr class="total"><th scope="row">any class</th>{footer}'
        f'<td class="total-col">{int(totals.sum()):,}</td></tr>'
    )
    return (
        f'<div class="scroll"><table class="heat"><thead><tr>{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def units_table(df: pd.DataFrame) -> str:
    """Every split counted all three ways, side by side — the page's reference table."""
    head = (
        "<tr><th>split</th><th>role</th><th>images</th><th>groups</th><th>farms</th>"
        "<th>images / group</th><th>sources</th></tr>"
    )
    body = []
    for split in SPLIT_ORDER:
        part = df[df["split"] == split]
        images, groups, farms = (count(part, u) for u in ("images", "groups", "farms"))
        sources = ", ".join(sorted(part["source_dataset"].unique()))
        body.append(
            f'<tr><th scope="row">{escape(split)}</th>'
            f'<td class="left muted">{escape(ROLE_OF[split])}</td>'
            f"<td>{images:,}</td><td>{groups:,}</td>"
            f"<td>{farms:,}</td><td>{images / groups if groups else 0:.1f}</td>"
            f'<td class="left muted">{escape(sources)}</td></tr>'
        )
    body.append(
        f'<tr class="total"><th scope="row">all</th><td class="left"></td>'
        f'<td>{count(df, "images"):,}</td><td>{count(df, "groups"):,}</td>'
        f'<td>{count(df, "farms"):,}</td>'
        f'<td>{count(df, "images") / count(df, "groups"):.1f}</td>'
        f'<td class="left"></td></tr>'
    )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def provenance_table(df: pd.DataFrame) -> str:
    """Which source file fed which split, and how many rows went each way."""
    head = "<tr><th>source dataset</th><th>source file</th><th>split</th><th>images</th></tr>"
    body = []
    grouped = df.groupby(
        ["source_dataset", "source_file", "split"], sort=False
    ).size().reset_index(name="rows")
    grouped = grouped.sort_values(
        ["source_dataset", "source_file", "split"], key=lambda s: s.map(
            lambda v: SPLIT_ORDER.index(v) if v in SPLIT_ORDER else -1
        ) if s.name == "split" else s
    )
    previous = None
    for row in grouped.itertuples():
        key = (row.source_dataset, row.source_file)
        first = key != previous
        previous = key
        dataset = f'<span class="swatch swatch-{row.source_dataset}"></span>{row.source_dataset}' if first else ""
        body.append(
            f'<tr><td class="left">{dataset}</td>'
            f'<td class="left"><code>{escape(row.source_file) if first else ""}</code></td>'
            f'<td class="left">{escape(row.split)}</td><td>{row.rows:,}</td></tr>'
        )
    return (
        f'<div class="scroll"><table class="data prov"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def overlap_table(df: pd.DataFrame) -> str:
    """Which splits share a farm or a source photograph, and how many rows are affected."""
    rows = df[df["overlap_with"].fillna("") != ""]
    if rows.empty:
        return '<p class="lede">No split shares a farm or source image with another.</p>'
    counts = (
        rows.groupby(["split", "overlap_with"]).size().reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    head = "<tr><th>split</th><th>also appears in</th><th>images</th></tr>"
    body = "".join(
        f'<tr><th scope="row">{escape(row.split)}</th>'
        f'<td class="left">{escape(row.overlap_with)}</td><td>{row.rows:,}</td></tr>'
        for row in counts.itertuples()
    )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def status_table(df: pd.DataFrame) -> str:
    """Labelled against placeholder rows, per split."""
    table = pd.crosstab(df["split"], df["label_status"]).reindex(SPLIT_ORDER, fill_value=0)
    statuses = ["labelled"] + [c for c in table.columns if c != "labelled"]
    head = "<tr><th>split</th>" + "".join(f"<th>{escape(s)}</th>" for s in statuses) + "</tr>"
    body = []
    for split in table.index:
        cells = "".join(
            f"<td>{int(table.loc[split, s]) if s in table.columns else 0:,}</td>"
            for s in statuses
        )
        body.append(f'<tr><th scope="row">{escape(split)}</th>{cells}</tr>')
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def multilabel_table(df: pd.DataFrame) -> str:
    """The class combinations carried by rows with more than one class."""
    multi = df[df["n_classes"] > 1]
    if multi.empty:
        return '<p class="lede">No row carries more than one class.</p>'
    counts = (
        multi.groupby(["crop_classes", "source_dataset"]).size()
        .reset_index(name="rows").sort_values("rows", ascending=False)
    )
    head = "<tr><th>classes</th><th>source</th><th>images</th></tr>"
    body = "".join(
        f'<tr><th scope="row">{escape(row.crop_classes)}</th>'
        f'<td class="left"><span class="swatch swatch-{row.source_dataset}"></span>'
        f'{escape(row.source_dataset)}</td><td>{row.rows:,}</td></tr>'
        for row in counts.itertuples()
    )
    return (
        f'<div class="scroll"><table class="data"><thead>{head}</thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def provenance_cards(df: pd.DataFrame) -> str:
    """One card per source: where it is built, what a row is, and how it was split.

    The three were split on three different principles — a curated 2022 hold-out for two
    of them, a geographic hold-out for the third — so a reader who assumes one rule for
    all three will misread the test numbers. That is why this sits on the page at all.
    """
    cards = []
    for number, name in enumerate(SOURCE_ORDER, 1):
        note = SOURCE_NOTES[name]
        part = df[df["source_dataset"] == name]
        landed = part.groupby("split").size().reindex(SPLIT_ORDER).dropna()
        rows = "".join(
            f'<tr><th scope="row">{escape(split)}</th><td>{int(n):,}</td></tr>'
            for split, n in landed.items()
        )
        cards.append(f"""
<div class="card prov-card">
  <div class="card-head">
    <h3><span class="swatch swatch-{name}"></span>{number}. <code>{escape(name)}</code></h3>
    <p class="repo"><a href="{escape(note['repo'])}">{escape(note['repo_name'])}</a>
    &middot; built by {inline_md(note['built_by'])}</p>
  </div>
  <div class="prose">
    <p class="prose-label">What a row is</p>
    {block_md(note['rows'])}
    <p class="prose-label">How it was split, upstream</p>
    {block_md(note['split'])}
  </div>
  <details>
    <summary>Where its {len(part):,} rows landed in this dataset</summary>
    <table class="data mini-split"><tbody>{rows}</tbody></table>
  </details>
</div>
""")
    return "".join(cards)


def tiles(df: pd.DataFrame) -> str:
    training = df[df["split"].isin(["train", "val"])]
    tests = df[df["split"].str.startswith("test_") & ~df["split"].str.endswith("_overlap")]
    pulled = df[df["split"].str.endswith("_overlap")]
    cards = [
        ("images", f"{len(df):,}", f"{count(df, 'groups'):,} groups · {count(df, 'farms'):,} farms"),
        ("train + val", f"{len(training):,}",
         f"{count(training, 'groups'):,} groups · {count(training, 'farms'):,} farms"),
        ("four test sets", f"{len(tests):,}",
         f"{count(tests, 'groups'):,} groups · {count(tests, 'farms'):,} farms"),
        ("pulled to overlap", f"{len(pulled):,}",
         "kept, not deleted — reachable from a test set"),
    ]
    return "".join(
        f'<div class="tile"><p class="tile-label">{escape(label)}</p>'
        f'<p class="tile-value">{escape(value)}</p>'
        f'<p class="tile-note">{escape(note)}</p></div>'
        for label, value, note in cards
    )


def legend() -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><i class="key-line" style="background:var(--src-{source})"></i>'
            f"{escape(source)}</span>"
            for source in SOURCE_ORDER
        )
        + "</div>"
    )


STYLE = """
:root {
  color-scheme: light;
  --plane: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --rule: rgba(11,11,11,0.10);
  --src-autocrops: #2a78d6;
  --src-generalisation: #eb6834;
  --src-historical: #1baf7a;
  --heat-0: #f4f7fb; --heat-1: #cde2fb; --heat-2: #9ec5f4; --heat-3: #6da7ec;
  --heat-4: #3987e5; --heat-5: #256abf; --heat-6: #0d366b;
  --heat-ink: #0b0b0b;
  --heat-ink-flip: #ffffff;
  --inbar-ink: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --rule: rgba(255,255,255,0.10);
    --src-autocrops: #3987e5;
    --src-generalisation: #d95926;
    --src-historical: #199e70;
    --heat-0: #1f2429; --heat-1: #0d366b; --heat-2: #184f95; --heat-3: #256abf;
    --heat-4: #3987e5; --heat-5: #6da7ec; --heat-6: #9ec5f4;
    --heat-ink: #f5f5f0;
    --heat-ink-flip: #0b0b0b;
    --inbar-ink: #0b0b0b;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0d0d0d;
  --surface: #1a1a19;
  --ink: #ffffff;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --grid: #2c2c2a;
  --axis: #383835;
  --rule: rgba(255,255,255,0.10);
  --src-autocrops: #3987e5;
  --src-generalisation: #d95926;
  --src-historical: #199e70;
  --heat-0: #1f2429; --heat-1: #0d366b; --heat-2: #184f95; --heat-3: #256abf;
  --heat-4: #3987e5; --heat-5: #6da7ec; --heat-6: #9ec5f4;
  --heat-ink: #f5f5f0;
  --heat-ink-flip: #0b0b0b;
  --inbar-ink: #0b0b0b;
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
.tile-value { margin: 0; font-size: 26px; font-weight: 600; }
.tile-note { margin: 0; font-size: 12.5px; color: var(--ink-2); }

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
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-left: auto; }
.legend span { display: flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--ink-2); }
.key-line { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
.scroll { overflow-x: auto; }
.swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; margin-right: 6px; }
.swatch-autocrops { background: var(--src-autocrops); }
.swatch-generalisation { background: var(--src-generalisation); }
.swatch-historical { background: var(--src-historical); }

.mark { cursor: default; }
.mark:hover path, .mark:hover rect { opacity: 0.78; }
.grid { stroke: var(--grid); stroke-width: 1; }
text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.rowlabel { font-size: 12.5px; fill: var(--ink-2); }
.note-strong { font-size: 11.5px; fill: var(--ink-2); font-weight: 600; font-variant-numeric: tabular-nums; }
.inbar { font-size: 11px; fill: var(--inbar-ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.axistitle { font-size: 11.5px; fill: var(--muted); }

table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
.data { width: 100%; }
.data th, .data td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); }
.data thead th { color: var(--muted); font-weight: 500; font-size: 11.5px; white-space: nowrap; }
.data tbody th { text-align: left; font-weight: 500; white-space: nowrap; }
.data td.left { text-align: left; }
.data tr.total th, .data tr.total td { font-weight: 650; border-top: 1px solid var(--axis); }
.muted { color: var(--muted); }
.prov code { font-size: 12px; }

.heat { border-collapse: separate; border-spacing: 0; }
.heat th, .heat td { padding: 0; }
.heat thead th { font-size: 10.5px; color: var(--muted); font-weight: 500; padding: 0 0 6px; }
.heat thead th span { display: block; writing-mode: vertical-rl; transform: rotate(180deg); }
.heat .corner { writing-mode: horizontal-tb; text-align: right; padding-right: 10px; white-space: nowrap; }
.heat tbody th {
  text-align: right; font-size: 12px; font-weight: 500; color: var(--ink-2);
  padding-right: 10px; white-space: nowrap;
}
.heat .cell {
  width: 58px; height: 26px; text-align: center; font-size: 11px; color: var(--heat-ink);
  background: var(--heat-0); border: 2px solid var(--surface); border-radius: 3px;
}
.heat .cell-flip { color: var(--heat-ink-flip); }
.heat .cell[style*="--step:1"] { background: var(--heat-1); }
.heat .cell[style*="--step:2"] { background: var(--heat-2); }
.heat .cell[style*="--step:3"] { background: var(--heat-3); }
.heat .cell[style*="--step:4"] { background: var(--heat-4); }
.heat .cell[style*="--step:5"] { background: var(--heat-5); }
.heat .cell[style*="--step:6"] { background: var(--heat-6); }
.heat .total-col {
  text-align: right; padding-left: 12px; font-size: 12px; color: var(--ink-2);
  font-weight: 600;
}
.heat tr.total th, .heat tr.total td { padding-top: 6px; font-weight: 650; color: var(--ink); }

.ramp { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--muted); }
.ramp i { width: 22px; height: 10px; border-radius: 2px; display: inline-block; }

.prov-card .card-head { align-items: baseline; gap: 6px 16px; }
.prov-card h3 { display: flex; align-items: center; }
.repo { margin: 0; font-size: 12.5px; color: var(--muted); margin-left: auto; }
.prose { display: flex; flex-direction: column; gap: 10px; max-width: 82ch; }
.prose p { margin: 0; color: var(--ink-2); font-size: 13.5px; }
.prose strong { color: var(--ink); font-weight: 600; }
.prose ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; }
.prose li { color: var(--ink-2); font-size: 13.5px; }
.prose-label {
  font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted) !important; margin-top: 4px !important;
}
details { font-size: 13px; color: var(--ink-2); }
summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
.mini-split { width: auto; margin-top: 8px; }
.mini-split th { padding-left: 0; }

.notes { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px 32px; }
.notes p { margin: 0; color: var(--ink-2); font-size: 13.5px; }
.notes strong { color: var(--ink); font-weight: 600; }
code {
  font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.9em;
  background: var(--plane); border: 1px solid var(--rule); border-radius: 3px; padding: 0 4px;
}
:focus-visible { outline: 2px solid var(--src-autocrops); outline-offset: 2px; }
"""


def build(dataset: Path) -> str:
    df = pd.read_csv(dataset, low_memory=False)
    df["overlap_with"] = df["overlap_with"].fillna("")
    df["farm_uid"] = df["farm_uid"].fillna("")

    ramp = "".join(f'<i style="background:var(--heat-{step})"></i>' for step in range(7))
    sections = [
        f"""
<header class="masthead">
  <p class="eyebrow">FLIP &middot; {escape(dataset.parent.name)}</p>
  <h1>What is in each split of the master dataset?</h1>
  <p class="standfirst">Three source datasets combined into one training corpus and four
  test sets. The dataset counts three different things &mdash; images, label groups and
  farms &mdash; and they do not move together, so every breakdown below is given by more
  than one of them. Built from <code>{escape(dataset)}</code> by
  <a href="{THIS_REPO}">{THIS_REPO_NAME}</a>.</p>
</header>
""",
        f'<div class="tiles">{tiles(df)}</div>',
        f"""
<section>
  <h2>Where the data came from</h2>
  <p class="lede">Three datasets, introduced in the order they were derived: the original
  pipeline first, then the two builds cut from its imagery. They were each split on a
  <strong>different principle</strong> &mdash; a curated 2022 FarmFinder hold-out for the
  first two, a geographic NSW/VIC hold-out for the third &mdash; so read each one's rule
  before comparing their test numbers.</p>
  {provenance_cards(df)}
  <div class="card">
    <div class="card-head"><h3>These sources are not independent</h3></div>
    <div class="prose">{block_md(NOT_INDEPENDENT)}</div>
  </div>
</section>
""",
        f"""
<section>
  <h2>The three units</h2>
  <p class="lede">An <strong>image</strong> is one row &mdash; a building crop, or a
  whole-farm photograph. A <strong>group</strong> is
  <code>dataset &times; identifier &times; imagery source</code>, the unit training draws
  on, so it follows the level the labels were assigned at. For <code>autocrops</code>, one
  farm-level label covers every crop of a farm in <em>one aerial capture</em> &mdash; a
  farm flown twice is two groups, because two flights are two photographs &mdash; capped
  at <strong>10</strong> rows, the builder's limit of ten building clusters per farm per
  capture. For <code>generalisation</code> and <code>historical</code>, each image carries
  its own label, so each is its own group. A <strong>farm</strong> is a distinct
  <code>farm_uid</code>, which only <code>autocrops</code> and
  <code>generalisation</code> carry &mdash; the historical pipeline has no farm identity,
  so it contributes images and groups but no farms.</p>
  <div class="card">{units_table(df)}</div>
</section>
""",
        f"""
<section>
  <h2>What each split is made of</h2>
  <p class="lede">The same nine splits counted three ways. <code>train</code> is dominated
  by <code>autocrops</code> by image, but the farm count shows how much of that is the
  same farms seen many times over.</p>
  <div class="card">
    <div class="card-head"><h3>Images &mdash; one row each</h3>{legend()}</div>
    <div class="scroll">{stacked_bars(by_split_source(df, "images"), "images")}</div>
    {source_table(by_split_source(df, "images"), "images", "split")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Groups &mdash; the training unit</h3>{legend()}</div>
    <div class="scroll">{stacked_bars(by_split_source(df, "groups"), "groups")}</div>
    {source_table(by_split_source(df, "groups"), "groups", "split")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farms &mdash; distinct <code>farm_uid</code></h3>{legend()}</div>
    <div class="scroll">{stacked_bars(by_split_source(df, "farms"), "farms")}</div>
    {source_table(by_split_source(df, "farms"), "farms", "split")}
  </div>
</section>
""",
        f"""
<section>
  <h2>Classes across the splits</h2>
  <p class="lede">A row carrying more than one class is counted under each of them. Cells
  are shaded by the class's share <em>of its own split</em>, not by raw count &mdash;
  otherwise the shading would just redraw the size gap between <code>train</code> and the
  test sets. The number in each cell is the count.</p>
  <div class="card">
    <div class="card-head">
      <h3>By image</h3>
      <div class="ramp">share of split{ramp}</div>
    </div>
    {class_matrix(class_by_split(df, "images"), "images")}
  </div>
  <div class="card">
    <div class="card-head">
      <h3>By group &mdash; how many independent labels back each class</h3>
      <div class="ramp">share of split{ramp}</div>
    </div>
    {class_matrix(class_by_split(df, "groups"), "groups")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Which source supplies each class</h3>{legend()}</div>
    <div class="scroll">{stacked_chart(class_table(df, "images"), "images", label_w=150, row_h=22, gap=7, axis_title="images", label="images per class by source dataset")}</div>
    {source_table(class_table(df, "images"), "images", "class")}
  </div>
</section>
""",
        f"""
<section>
  <h2>Why images and groups diverge</h2>
  <p class="lede">Group size is the whole of the difference. Every
  <code>generalisation</code> and <code>historical</code> group is a single image, while
  an <code>autocrops</code> group is one farm in one capture and averages several crops
  &mdash; so sampling by group and sampling by image give quite different class balances.
  Nothing exceeds ten.</p>
  <div class="card">
    <div class="card-head"><h3>Images per group</h3>{legend()}</div>
    <div class="scroll">{stacked_chart(group_size_table(df), "groups", width=620, label_w=120, row_h=24, gap=8, axis_title="groups", label="group size distribution by source dataset")}</div>
    {source_table(group_size_table(df), "groups", "images per group")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Rows carrying more than one class</h3></div>
    {multilabel_table(df)}
  </div>
</section>
""",
        f"""
<section>
  <h2>Provenance</h2>
  <p class="lede">Every source file, and where its rows ended up. Each row of each source
  appears exactly once in the output.</p>
  <div class="card">{provenance_table(df)}</div>
</section>
""",
        f"""
<section>
  <h2>Overlap and label status</h2>
  <p class="lede">Splits that share a farm or a source photograph, and the rows whose
  label is a placeholder rather than a class. The <code>*_overlap</code> splits are rows
  pulled out of training for reaching a test set &mdash; they were kept, not deleted.
  <code>test_gen_original</code> and <code>test_autocrop_gen_vic</code> share imagery by
  design and must never have their scores pooled.</p>
  <div class="card">
    <div class="card-head"><h3>Shared farms and source images</h3></div>
    {overlap_table(df)}
  </div>
  <div class="card">
    <div class="card-head"><h3>Label status</h3></div>
    {status_table(df)}
  </div>
</section>
""",
        """
<section>
  <h2>Reading these numbers</h2>
  <div class="notes">
    <p><strong>False is not a verified negative.</strong> The three sources carry different
    class lists, and a class a source never assessed is written <code>False</code>.
    <code>paddock</code> and <code>other_industrial</code> are only ever true on
    <code>generalisation</code> rows.</p>
    <p><strong>Farms are not comparable across sources.</strong> Only
    <code>autocrops</code> and <code>generalisation</code> carry a <code>farm_uid</code>,
    and they share 24 of them. Groups are keyed by source as well as identifier and
    capture, so they never span datasets and never span splits. <code>generalisation</code>
    groups by image because its labels are per-crop; its split integrity comes from its
    own farm-grouped geographic split upstream.</p>
    <p><strong>Class counts double-count multi-label rows.</strong> A row labelled
    <code>dairy,horse</code> appears under both, so the class columns sum to more than the
    split's row count.</p>
    <p><strong>Placeholder rows are still in the file.</strong> The
    <code>notavailable</code> and <code>otherlivestock</code> rows carry no positive label
    at all; filter on <code>label_status</code> before training or scoring.</p>
  </div>
</section>
""",
    ]
    return (
        "<title>FLIP master dataset summary</title>"
        f"<style>{STYLE}</style>"
        f'<div class="page">{"".join(sections)}</div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"the master dataset.csv to summarise (default: {DEFAULT_DATASET})")
    parser.add_argument("--file", type=Path, default=None,
                        help="where to write the html (default: summary.html beside --dataset)")
    args = parser.parse_args()

    destination = args.file or args.dataset.parent / "summary.html"
    destination.write_text(build(args.dataset), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
