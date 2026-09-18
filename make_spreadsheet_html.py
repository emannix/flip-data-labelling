"""Render the farms of a relabelling workbook as one HTML page for manual review.

Companion to `make_spreadsheet.py`: same `dataset.csv`, same `--sources` filter and the
same `labelled_sheets/` exclusion, so the page covers exactly the farms the workbook does,
in the same order. For each farm it shows

  1. the farm's metadata — the builder's `{farm_uid}_metadata.json` beside the crops
     (stock counts, PIC, WMS raster, property attributes, ...) with the empty fields
     dropped, plus a map link when the farm has a Lat/Long; and
  2. every crop in `dataset.csv` for that farm, grouped by the aerial raster it was cut
     from (`ecw_stem`), the farm's own WMS raster first, then by capture date.

The page lives with the data: by default it is written as `review.html` at the top of the
build directory (beside `dataset.csv`), and because browsers cannot show the JPEG-in-GeoTIFF
crops, each one gets a JPEG preview beside it (`..._building_0.tif` -> `..._building_0.jpg`)
that the page references by relative path. Nothing is copied elsewhere, and re-running only
makes the previews that are missing. The builder's own tooling globs `*.tif` and the
per-farm metadata json, so the previews do not disturb it. `--embed` inlines the previews
as data URIs instead (one self-contained file, but large). Clicking a crop opens it large
in a lightbox on the same page; arrow keys step through the farm's crops, Esc closes.

    .venv/bin/python make_spreadsheet_html.py \\
        --input /home/mannixe/FLIP/flip-geoimage-dataset-builder/output_gen_extra_test/dataset.csv \\
        --sources vic_pics
    xdg-open /home/mannixe/FLIP/flip-geoimage-dataset-builder/output_gen_extra_test/review.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from make_spreadsheet import DEFAULT_INPUT, DEFAULT_LABELLED, DEFAULT_SOURCES, labelled_farms, optional_path

DEFAULT_OUTPUT_NAME = "review.html"  # written beside dataset.csv unless --output says otherwise
DEFAULT_THUMBNAIL = 1000
DEFAULT_QUALITY = 82
DEFAULT_WORKERS = 8

# dataset.csv columns describing the crop rather than the farm, shown under each image.
CROP_FIELDS = ["crop_width", "crop_height", "crop_res", "crop_std", "crop_fill_frac"]
# dataset.csv columns describing the farm, shown in the farm header when present.
FARM_FIELDS = ["Farm_type", "processed_class", "source", "PFI", "pic", "WMS_NAME", "training_overlap_frac", "split"]
# metadata.json keys that are either huge, redundant with the header, or internal.
SKIP_META = {"geometry", "farm_index", "index", "farm_uid"}
# Values the builder writes for "nothing here".
EMPTY_VALUES = {"", "NaT", "nan", "NaN", "None", None}

DATE_RE = re.compile(r"(\d{4})(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d{2})", re.I)
MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), start=1)}


# --- selection: the same farms as make_spreadsheet.py -------------------------------


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def filter_sources(rows: list[dict], sources: list[str]) -> list[dict]:
    wanted = tuple(source.lower() for source in sources)
    return [row for row in rows if row["source"].lower().startswith(wanted)]


def drop_labelled(rows: list[dict], farms: set[str]) -> list[dict]:
    return [row for row in rows if row["farm_uid"] not in farms and row["PFI"] not in farms]


def group_by_farm(rows: list[dict]) -> dict[str, list[dict]]:
    """Rows per farm_uid, in first-seen order — the order the workbook uses."""
    farms: dict[str, list[dict]] = {}
    for row in rows:
        farms.setdefault(row["farm_uid"], []).append(row)
    return farms


# --- metadata --------------------------------------------------------------------------


def load_metadata(build_dir: Path, farm_uid: str) -> dict:
    path = build_dir / farm_uid / f"{farm_uid}_metadata.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def is_empty(value) -> bool:
    if isinstance(value, (list, dict)):
        return not value
    if isinstance(value, float) and value != value:  # NaN
        return True
    return value in EMPTY_VALUES


def format_value(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, list):
        return ", ".join(format_value(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={format_value(item)}" for key, item in value.items())
    return str(value)


def metadata_rows(meta: dict) -> list[tuple[str, str]]:
    """Non-empty scalar-ish fields of the metadata json, in the builder's order."""
    return [
        (key, format_value(value))
        for key, value in meta.items()
        if key not in SKIP_META and not is_empty(value)
    ]


def map_link(lat: str, long: str) -> str:
    try:
        lat_f, long_f = float(lat), float(long)
    except (TypeError, ValueError):
        return ""
    return f"https://www.google.com/maps/@{lat_f:.6f},{long_f:.6f},600m/data=!3m1!1e3"


# --- imagery ---------------------------------------------------------------------------


def capture_date(stem: str) -> tuple:
    match = DATE_RE.search(stem)
    if not match:
        return (9999, 99, 99)
    year, month, day = match.groups()
    return (int(year), MONTHS[month.lower()], int(day))


def group_by_source(rows: list[dict]) -> list[tuple[str, bool, list[dict]]]:
    """(raster stem, is the farm's own WMS raster, crops) — WMS raster first, then by date."""
    sources: dict[str, list[dict]] = {}
    for row in rows:
        sources.setdefault(row.get("ecw_stem") or "(unknown raster)", []).append(row)
    groups = []
    for stem, crops in sources.items():
        wms = any(crop.get("wms_match") == "True" for crop in crops)
        crops.sort(key=lambda crop: (int(crop["building_cluster"]) if crop["building_cluster"].isdigit() else 0, crop["image_path"]))
        groups.append((stem, wms, crops))
    groups.sort(key=lambda group: (not group[1], capture_date(group[0]), group[0]))
    return groups


def thumbnail_path(build_dir: Path, image_path: str) -> Path:
    """The preview sits beside its crop, same name with a .jpg suffix."""
    return build_dir / Path(image_path).with_suffix(".jpg")


def make_thumbnail(job: tuple[Path, Path, int, int]) -> str | None:
    """Write one thumbnail; returns an error message rather than raising, so one bad
    crop does not stop the run."""
    source, target, size, quality = job
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.thumbnail((size, size))
            image.convert("RGB").save(target, "JPEG", quality=quality, optimize=True)
    except Exception as error:  # noqa: BLE001 - reported, not fatal
        return f"{source}: {error}"
    return None


def build_thumbnails(rows: list[dict], build_dir: Path, size: int, quality: int, workers: int, force: bool) -> list[str]:
    jobs = []
    for row in rows:
        target = thumbnail_path(build_dir, row["image_path"])
        if force or not target.exists():
            jobs.append((build_dir / row["image_path"], target, size, quality))
    if not jobs:
        print(f"  previews: all {len(rows)} already beside their crops in {build_dir}")
        return []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        errors = [error for error in pool.map(make_thumbnail, jobs, chunksize=8) if error]
    print(f"  previews: {len(jobs) - len(errors)} written beside their crops in {build_dir}, {len(errors)} failed")
    return errors


def image_src(build_dir: Path, output: Path, image_path: str, embed: bool) -> str:
    target = thumbnail_path(build_dir, image_path)
    if not target.exists():
        return ""
    if embed:
        return "data:image/jpeg;base64," + base64.b64encode(target.read_bytes()).decode("ascii")
    return html.escape(os.path.relpath(target, output.parent))


# --- html ------------------------------------------------------------------------------

STYLE = """
:root { --ink:#1f2933; --muted:#616e7c; --line:#d9dee3; --band:#f4f6f8; --head:#44546a; --accent:#2a78d6; --wms:#1baf7a; }
* { box-sizing:border-box; }
body { margin:0; font:14px/1.45 Arial, Helvetica, sans-serif; color:var(--ink); background:#fff; }
a { color:var(--accent); }
header { position:sticky; top:0; z-index:5; background:var(--head); color:#fff; padding:8px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; }
header input { font:inherit; padding:4px 8px; border:1px solid #fff; border-radius:3px; min-width:260px; }
header .count { color:#cfd6e0; }
main { padding:0 16px 48px; }
table.index { border-collapse:collapse; margin:16px 0; font-size:13px; }
table.index th, table.index td { border:1px solid var(--line); padding:3px 8px; text-align:left; }
table.index th { background:var(--band); position:sticky; top:44px; }
table.index tr:nth-child(even) td { background:var(--band); }
section.farm { border-top:3px solid var(--head); margin-top:32px; padding-top:8px; }
section.farm.hidden, table.index tr.hidden { display:none; }
.farm-head { display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
.farm-head h2 { margin:0; font-size:18px; }
.farm-head .nav a { margin-left:8px; font-size:13px; }
.summary { display:flex; gap:6px 18px; flex-wrap:wrap; margin:6px 0 10px; color:var(--muted); font-size:13px; }
.summary b { color:var(--ink); }
.farm-body { display:grid; grid-template-columns:minmax(260px, 340px) 1fr; gap:20px; align-items:start; }
@media (max-width:900px) { .farm-body { grid-template-columns:1fr; } }
details.meta { border:1px solid var(--line); border-radius:4px; padding:6px 10px; background:var(--band); }
details.meta summary { cursor:pointer; font-weight:bold; }
table.meta { border-collapse:collapse; width:100%; font-size:12px; margin-top:6px; }
table.meta th, table.meta td { border-top:1px solid var(--line); padding:2px 6px; text-align:left; vertical-align:top; word-break:break-word; }
table.meta th { color:var(--muted); font-weight:normal; white-space:nowrap; width:38%; }
.source { margin-bottom:18px; }
.source h3 { margin:0 0 6px; font-size:14px; }
.source h3 .badge { display:inline-block; font-size:11px; padding:1px 6px; border-radius:3px; background:var(--wms); color:#fff; margin-left:8px; vertical-align:middle; }
.source h3 .n { color:var(--muted); font-weight:normal; margin-left:8px; }
.crops { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:10px; }
figure { margin:0; border:1px solid var(--line); border-radius:4px; overflow:hidden; background:var(--band); }
figure img { display:block; width:100%; height:auto; background:#e5e8ec; cursor:zoom-in; }
figure .missing { display:flex; align-items:center; justify-content:center; height:180px; color:var(--muted); }
figcaption { padding:4px 8px; font-size:12px; color:var(--muted); }
figcaption b { color:var(--ink); }
footer { color:var(--muted); font-size:12px; margin-top:32px; }
#lightbox { position:fixed; inset:0; z-index:20; background:rgba(15,20,26,.92); display:none; flex-direction:column; align-items:center; justify-content:center; }
#lightbox.open { display:flex; }
#lightbox img { max-width:96vw; max-height:88vh; object-fit:contain; background:#000; box-shadow:0 0 24px #000; cursor:zoom-out; }
#lightbox .cap { color:#fff; font-size:13px; margin-top:8px; text-align:center; max-width:96vw; }
#lightbox .cap b { color:#fff; }
#lightbox button { position:absolute; top:50%; transform:translateY(-50%); background:rgba(255,255,255,.15); color:#fff; border:0; font-size:28px; width:48px; height:72px; cursor:pointer; border-radius:4px; }
#lightbox button:hover { background:rgba(255,255,255,.3); }
#lightbox #prev { left:12px; } #lightbox #next { right:12px; }
#lightbox #close { top:12px; right:12px; transform:none; width:40px; height:40px; font-size:22px; }
"""

SCRIPT = """
const box = document.getElementById('filter');
const sections = [...document.querySelectorAll('section.farm')];
const rows = [...document.querySelectorAll('table.index tbody tr')];
const count = document.getElementById('count');
function apply() {
  const q = box.value.trim().toLowerCase();
  let shown = 0;
  sections.forEach((s, i) => {
    const hit = !q || s.dataset.search.includes(q);
    s.classList.toggle('hidden', !hit);
    rows[i].classList.toggle('hidden', !hit);
    shown += hit;
  });
  count.textContent = shown + ' / ' + sections.length + ' farms';
}
box.addEventListener('input', apply);
apply();

// Lightbox: click a crop to see it large in place; arrows step through the farm's crops.
const lightbox = document.getElementById('lightbox');
const lbImg = lightbox.querySelector('img');
const lbCap = lightbox.querySelector('.cap');
let gallery = [], current = -1;
function show(i) {
  if (i < 0 || i >= gallery.length) return;
  current = i;
  const img = gallery[i];
  lbImg.src = img.src;
  lbCap.innerHTML = img.dataset.caption + ' <span style="opacity:.7">(' + (i + 1) + ' / ' + gallery.length + ')</span>';
  lightbox.classList.add('open');
}
function close() { lightbox.classList.remove('open'); lbImg.src = ''; }
document.querySelectorAll('section.farm').forEach(s => {
  const imgs = [...s.querySelectorAll('figure img')];
  imgs.forEach((img, i) => img.addEventListener('click', () => { gallery = imgs; show(i); }));
});
document.getElementById('prev').addEventListener('click', e => { e.stopPropagation(); show(current - 1); });
document.getElementById('next').addEventListener('click', e => { e.stopPropagation(); show(current + 1); });
document.getElementById('close').addEventListener('click', e => { e.stopPropagation(); close(); });
lightbox.addEventListener('click', e => { if (e.target === lightbox || e.target === lbImg) close(); });
document.addEventListener('keydown', e => {
  if (!lightbox.classList.contains('open')) return;
  if (e.key === 'Escape') close();
  else if (e.key === 'ArrowLeft') show(current - 1);
  else if (e.key === 'ArrowRight') show(current + 1);
});
"""

LIGHTBOX = (
    '<div id="lightbox"><button id="close" title="close (Esc)">&times;</button>'
    '<button id="prev" title="previous (left arrow)">&lsaquo;</button><img alt="">'
    '<div class="cap"></div><button id="next" title="next (right arrow)">&rsaquo;</button></div>'
)


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def render_crop(crop: dict, src: str, stem: str) -> str:
    name = Path(crop["image_path"]).name
    details = " · ".join(
        f"{key.removeprefix('crop_')} {esc(crop[key])}" for key in CROP_FIELDS if crop.get(key) not in EMPTY_VALUES
    )
    if src:
        caption = f"<b>{esc(crop['farm_uid'])}</b> · {esc(stem)} · building {esc(crop['building_cluster'])} · {details}"
        image = (
            f'<img src="{src}" loading="lazy" alt="{esc(name)}" title="{esc(name)} (click to enlarge)" '
            f'data-caption="{esc(caption)}">'
        )
    else:
        image = '<div class="missing">preview missing</div>'
    return (
        f"<figure>{image}<figcaption><b>building {esc(crop['building_cluster'])}</b>"
        f" · {details}<br><span title=\"{esc(crop['image_path'])}\">{esc(name)}</span></figcaption></figure>"
    )


def render_farm(index: int, farm_uid: str, rows: list[dict], meta: dict, n_farms: int, src_for) -> str:
    first = rows[0]
    summary = "".join(
        f"<span>{esc(key)} <b>{esc(first[key])}</b></span>"
        for key in FARM_FIELDS
        if first.get(key) not in EMPTY_VALUES
    )
    link = map_link(first.get("Lat"), first.get("Long"))
    if link:
        summary += f'<span><a href="{link}" target="_blank">map ({esc(first["Lat"])}, {esc(first["Long"])})</a></span>'
    nav = '<span class="nav"><a href="#index">index</a>'
    if index > 0:
        nav += f'<a href="#farm-{index - 1}">previous</a>'
    if index < n_farms - 1:
        nav += f'<a href="#farm-{index + 1}">next</a>'
    nav += "</span>"

    meta_rows = metadata_rows(meta)
    if meta_rows:
        meta_html = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>" for key, value in meta_rows)
        meta_html = (
            f'<details class="meta" open><summary>metadata ({len(meta_rows)} fields)</summary>'
            f'<table class="meta">{meta_html}</table></details>'
        )
    else:
        meta_html = '<details class="meta"><summary>metadata: none found</summary></details>'

    sources_html = []
    for stem, wms, crops in group_by_source(rows):
        badge = '<span class="badge">farm\'s WMS raster</span>' if wms else ""
        figures = "".join(render_crop(crop, src_for(crop["image_path"]), stem) for crop in crops)
        sources_html.append(
            f'<div class="source"><h3>{esc(stem)}{badge}<span class="n">{len(crops)} crops</span></h3>'
            f'<div class="crops">{figures}</div></div>'
        )

    search = " ".join(
        [farm_uid, first.get("source", ""), first.get("Farm_type", ""), first.get("PFI", ""), first.get("pic", "")]
        + [value for _, value in meta_rows]
    ).lower()
    return (
        f'<section class="farm" id="farm-{index}" data-search="{esc(search)}">'
        f'<div class="farm-head"><h2>{index + 1}. {esc(farm_uid)}</h2>{nav}</div>'
        f'<div class="summary"><span>images <b>{len(rows)}</b></span>{summary}</div>'
        f'<div class="farm-body"><div>{meta_html}</div><div>{"".join(sources_html)}</div></div>'
        "</section>"
    )


def render_index(farms: dict[str, list[dict]]) -> str:
    body = "".join(
        f'<tr><td>{i + 1}</td><td><a href="#farm-{i}">{esc(uid)}</a></td><td>{esc(rows[0].get("Farm_type"))}</td>'
        f'<td>{esc(rows[0].get("source"))}</td><td>{esc(rows[0].get("pic") or rows[0].get("PFI"))}</td>'
        f'<td>{len(rows)}</td><td>{len({r.get("ecw_stem") for r in rows})}</td></tr>'
        for i, (uid, rows) in enumerate(farms.items())
    )
    return (
        '<table class="index" id="index"><thead><tr><th>#</th><th>Farm UID</th><th>Farm_type</th>'
        "<th>source</th><th>PIC / PFI</th><th>images</th><th>rasters</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_page(title: str, farms: dict[str, list[dict]], metadata: dict[str, dict], src_for, note: str) -> str:
    sections = "".join(
        render_farm(i, uid, rows, metadata[uid], len(farms), src_for) for i, (uid, rows) in enumerate(farms.items())
    )
    n_images = sum(len(rows) for rows in farms.values())
    return (
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{esc(title)}</title>"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><style>{STYLE}</style></head><body>"
        f"<header><h1>{esc(title)}</h1><input id=\"filter\" type=\"search\" placeholder=\"filter farms: uid, PIC, PFI, class, metadata\">"
        f"<span class=\"count\" id=\"count\"></span><span class=\"count\">{n_images} images</span></header>"
        f"<main>{render_index(farms)}{sections}<footer>{esc(note)}</footer></main>"
        f"{LIGHTBOX}<script>{SCRIPT}</script></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="the builder's dataset.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"html to write (default: {DEFAULT_OUTPUT_NAME} beside the input dataset.csv)",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=DEFAULT_SOURCES,
        help="reaches to keep, matched against the start of the source shapefile name "
        "(pass with no values to keep everything)",
    )
    parser.add_argument(
        "--exclude-labelled",
        type=optional_path,
        default=DEFAULT_LABELLED,
        help="directory of completed labelling workbooks whose farms are already done "
        "(pass an empty string to keep everything)",
    )
    parser.add_argument("--thumbnail", type=int, default=DEFAULT_THUMBNAIL, help="longest side of each preview in px")
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY, help="JPEG quality of the previews")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="processes used to make previews")
    parser.add_argument("--force", action="store_true", help="remake previews that already exist")
    parser.add_argument("--embed", action="store_true", help="inline the previews as data URIs (one large file)")
    parser.add_argument("--title", default=None, help="page title (default: the output file's stem)")
    args = parser.parse_args()

    build_dir = args.input.parent
    rows = read_rows(args.input)
    if args.sources:
        rows = filter_sources(rows, args.sources)
        if not rows:
            parser.error(f"no rows match sources {args.sources}")
    done = labelled_farms(args.exclude_labelled) if args.exclude_labelled else set()
    if done:
        rows = drop_labelled(rows, done)
        if not rows:
            parser.error(f"every matching farm is already labelled in {args.exclude_labelled}")

    farms = group_by_farm(rows)
    metadata = {uid: load_metadata(build_dir, uid) for uid in farms}
    output = args.output or build_dir / DEFAULT_OUTPUT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"{args.input} -> {output}")
    print(f"  sources: {', '.join(sorted({row['source'] for row in rows}))}")
    if args.exclude_labelled:
        print(f"  excluded: {len(done)} farm ids already labelled in {args.exclude_labelled}")
    print(f"  {len(farms)} farms, {len(rows)} images, {sum(bool(meta) for meta in metadata.values())} with metadata")
    errors = build_thumbnails(rows, build_dir, args.thumbnail, args.quality, args.workers, args.force)
    for error in errors[:20]:
        print(f"    {error}")

    title = args.title or f"{build_dir.name} review"
    note = (
        f"Built from {args.input}. Crops are the rows of dataset.csv for these farms (the builder has already "
        f"dropped blank crops); each preview is a {args.thumbnail}px JPEG beside its GeoTIFF. Click a crop to "
        "enlarge it; left/right arrows step through the farm's crops, Esc closes."
    )
    page = render_page(title, farms, metadata, lambda path: image_src(build_dir, output, path, args.embed), note)
    output.write_text(page, encoding="utf-8")
    size_mb = output.stat().st_size / 1e6
    print(f"  wrote {output} ({size_mb:.1f} MB{', embedded' if args.embed else ''})")


if __name__ == "__main__":
    main()
