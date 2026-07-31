"""Build the two-sheet relabelling workbook from the flat per-image spreadsheet.

Sheet 1 ("Farm labels"): one row per Farm PFI, with the original label columns and a
fixed-option confidence dropdown (High/Medium/Low) next to each label.

Sheet 2 ("Image labels"): one row per image, as the input sheet stands, with the
one-hot label columns replaced by a single dropdown holding the existing classes plus
Multiple Classes / Paddock / Other/Industrial / Ambiguous.
"""

import argparse
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

DEFAULT_INPUT = Path("original_new_2026_07_24_generalisation_relabel_clean.xlsx")
DEFAULT_OUTPUT = Path("output/2026_07_24_generalisation_relabel_farm_and_image_bega_nowra_freemans.xlsx")

# Columns in the input sheet that describe the farm/image rather than the label.
KEY_COLUMNS = [
    "source",
    "Farm PFI",
    "source_image_path",
    "Farm_type (previous)",
]
IMAGE_COLUMN = "image_path"
COMMENTS_COLUMN = "Comments"

CONFIDENCE_OPTIONS = ["High", "Medium", "Low"]
EXTRA_IMAGE_OPTIONS = [
    "Multiple Classes",
    "Paddock",
    "Other/Industrial",
    "Ambiguous",
]

FARM_SHEET = "Farm labels"
IMAGE_SHEET = "Image labels"
LISTS_SHEET = "Lists"

HEADER_FONT = Font(bold=True)
HEADER_ALIGNMENT = Alignment(vertical="center", wrap_text=True)


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read the input sheet into a header list and a list of row dicts."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    row_iter = sheet.iter_rows(values_only=True)
    header = [cell for cell in next(row_iter) if cell is not None]
    rows = [
        dict(zip(header, values))
        for values in row_iter
        if any(value is not None for value in values)
    ]
    workbook.close()
    return header, rows


def label_columns(header: list[str]) -> list[str]:
    """The class columns: everything between image_path and Comments."""
    start = header.index(IMAGE_COLUMN) + 1
    end = header.index(COMMENTS_COLUMN)
    return header[start:end]


def group_by_farm(rows: list[dict]) -> list[dict]:
    """One record per Farm PFI, keeping first-seen order and image counts."""
    farms: dict[object, dict] = {}
    for row in rows:
        pfi = row["Farm PFI"]
        farm = farms.get(pfi)
        if farm is None:
            farm = {key: row[key] for key in KEY_COLUMNS}
            farm["n_images"] = 0
            farms[pfi] = farm
        farm["n_images"] += 1
    return list(farms.values())


def write_header(sheet, header: list[str]) -> None:
    sheet.append(header)
    for cell in sheet[1]:
        cell.font = HEADER_FONT
        cell.alignment = HEADER_ALIGNMENT
    sheet.row_dimensions[1].height = 30


def set_widths(sheet, widths: dict[str, float]) -> None:
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width


def add_lists_sheet(workbook, image_options: list[str]):
    """Hidden sheet holding the dropdown options, referenced by the validations."""
    sheet = workbook.create_sheet(LISTS_SHEET)
    sheet["A1"] = "Confidence"
    sheet["B1"] = "Image label"
    for offset, value in enumerate(CONFIDENCE_OPTIONS, start=2):
        sheet.cell(row=offset, column=1, value=value)
    for offset, value in enumerate(image_options, start=2):
        sheet.cell(row=offset, column=2, value=value)
    for cell in sheet[1]:
        cell.font = HEADER_FONT
    sheet.column_dimensions["A"].width = 16
    sheet.column_dimensions["B"].width = 22
    sheet.sheet_state = "hidden"
    return sheet


def list_validation(formula_range: str, prompt: str) -> DataValidation:
    validation = DataValidation(
        type="list",
        formula1=f"={LISTS_SHEET}!{formula_range}",
        allow_blank=True,
        showDropDown=False,  # False means "show the in-cell dropdown arrow"
        showInputMessage=True,
        showErrorMessage=True,
    )
    validation.error = "Pick one of the listed options."
    validation.errorTitle = "Invalid entry"
    validation.prompt = prompt
    validation.promptTitle = "Select a value"
    return validation


def build_farm_sheet(workbook, farms: list[dict], labels: list[str]) -> None:
    sheet = workbook.create_sheet(FARM_SHEET)
    header = KEY_COLUMNS + ["n_images"]
    for label in labels:
        header += [label, f"{label} confidence"]
    header.append(COMMENTS_COLUMN)
    write_header(sheet, header)

    for farm in farms:
        sheet.append([farm[key] for key in KEY_COLUMNS] + [farm["n_images"]])

    last_row = sheet.max_row
    validation = list_validation(
        f"$A$2:$A${1 + len(CONFIDENCE_OPTIONS)}",
        "How confident are you in this farm-level label?",
    )
    sheet.add_data_validation(validation)
    for index, _ in enumerate(labels):
        # Label column then its confidence column, from the first label onwards.
        confidence_column = get_column_letter(len(KEY_COLUMNS) + 2 + 2 * index + 1)
        validation.add(f"{confidence_column}2:{confidence_column}{last_row}")

    set_widths(
        sheet,
        {"A": 24, "B": 14, "C": 55, "D": 20, "E": 10, get_column_letter(len(header)): 34},
    )
    for index in range(len(labels)):
        label_column = get_column_letter(len(KEY_COLUMNS) + 2 + 2 * index)
        set_widths(
            sheet,
            {
                label_column: 15,
                get_column_letter(len(KEY_COLUMNS) + 2 + 2 * index + 1): 12,
            },
        )
    sheet.freeze_panes = "F2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(header))}{last_row}"


def build_image_sheet(workbook, rows: list[dict], image_options: list[str]) -> None:
    sheet = workbook.create_sheet(IMAGE_SHEET)
    header = KEY_COLUMNS + [IMAGE_COLUMN, "Label", COMMENTS_COLUMN]
    write_header(sheet, header)

    for row in rows:
        sheet.append(
            [row[key] for key in KEY_COLUMNS]
            + [row[IMAGE_COLUMN], None, row.get(COMMENTS_COLUMN)]
        )

    last_row = sheet.max_row
    validation = list_validation(
        f"$B$2:$B${1 + len(image_options)}",
        "Pick the class shown in this image.",
    )
    sheet.add_data_validation(validation)
    label_column = get_column_letter(len(KEY_COLUMNS) + 2)
    validation.add(f"{label_column}2:{label_column}{last_row}")

    set_widths(
        sheet,
        {"A": 24, "B": 14, "C": 55, "D": 20, "E": 100, label_column: 20, "G": 34},
    )
    sheet.freeze_panes = "F2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(header))}{last_row}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    header, rows = read_rows(args.input)
    labels = label_columns(header)
    farms = group_by_farm(rows)
    image_options = labels + EXTRA_IMAGE_OPTIONS

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    add_lists_sheet(workbook, image_options)
    build_farm_sheet(workbook, farms, labels)
    build_image_sheet(workbook, rows, image_options)
    workbook.move_sheet(LISTS_SHEET, offset=2)
    workbook.active = 0
    workbook.save(args.output)

    print(f"{args.input} -> {args.output}")
    print(f"  {FARM_SHEET}: {len(farms)} farms x {len(labels)} labels (+ confidence)")
    print(f"  {IMAGE_SHEET}: {len(rows)} images, {len(image_options)} dropdown options")


if __name__ == "__main__":
    main()
