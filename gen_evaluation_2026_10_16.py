"""Score the comfev2 wave 9 FLIP arms on the gen-NSW hold-out, beside everything scored before.

Fourth of the dated successors to `gen_evaluation.py`. Every scoring and charting routine is
imported from `gen_evaluation_2026_09_04.py`, the paired macro and the run-state loader from
`gen_evaluation_2026_09_04_backbones.py`, the contrast card's lookup, the per-seed macro and
the training table from `gen_evaluation_2026_09_18.py`. What differs is that there are now
two hold-outs, and two rosters that meet on one of them.

*The question.* Every page so far scored models trained on the master build, which carries
every crop-labelled NSW row (the `generalisation` case studies and the `generalisation_extra`
parcel crops) in its training split, on the 1,146-crop VIC hold-out - where poultry has 3
positives, sheep 5 and commercial pig 8. On 2026-10-09 a second split set was cut
(`gen_dataset_master_gen_nsw_test.py`): every crop-labelled row moves to the test side,
`test_autocrop_gen_vic_nsw.csv` (2,646 crops, 1,048 farms: the VIC hold-out plus 1,500 NSW
crops, 52 poultry / 18 commercial pig / 134 backyard pig), and train / val keep farm-level
labels only. comfev2 wave 9 (`comfe-run-flip/plans/2026-10-09-comfev2-wave9-plan.md`) trained
eighteen arms on it - ComFe v1 at two head sizes, the v2 recipe F3 and its wave 8 readout
fix F4 with three variants, a single-label linear probe, four reference prototype methods
with their original losses, one-class ComFe models for the three target classes, and two
whole-farm-only arms - all on DINOv2 ViT-B/14 without registers, ten output classes, four
seeds, group pooling. The wave is judged on poultry, commercial pig and backyard pig.

Those arms are scored twice here:

    gen_vic_nsw   the primary hold-out, all 2,646 crops: paddock and other_industrial crops are
                  outside the ten-class vocabulary and count as negatives for every class, as
                  in the wave's own scorer (`tools/flip_perclass.py`, whose per-seed numbers the
                  `seed mean` columns reproduce). Only the wave 9 arms can be read here - every
                  earlier model trained on the NSW half.
    vic           the 1,146 VIC crops, the same rows as every earlier page, matched on
                  `group_id`: the wave 9 arms' predictions restricted to them, beside the
                  2026-09-18 page's eleven families re-read and the wave 7b F3 and wave 8 P1
                  (= F4) arms of the same ViT-B sweeps. Head for head, the only thing that
                  changes between a prior family and its wave 9 twin is the split (no NSW crop
                  labels, no paddock / other_industrial outputs, one label per group).

*The rosters.* Prior families keep their 2026-09-18 keys; wave 9 arms are `w9_` + the arm.

    prior, ViT-L +reg     lin_l comfe_l (09_04 master)  lin_m18 comfe_m18 (09_18 master)  lin_s3 comfe_s3 (SAM3)
    prior, ViT-B          lin_b comfe_b comfe_b04 f3_b f4_b (09_18 master, grouped)  lin_breg comfe_breg (+reg)
    wave 9, ComFe v1      w9_v1_0904 (production head)  w9_v1 (paper head)
    wave 9, v2 recipe     w9_f3  w9_f4  w9_f4_mlp  w9_f4_block  w9_d1
    wave 9, others        w9_lin (probe)  w9_eval w9_nppp w9_tesnet w9_vmf (references)
    wave 9, one class     w9_bin_poultry  w9_bin_cpig  w9_bin_bpig
    wave 9, relaunching   w9_noauto_v1  w9_noauto_f4  w9_nppp_aug (crashed on launch, no prediction yet)

*Same truth.* The VIC rows of `test_autocrop_gen_vic_nsw.csv` are checked against the
2026-09-04 `test_autocrop_gen_vic.csv` on `(ecw_stem, building_cluster, PFI)` and every
`binary_*` column at load time, so the VIC section is on exactly the truth of every earlier
page and its well-sampled macro is directly comparable. On the primary hold-out every
livestock class clears the sampling bar, so its well-sampled macro is the shared nine; the
wave's own headline, the three target classes, is reported as a fourth macro.

The run folders are read from `comfe-run-flip/view/flip_comfe_2026_09_18/` - `wave9/` is a
copy of `v_interpretable/wave9/` without checkpoints (`rsync -a --exclude checkpoints`), to be
refreshed when the three relaunched arms land.

Writes the CSVs (one folder per hold-out) *and* the self-contained HTML dashboard into
`output_eval_2026_10_16/`.

Usage:

    .venv/bin/python gen_evaluation_2026_10_16.py
    .venv/bin/python gen_evaluation_2026_10_16.py --bootstrap 10 --gallery none   # a quick pass
"""

from __future__ import annotations

import argparse
import re
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
from gen_evaluation_2026_09_18 import (
    MASTER_18,
    MASTER_18_TRAIN_CSV,
    SAM3_18_TRAIN_CSV,
    SWEEP_04,
    SWEEP_18,
    TEST_CSV_18,
    contrast_lookup,
    same_rows,
    seed_macro,
    training_card,
)

OUTPUT = Path("output_eval_2026_10_16")

# The primary hold-out and the split the wave 9 arms trained on. The `w9/` copies are what
# the runs actually read (filtered to labelled rows, plus `w9_class`); they are checked
# against the master's own copies at load time.
TEST_CSV_NSW = MASTER_18 / "test_autocrop_gen_vic_nsw.csv"
TEST_CSV_NSW_W9 = MASTER_18 / "w9" / "test_autocrop_gen_vic_nsw.csv"
W9_TRAIN_CSV = MASTER_18 / "train_minus_gen_nsw_df.csv"
NOAUTO_TRAIN_CSV = MASTER_18 / "train_no_autocrops_df.csv"

SWEEP_W9 = SWEEP_18 / "wave9"

# The gallery reads crops and whole-farm images off the 2026-09-18 master, which holds the
# imagery of both halves of the primary hold-out; the VIC crops are byte-identical copies.
base.MASTER = MASTER_18

# The ten `binary_*` columns the wave 9 arms were trained on, in the run config's order
# (7b's twelve less `other_industrial` and `paddock`). The class-index arms (probe,
# references) save ten softmax columns in the same order; their configs name `w9_class`
# rather than the list, so the order is asserted here rather than read back.
W9_CLASSES = [
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
TARGET_CLASSES = ["poultry", "commercialpig", "backyardpig"]
MACRO_TARGET = "MACRO (target 3)"

VITL = "DINOv2 ViT-L/14 w/ reg."
VITB = "DINOv2 ViT-B/14"
VITB_REG = "DINOv2 ViT-B/14 w/ reg."
GROUPED_04 = "09_04 master, grouped"
GROUPED_18 = "09_18 master, grouped"
SAM3_18 = "09_18 SAM3 relabel, per image"
GEN_NSW = "gen-NSW split, grouped"
NOAUTO = "whole-farm photographs only"

HOLDOUT_VIC = "vic"
HOLDOUT_NSW = "gen_vic_nsw"

# --------------------------------------------------------------------------------------
# the rosters
# --------------------------------------------------------------------------------------

# `head` is the family of head: `comfe` (ComFe v1), `recipe` (the comfev2 recipe: F3, F4
# and its variants, D1), `linear`, `reference` (the prototype-network references), `one`
# (ComFe one class + background). `group` is which roster the family sits in; `holdouts`
# which hold-outs it is scored on. `runs` and `match` locate the seed runs as on the
# earlier pages; `match` carries the launch-batch hash wherever a view folder holds an
# abandoned earlier batch of the same name.
PRIOR = [
    {
        "key": "lin_l",
        "label": "Linear probe, ViT-L +reg, 09_04 master",
        "head": "linear",
        "backbone": VITL,
        "data": GROUPED_04,
        "wave": "2026-09-11 backbone sweep",
        "detail": "one linear layer on frozen facebook/dinov2-with-registers-large, trained on original_master_2026_09_04 with PFI group batches; lin_l on the backbones and 09_18 pages",
        "classes": MASTER_CLASSES,
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
        "detail": "the production head (100 + 100 prototypes, class contrast on) on the same backbone, trained on original_master_2026_09_04 grouped; comfe_m on the 09_04 page, comfe_l since",
        "classes": MASTER_CLASSES,
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
        "detail": "lin_l's recipe on original_master_2026_09_18: the same farm-level labels pooled per group, plus the 1,017 generalisation_extra crops",
        "classes": MASTER_CLASSES,
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
        "detail": "comfe_l's recipe on original_master_2026_09_18, grouped",
        "classes": MASTER_CLASSES,
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
        "detail": "trained on sam3_train_df.csv: farm-labelled crops with no SAM3 building detection relabelled paddock, every label per image, no grouping",
        "classes": MASTER_CLASSES,
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
        "detail": "the production head on sam3_train_df.csv, per-image labels, no grouping",
        "classes": MASTER_CLASSES,
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
        "detail": "facebook/dinov2-base without registers; the grouped multi-label probe (BCE) with the DINOv2 paper's 13-lr x avgpool x blocks head sweep, early stopping on val macro mAP",
        "classes": MASTER_CLASSES,
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
        "detail": "ComFe v1 settings on the v2 code, the paper's head rule: 3 class prototypes per class (36 + 36 background), class contrast off, 5 image prototypes",
        "classes": MASTER_CLASSES,
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
        "detail": "the production head settings (100 + 100 prototypes, class contrast on) on the v2 code and ViT-B; w9_v1_0904's twin on the 09_18 master",
        "classes": MASTER_CLASSES,
        "runs": SWEEP_18 / "wave7b",
        "match": "flip_comfe_v2w7b_v1_0904_1-5",
    },
    {
        "key": "f3_b",
        "label": "ComFe v2 F3, ViT-B, 09_18 master",
        "head": "recipe",
        "backbone": VITB,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7b `F3`",
        "detail": "the pre-wave-8 recipe (Sinkhorn, m2f masked decoder, 20 image prototypes, mask-pooled class evidence, part matching, residual queries) on the 09_18 master; w9_f3's twin. First scored here: the wave 7 document carried only its log mAP",
        "classes": MASTER_CLASSES,
        "runs": SWEEP_18 / "wave7b",
        "match": "flip_comfe_v2w7b_F3_1-5",
    },
    {
        "key": "f4_b",
        "label": "ComFe v2 F4, ViT-B, 09_18 master",
        "head": "recipe",
        "backbone": VITB,
        "data": GROUPED_18,
        "wave": "comfev2 wave 8 `P1` (F3 + pool_proj linear)",
        "detail": "F3 with a learned linear projection of the mask-pooled feature before the class prototypes - the readout fix wave 8 adopted as F4 - on the 09_18 master; w9_f4's twin. First scored here",
        "classes": MASTER_CLASSES,
        "runs": SWEEP_18 / "wave8",
        "match": "comfe_v2w8_flip_P1_F3_pool_proj_linear_1-5",
    },
    {
        "key": "lin_breg",
        "label": "Linear probe, ViT-B +reg, 09_18 master",
        "head": "linear",
        "backbone": VITB_REG,
        "data": GROUPED_18,
        "wave": "comfev2 wave 7d `linear`",
        "detail": "lin_b on facebook/dinov2-with-registers-base, otherwise identical",
        "classes": MASTER_CLASSES,
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
        "detail": "comfe_b on facebook/dinov2-with-registers-base, otherwise identical",
        "classes": MASTER_CLASSES,
        "runs": SWEEP_18 / "wave7d",
        "match": "flip_comfe_v2w7d_v1_1-5",
    },
]
for spec in PRIOR:
    spec.update(group="prior", holdouts=[HOLDOUT_VIC], space="test")


def w9(key: str, arm: str, label: str, head: str, detail: str, classes: list[str] = W9_CLASSES, data: str = GEN_NSW) -> dict:
    return {
        "key": key,
        "label": label,
        "head": head,
        "backbone": VITB,
        "data": data,
        "wave": f"comfev2 wave 9 `{arm}`",
        "detail": detail,
        "classes": classes,
        "runs": SWEEP_W9,
        "match": f"flip_comfe_v2w9_{arm}_1-5",
        "group": "wave9",
        "holdouts": [HOLDOUT_NSW, HOLDOUT_VIC],
        "space": "test",
    }


WAVE9 = [
    w9("w9_v1_0904", "v1_0904", "ComFe v1, production head", "comfe",
       "the 09_18 production head on the v2 code (5 image prototypes, 100 + 100 prototypes, class contrast on, v1 decoder) on the ten columns, grouped: the wave's ComFe v1 baseline; comfe_b04 on the gen-NSW split"),
    w9("w9_v1", "v1", "ComFe v1, paper head", "comfe",
       "ComFe v1 at the paper's head rule (3 per class: 30 + 30 prototypes, class contrast off); comfe_b on the gen-NSW split"),
    w9("w9_f3", "F3", "ComFe v2 F3", "recipe",
       "the pre-wave-8 recipe: Sinkhorn, m2f 2 x 256 masked decoder, 20 image prototypes, mask-pooled class evidence, part matching 0.5, residual queries, no background target; the control F4 is read against"),
    w9("w9_f4", "F4", "ComFe v2 F4 (the candidate)", "recipe",
       "F3 + pool_proj linear, the wave 8 readout fix: the improved ComFe the wave asks about"),
    w9("w9_f4_mlp", "F4_mlp", "ComFe v2 F4, MLP readout", "recipe",
       "F3 + pool_proj mlp (wave 8 P2)"),
    w9("w9_f4_block", "F4_block", "ComFe v2 F4 + block", "recipe",
       "F4 with the appended backbone block trained from epoch 25 (wave 8 B1 on F4)"),
    w9("w9_d1", "D1", "ComFe v2 F3, decoder readout", "recipe",
       "F3 with class_embed: decoder - the class evidence read from the decoder output rather than the mask pool; the recipe's accuracy ceiling at the cost of the parts, a diagnostic rather than a candidate"),
    w9("w9_lin", "linear", "Linear probe, single-label", "linear",
       "the grouped linear probe with softmax + cross-entropy on one class index per group (w9_class), PFI batches of 64, 13 lrs x avgpool x {1, 4} blocks, selection on mean per-class accuracy, early stopping; lin_b on the gen-NSW split with the multi-label BCE head swapped for softmax"),
    w9("w9_eval", "eval_K5", "EvalProtoPNet K = 5", "reference",
       "the reference prototype network with its original single-label loss, logits max-pooled per group; 5 + 7 epochs"),
    w9("w9_nppp", "nppp_K5", "NPPP K = 5", "reference",
       "the reference method with its original loss, group-pooled; 1 + 5 epochs, no augmentation"),
    w9("w9_tesnet", "tesnet_K5", "TesNet K = 5", "reference",
       "the reference method with its original loss, group-pooled; 5 + 15 + 20 epochs, one push"),
    w9("w9_vmf", "vmf_J5", "vMFProto J = 5", "reference",
       "the reference method with its original loss, group-pooled; 1 + 19 epochs, no augmentation"),
    w9("w9_bin_poultry", "bin1_poultry", "ComFe F4, poultry only", "one",
       "F4 as one class + background on the single binary_poultry column, 3 + 3 prototypes; every other crop is a negative",
       classes=["poultry"]),
    w9("w9_bin_cpig", "bin1_commercialpig", "ComFe F4, commercial pig only", "one",
       "F4 as one class + background on binary_commercialpig",
       classes=["commercialpig"]),
    w9("w9_bin_bpig", "bin1_backyardpig", "ComFe F4, backyard pig only", "one",
       "F4 as one class + background on binary_backyardpig",
       classes=["backyardpig"]),
    w9("w9_noauto_v1", "noauto_v1_0904", "ComFe v1, production head, whole farms only", "comfe",
       "w9_v1_0904 trained on train_no_autocrops_df.csv, the 1,969 historical whole-farm photographs alone (singleton groups): what the building crops add. The 2026-10-09 launch crashed before training (the w9/ csvs were not in the tar) and is to be relaunched",
       data=NOAUTO),
    w9("w9_noauto_f4", "noauto_F4", "ComFe v2 F4, whole farms only", "recipe",
       "w9_f4 on the whole-farm photographs alone; crashed on launch as above, to be relaunched",
       data=NOAUTO),
    w9("w9_nppp_aug", "nppp_K5_aug", "NPPP K = 5 with its augmentation regime", "reference",
       "NPPP with its Augmentor regime online (30 + 150 epochs); the launch crashed in the shear augmentation on extreme-aspect crops, fixed, to be relaunched"),
]

MODELS = PRIOR + WAVE9

# The named contrasts each hold-out leads with, (left minus right) pairs read off the
# paired macro, grouped by the axis they vary. The per-class contrasts are the same for the
# classes the wave is judged on, read off the per-class pairs.
CONTRASTS_NSW = [
    ("Head: the v2 recipe against ComFe v1, on the gen-NSW split", [
        ("w9_f4", "w9_v1_0904", "F4 against the production head: the wave's question"),
        ("w9_f3", "w9_v1_0904", "F3 against the production head: before the readout fix"),
        ("w9_f4", "w9_f3", "F4 against F3: the readout fix alone"),
        ("w9_f4_mlp", "w9_f4", "MLP readout against linear"),
        ("w9_f4_block", "w9_f4", "+ the appended block"),
        ("w9_d1", "w9_f4", "decoder readout against the mask pool: what the parts cost"),
        ("w9_v1_0904", "w9_v1", "ComFe v1: production head against the paper's"),
    ]),
    ("Against the linear probe", [
        ("w9_f4", "w9_lin", "F4"),
        ("w9_v1_0904", "w9_lin", "ComFe v1, production head"),
        ("w9_eval", "w9_lin", "EvalProtoPNet"),
        ("w9_nppp", "w9_lin", "NPPP"),
        ("w9_tesnet", "w9_lin", "TesNet"),
        ("w9_vmf", "w9_lin", "vMFProto"),
    ]),
    ("The references against F4", [
        ("w9_eval", "w9_f4", "EvalProtoPNet"),
        ("w9_nppp", "w9_f4", "NPPP"),
        ("w9_tesnet", "w9_f4", "TesNet"),
        ("w9_vmf", "w9_f4", "vMFProto"),
    ]),
    ("What the building crops add (pending the relaunch)", [
        ("w9_v1_0904", "w9_noauto_v1", "ComFe v1, production head"),
        ("w9_f4", "w9_noauto_f4", "F4"),
    ]),
]
CLASS_CONTRASTS_NSW = [
    ("F4 against ComFe v1, the three target classes (adopted if within 0.02 or above on each)", [
        ("w9_f4", "w9_v1_0904", c, "") for c in TARGET_CLASSES
    ]),
    ("F3 to F4: where the readout fix moved the accuracy", [
        ("w9_f4", "w9_f3", c, "") for c in TARGET_CLASSES
    ]),
    ("One class + background against the ten-class F4, on its own class", [
        ("w9_bin_poultry", "w9_f4", "poultry", ""),
        ("w9_bin_cpig", "w9_f4", "commercialpig", ""),
        ("w9_bin_bpig", "w9_f4", "backyardpig", ""),
    ]),
    ("F4 against the probe", [
        ("w9_f4", "w9_lin", c, "") for c in TARGET_CLASSES
    ]),
]
CONTRASTS_VIC = [
    ("Split: the gen-NSW split against the 09_18 master, same head and backbone (ViT-B)", [
        ("w9_v1_0904", "comfe_b04", "ComFe v1, production head"),
        ("w9_v1", "comfe_b", "ComFe v1, paper head"),
        ("w9_f3", "f3_b", "ComFe v2 F3"),
        ("w9_f4", "f4_b", "ComFe v2 F4"),
        ("w9_lin", "lin_b", "linear probe (softmax single-label against BCE multi-label)"),
    ]),
    ("Head on the gen-NSW split, read on the VIC half", [
        ("w9_f4", "w9_v1_0904", "F4 against the production head"),
        ("w9_f4", "w9_f3", "the readout fix alone"),
        ("w9_f4", "w9_lin", "F4 against the probe"),
        ("w9_v1_0904", "w9_lin", "ComFe v1 against the probe"),
        ("w9_eval", "w9_lin", "EvalProtoPNet against the probe"),
    ]),
    ("Head on the 09_18 master, ViT-B (the recipe arms are new to this page)", [
        ("f4_b", "comfe_b04", "F4 against the production head"),
        ("f3_b", "comfe_b04", "F3 against the production head"),
        ("f4_b", "f3_b", "the readout fix alone"),
        ("comfe_b04", "lin_b", "ComFe v1, production head, against the probe"),
        ("f4_b", "lin_b", "F4 against the probe"),
    ]),
    ("The 2026-09-18 page's contrasts, repeated", [
        ("comfe_l", "lin_l", "head: ViT-L +reg, 09_04 master"),
        ("comfe_m18", "lin_m18", "head: ViT-L +reg, 09_18 master"),
        ("comfe_s3", "lin_s3", "head: ViT-L +reg, 09_18 SAM3 relabel"),
        ("comfe_breg", "lin_breg", "head: ViT-B +reg, 09_18 master"),
        ("comfe_m18", "comfe_l", "data: + generalisation_extra, ComFe ViT-L"),
        ("comfe_s3", "comfe_m18", "data: + SAM3 relabel, ComFe ViT-L"),
        ("comfe_b04", "comfe_m18", "backbone: ViT-B against ViT-L +reg, production head"),
        ("comfe_breg", "comfe_b", "backbone: + registers, ViT-B paper head"),
        ("lin_b", "lin_m18", "backbone: probe, ViT-B against ViT-L +reg"),
    ]),
]
CLASS_CONTRASTS_VIC = [
    ("Split: the gen-NSW split against the 09_18 master, on the VIC half's target classes (the wave 9 plan's Q2)", [
        ("w9_v1_0904", "comfe_b04", "commercialpig", "8 positives"),
        ("w9_v1_0904", "comfe_b04", "backyardpig", "71 positives"),
        ("w9_f4", "f4_b", "commercialpig", "8 positives"),
        ("w9_f4", "f4_b", "backyardpig", "71 positives"),
        ("w9_lin", "lin_b", "backyardpig", "71 positives"),
    ]),
]

# The all-pairs matrices are kept to a readable subset on the VIC section, where the two
# rosters together would make 300 columns; every pair is still in `evaluation_pairs.csv`.
PAIR_KEYS_VIC = [
    "lin_l", "comfe_l", "lin_m18", "comfe_m18", "lin_b", "comfe_b04", "f3_b", "f4_b",
    "w9_lin", "w9_v1_0904", "w9_f3", "w9_f4", "w9_eval",
]
PAIR_KEYS_NSW = [
    "w9_lin", "w9_v1_0904", "w9_v1", "w9_f3", "w9_f4", "w9_f4_mlp", "w9_f4_block", "w9_d1",
    "w9_eval", "w9_nppp", "w9_tesnet", "w9_vmf",
]

# The galleries show a few models per crop rather than the whole roster. The VIC gallery is
# picked on the 2026-09-04 ComFe as on every earlier page, so the same farms come up; the
# NSW gallery is picked on F4, the candidate, over the NSW farms only.
GALLERY_KEYS_VIC = ["comfe_l", "lin_l", "comfe_b04", "f4_b", "w9_v1_0904", "w9_f4", "w9_lin", "w9_eval"]
GALLERY_KEYS_NSW = ["w9_v1_0904", "w9_f4", "w9_f3", "w9_d1", "w9_lin", "w9_eval", "w9_nppp"]
REFERENCE_KEYS_VIC = ["comfe_l", "comfe_b04", "w9_f4"]
REFERENCE_KEYS_NSW = ["w9_f4", "w9_v1_0904", "w9_lin"]

# Colours. The eleven 2026-09-18 families keep that page's colours. Each head recipe is a
# hue family and the gen-NSW twin is its darker member, so a prior family and its wave 9
# twin sit together in the VIC charts: production head amber, paper head brown, F3 magenta,
# F4 violet, probe orange-red. The F4 variants, the references and the one-class models
# take hues of their own.
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
    "f3_b": ("#c43f9c", "#d66bb8"),
    "f4_b": ("#7a3fd1", "#a97ff0"),
    "lin_breg": ("#e34948", "#e66767"),
    "comfe_breg": ("#5e6b7a", "#9aa7b5"),
    "w9_v1_0904": ("#9a6400", "#e0b040"),
    "w9_v1": ("#55361a", "#9c7a56"),
    "w9_f3": ("#7f1f63", "#b35a99"),
    "w9_f4": ("#3f1a8a", "#8c6ce0"),
    "w9_f4_mlp": ("#1f6f8f", "#4fa3c4"),
    "w9_f4_block": ("#1f4fa8", "#5f8fe0"),
    "w9_d1": ("#6b6b1f", "#a8a84a"),
    "w9_lin": ("#a33f14", "#d4713f"),
    "w9_eval": ("#d4466f", "#e8759a"),
    "w9_nppp": ("#00897b", "#3fb8a8"),
    "w9_tesnet": ("#7a5548", "#a88578"),
    "w9_vmf": ("#8e24aa", "#ba68c8"),
    "w9_bin_poultry": ("#e0607e", "#f08aa4"),
    "w9_bin_cpig": ("#ef7f4f", "#ffa37a"),
    "w9_bin_bpig": ("#6d4c41", "#9c7b6f"),
    "w9_noauto_v1": ("#8f8f8f", "#9a9a9a"),
    "w9_noauto_f4": ("#6f6f6f", "#b0b0b0"),
    "w9_nppp_aug": ("#4f4f4f", "#c4c4c4"),
}
base.SERIES = SERIES
sweep.SERIES = SERIES

HEAD_NAMES = {
    "comfe": "ComFe v1",
    "recipe": "ComFe v2 recipe",
    "linear": "linear probe",
    "reference": "reference prototype network",
    "one": "ComFe v2 F4, one class + background",
}


# --------------------------------------------------------------------------------------
# loading: every family once, then restricted to each hold-out it is read on
# --------------------------------------------------------------------------------------


def restrict(model: dict, rows: np.ndarray) -> dict:
    """The same family read on a subset of its hold-out's rows."""
    return {
        **model,
        "seeds": [s[rows] for s in model["seeds"]],
        "ensemble": None if model["ensemble"] is None else model["ensemble"][rows],
    }


def load_families(test_vic: pd.DataFrame, test_nsw: pd.DataFrame, vic_rows: np.ndarray) -> dict[str, list[dict]]:
    """Every family loaded on its own hold-out, then placed on each hold-out it is read on."""
    by_holdout: dict[str, list[dict]] = {HOLDOUT_NSW: [], HOLDOUT_VIC: []}
    for spec in MODELS:
        if spec["group"] == "prior":
            model = load_model(spec, len(test_vic), test_vic["group_id"].to_numpy())
            by_holdout[HOLDOUT_VIC].append(model)
        else:
            model = load_model(spec, len(test_nsw), test_nsw["group_id"].to_numpy())
            by_holdout[HOLDOUT_NSW].append(model)
            by_holdout[HOLDOUT_VIC].append(restrict(model, vic_rows))
        note = f"{len(model['seeds'])} seed run(s)"
        if model["n_pending"]:
            note += f", {model['n_pending']} still training"
        if model["n_dead"]:
            note += f", {model['n_dead']} dead"
        if model["epochs"]:
            note += "; saved at epoch " + "/".join(str(e) for e in model["epochs"])
        print(f"  {model['key']:>14}: {note}")
        for name in model["pending_names"]:
            print(f"                  pending: {name}")
        for name in model["dead_names"]:
            print(f"                  dead: {name}")
    return by_holdout


# --------------------------------------------------------------------------------------
# scoring one hold-out
# --------------------------------------------------------------------------------------


def macro_row(table: pd.DataFrame, level: str, label: str, subset: list[str], models: list[dict]) -> pd.DataFrame:
    """One more macro row for `table`, the mean over `subset` of the per-class rows.

    Mirrors the macro block of `evaluate`; a model with no output for one of the classes
    has no entry.
    """
    rows = table[(table["level"] == level) & (table["class"].isin(subset))]
    macro = {"level": level, "class": label, "n": int(table["n"].iloc[0])}
    for model in models:
        if model["ensemble"] is None:
            continue
        key = model["key"]
        usable = rows[rows[f"{key}|AP"].notna()] if f"{key}|AP" in rows else rows.iloc[0:0]
        if len(usable) != len(subset):
            continue
        macro[f"{key}|AP"] = float(usable[f"{key}|AP"].mean())
        macro[f"{key}|AUC"] = float(usable[f"{key}|AUC"].mean())
        macro[f"{key}|n_classes"] = int(len(usable))
        macro[f"{key}|status"] = "scored"
        macro["positives"] = int(usable["positives"].sum())
        macro["prevalence"] = float(usable["prevalence"].mean())
        macro["n_train"] = int(usable["n_train"].sum())
        macro["n_train_master"] = int(usable["n_train_master"].sum())
    return pd.DataFrame([macro])


def score_holdout(
    name: str,
    test: pd.DataFrame,
    models: list[dict],
    crop_regions: np.ndarray,
    trained: dict[str, int],
    trained_master: dict[str, int],
    args: argparse.Namespace,
    extra_macros: list[tuple[str, list[str]]],
) -> dict:
    """Everything the earlier pages computed, for one hold-out, into one dict."""
    scored = [m for m in models if m["ensemble"] is not None]
    if not scored:
        raise SystemExit(f"{name}: no model has a saved prediction yet")
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)

    crop = crop_truth(test)
    farms = farm_frame(test)
    farm = farm_truth(farms)
    keys = test["farm_uid"].to_numpy()
    order = farms["farm_uid"].to_numpy()
    farm_regions = pd.Series(crop_regions).groupby(keys).first().reindex(order).to_numpy()
    print(f"{name}: {len(test)} crops, {len(farms)} farms, crop scores aggregated with {args.aggregation}")

    farm_models = []
    for model in models:
        if model["ensemble"] is None:
            farm_models.append({**model, "ensemble": None, "seeds": []})
            continue
        farm_models.append({
            **model,
            "ensemble": farm_scores(model["ensemble"], keys, order, args.aggregation),
            "seeds": [farm_scores(s, keys, order, args.aggregation) for s in model["seeds"]],
        })

    truths = {"crop": crop, "farm": farm}
    by_level = {"crop": models, "farm": farm_models}
    regions_of = {"crop": crop_regions, "farm": farm_regions}

    metrics, pairs, macro, regions, agreement, seeds = [], [], [], [], [], []
    for level in LEVELS:
        table, pair = evaluate(
            level, truths[level], by_level[level], args.bootstrap, args.seed, trained, trained_master
        )
        well = [
            c for c in SHARED_CLASSES if table.loc[table["class"] == c, "well_sampled"].any()
        ]
        subsets = {MACRO: ALL_CLASSES, MACRO_SHARED: SHARED_CLASSES, MACRO_SAMPLED: well}
        macro_sets = [(MACRO_SAMPLED, well), (MACRO_SHARED, SHARED_CLASSES)]
        for label, subset in extra_macros:
            usable = [c for c in subset if 0 < truths[level][c].sum() < len(truths[level])]
            if len(usable) != len(subset):
                continue
            table = pd.concat([table, macro_row(table, level, label, subset, by_level[level])], ignore_index=True)
            subsets[label] = subset
            macro_sets.append((label, subset))
        # A macro over a named subset needs every class of the subset: the one-class models
        # get no shared-9 or well-sampled macro rather than one over their single class.
        n_units = len(truths[level])
        for label, subset in subsets.items():
            if label == MACRO:
                continue
            evaluable = [c for c in subset if 0 < truths[level][c].sum() < n_units]
            for model in by_level[level]:
                if model["ensemble"] is None:
                    continue
                if any(base.column_of(model, c) is None for c in evaluable):
                    hit = (table["level"] == level) & (table["class"] == label)
                    for column in ("AP", "AUC", "n_classes", "status"):
                        if f"{model['key']}|{column}" in table.columns:
                            table.loc[hit, f"{model['key']}|{column}"] = np.nan
        metrics.append(table)
        pairs.append(pair)
        per_seed = seed_macro(level, truths[level], by_level[level], subsets)
        for label, subset in subsets.items():
            if label == MACRO:
                continue
            evaluable = len([c for c in subset if 0 < truths[level][c].sum() < n_units])
            per_seed = per_seed[(per_seed["macro"] != label) | (per_seed["n_classes"] >= evaluable)]
        seeds.append(per_seed.reset_index(drop=True))
        for label, subset in macro_sets:
            macro.append(macro_pairs(
                level, truths[level], by_level[level], subset, label, args.bootstrap, args.seed
            ))
        regions.append(region_summary(
            level, truths[level], regions_of[level], by_level[level], args.bootstrap, args.seed
        ))
        found = top_k_agreement(truths[level], by_level[level])
        found.insert(0, "level", level)
        agreement.append(found)

    metrics = pd.concat(metrics, ignore_index=True)
    pairs = pd.concat(pairs, ignore_index=True)
    macro = pd.concat(macro, ignore_index=True)
    regions = pd.concat(regions, ignore_index=True)
    agreement = pd.concat(agreement, ignore_index=True)
    seeds = pd.concat(seeds, ignore_index=True)

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
        predicted, points = decisions(m, farm, args.farm_threshold)
        operating.append(points)
        confusions["farm"][m["key"]], denominators["farm"][m["key"]] = confusion_sets(
            farm, m, predicted, "no class marked"
        )
        coarse = merge_model(m)
        predicted, points = decisions(coarse, farm_merged, args.farm_threshold)
        operating_merged.append(points)
        confusions["farm_merged"][m["key"]], denominators["farm_merged"][m["key"]] = confusion_sets(
            farm_merged, coarse, predicted, "no class marked", merged_order()
        )
    operating = pd.concat(operating, ignore_index=True)
    operating.insert(0, "rule", args.farm_threshold)
    operating_merged = pd.concat(operating_merged, ignore_index=True)
    operating_merged.insert(0, "rule", args.farm_threshold)

    # ---- outputs -------------------------------------------------------------------
    metrics.to_csv(out / "evaluation.csv", index=False)
    pairs.to_csv(out / "evaluation_pairs.csv", index=False)
    macro.to_csv(out / "macro_pairs.csv", index=False)
    seeds.to_csv(out / "seed_macro.csv", index=False)
    regions.to_csv(out / "evaluation_by_region.csv", index=False)
    agreement.to_csv(out / "agreement.csv", index=False)
    for level, tables in confusions.items():
        pd.concat(tables, names=["model", "annotated"]).to_csv(out / f"confusion_{level}.csv")
    operating.to_csv(out / "operating_points_farm.csv", index=False)
    operating_merged.to_csv(out / "operating_points_farm_merged.csv", index=False)

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
                for cls in model["classes"]:
                    columns[f"{prefix}|{cls}"] = array[:, model["classes"].index(cls)]
        return columns

    pd.DataFrame({
        "region": crop_regions,
        "source": test["source"].to_numpy(),
        "farm_uid": keys,
        "PFI": test["PFI"].to_numpy(),
        "group_id": test["group_id"].to_numpy(),
        "image_path": test["image_path"].to_numpy(),
        "crop_label": test["crop_label"].to_numpy(),
        "crop_classes": test["crop_classes"].to_numpy(),
        **{f"true|{cls}": crop[cls].to_numpy() for cls in ALL_CLASSES},
        **score_columns(models),
    }).to_csv(out / "scores_crop.csv", index=False)
    pd.DataFrame({
        "region": farm_regions,
        "source": farms["source"].to_numpy(),
        "farm_uid": order,
        "PFI": farms["PFI"].to_numpy(),
        "n_crops": farms["n_crops"].to_numpy(),
        "farm_labels": farms["farm_labels"].to_numpy(),
        **{f"true|{cls}": farm[cls].to_numpy() for cls in ALL_CLASSES},
        **score_columns(farm_models),
    }).to_csv(out / "scores_farm.csv", index=False)

    return {
        "name": name,
        "test": test,
        "models": models,
        "farm_models": farm_models,
        "scored": scored,
        "crop": crop,
        "farms": farms,
        "farm": farm,
        "truths": truths,
        "crop_regions": crop_regions,
        "farm_regions": farm_regions,
        "metrics": metrics,
        "pairs": pairs,
        "macro": macro,
        "regions": regions,
        "agreement": agreement,
        "seeds": seeds,
        "confusions": confusions,
        "denominators": denominators,
        "operating": operating,
        "operating_merged": operating_merged,
        "out": out,
    }


def build_gallery(
    result: dict, reference_keys: list[str], shown_keys: list[str], farm_mask: np.ndarray | None
) -> tuple[str, list[dict], dict[int, list[str]], dict | None, pd.DataFrame]:
    """A farm gallery over (a subset of) the hold-out's farms, a few models per crop."""
    models = [m for m in result["models"] if m["key"] in shown_keys]
    farm_models = [m for m in result["farm_models"] if m["key"] in shown_keys]
    farms, farm = result["farms"], result["farm"]
    if farm_mask is not None:
        farms = farms[farm_mask].reset_index(drop=True)
        farm = farm[farm_mask].reset_index(drop=True)
        farm_models = [restrict(m, np.flatnonzero(farm_mask)) if m["ensemble"] is not None else m for m in farm_models]
    reference = next(
        (m for key in reference_keys for m in farm_models if m["key"] == key and m["ensemble"] is not None),
        None,
    )
    if reference is None:
        return "", [], {}, None, farms
    groups, reasons = choose_examples(farms, farm, reference, farm_models)
    gallery = farm_gallery(groups, reasons, farms, result["test"], result["crop"], models, farm_models)
    return gallery, groups, reasons, reference, farms


def examples_frame(groups: list[dict], reasons: dict, farms: pd.DataFrame, reference: dict | None) -> pd.DataFrame:
    return pd.DataFrame([
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
    ])


# --------------------------------------------------------------------------------------
# page pieces particular to this roster
# --------------------------------------------------------------------------------------


def roster_card(models: list[dict], holdout_names: dict[str, str]) -> str:
    """One card per family, naming the head, data and which hold-outs it is read on."""
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
        runs = f"{n_ready} seed run(s) scored, {len(model['classes'])} output class(es)"
        if epochs:
            runs += f"; saved at epoch {'/'.join(str(e) for e in epochs)}"
        if hours:
            runs += f"; {np.mean(hours):.1f} h per run to the start of testing"
        where = ", ".join(holdout_names[h] for h in model["holdouts"])
        cards.append(
            f'<div class="model"><p class="model-name">'
            f'<i class="key-line" style="background:var(--series-{model["key"]})"></i>'
            f'{escape(model["label"])}<span class="model-key">{escape(model["key"])}</span></p>'
            f'<p class="model-note"><b>{escape(HEAD_NAMES[model["head"]])}</b> on <b>{escape(model["backbone"])}</b>, '
            f'<b>{escape(model["data"])}</b> &mdash; {escape(model["wave"])}. {escape(model["detail"])}</p>'
            f'<p class="model-runs">{runs}; read on {escape(where)} {" ".join(states)}</p></div>'
        )
    return f'<div class="roster">{"".join(cards)}</div>'


def contrast_card(macro: pd.DataFrame, scored: list[dict], label: str, contrasts: list) -> str:
    """The named contrasts, crop and farm side by side, read off the paired macro."""
    present = {m["key"] for m in scored}
    levels = [level for level in LEVELS if level in set(macro[macro["macro"] == label]["level"])]
    head = "".join(f"<th>{escape(level)}</th>" for level in levels)
    blocks = []
    for title, pairs in contrasts:
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
                cells.append(delta_cell(entry["delta"], entry["lo"], entry["hi"], entry["clear"],
                                        f"{label}, {level}: {b} {entry['AP_b']:.3f} minus {a} {entry['AP_a']:.3f} = {entry['delta']:+.3f} AP"))
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


def delta_cell(delta: float, lo: float, hi: float, clear: bool, caption: str, scale: float = 0.10) -> str:
    strength = min(abs(delta) / scale, 1.0)
    pole = "up" if delta >= 0 else "down"
    mark = ' <span class="clear">&#9679;</span>' if clear else ""
    bounds = "" if pd.isna(lo) else f'<span class="muted">[{lo:+.3f}, {hi:+.3f}]</span>'
    return (
        f'<td class="delta-cell" style="--tint:{strength:.3f}" data-pole="{pole}" '
        f'title="{escape(caption)}"><span class="delta-value">{delta:+.3f}{mark}</span>'
        f"<br>{bounds}</td>"
    )


def class_pair(pairs: pd.DataFrame, level: str, name: str, b: str, a: str) -> dict | None:
    """The per-class pair row for b minus a, whichever way round it was stored."""
    rows = pairs[(pairs["level"] == level) & (pairs["class"] == name)]
    hit = rows[(rows["b"] == b) & (rows["a"] == a)]
    sign = 1.0
    if hit.empty:
        hit = rows[(rows["b"] == a) & (rows["a"] == b)]
        sign = -1.0
    if hit.empty:
        return None
    row = hit.iloc[0]
    lo, hi = row["delta_lo"], row["delta_hi"]
    if sign < 0 and pd.notna(lo):
        lo, hi = -hi, -lo
    return {"delta": sign * float(row["delta_AP"]), "lo": lo, "hi": hi, "clear": bool(row["clear_of_zero"]),
            "positives": int(row["positives"])}


def class_contrast_card(metrics: pd.DataFrame, pairs: pd.DataFrame, scored: list[dict], contrasts: list) -> str:
    """Named per-class contrasts: AP(left), AP(right), the paired difference, per level."""
    present = {m["key"] for m in scored}
    levels = list(LEVELS)
    head = "".join(
        f"<th>{escape(level)} AP left</th><th>{escape(level)} AP right</th><th>{escape(level)} difference</th>"
        for level in levels
    )
    blocks = []
    for title, items in contrasts:
        body = []
        for b, a, name, note in items:
            if b not in present or a not in present:
                continue
            cells = []
            for level in levels:
                row = metrics[(metrics["level"] == level) & (metrics["class"] == name)]
                ap_b = row[f"{b}|AP"].iloc[0] if not row.empty and f"{b}|AP" in row else np.nan
                ap_a = row[f"{a}|AP"].iloc[0] if not row.empty and f"{a}|AP" in row else np.nan
                cells.append(f"<td>{'-' if pd.isna(ap_b) else f'{ap_b:.3f}'}</td>")
                cells.append(f"<td>{'-' if pd.isna(ap_a) else f'{ap_a:.3f}'}</td>")
                entry = class_pair(pairs, level, name, b, a)
                if entry is None:
                    cells.append("<td>-</td>")
                else:
                    cells.append(delta_cell(entry["delta"], entry["lo"], entry["hi"], entry["clear"],
                                            f"{name}, {level}: {b} minus {a} = {entry['delta']:+.3f} AP", scale=0.20))
            crop_row = metrics[(metrics["level"] == "crop") & (metrics["class"] == name)]
            positives = int(crop_row["positives"].iloc[0]) if not crop_row.empty else 0
            body.append(
                f'<tr><th scope="row"><span class="pairhead">'
                f'<i class="key-line" style="background:var(--series-{b})"></i>'
                f'&minus;<i class="key-line" style="background:var(--series-{a})"></i> '
                f"{escape(b)} &minus; {escape(a)}</span><br>"
                f'<b>{escape(name)}</b> <span class="muted">{positives} crop positives{(" &middot; " + escape(note)) if note else ""}</span></th>'
                f'{"".join(cells)}</tr>'
            )
        if body:
            blocks.append(
                f'<tr class="group-row"><th colspan="{3 * len(levels) + 1}">{escape(title)}</th></tr>'
                + "".join(body)
            )
    if not blocks:
        return ""
    return (
        f'<div class="scroll"><table class="data pairs"><thead><tr><th>pair, class</th>{head}</tr></thead>'
        f"<tbody>{''.join(blocks)}</tbody></table></div>"
    )


def macro_tiles_for(metrics: pd.DataFrame, seeds: pd.DataFrame, models: list[dict], columns: list[tuple[str, str]], rank_on: str, note: str) -> str:
    """Headline tiles: the seed-ensemble macro and the per-seed mean ± sd, per macro, per level."""
    scored = [m for m in models if m["ensemble"] is not None]
    tiles = []
    for level in LEVELS:
        rows = {label: metrics[(metrics["level"] == level) & (metrics["class"] == label)] for label, _ in columns}
        if all(r.empty for r in rows.values()):
            continue
        head = "".join(
            f'<th{"" if label == rank_on else " class=muted"}>{escape(title)}</th>'
            f'<th class="muted">seeds</th>'
            for label, title in columns
        )
        body = []
        for m in scored:
            key = m["key"]
            cells = []
            any_value = False
            for label, _ in columns:
                row = rows[label]
                value = row.iloc[0].get(f"{key}|AP") if not row.empty else np.nan
                any_value |= pd.notna(value)
                cells.append(
                    f'<td{"" if label == rank_on else " class=muted"}>{"-" if pd.isna(value) else f"{value:.3f}"}</td>'
                )
                seed = seeds[(seeds["level"] == level) & (seeds["macro"] == label) & (seeds["model"] == key)]
                if seed.empty:
                    cells.append('<td class="muted">-</td>')
                else:
                    s = seed.iloc[0]
                    sd = "" if pd.isna(s["AP_seed_sd"]) else f" &plusmn; {s['AP_seed_sd']:.3f}"
                    cells.append(f'<td class="muted">{s["AP_seed_mean"]:.3f}{sd}</td>')
            if not any_value:
                continue
            body.append(
                f'<tr><th scope="row"><i class="key-line" style="background:var(--series-{key})"></i>'
                f"{escape(key)}</th>{''.join(cells)}</tr>"
            )
        tiles.append(
            f'<div class="tile tile-wide"><p class="tile-label">macro AP, {escape(level)} level</p>'
            f'<div class="scroll"><table class="mini"><thead><tr><th></th>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>"
            f'<p class="tile-note">{note}</p></div>'
        )
    return "".join(tiles)


def restrict_pairs(pairs: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return pairs[pairs["a"].isin(keys) & pairs["b"].isin(keys)]


def w9_metrics_table(metrics: pd.DataFrame, models: list[dict], level: str) -> str:
    """The base table with its training columns renamed for the gen-NSW split."""
    html = metrics_table(metrics, models, level)
    html = html.replace("train<br>2026-08-21", "train<br>gen-NSW split").replace("train<br>master", "train<br>09_18 master")
    return re.sub(
        r'<p class="table-note">.*?</p>',
        f'<p class="table-note"><b>train</b> is the number of crops carrying that class in '
        f'<code>train_minus_gen_nsw_df.csv</code> (what the wave 9 arms trained on) and in the '
        f'2026-09-18 master\'s <code>train_df.csv</code>. Dimmed rows fall short of '
        f'{MIN_TRAIN_EXAMPLES} training crops in either or of {MIN_POSITIVES} test positives; '
        f'an AP cell is marked when its own 95&nbsp;% interval is at least {WIDE_INTERVAL:.2f} wide. '
        f'<code>paddock</code> and <code>other_industrial</code> are outside the ten-class '
        f'vocabulary, so every model shows <em>no output</em> there: those crops are negatives '
        f'for every class.</p>',
        html, count=1, flags=re.DOTALL,
    )


def banner_for(models: list[dict]) -> str:
    waiting = [m for m in models if m["ensemble"] is None or m["n_pending"] or m["n_dead"]]
    if not waiting:
        return ""
    parts = []
    for m in waiting:
        if m["n_pending"]:
            parts.append(f"{escape(m['label'])} ({m['n_pending']} run(s) still training)")
        elif m["n_dead"]:
            parts.append(f"{escape(m['label'])} ({m['n_dead']} dead run(s) left out)")
        else:
            parts.append(f"{escape(m['label'])} (nothing saved yet)")
    return (
        f'<div class="banner"><strong>Incomplete sweep.</strong> {"; ".join(parts)}. '
        f"Everything below is scored on the runs that have landed."
        f"</div>"
    )


def holdout_sections(
    result: dict,
    title: str,
    intro: str,
    tiles: str,
    contrasts: list,
    class_contrasts: list,
    pair_keys: list[str],
    gallery: str,
    gallery_intro: str,
    reference: dict | None,
    macro_labels: list[str],
    table_of,
    region_word: str,
    aggregation: str,
    farm_rule: str,
    anchor: str,
) -> str:
    """The run of sections one hold-out gets: tiles, contrasts, charts, pairs, reaches, curves, leaks, gallery, full results."""
    metrics, pairs, macro, regions = result["metrics"], result["pairs"], result["macro"], result["regions"]
    models, scored, truths = result["models"], result["scored"], result["truths"]
    n_crops, n_farms = len(result["test"]), len(result["farms"])
    if farm_rule == "prevalence":
        rule_note = (
            "A class is predicted for a farm when the farm is among the <em>k</em> "
            "highest-scoring farms for that class, with <em>k</em> the number of farms "
            "annotated with it &mdash; the operating point at prevalence. Rerun with "
            "<code>--farm-threshold 0.5</code> to see a fixed score cut instead."
        )
    else:
        rule_note = (
            f"A class is predicted for a farm when its score exceeds {escape(farm_rule)}, "
            f"whatever the class or model."
        )
    merged_caption = "; ".join(f"{g} = {' + '.join(ms)}" for g, ms in base.CLASS_GROUPS.items())
    ramp = "".join(f'<i style="background:var(--heat-{step})"></i>' for step in range(7))
    diag_key = '<div class="ramp diag-key"><i></i>annotated class is the top class</div>'
    pair_models = [m for m in scored if m["key"] in pair_keys]
    pairs_shown = restrict_pairs(pairs, pair_keys)
    macro_shown = restrict_pairs(macro, pair_keys)
    pr_models = [m for m in scored if m["head"] != "one"]

    contrast_cards = "".join(
        f'<div class="card"><div class="card-head"><h3>{escape(label)}</h3></div>'
        f"{contrast_card(macro, scored, label, contrasts)}</div>"
        for label in macro_labels
    )
    class_card = class_contrast_card(metrics, pairs, scored, class_contrasts)
    if class_card:
        class_card = (
            '<div class="card"><div class="card-head"><h3>By class</h3></div>'
            '<p class="lede">AP of the left and right model on the named class and their paired '
            'difference, crop and farm level. Shaded to &plusmn;0.20.</p>' + class_card + "</div>"
        )
    gallery_section = ""
    if gallery:
        gallery_section = f"""
<section id="{anchor}-farms">
  <h2>Real farms &mdash; {escape(title)}</h2>
  <p class="lede">{gallery_intro} Farms are picked per annotated class to span
  {escape(reference["key"] if reference else "the reference model")}&rsquo;s confidence
  &mdash; its highest-, median- and lowest-scoring farm for that class &mdash; and each group
  then adds the {escape(EXAMPLES_CONFUSED)} farms <em>not</em> annotated with that class that
  the shown models together rank highest for it. Click any image to enlarge it; the arrow
  keys step through the gallery, Escape closes it, and a group heading collapses its group.</p>
  <p class="gallery-key"><span><b>&#10003;</b> top class is annotated</span>
  <span><b>&#10007;</b> top class is not annotated; the annotated class&rsquo;s own score
  follows</span><span><b>&#8709;</b> the model has no output for any annotated class, so its
  top class is forced</span><span><b>&middot;</b> no annotation to compare with</span></p>
  {gallery}
</section>
"""
    return f"""
<section id="{anchor}">
  <h2>{escape(title)}</h2>
  <p class="lede">{intro}</p>
  <div class="tiles">{tiles}{agreement_card(result["agreement"], pr_models)}</div>
</section>
<section id="{anchor}-contrasts">
  <h2>The named contrasts &mdash; {escape(title)}</h2>
  <p class="lede">Each cell is the macro AP of the left model minus the right model&rsquo;s,
  both computed on the <em>same</em> bootstrap resamples, so the difference carries its own
  95&nbsp;% interval; a &#9679; marks an interval clear of zero. Within each block everything
  but the named thing is held fixed.</p>
  {contrast_cards}
  {class_card}
</section>
<section id="{anchor}-ap">
  <h2>Average precision by class &mdash; {escape(title)}</h2>
  <p class="lede">A random ranker scores the prevalence, marked on each row &mdash; a bar that
  stops near the tick has found nothing. Whiskers are 95&nbsp;% bootstrap intervals over the
  evaluation units. Open circles are the individual seed runs.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level &mdash; one annotated crop per unit</h3>
    {legend(models)}</div>
    <div class="scroll">{ap_chart(metrics[metrics["level"] == "crop"], scored, "crop")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; {escape(aggregation)} over each farm's crops</h3>
    {legend(models)}</div>
    <p class="lede">Farm labels come from the workbook&rsquo;s &ldquo;Farm labels&rdquo; sheet,
    which records livestock only; a farm or parcel with no farm label is a negative for every
    class.</p>
    <div class="scroll">{ap_chart(metrics[metrics["level"] == "farm"], scored, "farm")}</div>
  </div>
</section>
<section id="{anchor}-pairs">
  <h2>Every pair &mdash; {escape(title)}</h2>
  <p class="lede">The paired macro and the per-class differences for every pair among
  {escape(", ".join(pair_keys))}; every other pair is in <code>evaluation_pairs.csv</code> and
  <code>macro_pairs.csv</code>. A &#9679; marks an interval clear of zero.</p>
  <div class="card">
    <div class="card-head"><h3>{escape(macro_labels[0])}</h3></div>
    {macro_pair_card(macro_shown, pair_models, macro_labels[0])}
  </div>
  <div class="card">
    <div class="card-head"><h3>Crop level, by class</h3></div>
    {pair_table(pairs_shown, pair_models, "crop")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level, by class</h3></div>
    {pair_table(pairs_shown, pair_models, "farm")}
  </div>
</section>
<section id="{anchor}-regions">
  <h2>By {escape(region_word)} &mdash; {escape(title)}</h2>
  <p class="lede">Each {escape(region_word)} carries its own class mix and prevalence
  baseline; read the models against each other <em>within</em> one. A missing whisker means
  too few bootstrap draws kept every class.</p>
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
    <div class="card-head"><h3>Every class within each {escape(region_word)}, crop level</h3></div>
    {region_table(regions, models, "crop")}
  </div>
</section>
<section id="{anchor}-curves">
  <h2>The curves behind the numbers &mdash; {escape(title)}</h2>
  <p class="lede">Precision against recall at every threshold, crop level, one panel per model.
  The thin curves are the individual seed runs; the horizontal tick is the prevalence.</p>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level</h3>
      <div class="legend"><span><i class="key-seed"></i>thin line = one seed run</span>
      <span><i class="key-tick"></i>prevalence</span></div>
    </div>
    <div class="scroll">{pr_panels(truths["crop"], pr_models, "crop")}</div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Precision-recall, crop level, common classes merged &mdash; {merged_caption}</h3></div>
    <div class="scroll">{pr_panels(merge_truth(truths["crop"]), [merge_model(m) for m in pr_models], "crop", merged_order())}</div>
  </div>
</section>
<section id="{anchor}-leaks">
  <h2>Where the classes leak &mdash; {escape(title)}</h2>
  <p class="lede">The annotated class against the model&rsquo;s top-scoring class, shaded by row
  fraction. {escape(int(truths["crop"]["paddock"].sum()))} of the {escape(n_crops)} crops are
  paddock and {escape(int(truths["crop"]["other_industrial"].sum()))} other_industrial; a model
  with no output for them must put every one somewhere.</p>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <div class="confusions">
      {"".join(confusion_table(result["confusions"]["crop"][m["key"]], m) for m in pr_models)}
    </div>
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level &mdash; each annotated class against every class the model predicts</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <p class="lede">Farm labels are multi-label, so here the model&rsquo;s answer is a set
    too. {rule_note} The diagonal&rsquo;s row fraction is that class&rsquo;s <em>recall</em>.</p>
    <div class="confusions">
      {"".join(confusion_table(result["confusions"]["farm"][m["key"]], m, result["denominators"]["farm"][m["key"]]) for m in pr_models)}
    </div>
    <div class="card-head"><h3>Operating points behind the farm-level panel</h3></div>
    {operating_table(result["operating"], pr_models)}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level, common classes merged &mdash; {merged_caption}</h3>
      <div class="ramp">less{ramp}more of the row</div>{diag_key}
    </div>
    <div class="confusions">
      {"".join(confusion_table(result["confusions"]["farm_merged"][m["key"]], m, result["denominators"]["farm_merged"][m["key"]]) for m in pr_models)}
    </div>
    <div class="card-head"><h3>Operating points, merged classes</h3></div>
    {operating_table(result["operating_merged"], pr_models, merged_order())}
  </div>
</section>
{gallery_section}
<section id="{anchor}-full">
  <h2>Full results &mdash; {escape(title)}</h2>
  <div class="card">
    <div class="card-head"><h3>Crop level</h3></div>
    {table_of(metrics, models, "crop")}
  </div>
  <div class="card">
    <div class="card-head"><h3>Farm level ({escape(aggregation)} over crops)</h3></div>
    {table_of(metrics, models, "farm")}
  </div>
</section>
"""


def build_page(
    nsw: dict,
    vic: dict,
    galleries: dict[str, tuple[str, dict | None]],
    models: list[dict],
    args: argparse.Namespace,
    train_sizes: dict[str, int],
    train_by_class: dict[str, dict[str, int]],
) -> str:
    holdout_names = {HOLDOUT_NSW: "the gen-NSW hold-out", HOLDOUT_VIC: "the VIC hold-out"}
    n_w9 = sum(m["group"] == "wave9" and m["ensemble"] is not None for m in nsw["models"])
    n_prior = sum(m["group"] == "prior" and m["ensemble"] is not None for m in vic["models"])
    nsw_vic = int((nsw["crop_regions"] == "VIC").sum())
    nsw_nsw = int((nsw["crop_regions"] == "NSW").sum())
    target_pos = {c: int(nsw["crop"][c].sum()) for c in TARGET_CLASSES}
    target_pos_vic = {c: int(vic["crop"][c].sum()) for c in TARGET_CLASSES}

    masthead = f"""
<header class="masthead">
  <p class="eyebrow">2026-10-16 &middot; comfev2 wave 9 on FLIP &middot; two hold-outs &middot;
  {escape(len(nsw["test"]))} crops over {escape(len(nsw["farms"]))} farms, of which the
  {escape(len(vic["test"]))} VIC crops are every earlier page&rsquo;s hold-out</p>
  <h1>The v2 recipe against ComFe v1 and the probe on the gen-NSW split, with every earlier FLIP result beside it</h1>
  <p class="standfirst">Every earlier page scored models trained on the master build, which
  holds all the crop-labelled NSW rows, on the {escape(len(vic["test"]))}-crop VIC hold-out,
  where poultry has {target_pos_vic["poultry"]} positives and commercial pig
  {target_pos_vic["commercialpig"]}. The gen-NSW split set of 2026-10-09 moves every
  crop-labelled row to the test side: <code>test_autocrop_gen_vic_nsw</code> is the VIC
  hold-out plus {escape(nsw_nsw)} NSW crops ({target_pos["poultry"]} poultry,
  {target_pos["commercialpig"]} commercial pig, {target_pos["backyardpig"]} backyard pig), and
  train / val keep farm-level labels only. comfev2 wave 9 trained {n_w9} arms on it &mdash;
  ComFe v1 at two head sizes, the v2 recipe F3 and its readout fix F4 with three variants, a
  single-label probe, four reference prototype networks, and one-class ComFe models for the
  three target classes &mdash; all on DINOv2 ViT-B/14 without registers, ten output classes,
  four seeds. <b>The first half of the page</b> reads them on the primary hold-out, where
  only they can be read (every earlier model trained on its NSW half): paddock and
  industrial crops are negatives for every class, as in the wave&rsquo;s own scorer. <b>The
  second half</b> restricts their predictions to the VIC rows and puts them beside the
  {n_prior} earlier families, including the wave 7b F3 and wave 8 F4 arms on the 09_18
  master, so each head&rsquo;s split cost can be read with the backbone held fixed and the
  whole earlier story (head, data, backbone) stays on the page. Average precision by class
  throughout; the gen-NSW headline is the wave&rsquo;s three target classes, the VIC headline
  the same well-sampled six as every earlier page.</p>
</header>
"""
    nsw_tiles = macro_tiles_for(
        nsw["metrics"], nsw["seeds"], nsw["models"],
        [(MACRO_TARGET, "target 3"), (MACRO_SHARED, "shared 9")], MACRO_TARGET,
        f"<b>target 3</b> is the unweighted mean over poultry, commercial pig and backyard pig, "
        f"the classes the wave is judged on; <b>shared 9</b> over every livestock class, all of "
        f"which clear {MIN_POSITIVES} positives here (so the well-sampled macro of the earlier "
        f"pages is the shared nine on this hold-out). The seed-ensemble number is the headline; "
        f"<em>seeds</em> is the mean &plusmn; sd of the four seed runs' own macro, the footing of "
        f"the wave 9 document's tables (its mAP is the shared-9 seed mean). The one-class "
        f"models have no macro.",
    )
    vic_tiles = macro_tiles_for(
        vic["metrics"], vic["seeds"], vic["models"],
        [(MACRO_SAMPLED, "well sampled"), (MACRO_SHARED, "shared 9")], MACRO_SAMPLED,
        f"<b>well sampled</b> keeps the six livestock classes with at least {MIN_TRAIN_EXAMPLES} "
        f"training crops and {MIN_POSITIVES} test positives on this hold-out, the same six as "
        f"every earlier page; <b>shared 9</b> is every livestock class. Rank on the left-hand "
        f"column. The wave 9 arms are read on the same {escape(len(vic['test']))} crops, with "
        f"paddock and other_industrial outside their vocabulary.",
    )
    nsw_intro = (
        f"All {escape(len(nsw['test']))} crops, {escape(nsw_vic)} VIC and {escape(nsw_nsw)} NSW, "
        f"over {escape(len(nsw['farms']))} farms and parcels. The wave 9 arms trained on "
        f"<code>train_minus_gen_nsw_df.csv</code>, which carries farm-level labels only, so every "
        f"crop here is out of their training data and out of their region for the NSW half. "
        f"The {escape(int(nsw['crop']['paddock'].sum()))} paddock and "
        f"{escape(int(nsw['crop']['other_industrial'].sum()))} other_industrial crops have no class "
        f"in the ten-class vocabulary and count as negatives for every class: whether a model "
        f"fires on a shed-less field or an industrial site is the production question. "
        f"Checkpoint: the best on validation macro mAP, the training job's own test pass."
    )
    vic_intro = (
        f"The {escape(len(vic['test']))} VIC crops of every earlier page, matched row for row on "
        f"the key and every label. The thirteen prior families trained on the master builds, "
        f"which include the NSW crop labels; the wave 9 arms did not, and emit ten classes "
        f"rather than twelve. Head for head on ViT-B, the only change between a prior family "
        f"and its <code>w9_</code> twin is the split."
    )
    gallery_nsw, ref_nsw = galleries[HOLDOUT_NSW]
    gallery_vic, ref_vic = galleries[HOLDOUT_VIC]

    sections = [
        masthead,
        banner_for(models),
        f"""
<section id="roster">
  <h2>The roster</h2>
  <p class="lede">Prior families keep their 2026-09-18 keys; wave 9 arms are <code>w9_</code>
  plus the arm name. Every family is four seed runs where it is complete.</p>
  <h3>Wave 9, on the gen-NSW split &mdash; read on both hold-outs</h3>
  {roster_card([m for m in nsw["models"] if m["group"] == "wave9"], holdout_names)}
  <h3>Prior families, on the master builds &mdash; read on the VIC hold-out only</h3>
  {roster_card([m for m in vic["models"] if m["group"] == "prior"], holdout_names)}
</section>
""",
        holdout_sections(
            nsw, "the gen-NSW hold-out", nsw_intro, nsw_tiles, CONTRASTS_NSW, CLASS_CONTRASTS_NSW,
            PAIR_KEYS_NSW, gallery_nsw,
            "NSW farms and parcels only, where the target classes live; every building crop of a farm, "
            "its hand annotation in bold, and each shown model&rsquo;s top-scoring class with its score.",
            ref_nsw, [MACRO_TARGET, MACRO_SHARED], w9_metrics_table, "state", args.aggregation,
            args.farm_threshold, "nsw",
        ),
        holdout_sections(
            vic, "the VIC hold-out", vic_intro, vic_tiles, CONTRASTS_VIC, CLASS_CONTRASTS_VIC,
            PAIR_KEYS_VIC, gallery_vic,
            "The same farms as on every earlier page (picked on the 2026-09-04 ComFe), the wave 9 "
            "arms read on them beside the earlier families.",
            ref_vic, [MACRO_SAMPLED, MACRO_SHARED], metrics_table, "reach", args.aggregation,
            args.farm_threshold, "vic",
        ),
        f"""
<section id="training">
  <h2>What each build trained on</h2>
  <p class="lede">Crops carrying each class in the training split of each build. The gen-NSW
  split is the 09_18 master less every generalisation and generalisation_extra row and less
  the autocrops rows on their farms, so it has no crop-level labels and no paddock or
  other_industrial rows at all; the whole-farm split is its historical photographs alone.</p>
  <div class="card">
    {training_card(train_by_class, train_sizes)}
  </div>
</section>
""",
        f"""
<section>
  <h2>How this was scored</h2>
  <div class="notes">
    <p><strong>Two hold-outs, one truth.</strong> The <code>binary_*</code> columns of
    <code>original_master_2026_09_18/test_autocrop_gen_vic_nsw.csv</code> (checked against
    the <code>w9/</code> copy the runs read, row for row on the key and every label) and of the
    2026-09-04 <code>test_autocrop_gen_vic.csv</code>. The VIC rows of the first are matched
    to the second on <code>group_id</code> and checked on the key and every label, so the VIC
    section is on exactly the truth of every earlier page. Both hold-outs are scored at crop
    and farm level, farms as before (<code>farm_uid</code>, workbook farm labels; a parcel with
    no farm label is a negative throughout).</p>
    <p><strong>Where the wave 9 arms are read.</strong> Each wave 9 run saved one prediction
    for the whole primary hold-out, keyed on <code>group_id</code>; the VIC section is those
    rows restricted, not a second run. Their ten output classes are the run config's
    <code>binary_*</code> list for the ComFe arms (read back and checked) and the same order
    for the class-index arms (probe, references), whose softmax columns are saved in class
    order. A class a model does not emit is <em>no output</em> everywhere, never zero; a
    macro over a class subset needs every class of the subset.</p>
    <p><strong>Negatives by construction.</strong> On the primary hold-out, the
    {escape(int(nsw["crop"]["paddock"].sum()) + int(nsw["crop"]["other_industrial"].sum()))}
    paddock and other_industrial crops carry no class in the vocabulary and are negatives
    for every class, as in <code>tools/flip_perclass.py</code>; the per-seed means in the
    tiles and the <code>seed_macro.csv</code> reproduce that scorer's numbers (its mAP is the
    shared-9 seed mean, over the nine classes with a positive).</p>
    <p><strong>Four macros.</strong> <em>MACRO (target 3)</em> is the mean over poultry,
    commercial pig and backyard pig, the classes wave 9 is judged on. <em>MACRO (shared 9)</em>
    is every livestock class. <em>MACRO (well sampled)</em> keeps the shared classes with at
    least {MIN_TRAIN_EXAMPLES} training crops in both the 2026-08-21 and the master split and
    {MIN_POSITIVES} test positives: six on the VIC hold-out, as on every earlier page, and all
    nine on the primary one. <em>MACRO (evaluable)</em> is each model over its own scoreable
    classes. Each paired macro is computed on every bootstrap resample and differenced; a
    draw that loses every positive of one class is dropped for both models, and the interval
    is withheld if fewer than half survive.</p>
    <p><strong>Prior families.</strong> The eleven 2026-09-18 families are re-read from the
    same run directories and repeat that page exactly. <code>f3_b</code> (wave 7b
    <code>F3</code>) and <code>f4_b</code> (wave 8 <code>P1</code>, F3 + pool_proj linear, the
    arm adopted as F4) are scored here for the first time, so the two comfev2 recipe arms have
    a twin on the 09_18 master and the split cost can be read for the recipe as well as for
    ComFe v1. The sam3 pair's two abandoned 2026-09-25 launches are matched out by batch hash.</p>
    <p><strong>Dead runs.</strong> A run with no saved prediction is <em>training</em> if
    its log or a tensorboard event file changed in the last {STALE_HOURS} hours and
    <em>dead</em> otherwise. The two <code>noauto</code> arms and <code>nppp_K5_aug</code>
    crashed on launch (missing csvs in the tar; a shear augmentation on extreme-aspect crops)
    and are to be relaunched; their families stay in the roster so the page fills in once the
    view folder is refreshed.</p>
    <p><strong>Thin classes.</strong> On the VIC hold-out poultry rests on 3 crops, sheep on
    5, commercial pig on 8; their pairwise differences there are unreadable, which is what
    the primary hold-out is for ({target_pos["poultry"]} / 12 / {target_pos["commercialpig"]}).
    Commercial pig is still 18 positives: read it with its interval.</p>
    <p><strong>Uncertainty.</strong> Percentile bootstrap over the evaluation units, paired
    across models, {args.bootstrap} resamples. An interval is withheld when fewer than half
    the draws were usable.</p>
  </div>
</section>
""",
    ]
    return (
        "<title>2026-10-16 evaluation</title>"
        f"<style>{stylesheet()}</style>"
        f'<div class="page">{"".join(sections)}</div>'
        + LIGHTBOX
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def console_summary(result: dict, contrasts: list, macro_labels: list[str]) -> None:
    metrics, seeds, macro, scored = result["metrics"], result["seeds"], result["macro"], result["scored"]
    for level in LEVELS:
        title = f"{result['name']}: {level} level - average precision per class"
        print(f"\n{title}\n" + "-" * len(title))
        header = f"{'class':<22}{'pos':>5}{'prev':>7}"
        for model in scored:
            header += f"{model['key']:>13}"
        print(header)
        for _, row in metrics[metrics["level"] == level].iterrows():
            positives = "" if pd.isna(row.get("positives")) else int(row["positives"])
            prevalence = "" if pd.isna(row.get("prevalence")) else f"{row['prevalence']:.3f}"
            mark = "" if row.get("well_sampled", True) or str(row["class"]).startswith("MACRO") else " *"
            line = f"{str(row['class']) + mark:<22}{positives:>5}{prevalence:>7}"
            for model in scored:
                value = row.get(f"{model['key']}|AP")
                line += f"{'-':>13}" if pd.isna(value) else f"{value:>13.3f}"
            print(line)
        for label in macro_labels:
            rows = seeds[(seeds["level"] == level) & (seeds["macro"] == label)]
            if rows.empty:
                continue
            print(f"{label}, mean ± sd of per-seed AP:")
            print("  " + "  ".join(
                f"{row['model']} {row['AP_seed_mean']:.3f}±{row['AP_seed_sd']:.3f}"
                for _, row in rows.iterrows()
            ))
    title = f"{result['name']}: {macro_labels[0]} - the named contrasts"
    print(f"\n{title}\n" + "-" * len(title))
    present = {m["key"] for m in scored}
    for heading, items in contrasts:
        print(heading)
        for b, a, note in items:
            if b not in present or a not in present:
                continue
            for level in LEVELS:
                entry = contrast_lookup(macro, macro_labels[0], level, b, a)
                if entry is None:
                    continue
                mark = " *" if entry["clear"] else ""
                bounds = "" if pd.isna(entry["lo"]) else f" [{entry['lo']:+.3f}, {entry['hi']:+.3f}]"
                print(
                    f"  {level:<5} {b + ' - ' + a:<26} {entry['delta']:+.3f}{bounds}{mark}"
                    f"   ({b} {entry['AP_b']:.3f}, {a} {entry['AP_a']:.3f})  {note}"
                )
    print("* = interval clear of zero")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aggregation", default="max", choices=["max", "mean", "top2"],
                        help="how a farm's per-crop scores become one farm score (default max)")
    parser.add_argument("--farm-threshold", default="prevalence",
                        help="'prevalence' (default) predicts each class for as many farms as are "
                             "annotated with it; a number predicts every class scoring above it")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gallery", default="both", choices=["both", "vic", "nsw", "none"],
                        help="which farm galleries to build (the slow part)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--file", type=Path, default=None, help="where to write the dashboard")
    args = parser.parse_args()
    if args.farm_threshold != "prevalence":
        try:
            float(args.farm_threshold)
        except ValueError:
            raise SystemExit(f"--farm-threshold must be 'prevalence' or a number, not {args.farm_threshold!r}")

    # ---- the two hold-outs ----------------------------------------------------------
    test_vic = pd.read_csv(TEST_CSV, keep_default_na=False, low_memory=False)
    test_nsw = pd.read_csv(TEST_CSV_NSW, keep_default_na=False, low_memory=False)
    print(f"{TEST_CSV.name}: {len(test_vic)} crops, {test_vic['farm_uid'].nunique()} farms")
    print(f"{TEST_CSV_NSW.name}: {len(test_nsw)} crops, {test_nsw['farm_uid'].nunique()} farms, "
          + ", ".join(f"{r} {n}" for r, n in test_nsw["region"].value_counts().items()))

    if GEN_TEST_CSV.exists():
        gen_test = pd.read_csv(GEN_TEST_CSV, keep_default_na=False)
        if not (len(gen_test) == len(test_vic) and all(
            gen_test[k].astype(str).tolist() == test_vic[k].astype(str).tolist() for k in JOIN_KEY
        )):
            raise SystemExit(f"{TEST_CSV} is not {GEN_TEST_CSV} row for row")
    other = pd.read_csv(TEST_CSV_18, keep_default_na=False, low_memory=False)
    if not same_rows(test_vic, other):
        raise SystemExit(f"{TEST_CSV_18} is not {TEST_CSV} row for row on the key and labels")
    w9_copy = pd.read_csv(TEST_CSV_NSW_W9, keep_default_na=False, low_memory=False)
    if not same_rows(test_nsw, w9_copy) or not test_nsw["group_id"].astype(str).equals(w9_copy["group_id"].astype(str)):
        raise SystemExit(f"{TEST_CSV_NSW_W9} is not {TEST_CSV_NSW} row for row on the key, labels and group_id")
    print(f"w9/{TEST_CSV_NSW_W9.name} matches {TEST_CSV_NSW.name} row for row, labels included")

    # The VIC rows inside the primary hold-out, by group_id, checked on the key and labels.
    position = pd.Series(np.arange(len(test_nsw)), index=test_nsw["group_id"].astype(str))
    if position.index.duplicated().any():
        raise SystemExit(f"{TEST_CSV_NSW}: group_id is not unique")
    vic_rows = position.reindex(test_vic["group_id"].astype(str)).to_numpy()
    if np.isnan(vic_rows).any():
        raise SystemExit(f"{int(np.isnan(vic_rows).sum())} VIC crops are not in {TEST_CSV_NSW.name}")
    vic_rows = vic_rows.astype(int)
    if not same_rows(test_vic, test_nsw.iloc[vic_rows].reset_index(drop=True)):
        raise SystemExit(f"the VIC rows of {TEST_CSV_NSW.name} differ from {TEST_CSV.name} on the key or labels")
    print(f"the {len(vic_rows)} VIC crops of {TEST_CSV_NSW.name} match {TEST_CSV.name} on the key and every label")

    # ---- training counts --------------------------------------------------------------
    trained = train_counts(pd.read_csv(GEN_TRAIN_CSV, keep_default_na=False))
    train_sizes, train_by_class = {}, {}
    for label, path in (
        ("09_04 master", MASTER_TRAIN_CSV),
        ("09_18 master", MASTER_18_TRAIN_CSV),
        ("09_18 SAM3", SAM3_18_TRAIN_CSV),
        ("gen-NSW split", W9_TRAIN_CSV),
        ("whole farms only", NOAUTO_TRAIN_CSV),
    ):
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        train_sizes[label] = len(frame)
        train_by_class[label] = train_counts(frame)
        print(f"{label}: {len(frame)} crops, per class: " + ", ".join(
            f"{c} {train_by_class[label][c]}" for c in ALL_CLASSES if train_by_class[label][c]
        ))

    # ---- the families -----------------------------------------------------------------
    by_holdout = load_families(test_vic, test_nsw, vic_rows)
    args.output.mkdir(parents=True, exist_ok=True)

    nsw = score_holdout(
        HOLDOUT_NSW, test_nsw, by_holdout[HOLDOUT_NSW],
        test_nsw["region"].astype(str).str.upper().to_numpy(),
        train_by_class["gen-NSW split"], train_by_class["09_18 master"], args,
        extra_macros=[(MACRO_TARGET, TARGET_CLASSES)],
    )
    vic = score_holdout(
        HOLDOUT_VIC, test_vic, by_holdout[HOLDOUT_VIC],
        test_vic["source"].map(region_name).to_numpy(),
        trained, train_by_class["09_04 master"], args,
        extra_macros=[(MACRO_TARGET, TARGET_CLASSES)],
    )

    console_summary(nsw, CONTRASTS_NSW, [MACRO_TARGET, MACRO_SHARED])
    console_summary(vic, CONTRASTS_VIC, [MACRO_SAMPLED, MACRO_SHARED])

    # ---- galleries ----------------------------------------------------------------------
    galleries: dict[str, tuple[str, dict | None]] = {HOLDOUT_NSW: ("", None), HOLDOUT_VIC: ("", None)}
    if args.gallery in ("both", "nsw"):
        gallery, groups, reasons, reference, farms = build_gallery(
            nsw, REFERENCE_KEYS_NSW, GALLERY_KEYS_NSW, nsw["farm_regions"] == "NSW"
        )
        galleries[HOLDOUT_NSW] = (gallery, reference)
        examples_frame(groups, reasons, farms, reference).to_csv(nsw["out"] / "examples.csv", index=False)
        print(f"\ngallery {HOLDOUT_NSW}: {len(reasons)} farms over {len(groups)} groups, "
              f"selected on {reference['key'] if reference else 'nothing'}")
    if args.gallery in ("both", "vic"):
        gallery, groups, reasons, reference, farms = build_gallery(
            vic, REFERENCE_KEYS_VIC, GALLERY_KEYS_VIC, None
        )
        galleries[HOLDOUT_VIC] = (gallery, reference)
        examples_frame(groups, reasons, farms, reference).to_csv(vic["out"] / "examples.csv", index=False)
        print(f"gallery {HOLDOUT_VIC}: {len(reasons)} farms over {len(groups)} groups, "
              f"selected on {reference['key'] if reference else 'nothing'}")

    # ---- roster and training tables ----------------------------------------------------
    pd.DataFrame([
        {
            "key": m["key"],
            "model": m["label"],
            "group": m["group"],
            "head": m["head"],
            "backbone": m["backbone"],
            "data": m["data"],
            "wave": m["wave"],
            "holdouts": " ".join(m["holdouts"]),
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
        for m in by_holdout[HOLDOUT_NSW] + [m for m in by_holdout[HOLDOUT_VIC] if m["group"] == "prior"]
    ]).to_csv(args.output / "models.csv", index=False)
    pd.DataFrame([
        {"build": label, "crops": train_sizes[label], **{c: train_by_class[label].get(c, 0) for c in ALL_CLASSES}}
        for label in train_sizes
    ]).to_csv(args.output / "training_counts.csv", index=False)

    page = build_page(nsw, vic, galleries, by_holdout[HOLDOUT_NSW] + [m for m in by_holdout[HOLDOUT_VIC] if m["group"] == "prior"],
                      args, train_sizes, train_by_class)
    target = args.file or args.output / "evaluation_dashboard.html"
    target.write_text(page)
    print(f"\nwrote {target} and the CSVs in {args.output}/{{{HOLDOUT_NSW},{HOLDOUT_VIC}}}/")


if __name__ == "__main__":
    main()
