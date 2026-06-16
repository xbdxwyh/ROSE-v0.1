#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the release-native ROSE global-click bridge."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rose_analysis_utils import (
    SUBSET_ORDER,
    infer_model,
    import_release_evaluator,
    load_main_eval,
    mean,
    metadata_path,
    pct,
    read_jsonl,
    unique_index,
    write_csv,
    write_json,
    write_jsonl,
)

BRIDGE_TEMPLATE = "T1B_CLICK_GLOBAL"
SOURCE_TEMPLATE = "T1_COUNT_GLOBAL"


def summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    card_rows = [r for r in rows if r.get("cardinality_exact") == 1]
    paired = [r for r in rows if r.get("base_gcnt_matched") == 1]
    gcnt_correct = [r for r in paired if r.get("base_gcnt_pass") == 1]

    out: Dict[str, Any] = {
        "n": len(rows),
        "gclk_pass_rate": mean(r.get("pass") for r in rows),
        "gclk_soft": mean(r.get("unified_soft") for r in rows),
        "valid_rate": mean(r.get("format_valid") for r in rows),
        "api_error_rate": mean(r.get("api_error") for r in rows),
        "cardinality_exact_rate": mean(r.get("cardinality_exact") for r in rows),
        "cardinality_exact_nonstrict_rate": mean(
            r.get("cardinality_exact_nonstrict") for r in rows
        ),
        "cardinality_mae": mean(r.get("cardinality_abs_error") for r in rows),
        "cardinality_bias": mean(r.get("cardinality_error") for r in rows),
        "location_exact_given_cardinality": (
            mean(r.get("pass") for r in card_rows) if card_rows else None
        ),
        "click_precision": mean(r.get("click_precision") for r in rows),
        "click_recall": mean(r.get("click_recall") for r in rows),
        "click_f1_strict": mean(r.get("click_f1_strict") for r in rows),
        "click_iou_strict": mean(r.get("click_iou_strict") for r in rows),
        "invalid_coordinate_sample_rate": mean(
            int((r.get("invalid_coordinate_count") or 0) > 0) for r in rows
        ),
        "duplicate_click_sample_rate": mean(
            int((r.get("duplicate_click_count") or 0) > 0) for r in rows
        ),
        "n_cardinality_exact": len(card_rows),
        "n_main_gcnt_matched": len(paired),
        "n_main_gcnt_correct": len(gcnt_correct),
    }

    if paired:
        gcnt = mean(r.get("base_gcnt_pass") for r in paired)
        gclk = mean(r.get("pass") for r in paired)
        out.update({
            "gcnt_pass_rate": gcnt,
            "gclk_pass_on_paired": gclk,
            "gclk_minus_gcnt": (
                float(gclk) - float(gcnt)
                if gclk is not None and gcnt is not None else None
            ),
            "gcnt_to_gclk_drop": (
                float(gcnt) - float(gclk)
                if gclk is not None and gcnt is not None else None
            ),
            "both_correct_rate": mean(
                int(r.get("base_gcnt_pass") == 1 and r.get("pass") == 1)
                for r in paired
            ),
            "gcnt_correct_gclk_wrong_rate": mean(
                int(r.get("base_gcnt_pass") == 1 and r.get("pass") != 1)
                for r in paired
            ),
        })
    else:
        out.update({
            "gcnt_pass_rate": None,
            "gclk_pass_on_paired": None,
            "gclk_minus_gcnt": None,
            "gcnt_to_gclk_drop": None,
            "both_correct_rate": None,
            "gcnt_correct_gclk_wrong_rate": None,
        })

    if gcnt_correct:
        out.update({
            "gclk_pass_given_gcnt_correct": mean(
                r.get("pass") for r in gcnt_correct
            ),
            "cardinality_exact_given_gcnt_correct": mean(
                r.get("cardinality_exact") for r in gcnt_correct
            ),
            "click_f1_given_gcnt_correct": mean(
                r.get("click_f1_strict") for r in gcnt_correct
            ),
        })
    else:
        out.update({
            "gclk_pass_given_gcnt_correct": None,
            "cardinality_exact_given_gcnt_correct": None,
            "click_f1_given_gcnt_correct": None,
        })
    return out


def macro_summary(by_subset: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metrics = sorted({
        key
        for summary in by_subset.values()
        for key in summary
        if key != "n"
    })
    out: Dict[str, Any] = {
        "n": sum(int(summary.get("n") or 0) for summary in by_subset.values()),
        "num_subsets": len(by_subset),
    }
    for metric in metrics:
        out[metric] = mean(
            summary.get(metric)
            for summary in by_subset.values()
            if summary.get(metric) is not None
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate T1B_CLICK_GLOBAL predictions and pair them with main T1."
    )
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--main_eval",
        type=Path,
        default=None,
        help="Main per_item_eval.jsonl or a directory containing it.",
    )
    p.add_argument("--main_model", default="")
    p.add_argument("--model", default="")
    p.add_argument("--allow_partial", action="store_true")
    p.add_argument("--near_miss_l1", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    core = import_release_evaluator()

    dataset_root = args.dataset_root.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = read_jsonl(metadata_path(dataset_root, args.split))
    bridge_rows = [
        row for row in metadata_rows
        if row.get("task_template") == BRIDGE_TEMPLATE
    ]
    if len(bridge_rows) != len(metadata_rows):
        raise ValueError(
            f"Expected only {BRIDGE_TEMPLATE} rows, found "
            f"{len(metadata_rows) - len(bridge_rows)} other rows."
        )

    metadata_by_id, _ = unique_index(bridge_rows, "item_id")
    prediction_rows = read_jsonl(predictions_path)
    predictions_by_id, duplicate_predictions = unique_index(
        prediction_rows, "item_id", keep_last=True
    )
    model = infer_model(prediction_rows, args.model)

    expected = set(metadata_by_id)
    predicted = set(predictions_by_id)
    matched = expected & predicted
    missing = sorted(expected - predicted)
    extra = sorted(predicted - expected)

    if missing and not args.allow_partial:
        raise ValueError(
            f"Missing {len(missing)} predictions. Use --allow_partial for samples."
        )
    if not matched:
        raise ValueError("No prediction item IDs match the bridge metadata.")

    eval_rows: List[dict] = []
    for item_id in sorted(matched):
        prediction = dict(predictions_by_id[item_id])
        prediction.setdefault("model", model)
        item = metadata_by_id[item_id]
        ev = core.evaluate_one(
            prediction,
            item,
            near_miss_l1=args.near_miss_l1,
        )
        clean = int(
            ev.get("format_valid") == 1
            and int(ev.get("invalid_coordinate_count") or 0) == 0
            and int(ev.get("duplicate_click_count") or 0) == 0
        )
        pred_card = int(ev.get("pred_click_count_unique") or 0)
        gt_card = int(ev.get("gt_count") or 0)
        ev.update({
            "source_item_id": item.get("source_item_id"),
            "bridge_type": item.get("bridge_type"),
            "clean_action": clean,
            "predicted_click_cardinality": pred_card,
            "target_click_cardinality": gt_card,
            "cardinality_error": pred_card - gt_card,
            "cardinality_abs_error": abs(pred_card - gt_card),
            "cardinality_exact_nonstrict": int(pred_card == gt_card),
            "cardinality_exact": int(clean == 1 and pred_card == gt_card),
        })
        eval_rows.append(ev)

    main_rows = load_main_eval(args.main_eval, args.main_model or model)
    t1_rows = [
        row for row in main_rows
        if row.get("task_template") == SOURCE_TEMPLATE
    ]
    by_item = {
        str(row.get("item_id")): row
        for row in t1_rows if row.get("item_id") is not None
    }
    by_scene = {
        (str(row.get("rose_subset")), str(row.get("scene_id"))): row
        for row in t1_rows if row.get("scene_id") is not None
    }

    for row in eval_rows:
        base = by_item.get(str(row.get("source_item_id")))
        if base is None:
            base = by_scene.get(
                (str(row.get("rose_subset")), str(row.get("scene_id")))
            )
        row.update({
            "base_gcnt_matched": int(base is not None),
            "base_gcnt_item_id": base.get("item_id") if base else None,
            "base_gcnt_pass": int(base.get("pass") or 0) if base else None,
            "base_gcnt_soft": base.get("unified_soft") if base else None,
        })

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in eval_rows:
        buckets[str(row.get("rose_subset"))].append(row)

    by_subset = {
        subset: summarize(buckets[subset])
        for subset in SUBSET_ORDER if subset in buckets
    }
    macro = macro_summary(by_subset)
    micro = summarize(eval_rows)

    summary = {
        "task_template": BRIDGE_TEMPLATE,
        "model": model,
        "split": args.split,
        "num_expected": len(expected),
        "num_predictions_loaded": len(prediction_rows),
        "num_predictions_used": len(eval_rows),
        "num_missing": len(missing),
        "num_extra": len(extra),
        "num_duplicate_prediction_rows": duplicate_predictions,
        "main_eval": str(args.main_eval) if args.main_eval else None,
        "macro_equal_subset": macro,
        "micro_pooled": micro,
        "by_subset": by_subset,
        "metric_notes": {
            "G-Clk": "Strict exact-set PASS for global odd-cell clicking.",
            "Card-Exact": (
                "Unique valid click count equals the target count and the "
                "formal action is clean."
            ),
            "Loc-Exact|Card-Exact": (
                "Exact-set PASS among samples with correct click cardinality."
            ),
            "G-Clk|G-Cnt correct": (
                "Bridge PASS on paired scenes whose original T1 count is correct."
            ),
        },
    }

    write_jsonl(out_dir / "per_item_global_click_eval.jsonl", eval_rows)
    write_jsonl(
        out_dir / "global_click_error_cases.jsonl",
        [row for row in eval_rows if row.get("pass") != 1],
    )
    write_json(out_dir / "global_click_summary.json", summary)

    subset_csv = []
    for subset, metrics in by_subset.items():
        subset_csv.append({
            "model": model,
            "subset": subset,
            **metrics,
        })
    write_csv(out_dir / "global_click_by_subset.csv", subset_csv)

    paper = {
        "Model": model,
        "G-Cnt": pct(macro.get("gcnt_pass_rate")),
        "G-Clk": pct(macro.get("gclk_pass_rate")),
        "G-Clk | G-Cnt correct": pct(
            macro.get("gclk_pass_given_gcnt_correct")
        ),
        "Card-Exact": pct(macro.get("cardinality_exact_rate")),
        "Card-Exact | G-Cnt correct": pct(
            macro.get("cardinality_exact_given_gcnt_correct")
        ),
        "Loc-Exact | Card-Exact": pct(
            macro.get("location_exact_given_cardinality")
        ),
        "C-F1": pct(macro.get("click_f1_strict")),
        "VALID": pct(macro.get("valid_rate")),
        "G-Cnt to G-Clk drop": pct(macro.get("gcnt_to_gclk_drop")),
    }
    write_csv(out_dir / "global_click_paper_table.csv", [paper])
    write_json(out_dir / "global_click_paper_table.json", paper)

    print("=" * 92)
    print("ROSE global-click bridge evaluation")
    print("=" * 92)
    print(f"Model:       {model}")
    print(f"Evaluated:   {len(eval_rows)}/{len(expected)}")
    print(f"G-Clk:       {pct(macro.get('gclk_pass_rate'))}%")
    print(f"Card-Exact:  {pct(macro.get('cardinality_exact_rate'))}%")
    print(f"C-F1:        {pct(macro.get('click_f1_strict'))}%")
    if macro.get("gcnt_pass_rate") is not None:
        print(f"G-Cnt:       {pct(macro.get('gcnt_pass_rate'))}%")
        print(
            "G-Clk|G-Cnt: "
            f"{pct(macro.get('gclk_pass_given_gcnt_correct'))}%"
        )
    print(f"Output:      {out_dir}")
    print("=" * 92)


if __name__ == "__main__":
    main()
