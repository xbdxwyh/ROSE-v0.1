#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the release-native ROSE matched local-count bridge."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

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

BRIDGE_TEMPLATE = "T2B_COUNT_LOCAL_MATCHED"
SOURCE_TEMPLATE = "T3_CLICK_LOCAL_NUMERIC"
REGION_CASE_ORDER = ["zero", "partial", "all"]


def click_clean(row: dict) -> int:
    return int(
        row.get("format_valid") == 1
        and int(row.get("invalid_coordinate_count") or 0) == 0
        and int(row.get("duplicate_click_count") or 0) == 0
    )


def summarize(rows: Sequence[dict]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}

    paired = [r for r in rows if r.get("base_lclk_matched") == 1]
    count_correct = [r for r in paired if r.get("pass") == 1]
    card_exact = [
        r for r in paired
        if r.get("base_lclk_cardinality_exact") == 1
    ]
    count_correct_card = [
        r for r in count_correct
        if r.get("base_lclk_cardinality_exact") == 1
    ]
    comparable = [
        r for r in paired
        if r.get("cross_task_cardinality_consistent") is not None
    ]

    out: Dict[str, Any] = {
        "n": len(rows),
        "mlcnt_pass_rate": mean(r.get("pass") for r in rows),
        "mlcnt_soft": mean(r.get("unified_soft") for r in rows),
        "mlcnt_valid_rate": mean(r.get("format_valid") for r in rows),
        "mlcnt_count_mae": mean(
            r.get("count_mae")
            for r in rows if r.get("pred_count") is not None
        ),
        "mlcnt_count_bias": mean(
            r.get("count_bias")
            for r in rows if r.get("pred_count") is not None
        ),
        "n_main_lclk_matched": len(paired),
        "n_mlcnt_correct": len(count_correct),
    }

    if not paired:
        out.update({
            "lclk_pass_rate": None,
            "lclk_soft": None,
            "lclk_valid_rate": None,
            "lclk_cardinality_exact_rate": None,
            "lclk_location_exact_given_cardinality": None,
            "lclk_click_f1_strict": None,
            "mlcnt_to_lclk_drop": None,
            "lclk_pass_given_mlcnt_correct": None,
            "transfer_failure_given_mlcnt_correct": None,
            "lclk_cardinality_exact_given_mlcnt_correct": None,
            "lclk_click_f1_given_mlcnt_correct": None,
            "lclk_location_exact_given_cardinality_and_mlcnt_correct": None,
            "cross_task_cardinality_consistency_rate": None,
            "both_correct_rate": None,
            "mlcnt_correct_lclk_wrong_rate": None,
        })
        return out

    mlcnt = mean(r.get("pass") for r in paired)
    lclk = mean(r.get("base_lclk_pass") for r in paired)
    out.update({
        "mlcnt_pass_on_paired": mlcnt,
        "lclk_pass_rate": lclk,
        "lclk_soft": mean(r.get("base_lclk_soft") for r in paired),
        "lclk_valid_rate": mean(r.get("base_lclk_format_valid") for r in paired),
        "lclk_cardinality_exact_rate": mean(
            r.get("base_lclk_cardinality_exact") for r in paired
        ),
        "lclk_location_exact_given_cardinality": (
            mean(r.get("base_lclk_pass") for r in card_exact)
            if card_exact else None
        ),
        "lclk_click_f1_strict": mean(
            r.get("base_lclk_click_f1_strict") for r in paired
        ),
        "mlcnt_to_lclk_drop": (
            float(mlcnt) - float(lclk)
            if mlcnt is not None and lclk is not None else None
        ),
        "both_correct_rate": mean(
            int(r.get("pass") == 1 and r.get("base_lclk_pass") == 1)
            for r in paired
        ),
        "mlcnt_correct_lclk_wrong_rate": mean(
            int(r.get("pass") == 1 and r.get("base_lclk_pass") != 1)
            for r in paired
        ),
        "cross_task_cardinality_consistency_rate": (
            mean(r.get("cross_task_cardinality_consistent") for r in comparable)
            if comparable else None
        ),
    })

    if count_correct:
        conditioned = mean(r.get("base_lclk_pass") for r in count_correct)
        out.update({
            "lclk_pass_given_mlcnt_correct": conditioned,
            "transfer_failure_given_mlcnt_correct": (
                1.0 - float(conditioned)
                if conditioned is not None else None
            ),
            "lclk_cardinality_exact_given_mlcnt_correct": mean(
                r.get("base_lclk_cardinality_exact") for r in count_correct
            ),
            "lclk_click_f1_given_mlcnt_correct": mean(
                r.get("base_lclk_click_f1_strict") for r in count_correct
            ),
        })
    else:
        out.update({
            "lclk_pass_given_mlcnt_correct": None,
            "transfer_failure_given_mlcnt_correct": None,
            "lclk_cardinality_exact_given_mlcnt_correct": None,
            "lclk_click_f1_given_mlcnt_correct": None,
        })

    out["lclk_location_exact_given_cardinality_and_mlcnt_correct"] = (
        mean(r.get("base_lclk_pass") for r in count_correct_card)
        if count_correct_card else None
    )
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
        description=(
            "Evaluate T2B_COUNT_LOCAL_MATCHED and pair it with original T3."
        )
    )
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument(
        "--main_eval",
        type=Path,
        required=True,
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
            f"Expected only {BRIDGE_TEMPLATE} rows, found other templates."
        )

    metadata_by_id, _ = unique_index(bridge_rows, "item_id")
    prediction_rows = read_jsonl(predictions_path)
    predictions_by_id, duplicate_predictions = unique_index(
        prediction_rows, "item_id", keep_last=True
    )
    model = infer_model(prediction_rows, args.model)

    expected = set(metadata_by_id)
    predicted = set(predictions_by_id)
    matched_ids = expected & predicted
    missing = sorted(expected - predicted)
    extra = sorted(predicted - expected)

    if missing and not args.allow_partial:
        raise ValueError(
            f"Missing {len(missing)} predictions. Use --allow_partial for samples."
        )
    if not matched_ids:
        raise ValueError("No prediction item IDs match the bridge metadata.")

    eval_rows: List[dict] = []
    for item_id in sorted(matched_ids):
        prediction = dict(predictions_by_id[item_id])
        prediction.setdefault("model", model)
        item = metadata_by_id[item_id]
        ev = core.evaluate_one(
            prediction,
            item,
            near_miss_l1=args.near_miss_l1,
        )
        ev.update({
            "source_item_id": item.get("source_item_id"),
            "source_click_item_id": item.get("source_click_item_id"),
            "bridge_type": item.get("bridge_type"),
        })
        eval_rows.append(ev)

    main_rows = load_main_eval(args.main_eval, args.main_model or model)
    t3_rows = [
        row for row in main_rows
        if row.get("task_template") == SOURCE_TEMPLATE
    ]
    by_item = {
        str(row.get("item_id")): row
        for row in t3_rows if row.get("item_id") is not None
    }
    by_scene = {
        (str(row.get("rose_subset")), str(row.get("scene_id"))): row
        for row in t3_rows if row.get("scene_id") is not None
    }

    target_mismatches = 0
    for row in eval_rows:
        base = by_item.get(str(row.get("source_click_item_id")))
        if base is None:
            base = by_item.get(str(row.get("source_item_id")))
        if base is None:
            base = by_scene.get(
                (str(row.get("rose_subset")), str(row.get("scene_id")))
            )

        if base is None:
            row["base_lclk_matched"] = 0
            continue

        clean = click_clean(base)
        gt_count = int(base.get("gt_count") or 0)
        pred_card = int(base.get("pred_click_count_unique") or 0)
        card_exact = int(clean == 1 and pred_card == gt_count)

        if int(row.get("gt_count") or 0) != gt_count:
            target_mismatches += 1

        bridge_pred = row.get("pred_count")
        cross_consistency = None
        if bridge_pred is not None and clean == 1:
            cross_consistency = int(int(bridge_pred) == pred_card)

        row.update({
            "base_lclk_matched": 1,
            "base_lclk_item_id": base.get("item_id"),
            "base_lclk_pass": int(base.get("pass") or 0),
            "base_lclk_soft": base.get("unified_soft"),
            "base_lclk_format_valid": int(base.get("format_valid") or 0),
            "base_lclk_clean_action": clean,
            "base_lclk_gt_count": gt_count,
            "base_lclk_pred_click_cardinality": pred_card,
            "base_lclk_cardinality_exact": card_exact,
            "base_lclk_click_f1_strict": base.get("click_f1_strict"),
            "base_lclk_region_violation_count": base.get(
                "region_violation_count"
            ),
            "cross_task_cardinality_consistent": cross_consistency,
        })

    if target_mismatches:
        raise ValueError(
            f"Found {target_mismatches} bridge/T3 target-count mismatches."
        )

    subset_buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in eval_rows:
        subset_buckets[str(row.get("rose_subset"))].append(row)

    by_subset = {
        subset: summarize(subset_buckets[subset])
        for subset in SUBSET_ORDER if subset in subset_buckets
    }
    macro = macro_summary(by_subset)
    micro = summarize(eval_rows)

    # Region-case analysis: equal-subset macro within each case.
    region_case_buckets: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in eval_rows:
        case = str(row.get("region_case") or "unknown")
        subset = str(row.get("rose_subset"))
        region_case_buckets[case][subset].append(row)

    by_region_case: Dict[str, Any] = {}
    region_case_csv: List[dict] = []
    ordered_cases = REGION_CASE_ORDER + sorted(
        case for case in region_case_buckets if case not in REGION_CASE_ORDER
    )
    for case in ordered_cases:
        if case not in region_case_buckets:
            continue
        subset_summaries = {
            subset: summarize(rows)
            for subset, rows in region_case_buckets[case].items()
        }
        case_macro = macro_summary(subset_summaries)
        by_region_case[case] = {
            "macro_equal_subset": case_macro,
            "by_subset": subset_summaries,
        }
        region_case_csv.append({
            "model": model,
            "region_case": case,
            **case_macro,
        })

    summary = {
        "task_template": BRIDGE_TEMPLATE,
        "source_task_template": SOURCE_TEMPLATE,
        "model": model,
        "split": args.split,
        "num_expected": len(expected),
        "num_predictions_loaded": len(prediction_rows),
        "num_predictions_used": len(eval_rows),
        "num_missing": len(missing),
        "num_extra": len(extra),
        "num_duplicate_prediction_rows": duplicate_predictions,
        "main_eval": str(args.main_eval),
        "macro_equal_subset": macro,
        "micro_pooled": micro,
        "by_subset": by_subset,
        "by_region_case": by_region_case,
        "metric_notes": {
            "mL-Cnt": (
                "Strict COUNT PASS on the exact same numeric region and target "
                "set as the paired original T3 local-click item."
            ),
            "L-Clk|mL-Cnt correct": (
                "Original T3 exact-set PASS restricted to pairs whose matched "
                "local count is correct."
            ),
            "Transfer failure": "1 - L-Clk|mL-Cnt correct.",
        },
    }

    write_jsonl(out_dir / "per_item_matched_count_eval.jsonl", eval_rows)
    write_jsonl(
        out_dir / "matched_count_error_cases.jsonl",
        [row for row in eval_rows if row.get("pass") != 1],
    )
    write_jsonl(
        out_dir / "mlcnt_correct_lclk_wrong.jsonl",
        [
            row for row in eval_rows
            if row.get("pass") == 1
            and row.get("base_lclk_matched") == 1
            and row.get("base_lclk_pass") != 1
        ],
    )
    write_json(out_dir / "matched_count_summary.json", summary)

    subset_csv = [
        {"model": model, "subset": subset, **metrics}
        for subset, metrics in by_subset.items()
    ]
    write_csv(out_dir / "matched_count_by_subset.csv", subset_csv)
    write_csv(out_dir / "matched_count_by_region_case.csv", region_case_csv)

    paper = {
        "Model": model,
        "mL-Cnt": pct(macro.get("mlcnt_pass_rate")),
        "L-Clk": pct(macro.get("lclk_pass_rate")),
        "L-Clk | mL-Cnt correct": pct(
            macro.get("lclk_pass_given_mlcnt_correct")
        ),
        "Card-Exact": pct(macro.get("lclk_cardinality_exact_rate")),
        "Card-Exact | mL-Cnt correct": pct(
            macro.get("lclk_cardinality_exact_given_mlcnt_correct")
        ),
        "Loc-Exact | Card-Exact": pct(
            macro.get("lclk_location_exact_given_cardinality")
        ),
        "Loc-Exact | Card & mL-Cnt": pct(
            macro.get(
                "lclk_location_exact_given_cardinality_and_mlcnt_correct"
            )
        ),
        "C-F1": pct(macro.get("lclk_click_f1_strict")),
        "mL-Cnt VALID": pct(macro.get("mlcnt_valid_rate")),
        "L-Clk VALID": pct(macro.get("lclk_valid_rate")),
        "mL-Cnt to L-Clk drop": pct(macro.get("mlcnt_to_lclk_drop")),
        "Transfer failure": pct(
            macro.get("transfer_failure_given_mlcnt_correct")
        ),
    }
    write_csv(out_dir / "matched_count_paper_table.csv", [paper])
    write_json(out_dir / "matched_count_paper_table.json", paper)

    print("=" * 96)
    print("ROSE matched local-count bridge evaluation")
    print("=" * 96)
    print(f"Model:          {model}")
    print(f"Evaluated:      {len(eval_rows)}/{len(expected)}")
    print(f"mL-Cnt:         {pct(macro.get('mlcnt_pass_rate'))}%")
    print(f"L-Clk:          {pct(macro.get('lclk_pass_rate'))}%")
    print(
        "L-Clk|mL-Cnt:  "
        f"{pct(macro.get('lclk_pass_given_mlcnt_correct'))}%"
    )
    print(
        "Transfer fail: "
        f"{pct(macro.get('transfer_failure_given_mlcnt_correct'))}%"
    )
    print(f"Output:         {out_dir}")
    print("=" * 96)


if __name__ == "__main__":
    main()
