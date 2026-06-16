#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-scene conditioned consistency analysis for release-native ROSE evals."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from rose_analysis_utils import (
    SUBSET_ORDER,
    mean,
    pct,
    read_jsonl,
    write_csv,
    write_json,
)

T1 = "T1_COUNT_GLOBAL"
TEMPLATE_ORDER = [
    "T1_COUNT_GLOBAL",
    "T2_COUNT_LOCAL_NUMERIC",
    "T3_CLICK_LOCAL_NUMERIC",
    "T4_CLICK_VISUAL_REGION",
    "T5_CLICK_COUNT_EXCLUSION",
]

TEMPLATE_SHORT = {
    "T1_COUNT_GLOBAL": "T1",
    "T2_COUNT_LOCAL_NUMERIC": "L-Cnt|T1",
    "T3_CLICK_LOCAL_NUMERIC": "L-Clk|T1",
    "T4_CLICK_VISUAL_REGION": "V-Clk|T1",
    "T5_CLICK_COUNT_EXCLUSION": "Excl-CS|T1",
}

ACTION_TEMPLATES = [
    "T3_CLICK_LOCAL_NUMERIC",
    "T4_CLICK_VISUAL_REGION",
    "T5_CLICK_COUNT_EXCLUSION",
]


def discover_files(paths: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("per_item_eval.jsonl")))
        else:
            raise FileNotFoundError(path)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No per_item_eval.jsonl files found.")
    return files


def collect(paths: Sequence[Path], split: str) -> Dict[str, List[dict]]:
    data: Dict[str, List[dict]] = defaultdict(list)
    for file in discover_files(paths):
        for row in read_jsonl(file):
            if split and str(row.get("rose_split")) != split:
                continue
            model = str(row.get("model") or "").strip()
            subset = str(row.get("rose_subset") or "").strip()
            template = str(row.get("task_template") or "").strip()
            if not model or subset not in SUBSET_ORDER:
                continue
            if template not in TEMPLATE_ORDER:
                continue
            data[model].append(row)
    return data


def scene_template_map(rows: Sequence[dict]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """subset -> scene -> template -> any-pass."""
    out: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        subset = str(row.get("rose_subset"))
        scene_id = str(row.get("scene_id"))
        template = str(row.get("task_template"))
        passed = int(row.get("pass") or 0)
        old = out[subset][scene_id].get(template, 0)
        out[subset][scene_id][template] = max(old, passed)
    return out


def analyze_model(model: str, rows: Sequence[dict]) -> Dict[str, Any]:
    mapping = scene_template_map(rows)
    by_subset: Dict[str, Any] = {}

    for subset in SUBSET_ORDER:
        scenes = mapping.get(subset, {})
        t1_scenes = [
            scene_id for scene_id, values in scenes.items()
            if T1 in values
        ]
        t1_correct = [
            scene_id for scene_id in t1_scenes
            if scenes[scene_id].get(T1) == 1
        ]

        rec: Dict[str, Any] = {
            "n_t1_scenes": len(t1_scenes),
            "n_t1_correct_scenes": len(t1_correct),
            "T1": (
                len(t1_correct) / len(t1_scenes)
                if t1_scenes else None
            ),
        }

        for template in TEMPLATE_ORDER[1:]:
            available = [
                scene_id for scene_id in t1_correct
                if template in scenes[scene_id]
            ]
            missing = len(t1_correct) - len(available)
            values = [
                scenes[scene_id][template] for scene_id in available
            ]
            key = TEMPLATE_SHORT[template]
            rec[key] = mean(values)
            rec[f"n_{key}"] = len(values)
            rec[f"missing_{key}"] = missing

        action_values = [
            rec.get(TEMPLATE_SHORT[template])
            for template in ACTION_TEMPLATES
            if rec.get(TEMPLATE_SHORT[template]) is not None
        ]
        rec["Action Ret."] = mean(action_values)
        rec["Action Fail."] = (
            1.0 - rec["Action Ret."]
            if rec["Action Ret."] is not None else None
        )
        by_subset[subset] = rec

    metric_keys = [
        "T1",
        "L-Cnt|T1",
        "L-Clk|T1",
        "V-Clk|T1",
        "Excl-CS|T1",
        "Action Ret.",
        "Action Fail.",
    ]
    macro = {
        key: mean(
            by_subset[subset].get(key)
            for subset in SUBSET_ORDER
            if by_subset[subset].get(key) is not None
        )
        for key in metric_keys
    }

    return {
        "model": model,
        "n_items": len(rows),
        "macro_equal_subset": macro,
        "by_subset": by_subset,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Condition T2-T5 performance on same-scene T1_COUNT_GLOBAL success."
        )
    )
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="per_item_eval.jsonl file(s) or directories containing them.",
    )
    p.add_argument("--split", default="test")
    p.add_argument("--models", default="")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--digits", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = collect(args.paths, args.split)

    if args.models.strip():
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        models = [model for model in requested if model in data]
        missing = [model for model in requested if model not in data]
        for model in missing:
            print(f"[WARN] Model not found: {model}")
    else:
        models = sorted(data)

    records = [analyze_model(model, data[model]) for model in models]
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "split": args.split,
        "condition": "same-scene T1_COUNT_GLOBAL pass == 1",
        "num_models": len(records),
        "models": records,
        "notes": {
            "macro": "Equal average over available ROSE subsets.",
            "paired_denominator": (
                "Each conditioned metric uses T1-correct scenes where the "
                "corresponding target template is available. Missing pairs are "
                "reported separately rather than counted as failures."
            ),
            "Action Ret.": "Mean of L-Clk|T1, V-Clk|T1 and Excl-CS|T1.",
        },
    }
    write_json(out_dir / f"scene_consistency_{args.split}.json", bundle)

    csv_rows: List[dict] = []
    latex_lines: List[str] = []
    for record in records:
        macro = record["macro_equal_subset"]
        csv_rows.append({
            "model": record["model"],
            **{
                key: pct(value, args.digits)
                for key, value in macro.items()
            },
        })
        cells = [
            record["model"],
            *[
                "--" if macro.get(key) is None
                else f"{100.0 * macro[key]:.{args.digits}f}"
                for key in [
                    "T1",
                    "L-Cnt|T1",
                    "L-Clk|T1",
                    "V-Clk|T1",
                    "Excl-CS|T1",
                    "Action Ret.",
                ]
            ],
        ]
        latex_lines.append(
            cells[0] + "\n& " + " & ".join(cells[1:]) + r" \\"
        )
        latex_lines.append("")

    write_csv(out_dir / f"scene_consistency_{args.split}.csv", csv_rows)
    (
        out_dir / f"scene_consistency_{args.split}_rows.tex"
    ).write_text("\n".join(latex_lines).rstrip() + "\n", encoding="utf-8")

    print("=" * 88)
    print("ROSE same-scene consistency analysis")
    print("=" * 88)
    for record in records:
        macro = record["macro_equal_subset"]
        def show(value):
            rendered = pct(value)
            return "--" if rendered is None else f"{rendered}%"

        print(
            f"{record['model']:<24} "
            f"T1={show(macro.get('T1'))}  "
            f"L-Cnt|T1={show(macro.get('L-Cnt|T1'))}  "
            f"ActionRet={show(macro.get('Action Ret.'))}"
        )
    print(f"Output: {out_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
