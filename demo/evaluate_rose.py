#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate ROSE v0.1 predictions against the public Hugging Face release format.

The evaluator reads one prediction JSONL file and one release metadata file:

  ROSE-v0.1-hf/
    metadata_dev.jsonl
    metadata_test.jsonl

Images are not required for evaluation.

Primary outputs
---------------
  per_item_eval.jsonl       Per-item strict and diagnostic metrics.
  error_cases.jsonl         Failed items only.
  evaluation_summary.json   Coverage, aggregate metrics, and paper-style tables.
  summary_by_subset.csv     Table-1 source metrics.
  summary_by_template.csv   Table-2 template metrics.
  table1_row.txt            One LaTeX-ready Table-1 row.
  table2_row.txt            One LaTeX-ready Table-2 row.

The definitions match the original ROSE evaluation scripts:
  PASS  = exact task success under the formal output grammar.
  VALID = grammar-valid output rate.
  SOFT  = COUNT: 1/(1+absolute count error)
          CLICK: strict click-set F1
          CLICK_COUNT: mean(strict click F1, count soft,
                            click-submit consistency)

Example: evaluate a 250-item partial test run
---------------------------------------------
  python evaluate_rose.py ^
    --dataset_root ".\\ROSE-v0.1-hf" ^
    --predictions ".\\outputs\\qwen36plus_test_50.jsonl" ^
    --split test ^
    --out_dir ".\\outputs\\eval_qwen36plus_test_250" ^
    --allow_partial

Example: evaluate a complete test submission
--------------------------------------------
  python evaluate_rose.py ^
    --dataset_root ".\\ROSE-v0.1-hf" ^
    --predictions ".\\predictions.jsonl" ^
    --split test ^
    --out_dir ".\\evaluation"

Dependencies
------------
No third-party package is needed for local evaluation.
When --dataset_root is omitted, install huggingface_hub so the metadata can be
retrieved automatically:

  pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_REPO_ID = "sysuwyh357/ROSE-v0.1"

SUBSET_ORDER = [
    "ROSE-ChineseGlyph",
    "ROSE-EmojiStyle",
    "ROSE-EmojiContent",
    "ROSE-PixelEdit",
    "ROSE-PixelContent",
]

SUBSET_SHORT = {
    "ROSE-ChineseGlyph": "ChineseGlyph",
    "ROSE-EmojiStyle": "EmojiStyle",
    "ROSE-EmojiContent": "EmojiContent",
    "ROSE-PixelEdit": "PixelEdit",
    "ROSE-PixelContent": "PixelContent",
}

TEMPLATE_ORDER = [
    "T1_COUNT_GLOBAL",
    "T2_COUNT_LOCAL_NUMERIC",
    "T3_CLICK_LOCAL_NUMERIC",
    "T4_CLICK_VISUAL_REGION",
    "T5_CLICK_COUNT_EXCLUSION",
]

TEMPLATE_SHORT = {
    "T1_COUNT_GLOBAL": "G-Cnt",
    "T2_COUNT_LOCAL_NUMERIC": "L-Cnt",
    "T3_CLICK_LOCAL_NUMERIC": "L-Clk",
    "T4_CLICK_VISUAL_REGION": "V-Clk",
    "T5_CLICK_COUNT_EXCLUSION": "Excl. C+S",
}

MODEL_DISPLAY = {
    "qwen3-vl-flash": "Qwen3-VL-Flash",
    "qwen3-vl-plus": "Qwen3-VL-Plus",
    "qwen3.6-plus": "Qwen3.6-Plus",
    "claude46sonnet": "Claude-Sonnet-4.6",
    "claude-sonnet-4.6": "Claude-Sonnet-4.6",
    "claude48opus": "Claude-Opus-4.8",
    "claude-opus-4.8": "Claude-Opus-4.8",
    "glm-4.6v": "GLM-4.6V",
    "glm46v": "GLM-4.6V",
    "glm-5v-turbo": "GLM-5V-Turbo",
    "glm5vturbo": "GLM-5V-Turbo",
    "gemini31pro": "Gemini-3.1-Pro",
    "gemini-3.1-pro": "Gemini-3.1-Pro",
    "gpt55none": "GPT-5.5",
    "gpt-5.5": "GPT-5.5",
    "human": "Human",
}

COORD_RE = re.compile(r"^\s*R\s*(\d+)\s*,\s*C\s*(\d+)\s*$", re.I)
COUNT_RE = re.compile(r"^\s*COUNT\s*\(\s*(\d+)\s*\)\s*$", re.I)
CLICK_RE = re.compile(
    r"^\s*CLICK\s*\(\s*R\s*(\d+)\s*,\s*C\s*(\d+)\s*\)\s*$",
    re.I,
)
SUBMIT_RE = re.compile(r"^\s*SUBMIT\s*\(\s*(\d+)\s*\)\s*$", re.I)
DONE_RE = re.compile(r"^\s*DONE\s*$", re.I)


# =============================================================================
# Basic I/O
# =============================================================================


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: Iterable[Any]) -> Optional[float]:
    vals = [v for v in (safe_float(x) for x in values) if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def pct(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(100.0 * value, digits)


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.{digits}f}"


# =============================================================================
# Dataset and prediction loading
# =============================================================================


def resolve_metadata_path(
    *,
    dataset_root: Optional[Path],
    repo_id: str,
    revision: Optional[str],
    split: str,
) -> Path:
    filename = f"metadata_{split}.jsonl"

    if dataset_root is not None:
        path = dataset_root.expanduser().resolve() / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing release metadata: {path}")
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required when --dataset_root is omitted. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": filename,
    }
    if revision:
        kwargs["revision"] = revision

    return Path(hf_hub_download(**kwargs)).resolve()


def unique_index(rows: Sequence[dict], key: str, source: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    duplicates: List[str] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            raise ValueError(f"Missing '{key}' in {source}")
        value_s = str(value)
        if value_s in out:
            duplicates.append(value_s)
        out[value_s] = row
    if duplicates:
        raise ValueError(
            f"Duplicate {key} values in {source}: {sorted(set(duplicates))[:10]}"
        )
    return out


def prediction_index(rows: Sequence[dict], source: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    duplicates: List[str] = []
    for row in rows:
        item_id = row.get("item_id")
        if item_id is None:
            raise ValueError(f"Prediction row without item_id in {source}")
        item_id_s = str(item_id)
        if item_id_s in out:
            duplicates.append(item_id_s)
        out[item_id_s] = row
    if duplicates:
        raise ValueError(
            "Duplicate item_id values in the prediction file. "
            f"First duplicates: {sorted(set(duplicates))[:10]}"
        )
    return out


def infer_model(predictions: Sequence[dict], override: str, source: Path) -> str:
    if override.strip():
        return override.strip()

    models = {
        str(row.get("model")).strip()
        for row in predictions
        if row.get("model") is not None and str(row.get("model")).strip()
    }
    if len(models) == 1:
        return next(iter(models))
    if len(models) > 1:
        raise ValueError(
            f"Multiple model names found in {source}: {sorted(models)}. "
            "Pass --model explicitly."
        )
    return source.stem


# =============================================================================
# Formal answer parser
# =============================================================================


@dataclass
class ParsedAnswer:
    format_valid: bool
    parse_error: Optional[str]
    pred_count: Optional[int]
    submit_count: Optional[int]
    click_actions: List[Tuple[int, int]]
    invalid_action_tokens: List[str]
    end_token: Optional[str]


def parse_formal_answer(raw_text: Optional[str], task_mode: str) -> ParsedAnswer:
    """Parse the exact ROSE output grammar; surrounding prose is invalid."""
    text = "" if raw_text is None else str(raw_text).strip()
    if not text:
        return ParsedAnswer(False, "empty", None, None, [], [], None)

    if task_mode == "COUNT":
        match = COUNT_RE.fullmatch(text)
        if not match:
            return ParsedAnswer(
                False,
                "expected_COUNT_only",
                None,
                None,
                [],
                [text],
                None,
            )
        return ParsedAnswer(True, None, int(match.group(1)), None, [], [], "COUNT")

    parts = [part.strip() for part in text.split(";") if part.strip()]

    if task_mode == "CLICK":
        if len(parts) == 1 and DONE_RE.fullmatch(parts[0]):
            return ParsedAnswer(True, None, None, None, [], [], "DONE")

        if not parts or not DONE_RE.fullmatch(parts[-1]):
            return ParsedAnswer(
                False,
                "missing_DONE",
                None,
                None,
                [],
                parts,
                None,
            )

        clicks: List[Tuple[int, int]] = []
        invalid: List[str] = []
        for token in parts[:-1]:
            match = CLICK_RE.fullmatch(token)
            if match:
                clicks.append((int(match.group(1)), int(match.group(2))))
            else:
                invalid.append(token)

        if invalid:
            return ParsedAnswer(
                False,
                "invalid_CLICK_token",
                None,
                None,
                clicks,
                invalid,
                "DONE",
            )
        return ParsedAnswer(True, None, None, None, clicks, [], "DONE")

    if task_mode == "CLICK_COUNT":
        if not parts:
            return ParsedAnswer(False, "empty", None, None, [], [], None)

        submit_match = SUBMIT_RE.fullmatch(parts[-1])
        if not submit_match:
            return ParsedAnswer(
                False,
                "missing_SUBMIT",
                None,
                None,
                [],
                parts,
                None,
            )

        clicks = []
        invalid = []
        for token in parts[:-1]:
            match = CLICK_RE.fullmatch(token)
            if match:
                clicks.append((int(match.group(1)), int(match.group(2))))
            else:
                invalid.append(token)

        submit_count = int(submit_match.group(1))
        if invalid:
            return ParsedAnswer(
                False,
                "invalid_CLICK_token",
                None,
                submit_count,
                clicks,
                invalid,
                "SUBMIT",
            )
        return ParsedAnswer(
            True,
            None,
            None,
            submit_count,
            clicks,
            [],
            "SUBMIT",
        )

    return ParsedAnswer(
        False,
        f"unknown_task_mode:{task_mode}",
        None,
        None,
        [],
        [text],
        None,
    )


# =============================================================================
# Coordinate and region helpers
# =============================================================================


def parse_coord(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    match = COORD_RE.fullmatch(str(value))
    if not match:
        raise ValueError(f"Invalid coordinate label: {value}")
    return int(match.group(1)), int(match.group(2))


def coord_str(cell: Tuple[int, int]) -> str:
    return f"R{cell[0]},C{cell[1]}"


def coord_set(values: Any) -> Set[Tuple[int, int]]:
    if values is None:
        return set()
    if isinstance(values, str):
        # Accept one coordinate string or a free-form string containing several labels.
        matches = re.findall(r"R\s*(\d+)\s*,\s*C\s*(\d+)", values, flags=re.I)
        return {(int(r), int(c)) for r, c in matches}
    if isinstance(values, Sequence):
        return {parse_coord(value) for value in values}
    raise ValueError(f"Unsupported coordinate collection: {type(values).__name__}")


def compute_region_cells(row: dict) -> Set[Tuple[int, int]]:
    rows = int(row["rows"])
    cols = int(row["cols"])
    region_type = str(row.get("region_type"))
    params = dict(row.get("region_params") or {})

    all_cells = {
        (r, c)
        for r in range(1, rows + 1)
        for c in range(1, cols + 1)
    }

    if region_type == "GLOBAL":
        return all_cells
    if region_type == "ROW_RANGE":
        r1, r2 = int(params["r1"]), int(params["r2"])
        return {
            (r, c)
            for r in range(r1, r2 + 1)
            for c in range(1, cols + 1)
        }
    if region_type == "COL_RANGE":
        c1, c2 = int(params["c1"]), int(params["c2"])
        return {
            (r, c)
            for r in range(1, rows + 1)
            for c in range(c1, c2 + 1)
        }
    if region_type in {"RECTANGLE", "VISUAL_REGION"}:
        r1, r2 = int(params["r1"]), int(params["r2"])
        c1, c2 = int(params["c1"]), int(params["c2"])
        return {
            (r, c)
            for r in range(r1, r2 + 1)
            for c in range(c1, c2 + 1)
        }
    if region_type == "EXCLUSION":
        nested = dict(row)
        nested["region_type"] = str(params["exclude_type"])
        nested["region_params"] = dict(params["exclude_params"])
        return all_cells - compute_region_cells(nested)

    raise ValueError(f"Unknown region_type: {region_type}")


def split_valid_clicks(
    clicks: Sequence[Tuple[int, int]],
    rows: int,
    cols: int,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    valid: List[Tuple[int, int]] = []
    invalid: List[Tuple[int, int]] = []
    for cell in clicks:
        if 1 <= cell[0] <= rows and 1 <= cell[1] <= cols:
            valid.append(cell)
        else:
            invalid.append(cell)
    return valid, invalid


def precision_recall_f1(
    pred: Set[Tuple[int, int]],
    gold: Set[Tuple[int, int]],
) -> Tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0

    overlap = len(pred & gold)
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def jaccard(pred: Set[Tuple[int, int]], gold: Set[Tuple[int, int]]) -> float:
    if not pred and not gold:
        return 1.0
    union = pred | gold
    return len(pred & gold) / len(union) if union else 0.0


def l1(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def near_miss_count(
    missed_gold: Set[Tuple[int, int]],
    pred: Set[Tuple[int, int]],
    radius: int,
) -> int:
    return sum(
        1
        for gold_cell in missed_gold
        if any(l1(gold_cell, pred_cell) <= radius for pred_cell in pred)
    )


def count_soft(error_abs: Optional[int]) -> float:
    if error_abs is None:
        return 0.0
    return 1.0 / (1.0 + float(error_abs))


# =============================================================================
# Per-item evaluation
# =============================================================================


def evaluate_one(
    prediction: dict,
    item: dict,
    *,
    near_miss_l1: int,
) -> dict:
    task_mode = str(item.get("task_mode"))
    task_template = str(item.get("task_template"))
    region_type = str(item.get("region_type"))

    rows = int(item["rows"])
    cols = int(item["cols"])
    region_cells = compute_region_cells(item)

    parsed = parse_formal_answer(prediction.get("raw_text"), task_mode)

    gt_count = int(item.get("ground_truth_count") or 0)
    gt_clicks = coord_set(item.get("ground_truth_clicks") or [])
    all_targets = coord_set(
        item.get("all_target_cells")
        or item.get("target_cells")
        or []
    )
    region_targets = coord_set(
        item.get("region_target_cells")
        or item.get("ground_truth_clicks")
        or []
    )
    out_region_targets = all_targets - region_targets

    valid_clicks, invalid_clicks = split_valid_clicks(
        parsed.click_actions,
        rows,
        cols,
    )
    pred_click_set = set(valid_clicks)
    duplicate_click_count = len(valid_clicks) - len(pred_click_set)

    click_applicable = task_mode in {"CLICK", "CLICK_COUNT"}
    count_applicable = task_mode in {"COUNT", "CLICK_COUNT"}
    submit_applicable = task_mode == "CLICK_COUNT"

    precision, recall, click_f1 = precision_recall_f1(pred_click_set, gt_clicks)
    click_iou = jaccard(pred_click_set, gt_clicks)

    click_clean = (
        parsed.format_valid
        and not invalid_clicks
        and duplicate_click_count == 0
    )
    click_f1_strict = click_f1 if click_applicable and click_clean else (
        0.0 if click_applicable else None
    )
    click_iou_strict = click_iou if click_applicable and click_clean else (
        0.0 if click_applicable else None
    )

    if task_mode == "COUNT":
        pred_count = parsed.pred_count
    elif task_mode == "CLICK_COUNT":
        pred_count = parsed.submit_count
    else:
        pred_count = None

    count_error = abs(pred_count - gt_count) if pred_count is not None else None
    count_bias = pred_count - gt_count if pred_count is not None else None
    count_exact = int(pred_count == gt_count) if count_applicable and pred_count is not None else (
        0 if count_applicable else None
    )
    count_soft_score = count_soft(count_error) if count_applicable else None

    click_exact = int(pred_click_set == gt_clicks) if click_applicable else None
    click_exact_strict = int(click_clean and pred_click_set == gt_clicks) if click_applicable else None

    submit_exact = None
    click_submit_consistency = None
    if submit_applicable:
        submit_exact = int(parsed.submit_count == gt_count) if parsed.submit_count is not None else 0
        click_submit_consistency = (
            int(parsed.submit_count == len(pred_click_set))
            if parsed.submit_count is not None
            else 0
        )

    passed = 0
    if task_mode == "COUNT":
        passed = int(parsed.format_valid and parsed.pred_count == gt_count)
    elif task_mode == "CLICK":
        passed = int(
            parsed.format_valid
            and not invalid_clicks
            and duplicate_click_count == 0
            and pred_click_set == gt_clicks
        )
    elif task_mode == "CLICK_COUNT":
        passed = int(
            parsed.format_valid
            and not invalid_clicks
            and duplicate_click_count == 0
            and pred_click_set == gt_clicks
            and parsed.submit_count == gt_count
            and parsed.submit_count == len(pred_click_set)
        )

    if task_mode == "COUNT":
        unified_soft = count_soft_score or 0.0
    elif task_mode == "CLICK":
        unified_soft = click_f1_strict or 0.0
    elif task_mode == "CLICK_COUNT":
        unified_soft = (
            (click_f1_strict or 0.0)
            + (count_soft_score or 0.0)
            + float(click_submit_consistency or 0)
        ) / 3.0
    else:
        unified_soft = 0.0

    region_violations = pred_click_set - region_cells
    outside_distractor_hits = pred_click_set & out_region_targets
    missed_targets = gt_clicks - pred_click_set
    false_positive_clicks = pred_click_set - gt_clicks

    global_count = len(all_targets)
    global_default_error = 0
    if task_template in {
        "T2_COUNT_LOCAL_NUMERIC",
        "T3_CLICK_LOCAL_NUMERIC",
        "T4_CLICK_VISUAL_REGION",
        "T5_CLICK_COUNT_EXCLUSION",
    }:
        if task_mode == "COUNT":
            if pred_count is not None and gt_count != global_count:
                global_default_error = int(pred_count == global_count)
        elif task_mode == "CLICK":
            if gt_clicks != all_targets:
                global_default_error = int(pred_click_set == all_targets)
        elif task_mode == "CLICK_COUNT":
            click_global = gt_clicks != all_targets and pred_click_set == all_targets
            count_global = (
                pred_count is not None
                and gt_count != global_count
                and pred_count == global_count
            )
            global_default_error = int(click_global or count_global)

    exclusion_failure = 0
    if region_type == "EXCLUSION":
        if click_applicable:
            exclusion_failure = int(bool(pred_click_set & out_region_targets))
        elif pred_count is not None and gt_count != global_count:
            exclusion_failure = int(pred_count == global_count)

    api_error = int(prediction.get("error") not in {None, "", "None"})
    near_miss = near_miss_count(missed_targets, pred_click_set, near_miss_l1) if click_applicable else None

    return {
        "item_id": item.get("item_id"),
        "scene_id": item.get("scene_id"),
        "image_id": item.get("image_id"),
        "model": prediction.get("model"),
        "rose_subset": item.get("rose_subset"),
        "rose_split": item.get("split"),
        "primitive_type": item.get("primitive_type"),
        "task_template": task_template,
        "task_mode": task_mode,
        "region_type": region_type,
        "region_case": item.get("region_case"),
        "cue_type": item.get("cue_type"),
        "difficulty": item.get("difficulty"),
        "rows": rows,
        "cols": cols,
        "region_cell_count": len(region_cells),
        "raw_text": prediction.get("raw_text"),
        "api_error": api_error,
        "api_error_text": prediction.get("error"),
        "format_valid": int(parsed.format_valid),
        "format_error": int(not parsed.format_valid),
        "parse_error": parsed.parse_error,
        "invalid_action_tokens": parsed.invalid_action_tokens,
        "pass": passed,
        "unified_soft": unified_soft,
        "count_applicable": int(count_applicable),
        "click_applicable": int(click_applicable),
        "submit_applicable": int(submit_applicable),
        "gt_count": gt_count,
        "pred_count": pred_count,
        "count_exact": count_exact,
        "count_mae": count_error,
        "count_bias": count_bias,
        "count_soft": count_soft_score,
        "gt_clicks": sorted(coord_str(x) for x in gt_clicks),
        "pred_clicks_all": [coord_str(x) for x in parsed.click_actions],
        "pred_clicks_valid": [coord_str(x) for x in valid_clicks],
        "pred_clicks_valid_unique": sorted(coord_str(x) for x in pred_click_set),
        "pred_clicks_invalid": [coord_str(x) for x in invalid_clicks],
        "pred_click_count_raw": len(parsed.click_actions),
        "pred_click_count_valid": len(valid_clicks),
        "pred_click_count_unique": len(pred_click_set),
        "invalid_coordinate_count": len(invalid_clicks) if click_applicable else None,
        "duplicate_click_count": duplicate_click_count if click_applicable else None,
        "click_exact_set": click_exact,
        "click_exact_set_strict": click_exact_strict,
        "click_precision": precision if click_applicable else None,
        "click_recall": recall if click_applicable else None,
        "click_f1": click_f1 if click_applicable else None,
        "click_f1_strict": click_f1_strict,
        "click_iou": click_iou if click_applicable else None,
        "click_iou_strict": click_iou_strict,
        "submit_count": parsed.submit_count,
        "submit_exact": submit_exact,
        "click_submit_consistency": click_submit_consistency,
        "region_violation_count": len(region_violations) if click_applicable else None,
        "region_ok": int(not region_violations) if click_applicable else None,
        "outside_distractor_hit_count": len(outside_distractor_hits) if click_applicable else None,
        "missed_target_count": len(missed_targets) if click_applicable else None,
        "false_positive_count": len(false_positive_clicks) if click_applicable else None,
        "over_click_count": max(0, len(pred_click_set) - gt_count) if click_applicable else None,
        "under_click_count": max(0, gt_count - len(pred_click_set)) if click_applicable else None,
        "near_miss_count": near_miss,
        "near_miss_rate": (
            near_miss / max(1, len(gt_clicks))
            if click_applicable and near_miss is not None
            else None
        ),
        "global_default_error": global_default_error,
        "exclusion_failure": exclusion_failure,
        "all_target_cells": sorted(coord_str(x) for x in all_targets),
        "region_target_cells": sorted(coord_str(x) for x in region_targets),
        "out_region_target_cells": sorted(coord_str(x) for x in out_region_targets),
        "ground_truth_formal_answer": item.get("ground_truth_formal_answer"),
    }


# =============================================================================
# Aggregation
# =============================================================================


def summary_for(rows: Sequence[dict]) -> dict:
    if not rows:
        return {"n": 0}

    count_rows = [row for row in rows if row.get("count_applicable") == 1]
    click_rows = [row for row in rows if row.get("click_applicable") == 1]
    submit_rows = [row for row in rows if row.get("submit_applicable") == 1]
    exclusion_rows = [row for row in rows if row.get("region_type") == "EXCLUSION"]

    return {
        "n": len(rows),
        "pass_rate": mean(row.get("pass") for row in rows),
        "unified_soft": mean(row.get("unified_soft") for row in rows),
        "valid_rate": mean(row.get("format_valid") for row in rows),
        "format_error_rate": mean(row.get("format_error") for row in rows),
        "api_error_rate": mean(row.get("api_error") for row in rows),
        "n_count_applicable": len(count_rows),
        "count_exact_rate": mean(row.get("count_exact") for row in count_rows),
        "count_mae_parsed": mean(
            row.get("count_mae")
            for row in count_rows
            if row.get("pred_count") is not None
        ),
        "count_bias_parsed": mean(
            row.get("count_bias")
            for row in count_rows
            if row.get("pred_count") is not None
        ),
        "count_soft": mean(row.get("count_soft") for row in count_rows),
        "n_click_applicable": len(click_rows),
        "click_exact_set_rate": mean(row.get("click_exact_set") for row in click_rows),
        "click_exact_set_strict_rate": mean(
            row.get("click_exact_set_strict") for row in click_rows
        ),
        "click_precision": mean(row.get("click_precision") for row in click_rows),
        "click_recall": mean(row.get("click_recall") for row in click_rows),
        "click_f1": mean(row.get("click_f1") for row in click_rows),
        "click_f1_strict": mean(row.get("click_f1_strict") for row in click_rows),
        "click_iou": mean(row.get("click_iou") for row in click_rows),
        "click_iou_strict": mean(row.get("click_iou_strict") for row in click_rows),
        "region_ok_rate": mean(row.get("region_ok") for row in click_rows),
        "region_violation_sample_rate": mean(
            int((row.get("region_violation_count") or 0) > 0)
            for row in click_rows
        ),
        "invalid_coordinate_sample_rate": mean(
            int((row.get("invalid_coordinate_count") or 0) > 0)
            for row in click_rows
        ),
        "duplicate_click_sample_rate": mean(
            int((row.get("duplicate_click_count") or 0) > 0)
            for row in click_rows
        ),
        "over_click_sample_rate": mean(
            int((row.get("over_click_count") or 0) > 0)
            for row in click_rows
        ),
        "under_click_sample_rate": mean(
            int((row.get("under_click_count") or 0) > 0)
            for row in click_rows
        ),
        "near_miss_rate": mean(row.get("near_miss_rate") for row in click_rows),
        "n_submit_applicable": len(submit_rows),
        "submit_exact_rate": mean(row.get("submit_exact") for row in submit_rows),
        "click_submit_consistency_rate": mean(
            row.get("click_submit_consistency") for row in submit_rows
        ),
        "global_default_error_rate": mean(
            row.get("global_default_error")
            for row in rows
            if row.get("task_template") != "T1_COUNT_GLOBAL"
        ),
        "n_exclusion": len(exclusion_rows),
        "exclusion_failure_rate": mean(
            row.get("exclusion_failure") for row in exclusion_rows
        ),
    }


def group_summary(rows: Sequence[dict], key: str) -> Dict[str, dict]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key))].append(row)
    return {value: summary_for(bucket) for value, bucket in sorted(buckets.items())}


def macro_over_subsets(
    rows: Sequence[dict],
    selector,
    value_key: str,
) -> Optional[float]:
    values: List[float] = []
    for subset in SUBSET_ORDER:
        subset_rows = [
            row
            for row in rows
            if row.get("rose_subset") == subset and selector(row)
        ]
        value = mean(row.get(value_key) for row in subset_rows)
        if value is not None:
            values.append(value)
    return mean(values)


def paper_table1(rows: Sequence[dict]) -> dict:
    by_subset: Dict[str, dict] = {}
    for subset in SUBSET_ORDER:
        subset_rows = [row for row in rows if row.get("rose_subset") == subset]
        by_subset[subset] = {
            "n": len(subset_rows),
            "PASS": mean(row.get("pass") for row in subset_rows),
            "SOFT": mean(row.get("unified_soft") for row in subset_rows),
            "VALID": mean(row.get("format_valid") for row in subset_rows),
        }

    return {
        "by_subset": by_subset,
        "Avg. PASS": mean(
            by_subset[subset]["PASS"]
            for subset in SUBSET_ORDER
            if by_subset[subset]["PASS"] is not None
        ),
        "Avg. SOFT": mean(
            by_subset[subset]["SOFT"]
            for subset in SUBSET_ORDER
            if by_subset[subset]["SOFT"] is not None
        ),
        "VALID": mean(
            by_subset[subset]["VALID"]
            for subset in SUBSET_ORDER
            if by_subset[subset]["VALID"] is not None
        ),
    }


def paper_table2(rows: Sequence[dict]) -> dict:
    template_pass: Dict[str, Optional[float]] = {}
    for template in TEMPLATE_ORDER:
        template_pass[TEMPLATE_SHORT[template]] = macro_over_subsets(
            rows,
            lambda row, template=template: row.get("task_template") == template,
            "pass",
        )

    click_f1 = macro_over_subsets(
        rows,
        lambda row: row.get("click_applicable") == 1,
        "click_f1_strict",
    )
    region_ok = macro_over_subsets(
        rows,
        lambda row: row.get("click_applicable") == 1,
        "region_ok",
    )

    avg_pass = mean(
        value
        for value in template_pass.values()
        if value is not None
    )

    return {
        **template_pass,
        "C-F1": click_f1,
        "R-OK": region_ok,
        "Avg. PASS": avg_pass,
    }


def scene_conditioned_action(rows: Sequence[dict]) -> dict:
    """Available-row version of the T1-correct -> action analysis."""
    by_subset: Dict[str, dict] = {}
    macro_values: Dict[str, Optional[float]] = {}

    for subset in SUBSET_ORDER:
        subset_rows = [row for row in rows if row.get("rose_subset") == subset]
        t1_correct_scenes = {
            str(row.get("scene_id"))
            for row in subset_rows
            if row.get("task_template") == "T1_COUNT_GLOBAL"
            and row.get("pass") == 1
        }

        record: Dict[str, Any] = {
            "n_t1_correct_scenes": len(t1_correct_scenes),
        }
        for template in TEMPLATE_ORDER[2:]:
            matched = [
                row
                for row in subset_rows
                if row.get("task_template") == template
                and str(row.get("scene_id")) in t1_correct_scenes
            ]
            record[TEMPLATE_SHORT[template]] = {
                "n": len(matched),
                "pass_rate": mean(row.get("pass") for row in matched),
            }
        by_subset[subset] = record

    for template in TEMPLATE_ORDER[2:]:
        values = [
            by_subset[subset][TEMPLATE_SHORT[template]]["pass_rate"]
            for subset in SUBSET_ORDER
            if by_subset[subset][TEMPLATE_SHORT[template]]["pass_rate"] is not None
        ]
        macro_values[TEMPLATE_SHORT[template]] = mean(values)

    return {
        "by_subset": by_subset,
        "macro": {
            **macro_values,
            "Action Avg.": mean(
                value for value in macro_values.values() if value is not None
            ),
        },
        "note": (
            "Computed only from scenes for which both sampled T1 and action rows are "
            "available. A random partial run may have a small denominator."
        ),
    }


def percentage_tree(value: Any) -> Any:
    """Convert rate-valued fields to percentages while preserving counts."""
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                out[key] = percentage_tree(child)
            elif child is None:
                out[key] = None
            elif isinstance(child, (int, float)) and (
                "rate" in key.lower()
                or key in {
                    "PASS",
                    "SOFT",
                    "VALID",
                    "Avg. PASS",
                    "Avg. SOFT",
                    "C-F1",
                    "R-OK",
                    "G-Cnt",
                    "L-Cnt",
                    "L-Clk",
                    "V-Clk",
                    "Excl. C+S",
                    "Action Avg.",
                }
            ):
                out[key] = pct(float(child))
            else:
                out[key] = child
        return out
    if isinstance(value, list):
        return [percentage_tree(x) for x in value]
    return value


# =============================================================================
# Paper-row rendering
# =============================================================================


def display_model_name(model: str, override: str) -> str:
    return override.strip() or MODEL_DISPLAY.get(model, model)


def table1_latex_row(model_name: str, table: dict, digits: int) -> str:
    cells = [model_name]
    for subset in SUBSET_ORDER:
        record = table["by_subset"][subset]
        cells.extend([
            fmt_pct(record.get("PASS"), digits),
            fmt_pct(record.get("SOFT"), digits),
        ])
    cells.extend([
        fmt_pct(table.get("Avg. PASS"), digits),
        fmt_pct(table.get("Avg. SOFT"), digits),
        fmt_pct(table.get("VALID"), digits),
    ])
    return cells[0] + "\n& " + " & ".join(cells[1:]) + r" \\" 


def table2_latex_row(model_name: str, table: dict, digits: int) -> str:
    keys = [
        "G-Cnt",
        "L-Cnt",
        "L-Clk",
        "V-Clk",
        "Excl. C+S",
        "C-F1",
        "R-OK",
        "Avg. PASS",
    ]
    cells = [fmt_pct(table.get(key), digits) for key in keys]
    return model_name + "\n& " + " & ".join(cells) + r" \\" 


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ROSE predictions using metadata_dev/test.jsonl from the "
            "public release."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", type=str, default="")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help=(
            "Evaluate only available predictions. Without this flag, every item "
            "in the selected split must be present."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional model-name override.",
    )
    parser.add_argument(
        "--display_name",
        type=str,
        default="",
        help="Optional display name used in LaTeX rows.",
    )
    parser.add_argument("--digits", type=int, default=1)
    parser.add_argument("--near_miss_l1", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictions_path = args.predictions.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_metadata_path(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        revision=args.revision or None,
        split=args.split,
    )

    metadata_rows = read_jsonl(metadata_path)
    metadata_rows = [
        row
        for row in metadata_rows
        if str(row.get("split", args.split)) == args.split
    ]
    metadata_by_id = unique_index(metadata_rows, "item_id", metadata_path)

    prediction_rows = read_jsonl(predictions_path)
    predictions_by_id = prediction_index(prediction_rows, predictions_path)
    model = infer_model(prediction_rows, args.model, predictions_path)
    model_name = display_model_name(model, args.display_name)

    expected_ids = set(metadata_by_id)
    predicted_ids = set(predictions_by_id)
    matched_ids = expected_ids & predicted_ids
    missing_ids = sorted(expected_ids - predicted_ids)
    extra_ids = sorted(predicted_ids - expected_ids)

    if missing_ids and not args.allow_partial:
        raise ValueError(
            f"Missing {len(missing_ids)} predictions for the {args.split} split. "
            "Use --allow_partial for a sampled/demo run. "
            f"First missing IDs: {missing_ids[:10]}"
        )
    if not matched_ids:
        raise ValueError("No prediction item_id matched the release metadata.")

    eval_rows: List[dict] = []
    for item_id in sorted(matched_ids):
        prediction = dict(predictions_by_id[item_id])
        prediction.setdefault("model", model)
        item = metadata_by_id[item_id]
        eval_rows.append(
            evaluate_one(
                prediction,
                item,
                near_miss_l1=args.near_miss_l1,
            )
        )

    by_subset = group_summary(eval_rows, "rose_subset")
    by_template = group_summary(eval_rows, "task_template")
    by_task_mode = group_summary(eval_rows, "task_mode")
    by_region_type = group_summary(eval_rows, "region_type")

    table1 = paper_table1(eval_rows)
    table2 = paper_table2(eval_rows)
    conditioned = scene_conditioned_action(eval_rows)

    expected_by_subset = Counter(
        str(row.get("rose_subset")) for row in metadata_rows
    )
    evaluated_by_subset = Counter(
        str(row.get("rose_subset")) for row in eval_rows
    )
    expected_by_template = Counter(
        str(row.get("task_template")) for row in metadata_rows
    )
    evaluated_by_template = Counter(
        str(row.get("task_template")) for row in eval_rows
    )

    coverage_by_subset = []
    for subset in SUBSET_ORDER:
        expected = expected_by_subset.get(subset, 0)
        evaluated = evaluated_by_subset.get(subset, 0)
        coverage_by_subset.append({
            "rose_subset": subset,
            "expected": expected,
            "evaluated": evaluated,
            "coverage_rate": evaluated / expected if expected else None,
        })

    coverage_by_template = []
    for template in TEMPLATE_ORDER:
        expected = expected_by_template.get(template, 0)
        evaluated = evaluated_by_template.get(template, 0)
        coverage_by_template.append({
            "task_template": template,
            "task_short": TEMPLATE_SHORT[template],
            "expected": expected,
            "evaluated": evaluated,
            "coverage_rate": evaluated / expected if expected else None,
        })

    complete_submission = not missing_ids and not extra_ids

    summary = {
        "evaluator": "ROSE release evaluator v1",
        "model": model,
        "display_name": model_name,
        "split": args.split,
        "dataset_repo_id": args.repo_id,
        "metadata_jsonl": str(metadata_path),
        "predictions_jsonl": str(predictions_path),
        "coverage": {
            "complete_submission": complete_submission,
            "allow_partial": args.allow_partial,
            "num_expected": len(expected_ids),
            "num_prediction_rows": len(prediction_rows),
            "num_evaluated": len(eval_rows),
            "num_missing": len(missing_ids),
            "num_extra": len(extra_ids),
            "coverage_rate": len(eval_rows) / len(expected_ids),
            "by_subset": coverage_by_subset,
            "by_template": coverage_by_template,
            "missing_item_ids": missing_ids,
            "extra_item_ids": extra_ids,
        },
        "overall": summary_for(eval_rows),
        "by_subset": by_subset,
        "by_template": by_template,
        "by_task_mode": by_task_mode,
        "by_region_type": by_region_type,
        "paper_metrics": {
            "table1": table1,
            "table2": table2,
            "scene_conditioned_action_available_rows": conditioned,
        },
        "paper_metrics_percent": percentage_tree({
            "table1": table1,
            "table2": table2,
            "scene_conditioned_action_available_rows": conditioned,
        }),
        "metric_notes": {
            "PASS": (
                "Strict exact success: formal grammar must be valid and the "
                "count/click set/submission must be exactly correct."
            ),
            "VALID": "Exact formal-grammar validity, independent of semantic correctness.",
            "SOFT": (
                "COUNT=count soft; CLICK=strict click F1; "
                "CLICK_COUNT=mean(strict click F1, count soft, "
                "click-submit consistency)."
            ),
            "Table 1": (
                "Subset scores are item means. Avg. PASS/SOFT and VALID are "
                "macro averages over visual subsets with available rows."
            ),
            "Table 2": (
                "Each template PASS, C-F1, and R-OK is macro-averaged over "
                "visual subsets with available rows. Avg. PASS is the mean of "
                "the five template PASS values."
            ),
            "partial_warning": (
                "Metrics from a sampled run are diagnostic and are not directly "
                "comparable to the full official test results."
            ) if not complete_submission else None,
        },
    }

    per_item_path = out_dir / "per_item_eval.jsonl"
    error_cases_path = out_dir / "error_cases.jsonl"
    summary_path = out_dir / "evaluation_summary.json"
    subset_csv_path = out_dir / "summary_by_subset.csv"
    template_csv_path = out_dir / "summary_by_template.csv"
    coverage_csv_path = out_dir / "coverage_by_subset_template.csv"
    table1_path = out_dir / "table1_row.txt"
    table2_path = out_dir / "table2_row.txt"

    write_jsonl(per_item_path, eval_rows)
    write_jsonl(error_cases_path, [row for row in eval_rows if row.get("pass") != 1])
    write_json(summary_path, summary)

    subset_csv_rows = []
    for subset in SUBSET_ORDER:
        record = by_subset.get(subset, {"n": 0})
        subset_csv_rows.append({
            "rose_subset": subset,
            "subset_short": SUBSET_SHORT[subset],
            **record,
        })
    write_csv(subset_csv_path, subset_csv_rows)

    template_csv_rows = []
    for template in TEMPLATE_ORDER:
        record = by_template.get(template, {"n": 0})
        template_csv_rows.append({
            "task_template": template,
            "task_short": TEMPLATE_SHORT[template],
            **record,
        })
    write_csv(template_csv_path, template_csv_rows)

    coverage_rows = [
        {"group": "subset", **row}
        for row in coverage_by_subset
    ] + [
        {"group": "template", **row}
        for row in coverage_by_template
    ]
    write_csv(coverage_csv_path, coverage_rows)

    table1_text = table1_latex_row(model_name, table1, args.digits)
    table2_text = table2_latex_row(model_name, table2, args.digits)
    table1_path.write_text(table1_text + "\n", encoding="utf-8")
    table2_path.write_text(table2_text + "\n", encoding="utf-8")

    print("=" * 92)
    print("ROSE release evaluation")
    print("=" * 92)
    print(f"Model:             {model_name} ({model})")
    print(f"Split:             {args.split}")
    print(f"Expected items:    {len(expected_ids)}")
    print(f"Prediction rows:   {len(prediction_rows)}")
    print(f"Evaluated items:   {len(eval_rows)}")
    print(f"Missing items:     {len(missing_ids)}")
    print(f"Extra items:       {len(extra_ids)}")
    print(f"Complete:          {complete_submission}")
    print("-" * 92)
    print("Table 1 row:")
    print(table1_text)
    print("-" * 92)
    print("Table 2 row:")
    print(table2_text)
    print("-" * 92)
    print(f"PASS:              {fmt_pct(summary['overall'].get('pass_rate'), args.digits)}")
    print(f"SOFT:              {fmt_pct(summary['overall'].get('unified_soft'), args.digits)}")
    print(f"VALID:             {fmt_pct(summary['overall'].get('valid_rate'), args.digits)}")
    if not complete_submission:
        print(
            "[WARN] Partial/sample evaluation: use these values for pipeline "
            "checking, not as official full-test results."
        )
    print("=" * 92)
    print(f"Per-item metrics:  {per_item_path}")
    print(f"Summary:           {summary_path}")
    print(f"Table 1 row:       {table1_path}")
    print(f"Table 2 row:       {table2_path}")
    print("=" * 92)


if __name__ == "__main__":
    main()
