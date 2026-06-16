#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for the release-native ROSE analysis scripts."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SUBSET_ORDER = [
    "ROSE-ChineseGlyph",
    "ROSE-EmojiStyle",
    "ROSE-EmojiContent",
    "ROSE-PixelEdit",
    "ROSE-PixelContent",
]

SUBSET_SHORT = {
    "ROSE-ChineseGlyph": "Glyph",
    "ROSE-EmojiStyle": "E-Style",
    "ROSE-EmojiContent": "E-Content",
    "ROSE-PixelEdit": "P-Edit",
    "ROSE-PixelContent": "P-Content",
}

SUBSET_ALIASES = {
    "glyph": "ROSE-ChineseGlyph",
    "chinese": "ROSE-ChineseGlyph",
    "chineseglyph": "ROSE-ChineseGlyph",
    "chinese_glyph": "ROSE-ChineseGlyph",
    "estyle": "ROSE-EmojiStyle",
    "e-style": "ROSE-EmojiStyle",
    "emojistyle": "ROSE-EmojiStyle",
    "emoji_style": "ROSE-EmojiStyle",
    "econtent": "ROSE-EmojiContent",
    "e-content": "ROSE-EmojiContent",
    "emojicontent": "ROSE-EmojiContent",
    "emoji_content": "ROSE-EmojiContent",
    "pedit": "ROSE-PixelEdit",
    "p-edit": "ROSE-PixelEdit",
    "pixeledit": "ROSE-PixelEdit",
    "pixel_edit": "ROSE-PixelEdit",
    "pcontent": "ROSE-PixelContent",
    "p-content": "ROSE-PixelContent",
    "pixelcontent": "ROSE-PixelContent",
    "pixel_content": "ROSE-PixelContent",
}


def import_release_evaluator():
    """Import the main evaluator from demo/evaluate_rose.py."""
    import importlib.util

    repo_root = Path(__file__).resolve().parents[1]
    evaluator_path = repo_root / "demo" / "evaluate_rose.py"
    if not evaluator_path.exists():
        raise ImportError(f"Missing main evaluator: {evaluator_path}")

    module_name = "_rose_main_evaluator"
    spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator module: {evaluator_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text or "").split(",") if x.strip()]


def normalize_subset_token(text: str) -> str:
    return (
        str(text).strip().lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def resolve_subsets(requested: str, available: Sequence[str]) -> List[str]:
    available_set = set(available)
    tokens = parse_csv(requested)
    if not tokens or any(normalize_subset_token(x) in {"all", "*"} for x in tokens):
        ordered = [x for x in SUBSET_ORDER if x in available_set]
        ordered.extend(sorted(x for x in available if x not in ordered))
        return ordered

    out: List[str] = []
    for token in tokens:
        value: Optional[str] = None
        if token in available_set:
            value = token
        else:
            key = normalize_subset_token(token)
            for candidate in available:
                if normalize_subset_token(candidate) == key:
                    value = candidate
                    break
            if value is None:
                for alias, candidate in SUBSET_ALIASES.items():
                    if normalize_subset_token(alias) == key and candidate in available_set:
                        value = candidate
                        break
        if value is None:
            raise ValueError(
                f"Unknown subset '{token}'. Available: {', '.join(sorted(available_set))}"
            )
        if value not in out:
            out.append(value)
    return out


def metadata_path(dataset_root: Path, split: str) -> Path:
    return dataset_root / f"metadata_{split}.jsonl"


def coord_label(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return f"R{int(value[0])},C{int(value[1])}"
        except Exception:
            return None
    text = str(value).strip().upper().replace(" ", "")
    if text.startswith("R") and ",C" in text:
        try:
            left, right = text[1:].split(",C", 1)
            return f"R{int(left)},C{int(right)}"
        except Exception:
            return None
    return None


def coord_sort_key(label: str) -> Tuple[int, int]:
    normalized = coord_label(label)
    if normalized is None:
        return (10**9, 10**9)
    left, right = normalized[1:].split(",C", 1)
    return int(left), int(right)


def normalize_coord_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [x.strip() for x in values.split(";") if x.strip()]
    out: List[str] = []
    seen = set()
    if isinstance(values, Sequence):
        for value in values:
            label = coord_label(value)
            if label is not None and label not in seen:
                seen.add(label)
                out.append(label)
    return sorted(out, key=coord_sort_key)


def formal_click_answer(cells: Sequence[str]) -> str:
    ordered = sorted(set(cells), key=coord_sort_key)
    if not ordered:
        return "DONE"
    return "; ".join(f"CLICK({cell})" for cell in ordered) + "; DONE"


def combine_prompt(global_prompt: Any, local_prompt: Any) -> str:
    return "\n\n".join(
        x for x in [str(global_prompt or "").strip(), str(local_prompt or "").strip()]
        if x
    )


def safe_relative_image(row: Mapping[str, Any]) -> Path:
    raw = row.get("image")
    if not raw:
        raise ValueError(f"Item {row.get('item_id')} has no release 'image' field")
    rel = Path(str(raw).replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe release image path: {raw}")
    return rel


def transfer_file(src: Path, dst: Path, mode: str) -> str:
    """Transfer one file and return the effective mode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"

    mode = mode.lower()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"

    if mode == "symlink":
        try:
            os.symlink(str(src.resolve()), str(dst))
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"

    raise ValueError(f"Unknown image mode: {mode}")


def materialize_images(
    source_root: Path,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    unique_paths = sorted({safe_relative_image(row).as_posix() for row in rows})
    for rel_text in unique_paths:
        rel = Path(rel_text)
        src = source_root / rel
        dst = output_root / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        effective = transfer_file(src, dst, mode)
        counts[effective] = counts.get(effective, 0) + 1
    return counts


def mean(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else None


def pct(value: Optional[float], digits: int = 1) -> Optional[float]:
    return None if value is None else round(100.0 * float(value), digits)


def infer_model(prediction_rows: Sequence[Mapping[str, Any]], override: str = "") -> str:
    if override.strip():
        return override.strip()
    models = {
        str(row.get("model")).strip()
        for row in prediction_rows
        if row.get("model") is not None and str(row.get("model")).strip()
    }
    if len(models) == 1:
        return next(iter(models))
    if len(models) > 1:
        raise ValueError(f"Multiple model names in prediction file: {sorted(models)}")
    return "model"


def unique_index(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    keep_last: bool = False,
) -> Tuple[Dict[str, dict], int]:
    out: Dict[str, dict] = {}
    duplicates = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            raise ValueError(f"Row is missing '{key}'")
        value_s = str(value)
        if value_s in out:
            duplicates += 1
            if not keep_last:
                raise ValueError(f"Duplicate {key}: {value_s}")
        out[value_s] = dict(row)
    return out, duplicates


def load_main_eval(path: Optional[Path], model: str = "") -> List[dict]:
    if path is None:
        return []
    if path.is_file():
        rows = read_jsonl(path)
    elif path.is_dir():
        files = sorted(path.rglob("per_item_eval.jsonl"))
        if not files:
            raise FileNotFoundError(f"No per_item_eval.jsonl under {path}")
        rows = []
        for file in files:
            rows.extend(read_jsonl(file))
    else:
        raise FileNotFoundError(path)

    if model:
        rows = [row for row in rows if str(row.get("model")) == model]
    return rows
