#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Reference Qwen runner for ROSE and its metadata-only derived tasks.

Supported dataset layouts
-------------------------
Local release directory:
  ROSE-v0.1-hf/
    metadata_dev.jsonl
    metadata_test.jsonl
    images/
      ROSE-ChineseGlyph/
      ROSE-EmojiStyle/
      ROSE-EmojiContent/
      ROSE-PixelEdit/
      ROSE-PixelContent/

Or automatically downloaded from:
  sysuwyh357/ROSE-v0.1

Typical usage
-------------
1) Validate 50 test samples, 10 from each subset:

  python run_qwen.py ^
    --dataset_root ".\ROSE-v0.1-hf" ^
    --split test ^
    --subsets all ^
    --samples_per_subset 10 ^
    --shuffle ^
    --seed 42 ^
    --dry_run

2) Run the same 50 samples with DashScope/Qwen:

  set DASHSCOPE_API_KEY=YOUR_KEY
  python run_qwen.py ^
    --dataset_root ".\ROSE-v0.1-hf" ^
    --split test ^
    --subsets all ^
    --samples_per_subset 10 ^
    --shuffle ^
    --seed 42 ^
    --model qwen3.6-plus ^
    --disable_thinking ^
    --output ".\outputs\qwen36plus_test_50.jsonl"

Dependencies
------------
  pip install openai pillow tqdm huggingface_hub
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
from tqdm import tqdm


DEFAULT_REPO_ID = "sysuwyh357/ROSE-v0.1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"

DEFAULT_SUBSET_ORDER = [
    "ROSE-ChineseGlyph",
    "ROSE-EmojiStyle",
    "ROSE-EmojiContent",
    "ROSE-PixelEdit",
    "ROSE-PixelContent",
]

SUBSET_ALIASES = {
    "chinese": "ROSE-ChineseGlyph",
    "glyph": "ROSE-ChineseGlyph",
    "chineseglyph": "ROSE-ChineseGlyph",
    "chinese_glyph": "ROSE-ChineseGlyph",
    "emoji_style": "ROSE-EmojiStyle",
    "emojistyle": "ROSE-EmojiStyle",
    "style": "ROSE-EmojiStyle",
    "emoji_content": "ROSE-EmojiContent",
    "emojicontent": "ROSE-EmojiContent",
    "pixel_edit": "ROSE-PixelEdit",
    "pixeledit": "ROSE-PixelEdit",
    "edit": "ROSE-PixelEdit",
    "pixel_content": "ROSE-PixelContent",
    "pixelcontent": "ROSE-PixelContent",
}

_IMAGE_DATA_URL_CACHE: Dict[str, str] = {}


# =============================================================================
# I/O
# =============================================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_no}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def completed_item_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            item_id = row.get("item_id")
            if item_id is not None:
                done.add(str(item_id))
    return done


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def normalize_key(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


def model_safe_name(model: str) -> str:
    return (
        str(model)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# =============================================================================
# Dataset loading and selection
# =============================================================================

def resolve_metadata_and_image_root(
    *,
    metadata_path: Optional[Path],
    dataset_root: Optional[Path],
    repo_id: str,
    revision: Optional[str],
    download_dir: Optional[Path],
    split: str,
) -> Tuple[Path, Optional[Path]]:
    """
    Resolve metadata independently from images.

    Modes:
      1. --metadata_path + --dataset_root:
         local GitHub metadata + local main ROSE images.
      2. --metadata_path only:
         local GitHub metadata + on-demand images from --repo_id.
      3. --dataset_root only:
         ordinary local release directory containing metadata and images.
      4. neither:
         main metadata and selected images are downloaded on demand from HF.
    """
    image_root: Optional[Path] = None

    if dataset_root is not None:
        image_root = dataset_root.expanduser().resolve()
        if not image_root.exists():
            raise FileNotFoundError(
                f"Dataset/image root does not exist: {image_root}"
            )

    if metadata_path is not None:
        resolved_metadata = metadata_path.expanduser().resolve()
        if not resolved_metadata.exists():
            raise FileNotFoundError(
                f"Metadata file does not exist: {resolved_metadata}"
            )
        return resolved_metadata, image_root

    filename = f"metadata_{split}.jsonl"

    if image_root is not None:
        resolved_metadata = image_root / filename
        if not resolved_metadata.exists():
            raise FileNotFoundError(
                f"Missing release metadata: {resolved_metadata}"
            )
        return resolved_metadata, image_root

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required when neither --dataset_root nor "
            "--metadata_path supplies a complete local dataset. Install it "
            "with: pip install huggingface_hub"
        ) from exc

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": filename,
    }
    if revision:
        kwargs["revision"] = revision
    if download_dir is not None:
        kwargs["cache_dir"] = str(download_dir.expanduser().resolve())

    print(f"[INFO] Resolving metadata from Hugging Face: {repo_id}/{filename}")
    resolved_metadata = Path(hf_hub_download(**kwargs)).resolve()
    return resolved_metadata, None


def resolve_subset_names(
    requested: str,
    available: Sequence[str],
) -> List[str]:
    available_set = set(available)
    tokens = parse_csv(requested)

    if not tokens or any(normalize_key(x) in {"all", "*"} for x in tokens):
        ordered = [x for x in DEFAULT_SUBSET_ORDER if x in available_set]
        ordered.extend(sorted(x for x in available if x not in set(ordered)))
        return ordered

    resolved: List[str] = []
    for token in tokens:
        value: Optional[str] = None

        if token in available_set:
            value = token
        else:
            key = normalize_key(token)

            for candidate in available:
                if normalize_key(candidate) == key:
                    value = candidate
                    break

            if value is None:
                for alias, candidate in SUBSET_ALIASES.items():
                    if normalize_key(alias) == key and candidate in available_set:
                        value = candidate
                        break

        if value is None:
            raise ValueError(
                f"Unknown subset '{token}'. Available: "
                f"{', '.join(sorted(available_set))}"
            )

        if value not in resolved:
            resolved.append(value)

    return resolved


def apply_value_filter(
    rows: List[dict],
    field: str,
    raw_values: str,
) -> List[dict]:
    keep = set(parse_csv(raw_values))
    if not keep:
        return rows
    return [row for row in rows if str(row.get(field)) in keep]


def select_rows(
    rows: List[dict],
    *,
    subsets: str,
    task_modes: str,
    region_types: str,
    cue_types: str,
    difficulties: str,
    primitive_types: str,
    task_templates: str,
    max_samples: int,
    samples_per_subset: int,
    shuffle: bool,
    seed: int,
) -> Tuple[List[dict], List[str]]:
    if max_samples > 0 and samples_per_subset > 0:
        raise ValueError(
            "Use only one of --max_samples and --samples_per_subset."
        )

    available_subsets = sorted({
        str(row.get("rose_subset"))
        for row in rows
        if row.get("rose_subset")
    })
    selected_subsets = resolve_subset_names(subsets, available_subsets)

    selected = [
        row for row in rows
        if str(row.get("rose_subset")) in set(selected_subsets)
    ]

    selected = apply_value_filter(selected, "task_mode", task_modes)
    selected = apply_value_filter(selected, "region_type", region_types)
    selected = apply_value_filter(selected, "cue_type", cue_types)
    selected = apply_value_filter(selected, "difficulty", difficulties)
    selected = apply_value_filter(selected, "primitive_type", primitive_types)
    selected = apply_value_filter(selected, "task_template", task_templates)

    rng = random.Random(seed)

    if samples_per_subset > 0:
        balanced: List[dict] = []
        for subset in selected_subsets:
            bucket = [
                row for row in selected
                if str(row.get("rose_subset")) == subset
            ]
            if shuffle:
                rng.shuffle(bucket)
            balanced.extend(bucket[:samples_per_subset])
        selected = balanced
    else:
        if shuffle:
            rng.shuffle(selected)
        if max_samples > 0:
            selected = selected[:max_samples]

    return selected, selected_subsets


_HF_IMAGE_PATH_CACHE: Dict[str, Path] = {}


def resolve_image_path(
    image_root: Optional[Path],
    row: dict,
    *,
    repo_id: str,
    revision: Optional[str],
    download_dir: Optional[Path],
) -> Path:
    image_value = row.get("image") or row.get("image_path")
    if not image_value:
        raise ValueError(
            f"Item {row.get('item_id')} has no 'image' or 'image_path' field."
        )

    normalized = str(image_value).replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"Unsafe image path for item {row.get('item_id')}: {image_value}"
        )

    if image_root is not None:
        image_path = (image_root / relative).resolve()
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image for item {row.get('item_id')}: {image_path}"
            )
        return image_path

    cache_key = f"{repo_id}@{revision or 'main'}::{normalized}"
    if cache_key in _HF_IMAGE_PATH_CACHE:
        return _HF_IMAGE_PATH_CACHE[cache_key]

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to fetch images from the main ROSE "
            "dataset. Install it with: pip install huggingface_hub"
        ) from exc

    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "filename": normalized,
    }
    if revision:
        kwargs["revision"] = revision
    if download_dir is not None:
        kwargs["cache_dir"] = str(download_dir.expanduser().resolve())

    image_path = Path(hf_hub_download(**kwargs)).resolve()
    _HF_IMAGE_PATH_CACHE[cache_key] = image_path
    return image_path


# =============================================================================
# Image encoding
# =============================================================================

def resize_if_needed(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side <= 0:
        return image

    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image

    scale = max_long_side / float(long_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def image_to_data_url(
    image_path: Path,
    *,
    image_format: str,
    image_quality: int,
    max_long_side: int,
) -> str:
    fmt = image_format.upper()
    if fmt == "AUTO":
        fmt = "PNG"

    cache_key = (
        f"{image_path.resolve()}::{fmt}::"
        f"{image_quality}::{max_long_side}"
    )
    if cache_key in _IMAGE_DATA_URL_CACHE:
        return _IMAGE_DATA_URL_CACHE[cache_key]

    with Image.open(image_path) as source:
        image = source.copy()

    image = resize_if_needed(image, max_long_side)
    buffer = BytesIO()

    if fmt == "PNG":
        image.save(buffer, format="PNG")
        mime = "image/png"
    elif fmt == "JPEG":
        image.convert("RGB").save(
            buffer,
            format="JPEG",
            quality=image_quality,
        )
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:{mime};base64,{encoded}"
    _IMAGE_DATA_URL_CACHE[cache_key] = data_url
    return data_url


# =============================================================================
# Prompt construction and API
# =============================================================================

def build_content(
    *,
    row: dict,
    image_path: Path,
    prompt_order: str,
    image_format: str,
    image_quality: int,
    max_image_long_side: int,
) -> List[dict]:
    prompt_global = str(row.get("prompt_global", "") or "").strip()
    prompt_local = str(row.get("prompt_local", "") or "").strip()

    if not prompt_global and not prompt_local:
        prompt = str(row.get("prompt", "") or "").strip()
        if not prompt:
            raise ValueError(
                f"Item {row.get('item_id')} has no usable prompt."
            )
        prompt_global = prompt

    image_part = {
        "type": "image_url",
        "image_url": {
            "url": image_to_data_url(
                image_path,
                image_format=image_format,
                image_quality=image_quality,
                max_long_side=max_image_long_side,
            )
        },
    }

    if prompt_order == "global_image_local":
        content: List[dict] = []
        if prompt_global:
            content.append({"type": "text", "text": prompt_global})
        content.append(image_part)
        if prompt_local:
            content.append({"type": "text", "text": prompt_local})
        return content

    combined = "\n\n".join(
        x for x in (prompt_global, prompt_local) if x
    )

    if prompt_order == "image_then_prompt":
        return [
            image_part,
            {"type": "text", "text": combined},
        ]

    if prompt_order == "prompt_then_image":
        return [
            {"type": "text", "text": combined},
            image_part,
        ]

    raise ValueError(f"Unknown prompt order: {prompt_order}")


def make_client(api_key: str, base_url: str):
    key = str(api_key or "").strip()
    if not key:
        raise ValueError("Missing API key.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for real API inference. "
            "Install it with: pip install openai"
        ) from exc
    return OpenAI(api_key=key, base_url=base_url)


def call_model(
    *,
    client: OpenAI,
    model: str,
    content: List[dict],
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
) -> Tuple[str, dict]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }

    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens

    if disable_thinking:
        kwargs["extra_body"] = {"enable_thinking": False}

    completion = client.chat.completions.create(**kwargs)
    message = completion.choices[0].message
    output = message.content

    if isinstance(output, list):
        parts: List[str] = []
        for part in output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        text = "\n".join(parts).strip()
    else:
        text = str(output or "").strip()

    raw = completion.model_dump() if hasattr(completion, "model_dump") else {}
    return text, raw


def robust_call(
    *,
    client: OpenAI,
    model: str,
    content: List[dict],
    temperature: float,
    max_tokens: int,
    disable_thinking: bool,
    max_retries: int,
    sleep_base: float,
) -> Tuple[str, dict]:
    attempts = max(1, max_retries + 1)
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            return call_model(
                client=client,
                model=model,
                content=content,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break

            wait = sleep_base * (2 ** attempt) + random.random()
            print(
                f"[WARN] Attempt {attempt + 1}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}; retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"API request failed after {attempts} attempt(s): {last_error}"
    )


# =============================================================================
# Output
# =============================================================================

def make_prediction_record(
    *,
    row: dict,
    model: str,
    split: str,
) -> dict:
    return {
        "item_id": row.get("item_id"),
        "scene_id": row.get("scene_id"),
        "image_id": row.get("image_id"),
        "model": model,
        "rose_subset": row.get("rose_subset"),
        "rose_split": split,
        "split": split,
        "primitive_type": row.get("primitive_type"),
        "task_template": row.get("task_template"),
        "task_mode": row.get("task_mode"),
        "region_type": row.get("region_type"),
        "region_case": row.get("region_case"),
        "cue_type": row.get("cue_type"),
        "difficulty": row.get("difficulty"),
        "raw_text": None,
        "error": None,
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Qwen-compatible multimodal API on the ROSE v0.1 "
            "Hugging Face release format."
        )
    )

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument(
        "--metadata_path",
        type=Path,
        default=None,
        help=(
            "Optional local metadata JSONL. This allows derived-task metadata "
            "to live in GitHub while images are resolved from --dataset_root "
            "or downloaded on demand from --repo_id."
        ),
    )
    dataset.add_argument(
        "--dataset_root",
        type=Path,
        default=None,
        help=(
            "Optional local main ROSE release root used for images. "
            "When --metadata_path is omitted, metadata_<split>.jsonl is also "
            "read from this root. When omitted, images are fetched on demand "
            "from --repo_id."
        ),
    )
    dataset.add_argument(
        "--repo_id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repository. Default: {DEFAULT_REPO_ID}",
    )
    dataset.add_argument(
        "--revision",
        type=str,
        default="",
        help="Optional branch, tag, or commit hash.",
    )
    dataset.add_argument(
        "--download_dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory for metadata/images.",
    )
    dataset.add_argument(
        "--split",
        choices=["dev", "test"],
        default="test",
    )
    dataset.add_argument(
        "--subsets",
        type=str,
        default="all",
        help="Comma-separated subset names/aliases, or all.",
    )

    filters = parser.add_argument_group("selection and filters")
    filters.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "Maximum number of rows across all selected subsets. "
            "0 means all. Do not combine with --samples_per_subset."
        ),
    )
    filters.add_argument(
        "--samples_per_subset",
        type=int,
        default=0,
        help=(
            "Take up to N rows from each selected subset. "
            "For all five subsets, 10 produces 50 rows."
        ),
    )
    filters.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before sampling.",
    )
    filters.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --shuffle. Default: 42.",
    )
    filters.add_argument("--task_modes", type=str, default="")
    filters.add_argument("--region_types", type=str, default="")
    filters.add_argument("--cue_types", type=str, default="")
    filters.add_argument("--difficulties", type=str, default="")
    filters.add_argument("--primitive_types", type=str, default="")
    filters.add_argument("--task_templates", type=str, default="")

    api = parser.add_argument_group("API")
    api.add_argument("--model", type=str, default=DEFAULT_MODEL)
    api.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    api.add_argument("--api_key", type=str, default="")
    api.add_argument(
        "--api_key_env",
        type=str,
        default="DASHSCOPE_API_KEY",
    )
    api.add_argument("--temperature", type=float, default=0.0)
    api.add_argument("--max_tokens", type=int, default=256)
    api.add_argument(
        "--max_retries",
        type=int,
        default=4,
        help="Retries after the first request.",
    )
    api.add_argument("--sleep_base", type=float, default=1.0)
    api.add_argument("--disable_thinking", action="store_true")

    image = parser.add_argument_group("image and prompt")
    image.add_argument(
        "--image_format",
        choices=["auto", "PNG", "JPEG"],
        default="auto",
    )
    image.add_argument("--image_quality", type=int, default=92)
    image.add_argument(
        "--max_image_long_side",
        type=int,
        default=0,
        help="0 keeps the original image size.",
    )
    image.add_argument(
        "--prompt_order",
        choices=[
            "global_image_local",
            "image_then_prompt",
            "prompt_then_image",
        ],
        default="global_image_local",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Prediction JSONL path. Default: "
            "outputs/<model>_<split>.jsonl"
        ),
    )
    output.add_argument("--resume", action="store_true")
    output.add_argument("--overwrite", action="store_true")
    output.add_argument("--save_raw", action="store_true")
    output.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate selection and image paths without API calls.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_path, image_root = resolve_metadata_and_image_root(
        metadata_path=args.metadata_path,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        revision=args.revision or None,
        download_dir=args.download_dir,
        split=args.split,
    )

    all_rows = read_jsonl(metadata_path)

    selected_rows, selected_subsets = select_rows(
        all_rows,
        subsets=args.subsets,
        task_modes=args.task_modes,
        region_types=args.region_types,
        cue_types=args.cue_types,
        difficulties=args.difficulties,
        primitive_types=args.primitive_types,
        task_templates=args.task_templates,
        max_samples=args.max_samples,
        samples_per_subset=args.samples_per_subset,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not selected_rows:
        raise ValueError("No rows matched the requested selection.")

    # Validate every image path before any paid API request.
    invalid_images: List[Tuple[str, str]] = []
    for row in selected_rows:
        try:
            resolve_image_path(
                image_root,
                row,
                repo_id=args.repo_id,
                revision=args.revision or None,
                download_dir=args.download_dir,
            )
        except Exception as exc:
            invalid_images.append((str(row.get("item_id")), str(exc)))

    if invalid_images:
        preview = "\n".join(
            f"  {item_id}: {message}"
            for item_id, message in invalid_images[:10]
        )
        raise FileNotFoundError(
            f"{len(invalid_images)} selected row(s) have invalid images:\n"
            f"{preview}"
        )

    if args.output is None:
        output_path = (
            Path("outputs")
            / f"{model_safe_name(args.model)}_{args.split}.jsonl"
        ).resolve()
    else:
        output_path = args.output.expanduser().resolve()

    if output_path.exists() and args.overwrite:
        output_path.unlink()

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --resume or --overwrite."
        )

    done = completed_item_ids(output_path) if args.resume else set()
    selected_ids = {str(row.get("item_id")) for row in selected_rows}
    todo = [
        row for row in selected_rows
        if str(row.get("item_id")) not in done
    ]

    subset_counts = Counter(
        str(row.get("rose_subset")) for row in selected_rows
    )
    template_counts = Counter(
        str(row.get("task_template")) for row in selected_rows
    )

    print("=" * 92)
    print("ROSE v0.1 Qwen release runner")
    print("=" * 92)
    print(
        "Image source:       "
        + (str(image_root) if image_root is not None
           else f"Hugging Face cache ({args.repo_id})")
    )
    print(f"Metadata:           {metadata_path}")
    print(f"Split:              {args.split}")
    print(f"Model:              {args.model}")
    print(f"Rows in split:      {len(all_rows)}")
    print(f"Rows selected:      {len(selected_rows)}")
    print(f"Already completed:  {len(done & selected_ids)}")
    print(f"Rows to run:        {len(todo)}")
    print(f"Output:             {output_path}")
    print(f"Dry run:            {args.dry_run}")
    print("-" * 92)
    print("Selected subsets:")
    for subset in selected_subsets:
        print(f"  {subset:<24} {subset_counts.get(subset, 0):>6}")
    print("-" * 92)
    print("Selected task templates:")
    for template, count in sorted(template_counts.items()):
        print(f"  {template:<32} {count:>6}")
    print("=" * 92)

    if args.dry_run:
        print("[DRY RUN] Selection and image validation passed.")
        return

    api_key = (
        str(args.api_key or "").strip()
        or os.environ.get(args.api_key_env, "").strip()
    )
    if not api_key:
        raise ValueError(
            f"Missing API key. Set {args.api_key_env} or pass --api_key."
        )

    client = make_client(api_key=api_key, base_url=args.base_url)

    config_path = output_path.with_suffix(".config.json")
    config = {
        "metadata_jsonl": str(metadata_path),
        "image_root": str(image_root) if image_root is not None else None,
        "dataset_repo_id": args.repo_id,
        "dataset_revision": args.revision or None,
        "split": args.split,
        "subsets": selected_subsets,
        "samples_per_subset": args.samples_per_subset,
        "max_samples": args.max_samples,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "task_modes": args.task_modes,
        "region_types": args.region_types,
        "cue_types": args.cue_types,
        "difficulties": args.difficulties,
        "primitive_types": args.primitive_types,
        "task_templates": args.task_templates,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "disable_thinking": args.disable_thinking,
        "image_format": args.image_format,
        "image_quality": args.image_quality,
        "max_image_long_side": args.max_image_long_side,
        "prompt_order": args.prompt_order,
        "output_jsonl": str(output_path),
        "num_selected": len(selected_rows),
        "num_already_completed": len(done & selected_ids),
        "num_to_run": len(todo),
    }
    write_json(config_path, config)

    success = 0
    errors = 0

    for row in tqdm(
        todo,
        desc=f"ROSE | {args.model} | {args.split}",
    ):
        record = make_prediction_record(
            row=row,
            model=args.model,
            split=args.split,
        )

        try:
            image_path = resolve_image_path(
                image_root,
                row,
                repo_id=args.repo_id,
                revision=args.revision or None,
                download_dir=args.download_dir,
            )
            content = build_content(
                row=row,
                image_path=image_path,
                prompt_order=args.prompt_order,
                image_format=args.image_format,
                image_quality=args.image_quality,
                max_image_long_side=args.max_image_long_side,
            )
            raw_text, raw_response = robust_call(
                client=client,
                model=args.model,
                content=content,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                disable_thinking=args.disable_thinking,
                max_retries=args.max_retries,
                sleep_base=args.sleep_base,
            )
            record["raw_text"] = raw_text
            if args.save_raw:
                record["raw_response"] = raw_response
            success += 1

        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            errors += 1

        append_jsonl(output_path, record)

    final_done = completed_item_ids(output_path)
    missing_ids = sorted(selected_ids - final_done)

    summary = {
        "metadata_jsonl": str(metadata_path),
        "image_root": str(image_root) if image_root is not None else None,
        "dataset_repo_id": args.repo_id,
        "split": args.split,
        "model": args.model,
        "output_jsonl": str(output_path),
        "num_selected": len(selected_rows),
        "num_attempted_this_run": len(todo),
        "num_success_this_run": success,
        "num_errors_this_run": errors,
        "num_completed_after_run": len(final_done & selected_ids),
        "num_missing_after_run": len(missing_ids),
        "missing_item_ids": missing_ids,
    }
    summary_path = output_path.with_suffix(".summary.json")
    write_json(summary_path, summary)

    print("=" * 92)
    print("[DONE] ROSE Qwen reference run finished")
    print(f"Predictions: {output_path}")
    print(f"Config:      {config_path}")
    print(f"Summary:     {summary_path}")
    print(
        f"Completed:   {summary['num_completed_after_run']}/"
        f"{summary['num_selected']}"
    )
    print(f"API errors:  {errors}")
    print("=" * 92)


if __name__ == "__main__":
    main()
