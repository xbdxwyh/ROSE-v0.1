# ROSE analysis

This directory contains the public analysis code used for the perception-to-
action diagnostics. The two bridge datasets are metadata-only: their `image`
fields point to images in the main Hugging Face dataset
`sysuwyh357/ROSE-v0.1`.

## Files

```text
analysis/
├── data/
│   ├── global_click/metadata_test.jsonl
│   └── matched_local_count/metadata_test.jsonl
├── evaluate_global_click_bridge.py
├── evaluate_matched_local_count_bridge.py
├── analyze_scene_consistency.py
└── rose_analysis_utils.py
```

The generation/conversion scripts used during dataset construction are not
required to reproduce the released analyses and are intentionally excluded
from the public repository.

## Global-click bridge evaluation

First evaluate the model on the main benchmark so that a main
`per_item_eval.jsonl` is available. Then run:

```powershell
python .\analysis\evaluate_global_click_bridge.py `
  --dataset_root ".\analysis\data\global_click" `
  --predictions ".\demo\outputs\qwen36plus_global_click_test_50.jsonl" `
  --split test `
  --main_eval ".\demo\outputs\eval_qwen36plus_main\per_item_eval.jsonl" `
  --main_model qwen3.6-plus `
  --model qwen3.6-plus `
  --out_dir ".\demo\outputs\eval_qwen36plus_global_click_test_50" `
  --allow_partial
```

Key metrics include `G-Clk`, click cardinality accuracy, strict click F1, and
`G-Clk | G-Cnt correct`.

## Matched local-count bridge evaluation

```powershell
python .\analysis\evaluate_matched_local_count_bridge.py `
  --dataset_root ".\analysis\data\matched_local_count" `
  --predictions ".\demo\outputs\qwen36plus_matched_local_count_test_50.jsonl" `
  --split test `
  --main_eval ".\demo\outputs\eval_qwen36plus_main\per_item_eval.jsonl" `
  --main_model qwen3.6-plus `
  --model qwen3.6-plus `
  --out_dir ".\demo\outputs\eval_qwen36plus_matched_local_count_test_50" `
  --allow_partial
```

Key metrics include `mL-Cnt`, paired `L-Clk`, `L-Clk | mL-Cnt correct`,
cardinality diagnostics, transfer failure, and the `zero/partial/all`
region-case breakdown.

## Same-scene consistency

This analysis requires no additional API calls. It conditions T2-T5 results on
same-scene T1 success:

```powershell
python .\analysis\analyze_scene_consistency.py `
  ".\demo\outputs\eval_qwen36plus_main\per_item_eval.jsonl" `
  --split test `
  --out_dir ".\demo\outputs\scene_consistency"
```
