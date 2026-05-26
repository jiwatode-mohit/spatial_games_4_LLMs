# SCM GVGAI Repo (`scm/repo_final`)

This repository contains code for GVGAI spatial reasoning and positioning experiments, plus previously generated result artifacts.

## Layout

- `scripts/`: entry points for running experiments and producing aggregate plots.
- `project/`: core runtime, LLM clients, analysis helpers, and training scripts.
- `gym_gvgai/`: GVGAI Gym environment package.
- `experiments/`: generated spatial reasoning experiment outputs.
- `positioning/`: generated spatial positioning outputs.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Main Commands

Run spatial reasoning setup:

```bash
python scripts/run_spatial_reasoning_setup.py
```

Run positioning experiment:

```bash
python scripts/run_spatial_positioning_experiment.py --output_root positioning
```

Aggregate spatial reasoning results:

```bash
python scripts/aggregate_spatial_results.py --experiment_root experiments/spatial_reasoning_altmodels
```

Plot positioning results:

```bash
python scripts/plot_spatial_positioning_results.py --input_path positioning
```

## Existing Result Trees

- Spatial runs (main): `experiments/spatial_reasoning/runs`
- Spatial runs (alt models): `experiments/spatial_reasoning_altmodels/runs`
- Positioning runs: `positioning/20260315T203819Z__model_local-qwen3-*/`

## Cleanup Notes

- Normalized duplicate file extensions from `*.gif.gif` to `*.gif`.
- `.gitignore` was reduced to targeted patterns; it no longer ignores all `*.txt`, `*.png`, or `*.gif` files globally.
- Generated artifacts are now ignored by directory scope rather than broad extension rules.
