#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Set specific font sizes
# plt.rc('axes', titlesize=14) # Title font size
# plt.rc('axes', labelsize=14) # Axis label font size
# plt.rc('xtick', labelsize=12) # X-tick label font size
# plt.rc('ytick', labelsize=12) # Y-tick label font size

plt.rcParams.update({'font.size': 14})


def _label_bars_within_axes(
    ax: plt.Axes,
    bars: Sequence[object],
    values: Sequence[float],
    errors: Sequence[float] | None = None,
    *,
    fontsize: int = 14,
    inside_threshold_ratio: float = 0.24,
) -> None:
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1e-9)
    bottom_pad = 0.02 * yrange
    top_pad = 0.015 * yrange
    inside_threshold = inside_threshold_ratio * yrange

    px_to_data_y = abs(
        ax.transData.inverted().transform((0.0, 1.0))[1]
        - ax.transData.inverted().transform((0.0, 0.0))[1]
    )
    text_height_data = (fontsize * ax.figure.dpi / 72.0) * px_to_data_y
    min_inside_height = max(inside_threshold, text_height_data + bottom_pad + top_pad)

    if errors is None:
        errors = [0.0] * len(values)

    for bar, val, err in zip(bars, values, errors):
        if not np.isfinite(float(val)):
            continue
        height = float(bar.get_height())
        err_val = max(float(err), 0.0) if np.isfinite(float(err)) else 0.0
        x = bar.get_x() + bar.get_width() / 2.0
        label = f"{float(val):.2f}"
        # Keep enough room for the portion of the error bar that extends into the bar.
        inside_viable = height >= (err_val + min_inside_height)
        if inside_viable:
            y = min(height - err_val - bottom_pad, ymax - top_pad)
            y = max(y, ymin + top_pad)
            va = "top"
        else:
            # Keep labels near the error-bar cap but slightly lower to avoid visual overlap.
            y = min(height + err_val + 0.5 * top_pad, ymax - top_pad)
            y = max(y, ymin + top_pad)
            va = "bottom"
        ax.text(x, y, label, ha="center", va=va, fontsize=fontsize, color="black", clip_on=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and aggregate spatial positioning results from one run or a directory of runs."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Single run dir or a root directory containing multiple positioning run dirs.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


POSITION_ROW_COL_PATTERN = re.compile(
    r"position\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)",
    re.IGNORECASE,
)
POSITION_X_Y_PATTERN = re.compile(
    r"position\s*\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)",
    re.IGNORECASE,
)
BARE_POSITION_TUPLE_PATTERN = re.compile(
    r"(?:position\s*[:=]?\s*)?\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
    re.IGNORECASE,
)


def extract_feedback_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.lower().startswith("feedback:"):
            return line.strip()
    return ""


def parse_position_from_text(text: str) -> Tuple[str, str]:
    search_spaces = [extract_feedback_line(text), str(text or "")]
    for candidate in search_spaces:
        if not candidate:
            continue
        match = POSITION_ROW_COL_PATTERN.search(candidate)
        if match:
            return match.group(1), match.group(2)
        match = POSITION_X_Y_PATTERN.search(candidate)
        if match:
            return match.group(2), match.group(1)
        match = BARE_POSITION_TUPLE_PATTERN.search(candidate)
        if match:
            return match.group(2), match.group(1)
    return "", ""


def mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return (sum(vals) / len(vals)) if vals else 0.0


def sample_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    mu = sum(vals) / len(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def sem(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) <= 1:
        return 0.0
    return sample_std(vals) / math.sqrt(len(vals))


def display_model_name(model: str) -> str:
    name = str(model)
    if name.startswith("local-"):
        name = name[len("local-"):]
    if name.endswith("-vllm"):
        name = name[: -len("-vllm")]
    return name


def aggregate_rows(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        out: Dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        exact_vals = [to_float(r.get("exact_position_match", 0)) for r in items]
        distances = [to_float(r.get("euclidean_distance", "")) for r in items if str(r.get("euclidean_distance", "")).strip() != ""]
        out["cases"] = len(items)
        out["raw_contract_rate"] = round(mean_or_zero(to_float(r.get("raw_contract_ok", 0)) for r in items), 6)
        out["sanitized_contract_rate"] = round(mean_or_zero(to_float(r.get("sanitized_contract_ok", 0)) for r in items), 6)
        out["position_parse_rate"] = round(mean_or_zero(to_float(r.get("position_parse_success", 0)) for r in items), 6)
        out["row_match_rate"] = round(mean_or_zero(to_float(r.get("row_match", 0)) for r in items), 6)
        out["col_match_rate"] = round(mean_or_zero(to_float(r.get("col_match", 0)) for r in items), 6)
        out["exact_position_rate"] = round(mean_or_zero(exact_vals), 6)
        out["exact_position_std"] = round(sample_std(exact_vals), 6)
        out["exact_position_sem"] = round(sem(exact_vals), 6)
        out["mean_euclidean_distance"] = round(mean_or_zero(distances), 6)
        out["euclidean_distance_std"] = round(sample_std(distances), 6)
        out["euclidean_distance_sem"] = round(sem(distances), 6)
        out["parsed_mean_euclidean_distance"] = round(mean_or_zero(distances), 6)
        out["mean_latency_seconds"] = round(mean_or_zero(to_float(r.get("latency_seconds", 0.0)) for r in items), 6)
        out["mean_total_tokens"] = round(mean_or_zero(to_float(r.get("total_tokens", 0.0)) for r in items), 3)
        summary_rows.append(out)
    return summary_rows


def infer_model_from_run_dir(run_dir: Path) -> str:
    run_config = run_dir / "run_config.json"
    if run_config.exists():
        try:
            data = json.loads(run_config.read_text(encoding="utf-8"))
            model = str(data.get("model", "")).strip()
            if model:
                return model
        except Exception:
            pass
    name = run_dir.name
    marker = "__model_"
    return name.split(marker, 1)[1] if marker in name else name


def load_rows_from_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    csv_path = run_dir / "per_case_results.csv"
    rows = [{k: v for k, v in row.items()} for row in read_csv(csv_path)]
    model = infer_model_from_run_dir(run_dir)
    for row in rows:
        row.setdefault("model", model)
        row["level"] = to_int(row.get("level"))
        row["case_index"] = to_int(row.get("case_index"))
        predicted_row_raw = str(row.get("predicted_row", "")).strip()
        predicted_col_raw = str(row.get("predicted_col", "")).strip()
        if predicted_row_raw == "" or predicted_col_raw == "":
            reparsed_row, reparsed_col = parse_position_from_text(str(row.get("response_sanitized", "")))
            if reparsed_row == "" or reparsed_col == "":
                reparsed_row, reparsed_col = parse_position_from_text(str(row.get("response_raw", "")))
            if reparsed_row != "" and reparsed_col != "":
                predicted_row_raw, predicted_col_raw = reparsed_row, reparsed_col
                row["predicted_row"] = reparsed_row
                row["predicted_col"] = reparsed_col
        answer_row = to_int(row.get("answer_row"))
        answer_col = to_int(row.get("answer_col"))
        if predicted_row_raw != "" and predicted_col_raw != "":
            predicted_row = to_int(predicted_row_raw)
            predicted_col = to_int(predicted_col_raw)
            row["euclidean_distance"] = round(
                math.sqrt((predicted_row - answer_row) ** 2 + (predicted_col - answer_col) ** 2),
                6,
            )
        else:
            row["euclidean_distance"] = ""
        for key in (
            "raw_contract_ok",
            "sanitized_contract_ok",
            "action_parse_success",
            "position_parse_success",
            "row_match",
            "col_match",
            "exact_position_match",
            "had_think",
            "residual_think",
        ):
            row[key] = to_int(row.get(key))
        for key in ("latency_seconds", "input_tokens", "output_tokens", "total_tokens"):
            row[key] = to_float(row.get(key))
    return rows


def discover_run_dirs(input_path: Path) -> List[Path]:
    if (input_path / "per_case_results.csv").exists():
        return [input_path]
    return sorted({path.parent for path in input_path.rglob("per_case_results.csv")})


def ensure_output_dir(input_path: Path, run_dirs: Sequence[Path]) -> Path:
    if len(run_dirs) == 1 and run_dirs[0] == input_path:
        out_dir = input_path / "plots"
    else:
        out_dir = input_path / "aggregate_plots" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_exact_position_by_thinking(summary_by_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(summary_by_thinking, key=lambda r: str(r["thinking"]))
    labels = [str(r["thinking"]) for r in rows]
    vals = [to_float(r["exact_position_rate"]) for r in rows]
    colors = ["#1f6feb" if label == "off" else "#d94841" for label in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Position Match Rate")
    ax.set_title("Avatar Position Match by Thinking Mode")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_game_metrics(summary_by_game: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(summary_by_game, key=lambda r: str(r["game"]))
    labels = [str(r["game"]).replace("_v0", "") for r in rows]
    exact_vals = [to_float(r["exact_position_rate"]) for r in rows]
    parse_vals = [to_float(r["position_parse_rate"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars1 = ax.bar(x - width / 2, parse_vals, width, label="Position Parse Rate", color="#2a9d8f")
    bars2 = ax.bar(x + width / 2, exact_vals, width, label="Exact Position Match Rate", color="#264653")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Position Parsing vs Exact Position by Game")
    ax.legend(frameon=False)
    for bars, vals in ((bars1, parse_vals), (bars2, exact_vals)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_game_level_heatmap(summary_by_tgl: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = list(summary_by_tgl)
    games = sorted({str(r["game"]) for r in rows})
    levels = sorted({to_int(r["level"]) for r in rows})
    think_modes = sorted({str(r["thinking"]) for r in rows})
    fig, axes = plt.subplots(1, len(think_modes), figsize=(5.2 * len(think_modes), 4.6), sharey=True, constrained_layout=True)
    if len(think_modes) == 1:
        axes = [axes]
    for ax, thinking in zip(axes, think_modes):
        matrix = []
        for game in games:
            row_vals = []
            for level in levels:
                match = next((r for r in rows if str(r["thinking"]) == thinking and str(r["game"]) == game and to_int(r["level"]) == level), None)
                row_vals.append(to_float(match["exact_position_rate"]) if match else 0.0)
            matrix.append(row_vals)
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(levels)), [f"lvl{lvl}" for lvl in levels])
        ax.set_yticks(range(len(games)), [g.replace("_v0", "") for g in games])
        ax.set_title(f"Exact Position Match Rate\nthinking={thinking}")
        for i in range(len(games)):
            for j in range(len(levels)):
                ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.92)
    cbar.set_label("Exact Position Match Rate")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_latency_tokens(summary_by_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(summary_by_thinking, key=lambda r: str(r["thinking"]))
    labels = [str(r["thinking"]) for r in rows]
    latency = [to_float(r["mean_latency_seconds"]) for r in rows]
    tokens = [to_float(r["mean_total_tokens"]) for r in rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(5.8, 4.1))
    bars = ax1.bar(x, latency, width=0.55, color="#577590")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Mean Latency (s)", color="#577590")
    ax1.tick_params(axis="y", labelcolor="#577590")
    ax1.set_title("Latency and Token Cost by Thinking Mode")
    for bar, val in zip(bars, latency):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + max(latency or [0.0]) * 0.03 + 0.01, f"{val:.2f}", ha="center", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(x, tokens, color="#e76f51", marker="o", linewidth=2.0)
    ax2.set_ylabel("Mean Total Tokens", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")
    for xi, val in zip(x, tokens):
        ax2.text(xi, val + max(tokens or [0.0]) * 0.03 + 1.0, f"{val:.0f}", color="#e76f51", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_row_col_vs_exact(summary_by_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = sorted(summary_by_thinking, key=lambda r: str(r["thinking"]))
    labels = [str(r["thinking"]) for r in rows]
    row_vals = [to_float(r["row_match_rate"]) for r in rows]
    col_vals = [to_float(r["col_match_rate"]) for r in rows]
    exact_vals = [to_float(r["exact_position_rate"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for positions, vals, name, color in (
        (x - width, row_vals, "Row Match", "#43aa8b"),
        (x, col_vals, "Col Match", "#90be6d"),
        (x + width, exact_vals, "Exact Match", "#f8961e"),
    ):
        bars = ax.bar(positions, vals, width, label=name, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Row/Column/Exact Position Match by Thinking Mode")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_model_thinking_grouped(summary_by_model_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = list(summary_by_model_thinking)
    models = sorted({str(r["model"]) for r in rows})
    model_labels = [display_model_name(model) for model in models]
    think_modes = sorted({str(r["thinking"]) for r in rows})
    x = np.arange(len(models))
    width = 0.38 if len(think_modes) <= 2 else 0.8 / max(len(think_modes), 1)
    colors = {"off": "#7B61FF", "on": "#D4A017"}
    fig, ax = plt.subplots(figsize=(max(7.5, len(models) * 1.3), 2.6), constrained_layout=True)
    for idx, thinking in enumerate(think_modes):
        vals = []
        errs = []
        for model in models:
            match = next((r for r in rows if str(r["model"]) == model and str(r["thinking"]) == thinking), None)
            vals.append(to_float(match["exact_position_rate"]) if match else 0.0)
            errs.append(to_float(match["exact_position_sem"]) if match else 0.0)
        offset = (idx - (len(think_modes) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=f"thinking={thinking}",
            color=colors.get(thinking, None),
        )
        _label_bars_within_axes(ax, bars, vals, errs, fontsize=14)
    ax.set_xticks(x, model_labels,)
    ax.set_xlabel("Model",)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Position Match Rate",)
    # ax.set_title("Model-wise Exact Position Match Rate by Thinking Mode")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_model_thinking_heatmap(summary_by_model_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = list(summary_by_model_thinking)
    models = sorted({str(r["model"]) for r in rows})
    model_labels = [display_model_name(model) for model in models]
    think_modes = sorted({str(r["thinking"]) for r in rows})
    matrix = []
    for model in models:
        vals = []
        for thinking in think_modes:
            match = next((r for r in rows if str(r["model"]) == model and str(r["thinking"]) == thinking), None)
            vals.append(to_float(match["exact_position_rate"]) if match else 0.0)
        matrix.append(vals)
    fig, ax = plt.subplots(figsize=(4.2 + 1.2 * len(think_modes), max(4.0, len(models) * 0.4 + 1.5)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(think_modes)), [f"thinking={t}" for t in think_modes])
    ax.set_yticks(range(len(models)), model_labels)
    ax.set_title("Exact Position Match Rate Heatmap by Model and Thinking Mode")
    for i in range(len(models)):
        for j in range(len(think_modes)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.92, label="Exact Position Match Rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_model_thinking_distance(summary_by_model_thinking: Sequence[Dict[str, Any]], out_path: Path) -> None:
    rows = list(summary_by_model_thinking)
    models = sorted({str(r["model"]) for r in rows})
    model_labels = [display_model_name(model) for model in models]
    think_modes = sorted({str(r["thinking"]) for r in rows})
    x = np.arange(len(models))
    width = 0.38 if len(think_modes) <= 2 else 0.8 / max(len(think_modes), 1)
    colors = {"off": "#7B61FF", "on": "#D4A017"}
    fig, ax = plt.subplots(figsize=(max(7.5, len(models) * 1.3), 3.0), constrained_layout=True)
    for idx, thinking in enumerate(think_modes):
        vals = []
        errs = []
        for model in models:
            match = next((r for r in rows if str(r["model"]) == model and str(r["thinking"]) == thinking), None)
            vals.append(to_float(match["mean_euclidean_distance"]) if match else 0.0)
            errs.append(to_float(match["euclidean_distance_sem"]) if match else 0.0)
        offset = (idx - (len(think_modes) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            yerr=errs,
            capsize=3,
            label=f"thinking={thinking}",
            color=colors.get(thinking, None),
        )
        _label_bars_within_axes(ax, bars, vals, errs, fontsize=14)
    ax.set_xticks(x, model_labels,)
    ax.set_xlabel("Model",)
    ax.set_ylabel("Mean Euclidean Distance",)
    # ax.set_title("Model-wise Position Error by Thinking Mode")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_plots_readme(out_dir: Path, plot_paths: Sequence[Path], run_dirs: Sequence[Path]) -> None:
    lines = [
        "# Positioning Plots",
        "",
        f"- Source run count: `{len(run_dirs)}`",
        "",
        "Generated visuals:",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).resolve()
    run_dirs = discover_run_dirs(input_path)
    if not run_dirs:
        raise RuntimeError(f"No positioning runs found under {input_path}")

    out_dir = ensure_output_dir(input_path, run_dirs)
    all_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        all_rows.extend(load_rows_from_run_dir(run_dir))

    summary_by_model_thinking = aggregate_rows(all_rows, ["model", "thinking"])

    plot_paths = [
        out_dir / "model_exact_position_by_thinking.png",
        out_dir / "model_euclidean_distance_by_thinking.png",
    ]

    plot_model_thinking_grouped(summary_by_model_thinking, plot_paths[0])
    plot_model_thinking_distance(summary_by_model_thinking, plot_paths[1])
    print(f"Wrote {len(plot_paths)} plots to {out_dir}")


if __name__ == "__main__":
    main()
