#!/usr/bin/env python3
"""Aggregate spatial experiment results into unified tables, visuals, and report.

Usage:
- Main 8B sweep:
  `python scripts/aggregate_spatial_results.py --experiment_root experiments/spatial_reasoning`
- All-model combined sweep:
  `python scripts/aggregate_spatial_results.py --experiment_root experiments/spatial_reasoning_altmodels`

This script reads spatial records from results-index artifacts, enriches each run with
run-summary/step-metrics/media metadata, computes dense aggregate metrics across factors,
and emits a single combined report suitable for analysis and paper-ready inspection.
PNG plots are written for HTML embedding and matching PDF copies are exported alongside them.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from adjustText import adjust_text


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D



plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 19,
    }
)
# plt.rcParams.update({'font.size': 24})

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments" / "spatial_reasoning"
LATEX_PAPER_ROOT = ROOT.parents[1] / "paper" / "69b044b7cb82b99640606d95"
LATEX_SPATIAL_FIGURES_DIR = LATEX_PAPER_ROOT / "figures" / "spatial_reasoning"
LATEX_FIGURE_EXPORTS = {
    "winrate_heatmap_game_level.png": "fig_spatial_winrate_heatmap_game_level.png",
    "mean_time_per_step_by_mode.png": "fig_spatial_mean_time_per_step_by_mode.png",
}
MAIN_SPATIAL_MODEL = "local-qwen3-8b-vllm"
WRITE_CSV = False


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_read_json(path: Optional[Path]) -> Dict[str, object]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_env_id(env_id: str) -> Tuple[str, Optional[int]]:
    # gvgai-<game>-lvl<level>-v0
    if not env_id.startswith("gvgai-"):
        return env_id, None
    core = env_id[len("gvgai-") :]
    if "-lvl" not in core:
        return core, None
    game, tail = core.rsplit("-lvl", 1)
    digits = []
    for ch in tail:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    level = int("".join(digits)) if digits else None
    return game, level


def parse_llm_actions_from_run_dir(run_dir: Path) -> List[Optional[int]]:
    llm_dir = run_dir / "llm_io"
    if not llm_dir.exists():
        return []
    logs = sorted(p for p in llm_dir.glob("*.json") if "sampled_inputs_every_" not in p.name)
    if not logs:
        return []
    try:
        entries = json.loads(logs[-1].read_text(encoding="utf-8"))
    except Exception:
        return []
    responses = [e.get("content", "") for e in entries if e.get("role") == "response"]
    out: List[Optional[int]] = []
    for r in responses:
        text = str(r)
        # Exclude pure planning responses from action-parse metrics.
        has_plan = bool(re.search(r"\bPlanActions\s*:\s*\[[^\]]*\]", text, flags=re.IGNORECASE))
        has_action = bool(re.search(r"\bAction\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE))
        if has_plan and not has_action:
            continue
        m = re.search(r"\bAction\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE)
        out.append(int(m.group(1)) if m else None)
    return out


def compute_action_execution_metrics(
    step_actions: List[Optional[int]],
    llm_actions: List[Optional[int]],
    planning_horizon_x: int = 1,
) -> Dict[str, float]:
    if not step_actions or not llm_actions:
        return {
            "llm_action_parse_rate": 0.0,
            "action_execution_match_rate": 0.0,
            "action_execution_coverage": 0.0,
        }
    n = min(len(step_actions), len(llm_actions))
    if n <= 0:
        return {
            "llm_action_parse_rate": 0.0,
            "action_execution_match_rate": 0.0,
            "action_execution_coverage": 0.0,
        }

    # Planning mode queries actions sparsely; align by horizon sampling.
    if planning_horizon_x and int(planning_horizon_x) > 1:
        sampled_steps: List[Optional[int]] = []
        idx = 0
        while idx < len(step_actions) and len(sampled_steps) < n:
            sampled_steps.append(step_actions[idx])
            idx += int(planning_horizon_x)
        if len(sampled_steps) < n:
            sampled_steps.extend(step_actions[len(sampled_steps):n])
        aligned_steps = sampled_steps[:n]
    else:
        aligned_steps = step_actions[:n]

    llm_parsed = sum(1 for a in llm_actions[:n] if a is not None)
    matched = sum(
        1
        for i in range(n)
        if llm_actions[i] is not None
        and aligned_steps[i] is not None
        and int(llm_actions[i]) == int(aligned_steps[i])
    )
    return {
        "llm_action_parse_rate": llm_parsed / n,
        "action_execution_match_rate": matched / n,
        "action_execution_coverage": n / max(len(step_actions), 1),
    }


def discover_record_paths(experiment_root: Path, include_backups: bool) -> List[Path]:
    source_roots = [experiment_root]
    if experiment_root.name == "spatial_reasoning_altmodels":
        source_roots.append(ROOT / "experiments" / "spatial_reasoning")

    paths: List[Path] = []
    for root in source_roots:
        paths.extend(sorted((root / "results_index" / "records").glob("*.json")))
    if include_backups:
        backup_roots = sorted((ROOT / "backups").glob("spatial_archive*"))
        for br in backup_roots:
            paths.extend(sorted((br / "results_index_records_moved").glob("*.json")))
            moved_exp = br / "experiments_moved" / "results_index" / "records"
            paths.extend(sorted(moved_exp.glob("*.json")))
    # Deduplicate by full resolved path string.
    uniq: Dict[str, Path] = {}
    for p in paths:
        uniq[str(p.resolve())] = p
    return sorted(uniq.values(), key=lambda p: str(p))


def infer_model_filter_for_root(experiment_root: Path) -> Optional[set[str]]:
    root_name = experiment_root.name
    if root_name == "spatial_reasoning":
        return {MAIN_SPATIAL_MODEL}
    if root_name == "spatial_reasoning_altmodels":
        return None
    return None


def _resolve_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists():
        return p
    # Allow stale absolute paths in moved backups: try to map to workspace-relative suffix.
    s = str(path_str)
    marker = "/GVGAI_GYM_cog/"
    if marker in s:
        suffix = s.split(marker, 1)[1]
        candidate = ROOT / suffix
        if candidate.exists():
            return candidate
    return None


def _extract_gif_path(run_dir: Optional[Path]) -> Optional[Path]:
    if not run_dir or not run_dir.exists():
        return None
    candidates = [run_dir / "gameplay.gif", run_dir / "gameplay.gif.gif"]
    candidates.extend(sorted(run_dir.glob("gameplay*.gif*")))
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_step_actions(step_metrics: Dict[str, object]) -> List[Optional[int]]:
    actions = step_metrics.get("actions_by_step")
    if not isinstance(actions, list):
        return []
    out: List[Optional[int]] = []
    for item in actions:
        if isinstance(item, dict):
            val = item.get("action")
            out.append(_to_int(val, default=-1) if val is not None else None)
    return out


def _winner_flags(winner: str) -> Tuple[int, int, int]:
    is_win = 1 if winner == "PLAYER_WINS" else 0
    is_loss = 1 if winner == "PLAYER_LOSES" else 0
    is_terminal = 1 if winner in {"PLAYER_WINS", "PLAYER_LOSES"} else 0
    return is_win, is_loss, is_terminal


def _apply_spatial_winner_override(game: str, winner: str, total_reward: float) -> Tuple[str, str]:
    if game == "spatialgame2" and abs(total_reward - 3.0) < 1e-9 and winner == "PLAYER_LOSES":
        return "PLAYER_WINS", "spatialgame2_reward_3_override"
    return winner, ""


def add_normalized_reward_metrics(rows: List[Dict[str, object]]) -> None:
    game_max_reward: Dict[str, float] = {}
    for row in rows:
        game = str(row.get("game", ""))
        reward = _to_float(row.get("total_reward", 0.0), 0.0)
        if reward > game_max_reward.get(game, 0.0):
            game_max_reward[game] = reward

    for row in rows:
        game = str(row.get("game", ""))
        reward = _to_float(row.get("total_reward", 0.0), 0.0)
        scale = max(game_max_reward.get(game, 0.0), 1.0)
        row["reward_normalizer"] = round(scale, 6)
        row["normalized_total_reward"] = round(reward / scale, 6)


def normalize_record(record: Dict[str, object]) -> Dict[str, object]:
    env_id = str(record.get("game_env_id", ""))
    game = str(record.get("game", ""))
    level = record.get("level", None)
    if not game:
        game, parsed_level = parse_env_id(env_id)
        if level is None:
            level = parsed_level

    run_dir = _resolve_path(str(record.get("run_dir", "")))
    run_summary_path = _resolve_path(str(record.get("run_summary_path", "")))
    step_metrics_path = _resolve_path(str(record.get("step_metrics_json_path", "")))
    gif_path = _extract_gif_path(run_dir)

    run_summary = _safe_read_json(run_summary_path)
    step_metrics = _safe_read_json(step_metrics_path)
    advanced = step_metrics.get("advanced_step_metrics", {}) if isinstance(step_metrics, dict) else {}

    planning_summary = run_summary.get("planning_summary", {}) if isinstance(run_summary, dict) else {}
    token_aggregates = run_summary.get("token_aggregates", {}) if isinstance(run_summary, dict) else {}
    last_mentioned = run_summary.get("last_mentioned_action_summary", {}) if isinstance(run_summary, dict) else {}

    steps = _to_int(record.get("steps"), _to_int(run_summary.get("steps"), 0))
    total_reward = _to_float(record.get("total_reward"), _to_float(run_summary.get("total_reward"), 0.0))
    runtime_seconds = _to_float(record.get("runtime_seconds"), _to_float(run_summary.get("runtime", {}).get("runtime_seconds"), 0.0) if isinstance(run_summary.get("runtime"), dict) else 0.0)

    winner_raw = str(record.get("winner", run_summary.get("winner", "UNKNOWN")))
    winner, winner_override_reason = _apply_spatial_winner_override(game=game, winner=winner_raw, total_reward=total_reward)
    is_win, is_loss, is_terminal = _winner_flags(winner)

    meaningful_ratio = _to_float(
        record.get("meaningful_step_ratio"),
        _to_float(run_summary.get("meaningful_step_ratio"), _to_float(step_metrics.get("meaningful_step_ratio"), 0.0)),
    )

    effective_ratio = _to_float(advanced.get("effective_step_ratio"), 0.0)
    if effective_ratio <= 0.0:
        effective_ratio = meaningful_ratio

    planning_horizon = _to_int(record.get("planning_horizon_x"), _to_int(run_summary.get("planning_horizon_x"), 1))
    llm_actions = parse_llm_actions_from_run_dir(run_dir) if run_dir else []
    step_actions = _find_step_actions(step_metrics)
    exec_metrics = compute_action_execution_metrics(
        step_actions=step_actions,
        llm_actions=llm_actions,
        planning_horizon_x=planning_horizon,
    )

    sum_total_tokens = _to_float(
        record.get("sum_total_tokens"),
        _to_float(token_aggregates.get("sum_total_tokens"), 0.0),
    )
    sum_input_tokens = _to_float(
        record.get("sum_input_tokens"),
        _to_float(token_aggregates.get("sum_input_tokens"), 0.0),
    )
    sum_output_tokens = _to_float(
        record.get("sum_output_tokens"),
        _to_float(token_aggregates.get("sum_output_tokens"), 0.0),
    )

    parse_success_count = _to_int(
        record.get("action_parse_success_count"),
        _to_int(planning_summary.get("action_parse_success_count"), 0),
    )
    plan_queries = _to_int(record.get("plan_queries_count"), _to_int(planning_summary.get("plan_queries_count"), 0))
    queued_exec = _to_int(
        record.get("queued_actions_executed"),
        _to_int(planning_summary.get("queued_actions_executed"), 0),
    )

    reward_per_step = total_reward / max(steps, 1)
    reward_per_1k_tok = total_reward / max(sum_total_tokens / 1000.0, 1e-9)
    tokens_per_step = sum_total_tokens / max(steps, 1)
    steps_per_second = steps / max(runtime_seconds, 1e-9)
    time_per_step_seconds = runtime_seconds / max(steps, 1)
    effective_steps_per_second = (effective_ratio * steps) / max(runtime_seconds, 1e-9)

    mode = str(record.get("mode", run_summary.get("mode", "")))
    model = str(record.get("model_profile", run_summary.get("model", "")))

    return {
        "record_path": str(record.get("_record_path", "")),
        "timestamp_utc": str(record.get("timestamp_utc", "")),
        "status": str(record.get("status", "unknown")),
        "game_env_id": env_id,
        "game": game,
        "level": _to_int(level, -1),
        "run_id": _to_int(record.get("run_id"), _to_int(run_summary.get("run_id"), 0)),
        "mode": mode,
        "base_mode": str(record.get("base_mode", run_summary.get("base_mode", ""))),
        "model_profile": model,
        "qwen_thinking_mode": str(record.get("qwen_thinking_mode", run_summary.get("qwen_thinking_mode", ""))),
        "translator_mode": str(record.get("translator_mode", run_summary.get("translator_mode", ""))),
        "causal_mode": str(record.get("causal_mode", run_summary.get("causal_mode", ""))),
        "planning_mode": str(record.get("planning_mode", run_summary.get("planning_mode", ""))),
        "planning_horizon_x": planning_horizon,
        "winner": winner,
        "winner_raw": winner_raw,
        "winner_override_reason": winner_override_reason,
        "is_completed": 1 if str(record.get("status", "")) == "completed" else 0,
        "is_terminal": is_terminal,
        "is_win": is_win,
        "is_loss": is_loss,
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "runtime_seconds": round(runtime_seconds, 6),
        "meaningful_step_ratio": round(meaningful_ratio, 6),
        "effective_step_ratio": round(effective_ratio, 6),
        "action_entropy_windowed": round(_to_float(advanced.get("action_entropy_windowed"), 0.0), 6),
        "action_change_rate": round(_to_float(advanced.get("action_change_rate"), 0.0), 6),
        "repeat_state_repeat_action_rate": round(_to_float(advanced.get("repeat_state_repeat_action_rate"), 0.0), 6),
        "non_progress_streak_p95": round(_to_float(advanced.get("non_progress_streak_p95"), 0.0), 6),
        "catastrophic_step_rate": round(_to_float(advanced.get("catastrophic_step_rate"), 0.0), 6),
        "llm_action_parse_rate": round(exec_metrics["llm_action_parse_rate"], 6),
        "action_execution_match_rate": round(exec_metrics["action_execution_match_rate"], 6),
        "action_execution_coverage": round(exec_metrics["action_execution_coverage"], 6),
        "sum_input_tokens": round(sum_input_tokens, 6),
        "sum_output_tokens": round(sum_output_tokens, 6),
        "sum_total_tokens": round(sum_total_tokens, 6),
        "tokens_per_step": round(tokens_per_step, 6),
        "reward_per_step": round(reward_per_step, 6),
        "reward_per_1k_tokens": round(reward_per_1k_tok, 6),
        "steps_per_second": round(steps_per_second, 6),
        "time_per_step_seconds": round(time_per_step_seconds, 6),
        "effective_steps_per_second": round(effective_steps_per_second, 6),
        "plan_queries_count": plan_queries,
        "queued_actions_executed": queued_exec,
        "planning_parse_fail_count": _to_int(record.get("planning_parse_fail_count"), _to_int(planning_summary.get("planning_parse_fail_count"), 0)),
        "plan_parse_success_count": _to_int(record.get("plan_parse_success_count"), _to_int(planning_summary.get("plan_parse_success_count"), 0)),
        "plan_parse_partial_count": _to_int(record.get("plan_parse_partial_count"), _to_int(planning_summary.get("plan_parse_partial_count"), 0)),
        "action_parse_success_count": parse_success_count,
        "action_parse_fail_count": _to_int(record.get("action_parse_fail_count"), _to_int(planning_summary.get("action_parse_fail_count"), 0)),
        "action_fallback_count": _to_int(record.get("action_fallback_count"), _to_int(planning_summary.get("action_fallback_count"), 0)),
        "action_fallback_last_valid_count": _to_int(record.get("action_fallback_last_valid_count"), _to_int(planning_summary.get("action_fallback_last_valid_count"), 0)),
        "action_fallback_nil_count": _to_int(record.get("action_fallback_nil_count"), _to_int(planning_summary.get("action_fallback_nil_count"), 0)),
        "action_parse_tolerant_success_count": _to_int(record.get("action_parse_tolerant_success_count"), _to_int(planning_summary.get("action_parse_tolerant_success_count"), 0)),
        "action_parse_strict_success_count": _to_int(record.get("action_parse_strict_success_count"), _to_int(planning_summary.get("action_parse_strict_success_count"), 0)),
        "think_sanitized_count": _to_int(record.get("think_sanitized_count"), _to_int(planning_summary.get("think_sanitized_count"), 0)),
        "think_incomplete_retry_count": _to_int(record.get("think_incomplete_retry_count"), _to_int(planning_summary.get("think_incomplete_retry_count"), 0)),
        "think_incomplete_terminal_fail_count": _to_int(record.get("think_incomplete_terminal_fail_count"), _to_int(planning_summary.get("think_incomplete_terminal_fail_count"), 0)),
        "action_think_incomplete_retry_count": _to_int(record.get("action_think_incomplete_retry_count"), _to_int(planning_summary.get("action_think_incomplete_retry_count"), 0)),
        "action_think_incomplete_terminal_fail_count": _to_int(record.get("action_think_incomplete_terminal_fail_count"), _to_int(planning_summary.get("action_think_incomplete_terminal_fail_count"), 0)),
        "plan_think_incomplete_retry_count": _to_int(record.get("plan_think_incomplete_retry_count"), _to_int(planning_summary.get("plan_think_incomplete_retry_count"), 0)),
        "plan_think_incomplete_terminal_fail_count": _to_int(record.get("plan_think_incomplete_terminal_fail_count"), _to_int(planning_summary.get("plan_think_incomplete_terminal_fail_count"), 0)),
        "last_mentioned_action_total_count": _to_int(record.get("last_mentioned_action_total_count"), _to_int(last_mentioned.get("last_mentioned_action_total_count"), 0)),
        "last_mentioned_action_match_count": _to_int(record.get("last_mentioned_action_match_count"), _to_int(last_mentioned.get("last_mentioned_action_match_count"), 0)),
        "last_mentioned_action_match_rate": round(_to_float(record.get("last_mentioned_action_match_rate"), _to_float(last_mentioned.get("last_mentioned_action_match_rate"), 0.0)), 6),
        "run_dir": str(run_dir.resolve()) if run_dir else "",
        "run_summary_path": str(run_summary_path.resolve()) if run_summary_path else "",
        "step_metrics_json_path": str(step_metrics_path.resolve()) if step_metrics_path else "",
        "gif_path": str(gif_path.resolve()) if gif_path else "",
        "slurm_job_id": str(record.get("slurm_job_id", "")),
        "slurm_array_job_id": str(record.get("slurm_array_job_id", "")),
        "slurm_array_task_id": str(record.get("slurm_array_task_id", "")),
        "hostname": str(record.get("hostname", "")),
        "error_type": str(record.get("error_type", "")),
        "error_message": str(record.get("error_message", "")),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]], ordered_fields: Optional[Sequence[str]] = None) -> None:
    if not WRITE_CSV:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if ordered_fields:
        fieldnames = list(ordered_fields)
    else:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def aggregate_means(rows: Sequence[Dict[str, object]], group_fields: Sequence[str]) -> List[Dict[str, object]]:
    metrics = [
        "is_completed",
        "is_terminal",
        "is_win",
        "is_loss",
        "steps",
        "total_reward",
        "normalized_total_reward",
        "runtime_seconds",
        "meaningful_step_ratio",
        "effective_step_ratio",
        "action_entropy_windowed",
        "action_change_rate",
        "repeat_state_repeat_action_rate",
        "non_progress_streak_p95",
        "catastrophic_step_rate",
        "llm_action_parse_rate",
        "action_execution_match_rate",
        "action_execution_coverage",
        "sum_total_tokens",
        "tokens_per_step",
        "reward_per_step",
        "reward_per_1k_tokens",
        "steps_per_second",
        "time_per_step_seconds",
        "effective_steps_per_second",
        "plan_queries_count",
        "queued_actions_executed",
        "plan_parse_success_count",
        "plan_parse_partial_count",
        "planning_parse_fail_count",
        "action_parse_success_count",
        "action_parse_fail_count",
        "action_fallback_count",
        "action_fallback_last_valid_count",
        "action_fallback_nil_count",
        "action_parse_tolerant_success_count",
        "action_parse_strict_success_count",
        "action_think_incomplete_retry_count",
        "action_think_incomplete_terminal_fail_count",
        "plan_think_incomplete_retry_count",
        "plan_think_incomplete_terminal_fail_count",
        "last_mentioned_action_match_rate",
    ]

    buckets: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        key = tuple(r[g] for g in group_fields)
        buckets[key].append(r)

    out: List[Dict[str, object]] = []
    for key in sorted(buckets.keys()):
        group = buckets[key]
        row: Dict[str, object] = {group_fields[i]: key[i] for i in range(len(group_fields))}
        n = len(group)
        row["n"] = n
        for m in metrics:
            vals = [_to_float(g.get(m, 0.0), 0.0) for g in group]
            mean = sum(vals) / max(n, 1)
            var = sum((v - mean) ** 2 for v in vals) / max(n, 1)
            row[f"mean_{m}"] = round(mean, 6)
            row[f"std_{m}"] = round(math.sqrt(var), 6)
        # Semantic aliases
        row["completion_rate"] = row["mean_is_completed"]
        row["terminal_rate"] = row["mean_is_terminal"]
        row["win_rate"] = row["mean_is_win"]
        row["loss_rate"] = row["mean_is_loss"]
        out.append(row)
    return out


def _find_row(rows: Sequence[Dict[str, object]], predicate) -> Optional[Dict[str, object]]:
    for row in rows:
        if predicate(row):
            return row
    return None


def _metric_value(row: Optional[Dict[str, object]], key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    return _to_float(row.get(key), default)


def _latex_value(value: float) -> str:
    if not math.isfinite(value):
        return "0.00"
    return f"{value:.2f}"


def write_values_tex(
    out_dir: Path,
    summary_by_game: Sequence[Dict[str, object]],
    summary_by_game_thinking: Sequence[Dict[str, object]],
    summary_by_level: Sequence[Dict[str, object]],
    summary_by_thinking: Sequence[Dict[str, object]],
    summary_by_horizon: Sequence[Dict[str, object]],
    summary_by_causal: Sequence[Dict[str, object]],
) -> None:
    def game_win_rate(game_name: str) -> float:
        row = _find_row(summary_by_game, lambda r: str(r.get("game", "")) == game_name)
        return _metric_value(row, "win_rate")

    def level_win_rate(level: int) -> float:
        row = _find_row(summary_by_level, lambda r: _to_int(r.get("level", -1), -1) == level)
        return _metric_value(row, "win_rate")

    def thinking_diff(game_name: str) -> float:
        on_row = _find_row(
            summary_by_game_thinking,
            lambda r: str(r.get("game", "")) == game_name and str(r.get("qwen_thinking_mode", "")) == "on",
        )
        off_row = _find_row(
            summary_by_game_thinking,
            lambda r: str(r.get("game", "")) == game_name and str(r.get("qwen_thinking_mode", "")) == "off",
        )
        return _metric_value(on_row, "win_rate") - _metric_value(off_row, "win_rate")

    overall_thinking_on = _find_row(summary_by_thinking, lambda r: str(r.get("qwen_thinking_mode", "")) == "on")
    overall_thinking_off = _find_row(summary_by_thinking, lambda r: str(r.get("qwen_thinking_mode", "")) == "off")
    overall_thinking_delta = abs(_metric_value(overall_thinking_on, "win_rate") - _metric_value(overall_thinking_off, "win_rate"))

    def horizon_win_rate(horizon: int) -> float:
        row = _find_row(
            summary_by_horizon,
            lambda r: _to_int(r.get("planning_horizon_x", -1), -1) == horizon,
        )
        return _metric_value(row, "win_rate")

    causal_on = _find_row(summary_by_causal, lambda r: str(r.get("causal_mode", "")) == "on")
    causal_off = _find_row(summary_by_causal, lambda r: str(r.get("causal_mode", "")) == "off")
    causal_diff_win = abs(_metric_value(causal_on, "win_rate") - _metric_value(causal_off, "win_rate"))
    causal_diff_time = abs(
        _metric_value(causal_on, "mean_time_per_step_seconds") - _metric_value(causal_off, "mean_time_per_step_seconds")
    )

    values = [
        f"\\newcommand{{\\gameonewinrate}}{{{_latex_value(game_win_rate('spatialgame1')*100)}}} % Absolute win rate of spatialgame1",
        f"\\newcommand{{\\gametwowinrate}}{{{_latex_value(game_win_rate('spatialgame2')*100)}}} % Absolute win rate of spatialgame2",
        f"\\newcommand{{\\gamethreewinrate}}{{{_latex_value(game_win_rate('spatialgame3')*100)}}} % Absolute win rate of spatialgame3",
        f"\\newcommand{{\\levelzerowinrate}}{{{_latex_value(level_win_rate(0)*100)}}}  % Absolute win rate of all level 0",
        f"\\newcommand{{\\levelonewinrate}}{{{_latex_value(level_win_rate(1)*100)}}} % Absolute win rate of all level 1",
        f"\\newcommand{{\\leveltwowinrate}}{{{_latex_value(level_win_rate(2)*100)}}}  % Absolute win rate of level 2",
        f"\\newcommand{{\\levelthreewinrate}}{{{_latex_value(level_win_rate(3)*100)}}} % Absolute win rate of level 3",
        f"\\newcommand{{\\levelfourwinrate}}{{{_latex_value(level_win_rate(4)*100)}}} % Absolute win rate of level 4",
        f"\\newcommand{{\\diffthinkgameone}}{{{_latex_value(thinking_diff('spatialgame1')*100)}}} % Difference between the winrates for thinking - non thinking for spatialgame1",
        f"\\newcommand{{\\diffthinkgametwo}}{{{_latex_value(thinking_diff('spatialgame2')*100)}}} % Difference between the winrates for thinking - non thinking for spatialgame2",
        f"\\newcommand{{\\diffthinkgamethree}}{{{_latex_value(thinking_diff('spatialgame3')*100)}}} % Difference between the winrates for thinking - non thinking for spatialgame3",
        f"\\newcommand{{\\diffthinkoverall}}{{{_latex_value(overall_thinking_delta*100)}}} % Absolute Difference between the winrates for thinking - non thinking for all games combined",
        f"\\newcommand{{\\plantenwinrate}}{{{_latex_value(horizon_win_rate(10))}}}  % Absolute Winrate with planning horizon 10",
        f"\\newcommand{{\\planfivewinrate}}{{{_latex_value(horizon_win_rate(5))}}} % Absolute Winrate with planning horizon 5",
        f"\\newcommand{{\\planonewinrate}}{{{_latex_value(horizon_win_rate(1))}}} % Absolute Winrate with planning horizon 1",
        f"\\newcommand{{\\diffcausalwin}}{{{_latex_value(causal_diff_win)*100}}} % absolute difference in winrate causal on vs off",
        f"\\newcommand{{\\diffcausaltime}}{{{_latex_value(causal_diff_time)}}} % absolute difference in mean time/step causal on vs off",
    ]
    (out_dir / "values.tex").write_text("\n".join(values) + "\n", encoding="utf-8")


def rank_table(rows: Sequence[Dict[str, object]], key: str, top_k: int, reverse: bool = True) -> List[Dict[str, object]]:
    ranked = sorted(rows, key=lambda r: _to_float(r.get(key, 0.0), 0.0), reverse=reverse)
    return ranked[: max(top_k, 1)]


def maybe_extract_preview(gif_path: Path, out_png: Path) -> bool:
    try:
        import imageio.v2 as imageio

        frames = imageio.mimread(str(gif_path), memtest=False)
        if not frames:
            return False
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(out_png), frames[0])
        return True
    except Exception:
        return False


def _format_value_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _label_bars_within_axes(
    ax: plt.Axes,
    bars: Sequence[object],
    *,
    fontsize: int = 20,
    inside_threshold_ratio: float = 0.16,
) -> None:
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1e-9)
    bottom_pad = 0.02 * yrange
    top_pad = 0.015 * yrange
    inside_threshold = inside_threshold_ratio * yrange

    # Approximate text height in data units to avoid placing labels inside short bars.
    px_to_data_y = abs(
        ax.transData.inverted().transform((0.0, 1.0))[1]
        - ax.transData.inverted().transform((0.0, 0.0))[1]
    )
    text_height_data = (fontsize * ax.figure.dpi / 72.0) * px_to_data_y
    min_inside_height = max(inside_threshold, text_height_data + bottom_pad + top_pad)

    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        x = bar.get_x() + bar.get_width() / 2.0
        label = _format_value_label(height)

        if height >= min_inside_height:
            y = min(height - bottom_pad, ymax - top_pad)
            y = max(y, ymin + top_pad)
            va = "top"
        else:
            y = min(height + top_pad, ymax - top_pad)
            y = max(y, ymin + top_pad)
            va = "bottom"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va=va,
            fontsize=fontsize,
            color="black",
            clip_on=True,
        )


def _annotate_bar_values(ax: plt.Axes, max_bars: int = 32) -> None:
    bars = [patch for patch in ax.patches if hasattr(patch, "get_height")]
    if not bars or len(bars) > max_bars:
        return
    _label_bars_within_axes(ax, bars, fontsize=20)


def _annotate_line_values(ax: plt.Axes, max_total_points: int = 18) -> None:
    lines = [line for line in ax.lines if len(line.get_xdata()) > 0]
    total_points = sum(len(line.get_xdata()) for line in lines)
    if not lines or total_points > max_total_points:
        return
    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1e-9)
    for line in lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            y_float = float(y)
            if not np.isfinite(y_float):
                continue
            ax.text(
                float(x),
                y_float + 0.015 * yrange,
                _format_value_label(y_float),
                color=line.get_color(),
                ha="center",
                va="bottom",
                fontsize=20,
            )


def _reposition_texts_within_axes(
    fig: plt.Figure,
    ax: plt.Axes,
    texts: Sequence[plt.Text],
    pad_px: float = 6.0,
) -> None:
    """Gently nudge labels inside axes bounds in display-space."""
    if not texts:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    x0 = ax_bbox.x0 + pad_px
    y0 = ax_bbox.y0 + pad_px
    x1 = ax_bbox.x1 - pad_px
    y1 = ax_bbox.y1 - pad_px
    inv = ax.transData.inverted()

    for t in texts:
        tb = t.get_window_extent(renderer=renderer)
        dx = 0.0
        dy = 0.0
        if tb.x0 < x0:
            dx = x0 - tb.x0
        elif tb.x1 > x1:
            dx = x1 - tb.x1
        if tb.y0 < y0:
            dy = y0 - tb.y0
        elif tb.y1 > y1:
            dy = y1 - tb.y1
        if dx == 0.0 and dy == 0.0:
            continue
        cx = (tb.x0 + tb.x1) * 0.5
        cy = (tb.y0 + tb.y1) * 0.5
        new_x, new_y = inv.transform((cx + dx, cy + dy))
        t.set_position((float(new_x), float(new_y)))


def _annotate_heatmap_values(ax: plt.Axes, max_cells: int = 25) -> None:
    if not ax.images:
        return
    image = ax.images[0]
    data = np.asarray(image.get_array())
    if data.ndim != 2 or data.size > max_cells:
        return
    finite_vals = data[np.isfinite(data)]
    mid = float(np.nanmean(finite_vals)) if finite_vals.size else 0.0
    for (i, j), value in np.ndenumerate(data):
        if not np.isfinite(value):
            continue
        color = "black" if float(value) < mid else "black"
        ax.text(j, i, _format_value_label(float(value)), ha="center", va="center", fontsize=20, color=color)

def _force_black_axis_text(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def _force_black_colorbar_text(cbar) -> None:
    cbar.ax.tick_params(colors="black")
    cbar.ax.yaxis.label.set_color("black")


def _place_legends_below(fig: plt.Figure, axes: Sequence[plt.Axes]) -> None:
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
        legend_y = float(getattr(ax, "_legend_below_y", -0.18))
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=min(2, max(1, len(labels))),
            frameon=False,
            columnspacing=1.2,
            handletextpad=0.6,
        )


def _finalize_figure(fig: plt.Figure, out_path: Path, *, annotate: bool = True) -> None:
    axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    for ax in axes:
        ax.set_title("")
    _place_legends_below(fig, axes)
    if annotate:
        for ax in axes:
            _annotate_bar_values(ax)
            _annotate_line_values(ax)
            _annotate_heatmap_values(ax)
    try:
        if not fig.get_constrained_layout():
            fig.tight_layout()
    except RuntimeError:
        pass
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


from typing import Sequence, Dict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_win_rate_by_game_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "qwen_thinking_mode"])
    games = sorted({str(r["game"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 3))
    
    bar_containers = []
    for i, t in enumerate(think_modes):
        vals = []
        for g in games:
            match = [r for r in summary if str(r["game"]) == g and str(r["qwen_thinking_mode"]) == t]
            vals.append(_to_float(match[0]["win_rate"], 0.0) if match else 0.0)
        
        # Calculate offset for the grouped bars
        offset = (i - (len(think_modes) - 1) / 2) * width
        
        # Create the bars and store the container in 'rects'
        rects = ax.bar(x + offset, vals, width, label=f"thinking={t}")
        bar_containers.append(rects)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=0)
    ax.set_ylim(0, 1.1)  # Increased limit slightly to make room for labels above bars
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Game and Thinking Mode")
    for rects in bar_containers:
        _label_bars_within_axes(ax, rects, fontsize=14)

    # MOVE LEGEND: bbox_to_anchor moves it relative to the axes
    # (0.5, -0.2) centers it horizontally and pushes it below the X-axis
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.25), 
        ncol=len(think_modes), 
        frameon=False
    )

    _finalize_figure(fig, out_path, annotate=False)


def plot_completion_rate_by_game_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "qwen_thinking_mode"])
    games = sorted({str(r["game"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 5))
    for i, t in enumerate(think_modes):
        vals = []
        for g in games:
            match = [r for r in summary if str(r["game"]) == g and str(r["qwen_thinking_mode"]) == t]
            vals.append(_to_float(match[0]["completion_rate"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(think_modes) - 1) / 2) * width, vals, width, label=f"thinking={t}")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Completion Rate by Game and Thinking Mode",fontsize=16)
    ax._legend_below_y = -0.24
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path,)


def plot_reward_vs_level(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "qwen_thinking_mode", "level"])
    games = sorted({str(r["game"]) for r in summary})

    fig, axes = plt.subplots(len(games), 1, figsize=(9, max(3 * len(games), 4)), sharex=True)
    if len(games) == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        subset = [r for r in summary if str(r["game"]) == game]
        think_modes = sorted({str(r["qwen_thinking_mode"]) for r in subset})
        for t in think_modes:
            s2 = sorted([r for r in subset if str(r["qwen_thinking_mode"]) == t], key=lambda r: _to_int(r["level"], 0))
            xs = [_to_int(r["level"], 0) for r in s2]
            ys = [_to_float(r["mean_normalized_total_reward"], 0.0) for r in s2]
            ax.plot(xs, ys, marker="o", label=f"thinking={t}")
        ax.set_title(f"Mean Normalized Reward vs Level ({game})")
        ax.set_ylabel("Mean Normalized Reward")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    axes[-1].set_xlabel("Level")
    _finalize_figure(fig, out_path, annotate=False)


def plot_spatialgame2_causal_on_absolute_reward_by_level_thinking(
    rows: Sequence[Dict[str, object]], out_path: Path
) -> None:
    summary = aggregate_means(rows, ["game", "level", "qwen_thinking_mode", "causal_mode"])
    subset = [
        r
        for r in summary
        if str(r.get("game", "")) == "spatialgame2" and str(r.get("causal_mode", "")) == "on"
    ]
    if not subset:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.text(0.5, 0.5, "No spatialgame2 causal=on data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    levels = sorted({_to_int(r["level"], 0) for r in subset})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in subset})
    x = np.arange(len(levels))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(levels) * 1.5), 5))
    for i, thinking in enumerate(think_modes):
        vals = []
        for lvl in levels:
            row = next(
                (
                    r
                    for r in subset
                    if _to_int(r["level"], 0) == lvl
                    and str(r["qwen_thinking_mode"]) == thinking
                ),
                None,
            )
            vals.append(_to_float(row.get("mean_total_reward", 0.0), 0.0) if row else 0.0)
        ax.bar(x + (i - (len(think_modes) - 1) / 2) * width, vals, width, label=f"thinking={thinking}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(lvl) for lvl in levels])
    ax.set_xlabel("Level")
    ax.set_ylabel("Mean Absolute Reward")
    ax.grid(axis="y", alpha=0.25)
    _finalize_figure(fig, out_path, annotate=False)


def plot_win_rate_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "planning_horizon_x"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})

    fig, ax = plt.subplots(figsize=(8, 5))
    for t in think_modes:
        s2 = sorted([r for r in summary if str(r["qwen_thinking_mode"]) == t], key=lambda r: _to_int(r["planning_horizon_x"], 1))
        xs = [_to_int(r["planning_horizon_x"], 1) for r in s2]
        ys = [_to_float(r["win_rate"], 0.0) for r in s2]
        ax.plot(xs, ys, marker="o", label=f"thinking={t}")

    ax.set_xlabel("Planning Horizon")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Win Rate vs Planning Horizon")
    ax.grid(alpha=0.25)
    ax.set_xticks([1, 5, 10])
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_completion_rate_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "planning_horizon_x"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})
    x = np.arange(len(horizons))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(think_modes):
        vals = []
        for h in horizons:
            row = next(
                (
                    r
                    for r in summary
                    if str(r["qwen_thinking_mode"]) == t and _to_int(r["planning_horizon_x"], 1) == h
                ),
                None,
            )
            vals.append(_to_float(row["completion_rate"], 0.0) if row else 0.0)
        ax.bar(x + (i - (len(think_modes) - 1) / 2) * width, vals, width, label=f"thinking={t}")

    ax.set_xlabel("Planning Horizon")
    ax.set_ylabel("Completion Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Completion Rate vs Planning Horizon")
    ax.grid(alpha=0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_win_rate_by_game_causal(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 5))
    for i, c in enumerate(causal_modes):
        vals = []
        for g in games:
            match = [r for r in summary if str(r["game"]) == g and str(r["causal_mode"]) == c]
            vals.append(_to_float(match[0]["win_rate"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={c}")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Game and Causal Mode")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _finalize_figure(fig, out_path)


def plot_completion_rate_by_game_causal(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 5))
    for i, c in enumerate(causal_modes):
        vals = []
        for g in games:
            match = [r for r in summary if str(r["game"]) == g and str(r["causal_mode"]) == c]
            vals.append(_to_float(match[0]["completion_rate"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={c}")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Completion Rate by Game and Causal Mode", fontsize=16)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _finalize_figure(fig, out_path)


def plot_reward_by_causal_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    x = np.arange(len(think_modes))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(causal_modes):
        vals = []
        for t in think_modes:
            match = [r for r in summary if str(r["qwen_thinking_mode"]) == t and str(r["causal_mode"]) == c]
            vals.append(_to_float(match[0]["mean_normalized_total_reward"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={c}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"thinking={t}" for t in think_modes])
    ax.set_ylabel("Mean Normalized Reward")
    ax.set_title("Mean Normalized Reward by Thinking and Causal Mode")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_runtime_by_causal_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    x = np.arange(len(think_modes))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, causal in enumerate(causal_modes):
        vals = []
        for thinking in think_modes:
            match = [
                r
                for r in summary
                if str(r["qwen_thinking_mode"]) == thinking and str(r["causal_mode"]) == causal
            ]
            vals.append(_to_float(match[0]["mean_runtime_seconds"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={causal}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"thinking={t}" for t in think_modes])
    ax.set_ylabel("Mean Runtime (s)")
    ax.set_title("Mean Runtime by Thinking and Causal Mode")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_causal_on_off_summary_triplet(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["causal_mode"])
    causal_modes = [c for c in ["off", "on"] if any(str(r["causal_mode"]) == c for r in summary)]
    if not causal_modes:
        causal_modes = sorted({str(r["causal_mode"]) for r in summary})

    metrics = [
        ("win_rate", "Win Rate", (0.0, 1.0)),
        ("mean_runtime_seconds", "Mean Runtime (s)", None),
        ("completion_rate", "Completion Rate", (0.0, 1.0)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    colors = {"off": "tab:blue", "on": "tab:orange"}
    x = np.arange(len(causal_modes))
    for ax, (metric_key, ylabel, ylim) in zip(axes, metrics):
        vals = []
        for mode in causal_modes:
            row = next((r for r in summary if str(r["causal_mode"]) == mode), None)
            vals.append(_to_float(row.get(metric_key, 0.0), 0.0) if row else 0.0)
        bar_colors = [colors.get(mode, "tab:gray") for mode in causal_modes]
        ax.bar(x, vals, color=bar_colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"causal={m}" for m in causal_modes])
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_win_rate_by_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile"])
    models = [str(r["model_profile"]) for r in summary]
    vals = [_to_float(r["win_rate"], 0.0) for r in summary]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    ax.bar(range(len(models)), vals, color="tab:blue", alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Model")
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_completion_rate_by_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile"])
    models = [str(r["model_profile"]) for r in summary]
    vals = [_to_float(r["completion_rate"], 0.0) for r in summary]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    ax.bar(range(len(models)), vals, color="tab:cyan", alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Completion Rate by Model")
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_reward_by_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile"])
    models = [str(r["model_profile"]) for r in summary]
    vals = [_to_float(r["mean_normalized_total_reward"], 0.0) for r in summary]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    ax.bar(range(len(models)), vals, color="tab:orange", alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Mean Normalized Reward")
    ax.set_title("Mean Normalized Reward by Model")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_time_per_step_by_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile"])
    models = [str(r["model_profile"]) for r in summary]
    vals = [_to_float(r["mean_time_per_step_seconds"], 0.0) for r in summary]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.6), 5))
    ax.bar(range(len(models)), vals, color="tab:green", alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Mean Time per Step (s)")
    ax.set_title("Mean Time per Step by Model")
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_win_rate_by_game_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "model_profile"])
    games = sorted({str(r["game"]) for r in summary})
    models = sorted({str(r["model_profile"]) for r in summary})
    x = np.arange(len(games))
    width = max(0.16, 0.82 / max(len(models), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.4), 5))
    for i, model in enumerate(models):
        vals = []
        for game in games:
            match = [r for r in summary if str(r["game"]) == game and str(r["model_profile"]) == model]
            vals.append(_to_float(match[0]["win_rate"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Win Rate")
    ax.set_title("Win Rate by Game and Model")
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_completion_rate_by_game_model(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "model_profile"])
    games = sorted({str(r["game"]) for r in summary})
    models = sorted({str(r["model_profile"]) for r in summary})
    x = np.arange(len(games))
    width = max(0.16, 0.82 / max(len(models), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.4), 5))
    for i, model in enumerate(models):
        vals = []
        for game in games:
            match = [r for r in summary if str(r["game"]) == game and str(r["model_profile"]) == model]
            vals.append(_to_float(match[0]["completion_rate"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Completion Rate")
    ax.set_title("Completion Rate by Game and Model")
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_heatmap_model_level_metric(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    title: str,
    cmap: str = "YlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    summary = aggregate_means(rows, ["model_profile", "level"])
    models = sorted({str(r["model_profile"]) for r in summary})
    levels = sorted({_to_int(r["level"], 0) for r in summary})
    mat = np.zeros((len(models), len(levels)), dtype=float)
    for i, model in enumerate(models):
        for j, lvl in enumerate(levels):
            row = next(
                (r for r in summary if str(r["model_profile"]) == model and _to_int(r["level"], 0) == lvl),
                None,
            )
            mat[i, j] = _to_float(row.get(metric_key, 0.0), 0.0) if row else 0.0

    fig, ax = plt.subplots(
        figsize=(max(9.5, len(levels) * 1.35), max(4.8, len(models) * 1.1)),
        constrained_layout=True,
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(x) for x in levels])
    ax.set_xlabel("Level")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    _force_black_axis_text(ax)
    _force_black_colorbar_text(cbar)
    _finalize_figure(fig, out_path)


def plot_model_reward_vs_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile", "planning_horizon_x"])
    models = sorted({str(r["model_profile"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for model in models:
        subset = sorted(
            [r for r in summary if str(r["model_profile"]) == model],
            key=lambda r: _to_int(r["planning_horizon_x"], 1),
        )
        xs = [_to_int(r["planning_horizon_x"], 1) for r in subset]
        ys = [_to_float(r["mean_normalized_total_reward"], 0.0) for r in subset]
        ax.plot(xs, ys, marker="o", linewidth=2, label=model)

    ax.set_xlabel("Planning Horizon")
    ax.set_xticks([1, 5, 10])
    ax.set_ylabel("Mean Normalized Reward")
    ax.set_title("Mean Normalized Reward by Model and Planning Horizon")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path,)


def plot_causal_horizon_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})

    fig_height = max(3.6, len(causal_modes) * 1.45)
    fig, axes = plt.subplots(
        1,
        len(think_modes),
        figsize=(max(10, 4.8 * len(think_modes)), fig_height),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    for col, t in enumerate(think_modes):
        mat = np.zeros((len(causal_modes), len(horizons)), dtype=float)
        for i, c in enumerate(causal_modes):
            for j, h in enumerate(horizons):
                row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == t
                        and str(r["causal_mode"]) == c
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                mat[i, j] = _to_float(row["win_rate"], 0.0) if row else 0.0
        ax = axes[0][col]
        im = ax.imshow(mat, aspect="equal", cmap="YlGn", vmin=0.0, vmax=1.0)
        last_im = im
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_yticks(range(len(causal_modes)))
        if col == 0:
            ax.set_yticklabels([f"causal={c}" for c in causal_modes])
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Planning Horizon")
        ax.set_title(f"Win Rate Heatmap (thinking={t})")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="black",
                )
    # for ax in axes[0]:
    #     ax.set_xticks(range(len(horizons)))
    #     ax.set_xticklabels([str(h) for h in horizons])
    #     ax.set_title(f"Win Rate Heatmap (thinking={t})")
    if len(think_modes) > 1:
        fig.supylabel("Causal Mode")
    else:
        axes[0][0].set_ylabel("Causal Mode")
    for ax in axes[0]:
        _force_black_axis_text(ax)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes[0].tolist(), fraction=0.035, pad=0.03)
        cbar.set_label("Win Rate")
        _force_black_colorbar_text(cbar)
    _finalize_figure(fig, out_path, annotate=False)


def plot_causal_comparison_by_level(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    summary = aggregate_means(rows, ["game", "level", "qwen_thinking_mode", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})

    fig, axes = plt.subplots(len(games), 1, figsize=(10, max(3.4 * len(games), 4.5)), sharex=True)
    if len(games) == 1:
        axes = [axes]

    combos = [(thinking, causal) for thinking in think_modes for causal in causal_modes]
    if len(combos) <= 2:
        combo_colors = {
            combo: color for combo, color in zip(combos, ["tab:blue", "tab:orange"])
        }
    else:
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(combos), 1)))
        combo_colors = {combo: palette[idx % len(palette)] for idx, combo in enumerate(combos)}
    linestyles = {"off": "-", "on": "--"}
    markers = {"off": "o", "on": "s"}

    for ax, game in zip(axes, games):
        game_rows = [r for r in summary if str(r["game"]) == game]
        levels = sorted({_to_int(r["level"], 0) for r in game_rows})
        for thinking in think_modes:
            for causal in causal_modes:
                vals = []
                for lvl in levels:
                    row = next(
                        (
                            r
                            for r in game_rows
                            if _to_int(r["level"], 0) == lvl
                            and str(r["qwen_thinking_mode"]) == thinking
                            and str(r["causal_mode"]) == causal
                        ),
                        None,
                    )
                    vals.append(_to_float(row.get(metric_key, 0.0), 0.0) if row else float("nan"))
                ax.plot(
                    levels,
                    vals,
                    marker=markers.get(causal, "o"),
                    linestyle=linestyles.get(causal, "-"),
                    color=combo_colors[(thinking, causal)],
                    linewidth=2,
                    label=f"thinking={thinking}, causal={causal}",
                )
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Level")
    _finalize_figure(fig, out_path)


def plot_causal_completion_comparison_by_level_bar(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "level", "qwen_thinking_mode", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    combos = [(thinking, causal) for thinking in think_modes for causal in causal_modes]
    if len(combos) <= 2:
        combo_colors = {combo: color for combo, color in zip(combos, ["tab:blue", "tab:orange"])}
    else:
        palette = plt.cm.tab10(np.linspace(0, 1, max(len(combos), 1)))
        combo_colors = {combo: palette[idx % len(palette)] for idx, combo in enumerate(combos)}

    fig, axes = plt.subplots(len(games), 1, figsize=(10, max(3.6 * len(games), 4.5)), sharex=False)
    if len(games) == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        game_rows = [r for r in summary if str(r["game"]) == game]
        levels = sorted({_to_int(r["level"], 0) for r in game_rows})
        x = np.arange(len(levels))
        width = max(0.12, 0.86 / max(len(combos), 1))
        for i, (thinking, causal) in enumerate(combos):
            vals = []
            for lvl in levels:
                row = next(
                    (
                        r
                        for r in game_rows
                        if _to_int(r["level"], 0) == lvl
                        and str(r["qwen_thinking_mode"]) == thinking
                        and str(r["causal_mode"]) == causal
                    ),
                    None,
                )
                vals.append(_to_float(row.get("completion_rate", 0.0), 0.0) if row else 0.0)
            ax.bar(
                x + (i - (len(combos) - 1) / 2) * width,
                vals,
                width,
                color=combo_colors[(thinking, causal)],
                label=f"thinking={thinking}, causal={causal}",
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Completion Rate")
        ax.set_xticks(x)
        ax.set_xticklabels([str(lvl) for lvl in levels])
        ax.set_xlabel("Level")
        ax.grid(axis="y", alpha=0.25)

    _finalize_figure(fig, out_path)


def plot_scatter_runtime_tokens(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for mode, marker in [("on", "o"), ("off", "x")]:
        subset = [r for r in rows if str(r.get("qwen_thinking_mode", "")) == mode]
        xs = [_to_float(r.get("sum_total_tokens", 0.0), 0.0) for r in subset]
        ys = [_to_float(r.get("runtime_seconds", 0.0), 0.0) for r in subset]
        colors = ["tab:green" if _to_int(r.get("is_win", 0), 0) == 1 else "tab:red" for r in subset]
        ax.scatter(xs, ys, c=colors, marker=marker, alpha=0.6, label=f"thinking={mode}")

    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Token Usage (green=win, red=non-win)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_dense_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    combos = [
        (str(r["qwen_thinking_mode"]), str(r["causal_mode"]), _to_int(r["planning_horizon_x"], 1))
        for r in summary
    ]
    combos = sorted(set(combos), key=lambda t: (t[0], t[1], t[2]))

    metrics = [
        ("win_rate", "WinRate"),
        ("mean_normalized_total_reward", "NormReward"),
        ("mean_meaningful_step_ratio", "Meaningful"),
        ("mean_effective_step_ratio", "Effective"),
        ("mean_action_execution_match_rate", "ExecMatch"),
        ("mean_llm_action_parse_rate", "ParseRate"),
        ("mean_reward_per_1k_tokens", "Reward/1kTok"),
        ("mean_steps_per_second", "Steps/Sec"),
    ]

    mat = np.zeros((len(metrics), len(combos)), dtype=float)
    for j, c in enumerate(combos):
        row = next((r for r in summary if (str(r["qwen_thinking_mode"]), str(r["causal_mode"]), _to_int(r["planning_horizon_x"], 1)) == c), None)
        if row is None:
            continue
        for i, (key, _) in enumerate(metrics):
            mat[i, j] = _to_float(row.get(key, 0.0), 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(combos) * 1.1), 5.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[1] for m in metrics])
    labels = [f"T{t}-C{c}-H{h}" for t, c, h in combos]
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Dense Metric Heatmap by (Thinking, Causal, Horizon)")
    cbar = fig.colorbar(im, ax=ax)
    _force_black_axis_text(ax)
    _force_black_colorbar_text(cbar)
    _finalize_figure(fig, out_path)


def plot_steps_distribution(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [_to_int(r.get("steps", 0), 0) for r in rows]
    ax.hist(steps, bins=25, color="tab:blue", alpha=0.8)
    ax.set_title("Distribution of Episode Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    _finalize_figure(fig, out_path)


def _paper_metric_vector(row: Dict[str, object]) -> Dict[str, float]:
    return {
        "M1.1 Completion Rate": _to_float(row.get("completion_rate", 0.0), 0.0),
        "M1.2 Win Rate": _to_float(row.get("win_rate", 0.0), 0.0),
        "M1.3 Mean Time/Step (s)": _to_float(row.get("mean_time_per_step_seconds", 0.0), 0.0),
        "M1.4 Steps to Goal": _to_float(row.get("mean_steps", 0.0), 0.0),
    }


def write_paper_metrics_tables(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    out_rows: List[Dict[str, object]] = []
    for r in summary:
        base = {
            "thinking": str(r.get("qwen_thinking_mode", "")),
            "causal": str(r.get("causal_mode", "")),
            "planning_horizon_x": _to_int(r.get("planning_horizon_x", 1), 1),
            "n": _to_int(r.get("n", 0), 0),
        }
        base.update(_paper_metric_vector(r))
        out_rows.append(base)
    write_csv(out_dir / "paper_metrics_by_mode.csv", out_rows)


def write_paper_metrics_latex(paper_metric_plots: Dict[str, Path], paper_metrics_dir: Path) -> List[Path]:
    """Write LaTeX figure snippets for paper-metric plots in the same folder."""
    exported: List[Path] = []
    lines = [
        "% Auto-generated by scripts/aggregate_spatial_results.py",
        "% Include this file from your paper as needed.",
        "",
    ]

    for title, plot_path in sorted(paper_metric_plots.items(), key=lambda kv: kv[0]):
        pdf_name = plot_path.with_suffix(".pdf").name
        tex_name = f"{plot_path.stem}.tex"
        tex_path = paper_metrics_dir / tex_name
        label = f"fig:{_safe_slug(plot_path.stem)}"
        caption = title.replace("&", "\\&")
        tex_content = [
            "% Auto-generated figure snippet",
            "\\begin{figure}[t]",
            "  \\centering",
            f"  \\includegraphics[width=\\linewidth]{{{pdf_name}}}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "\\end{figure}",
            "",
        ]
        tex_path.write_text("\n".join(tex_content), encoding="utf-8")
        exported.append(tex_path)
        lines.append(f"\\input{{{tex_name}}}")

    index_path = paper_metrics_dir / "paper_metrics_figures.tex"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    exported.append(index_path)
    return exported


def plot_paper_m1_by_mode(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    summary = sorted(
        summary,
        key=lambda r: (
            str(r["qwen_thinking_mode"]),
            str(r["causal_mode"]),
            _to_int(r["planning_horizon_x"], 1),
        ),
    )
    headers = ["Thinking", "Causal", "Horizon", "M1.1", "M1.2", "M1.3 (s)", "M1.4"]
    table_rows: List[List[str]] = []
    for r in summary:
        table_rows.append(
            [
                str(r["qwen_thinking_mode"]),
                str(r["causal_mode"]),
                str(_to_int(r["planning_horizon_x"], 1)),
                f"{_to_float(r.get('completion_rate', 0.0), 0.0):.3f}",
                f"{_to_float(r.get('win_rate', 0.0), 0.0):.3f}",
                f"{_to_float(r.get('mean_time_per_step_seconds', 0.0), 0.0):.3f}",
                f"{_to_float(r.get('mean_steps', 0.0), 0.0):.2f}",
            ]
        )

    fig_h = max(4.5, 0.42 * (len(table_rows) + 1))
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef9")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fb")
    _finalize_figure(fig, out_path)


def _format_table_cell(value: Optional[object]) -> str:
    if value is None:
        return "NA"
    val = _to_float(value, float("nan"))
    if not np.isfinite(val):
        return "NA"
    return f"{val:.3f}"


def _draw_summary_table(ax: plt.Axes, title: str, headers: List[str], body: List[List[str]]) -> None:
    ax.axis("off")
    ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=16, fontweight="bold")
    tbl = ax.table(cellText=body, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef9")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fb")


def _model_and_avg_groups(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, Sequence[Dict[str, object]]]]:
    summary = aggregate_means(rows, ["model_profile"])
    model_names = _ordered_model_names_from_summary(summary) if summary else []
    groups: List[Tuple[str, Sequence[Dict[str, object]]]] = []
    for model in model_names:
        subset = [
            r for r in rows if str(r.get("model_profile", "")).replace("local-", "").replace("-vllm", "") == model
        ]
        if subset:
            groups.append((model, subset))
    groups.append(("Average", rows))
    return groups


def plot_summary_table_causal(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    headers = ["Causal", "WinRate", "Completion", "Mean Time/Step (s)"]
    groups = _model_and_avg_groups(rows)

    fig, axes = plt.subplots(len(groups), 1, figsize=(12.5, 2.0 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    for ax, (label, subset) in zip(axes, groups):
        causal_summary = aggregate_means(subset, ["causal_mode"])
        body = []
        for mode in ["off", "on"]:
            row = next((r for r in causal_summary if str(r.get("causal_mode", "")) == mode), None)
            body.append(
                [
                    mode,
                    _format_table_cell(row.get("win_rate") if row else None),
                    _format_table_cell(row.get("completion_rate") if row else None),
                    _format_table_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
        _draw_summary_table(ax, f"Causal Summary (Model: {label})", headers, body)

    fig.subplots_adjust(hspace=0.12, top=0.98, bottom=0.02)
    _finalize_figure(fig, out_path)


def plot_summary_table_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    headers = ["Thinking", "WinRate", "Completion", "Mean Time/Step (s)"]
    groups = _model_and_avg_groups(rows)

    fig, axes = plt.subplots(len(groups), 1, figsize=(12.5, 2.0 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    for ax, (label, subset) in zip(axes, groups):
        thinking_summary = aggregate_means(subset, ["qwen_thinking_mode"])
        body = []
        for mode in ["off", "on"]:
            row = next((r for r in thinking_summary if str(r.get("qwen_thinking_mode", "")) == mode), None)
            body.append(
                [
                    mode,
                    _format_table_cell(row.get("win_rate") if row else None),
                    _format_table_cell(row.get("completion_rate") if row else None),
                    _format_table_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
        _draw_summary_table(ax, f"Thinking Summary (Model: {label})", headers, body)

    fig.subplots_adjust(hspace=0.12, top=0.98, bottom=0.02)
    _finalize_figure(fig, out_path)


def plot_summary_table_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    headers = ["Horizon", "WinRate", "Completion", "Mean Time/Step (s)"]
    groups = _model_and_avg_groups(rows)

    fig, axes = plt.subplots(len(groups), 1, figsize=(12.5, 2.1 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    for ax, (label, subset) in zip(axes, groups):
        horizon_summary = aggregate_means(subset, ["planning_horizon_x"])
        body = []
        for h in [1, 5, 10]:
            row = next(
                (r for r in horizon_summary if _to_int(r.get("planning_horizon_x", 1), 1) == h),
                None,
            )
            body.append(
                [
                    str(h),
                    _format_table_cell(row.get("win_rate") if row else None),
                    _format_table_cell(row.get("completion_rate") if row else None),
                    _format_table_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
        _draw_summary_table(ax, f"Horizon Summary (Model: {label})", headers, body)

    fig.subplots_adjust(hspace=0.12, top=0.98, bottom=0.02)
    _finalize_figure(fig, out_path)


# def write_mode_summary_table_latex(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
#     def latex_cell(value: Optional[object]) -> str:
#         if value is None:
#             return "NA"
#         val = _to_float(value, float("nan"))
#         if not math.isfinite(val):
#             return "NA"
#         return f"{val:.3f}"

#     lines = [
#         "% Auto-generated by scripts/aggregate_spatial_results.py",
#         "\\begin{table}[t]",
#         "  \\centering",
#         "  \\caption{Average win rate, completion rate, and mean time per step by causal mode, thinking mode, and horizon (all runs combined).}",
#         "  \\label{tab:spatial_mode_summary}",
#         "  \\begin{tabular}{lrrr}",
#         "    \\toprule",
#         "    Causal & WinRate & Completion & Mean Time/Step (s) \\\\",
#         "    \\midrule",
#     ]

#     for mode in ["off", "on"]:
#         causal_summary = aggregate_means(rows, ["causal_mode"])
#         row = next((r for r in causal_summary if str(r.get("causal_mode", "")) == mode), None)
#         lines.append(
#             "    "
#             + " & ".join(
#                 [
#                     mode,
#                     latex_cell(row.get("win_rate") if row else None),
#                     latex_cell(row.get("completion_rate") if row else None),
#                     latex_cell(row.get("mean_time_per_step_seconds") if row else None),
#                 ]
#             )
#             + " \\\\"
#         )

#     lines.extend(
#         [
#             "    \\bottomrule",
#             "  \\end{tabular}",
#             "  \\vspace{0.6em}",
#             "  \\begin{tabular}{lrrr}",
#             "    \\toprule",
#             "    Thinking & WinRate & Completion & Mean Time/Step (s) \\\\",
#             "    \\midrule",
#         ]
#     )

#     for mode in ["off", "on"]:
#         thinking_summary = aggregate_means(rows, ["qwen_thinking_mode"])
#         row = next((r for r in thinking_summary if str(r.get("qwen_thinking_mode", "")) == mode), None)
#         lines.append(
#             "    "
#             + " & ".join(
#                 [
#                     mode,
#                     latex_cell(row.get("win_rate") if row else None),
#                     latex_cell(row.get("completion_rate") if row else None),
#                     latex_cell(row.get("mean_time_per_step_seconds") if row else None),
#                 ]
#             )
#             + " \\\\"
#         )

#     lines.extend(
#         [
#             "    \\bottomrule",
#             "  \\end{tabular}",
#             "  \\vspace{0.6em}",
#             "  \\begin{tabular}{lrrr}",
#             "    \\toprule",
#             "    Horizon & WinRate & Completion & Mean Time/Step (s) \\\\",
#             "    \\midrule",
#         ]
#     )

#     for h in [1, 5, 10]:
#         horizon_summary = aggregate_means(rows, ["planning_horizon_x"])
#         row = next(
#             (r for r in horizon_summary if _to_int(r.get("planning_horizon_x", 1), 1) == h),
#             None,
#         )
#         lines.append(
#             "    "
#             + " & ".join(
#                 [
#                     str(h),
#                     latex_cell(row.get("win_rate") if row else None),
#                     latex_cell(row.get("completion_rate") if row else None),
#                     latex_cell(row.get("mean_time_per_step_seconds") if row else None),
#                 ]
#             )
#             + " \\\\"
#         )

#     lines.extend(
#         [
#             "    \\bottomrule",
#             "  \\end{tabular}",
#             "\\end{table}",
#             "",
#         ]
#     )
#     out_path.write_text("\n".join(lines), encoding="utf-8")

def write_mode_summary_table_latex(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    def latex_cell(value: Optional[object]) -> str:
        if value is None:
            return "NA"
        val = _to_float(value, float("nan"))
        if not math.isfinite(val):
            return "NA"
        return f"{val:.3f}"

    lines = [
        "% Auto-generated by scripts/aggregate_spatial_results.py",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Average win rate, completion rate, and mean time per step by causal mode, thinking mode, and horizon (all runs combined).}",
        "  \\label{tab:spatial_mode_summary}",
        "  \\begin{tabular}{lrrr}",
        "    \\toprule",
        "    \\textbf{Category} & \\textbf{Win Rate} & \\textbf{Completion} & \\textbf{Mean Time/Step (s)} \\\\",
        "    \\midrule",
        "    \\textit{Causal} & & & \\\\",
    ]

    causal_summary = aggregate_means(rows, ["causal_mode"])
    for mode in ["off", "on"]:
        row = next((r for r in causal_summary if str(r.get("causal_mode", "")) == mode), None)
        lines.append(
            "    "
            + " & ".join(
                [
                    mode,
                    latex_cell(row.get("win_rate") if row else None),
                    latex_cell(row.get("completion_rate") if row else None),
                    latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "    \\cmidrule(lr){1-4}",
            "    \\textit{Thinking} & & & \\\\",
        ]
    )

    thinking_summary = aggregate_means(rows, ["qwen_thinking_mode"])
    for mode in ["off", "on"]:
        row = next((r for r in thinking_summary if str(r.get("qwen_thinking_mode", "")) == mode), None)
        lines.append(
            "    "
            + " & ".join(
                [
                    mode,
                    latex_cell(row.get("win_rate") if row else None),
                    latex_cell(row.get("completion_rate") if row else None),
                    latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "    \\cmidrule(lr){1-4}",
            "    \\textit{Horizon} & & & \\\\",
        ]
    )

    horizon_summary = aggregate_means(rows, ["planning_horizon_x"])
    for h in [1, 5, 10]:
        row = next(
            (r for r in horizon_summary if _to_int(r.get("planning_horizon_x", 1), 1) == h),
            None,
        )
        lines.append(
            "    "
            + " & ".join(
                [
                    str(h),
                    latex_cell(row.get("win_rate") if row else None),
                    latex_cell(row.get("completion_rate") if row else None),
                    latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                ]
            )
            + " \\\\"
        )

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_three_summary_tables_latex(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    def latex_cell(value: Optional[object]) -> str:
        if value is None:
            return "NA"
        val = _to_float(value, float("nan"))
        if not math.isfinite(val):
            return "NA"
        return f"{val:.3f}"

    def esc(text: str) -> str:
        return (
            text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("#", "\\#")
            .replace("{", "\\{")
            .replace("}", "\\}")
        )

    groups = _model_and_avg_groups(rows)
    lines: List[str] = [
        "% Auto-generated by scripts/aggregate_spatial_results.py",
        "% Requires \\usepackage{booktabs}",
        "",
    ]

    # Table 1: Causal summary
    lines.extend(
        [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{Causal summary by model.}",
            "  \\label{tab:spatial_summary_causal}",
            "  \\begin{tabular}{llrrr}",
            "    \\toprule",
            "    Model & Causal & WinRate & Completion & Mean Time/Step (s) \\\\",
            "    \\midrule",
        ]
    )
    for model_label, subset in groups:
        causal_summary = aggregate_means(subset, ["causal_mode"])
        for mode in ["off", "on"]:
            row = next((r for r in causal_summary if str(r.get("causal_mode", "")) == mode), None)
            lines.append(
                "    "
                + " & ".join(
                    [
                        esc(model_label),
                        mode,
                        latex_cell(row.get("win_rate") if row else None),
                        latex_cell(row.get("completion_rate") if row else None),
                        latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])

    # Table 2: Thinking summary
    lines.extend(
        [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{Thinking summary by model.}",
            "  \\label{tab:spatial_summary_thinking}",
            "  \\begin{tabular}{llrrr}",
            "    \\toprule",
            "    Model & Thinking & WinRate & Completion & Mean Time/Step (s) \\\\",
            "    \\midrule",
        ]
    )
    for model_label, subset in groups:
        thinking_summary = aggregate_means(subset, ["qwen_thinking_mode"])
        for mode in ["off", "on"]:
            row = next((r for r in thinking_summary if str(r.get("qwen_thinking_mode", "")) == mode), None)
            lines.append(
                "    "
                + " & ".join(
                    [
                        esc(model_label),
                        mode,
                        latex_cell(row.get("win_rate") if row else None),
                        latex_cell(row.get("completion_rate") if row else None),
                        latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])

    # Table 3: Horizon summary
    lines.extend(
        [
            "\\begin{table}[t]",
            "  \\centering",
            "  \\caption{Horizon summary by model.}",
            "  \\label{tab:spatial_summary_horizon}",
            "  \\begin{tabular}{llrrr}",
            "    \\toprule",
            "    Model & Horizon & WinRate & Completion & Mean Time/Step (s) \\\\",
            "    \\midrule",
        ]
    )
    for model_label, subset in groups:
        horizon_summary = aggregate_means(subset, ["planning_horizon_x"])
        for h in [1, 5, 10]:
            row = next(
                (r for r in horizon_summary if _to_int(r.get("planning_horizon_x", 1), 1) == h),
                None,
            )
            lines.append(
                "    "
                + " & ".join(
                    [
                        esc(model_label),
                        str(h),
                        latex_cell(row.get("win_rate") if row else None),
                        latex_cell(row.get("completion_rate") if row else None),
                        latex_cell(row.get("mean_time_per_step_seconds") if row else None),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}", ""])

    out_path.write_text("\n".join(lines), encoding="utf-8")

def plot_paper_m2_causal_deltas(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})
    metric_specs = [
        ("completion_rate", "M1.1"),
        ("win_rate", "M1.2"),
        ("mean_time_per_step_seconds", "M1.3"),
        ("mean_steps", "M1.4"),
    ]
    x = np.arange(len(horizons))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric_key, metric_label) in zip(axes.flatten(), metric_specs):
        for i, t in enumerate(think_modes):
            vals = []
            for h in horizons:
                on_row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == t
                        and str(r["causal_mode"]) == "on"
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                off_row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == t
                        and str(r["causal_mode"]) == "off"
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                on_v = _to_float(on_row.get(metric_key, 0.0), 0.0) if on_row else 0.0
                off_v = _to_float(off_row.get(metric_key, 0.0), 0.0) if off_row else 0.0
                vals.append(on_v - off_v)
            ax.bar(x + (i - (len(think_modes) - 1) / 2) * width, vals, width, label=f"thinking={t}")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_ylabel(f"Delta {metric_label} (on-off)")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Planning Horizon")
    _finalize_figure(fig, out_path)


def plot_paper_m3_thinking_deltas(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})
    metric_specs = [
        ("completion_rate", "M1.1"),
        ("win_rate", "M1.2"),
        ("mean_time_per_step_seconds", "M1.3"),
        ("mean_steps", "M1.4"),
    ]
    x = np.arange(len(horizons))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric_key, metric_label) in zip(axes.flatten(), metric_specs):
        for i, c in enumerate(causal_modes):
            vals = []
            for h in horizons:
                on_row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == "on"
                        and str(r["causal_mode"]) == c
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                off_row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == "off"
                        and str(r["causal_mode"]) == c
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                on_v = _to_float(on_row.get(metric_key, 0.0), 0.0) if on_row else 0.0
                off_v = _to_float(off_row.get(metric_key, 0.0), 0.0) if off_row else 0.0
                vals.append(on_v - off_v)
            ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={c}")
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        ax.set_ylabel(f"Delta {metric_label} (think on-off)")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_xlabel("Planning Horizon")
    _finalize_figure(fig, out_path)


def plot_paper_m3_horizon_comparison(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    combos = sorted({(str(r["qwen_thinking_mode"]), str(r["causal_mode"])) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})
    metric_specs = [
        ("completion_rate", "M1.1 Completion Rate"),
        ("win_rate", "M1.2 Win Rate"),
        ("mean_time_per_step_seconds", "M1.3 Mean Time/Step (s)"),
        ("mean_steps", "M1.4 Steps to Goal"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (metric_key, metric_label) in zip(axes.flatten(), metric_specs):
        for thinking, causal in combos:
            vals = []
            for h in horizons:
                row = next(
                    (
                        r
                        for r in summary
                        if str(r["qwen_thinking_mode"]) == thinking
                        and str(r["causal_mode"]) == causal
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                vals.append(_to_float(row.get(metric_key, 0.0), 0.0) if row else float("nan"))
            ax.plot(horizons, vals, marker="o", linewidth=2, label=f"T{thinking}-C{causal}")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)
        ax.set_xticks([1, 5, 10])
    for ax in axes[-1]:
        ax.set_xlabel("Planning Horizon")
    _finalize_figure(fig, out_path)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()


# def plot_heatmap_game_level_metric(
#     rows: Sequence[Dict[str, object]],
#     out_path: Path,
#     metric_key: str,
#     title: str,
#     cmap: str = "viridis",
#     vmin: Optional[float] = None,
#     vmax: Optional[float] = None,
# ) -> None:
#     summary = aggregate_means(rows, ["game", "level"])
#     games = sorted({str(r["game"]) for r in summary})
#     levels = sorted({_to_int(r["level"], 0) for r in summary})
#     mat = np.zeros((len(games), len(levels)), dtype=float)
#     for i, g in enumerate(games):
#         for j, lvl in enumerate(levels):
#             row = next(
#                 (r for r in summary if str(r["game"]) == g and _to_int(r["level"], 0) == lvl),
#                 None,
#             )
#             mat[i, j] = _to_float(row.get(metric_key, 0.0), 0.0) if row else 0.0

#     row_avgs = np.nanmean(mat, axis=1)
#     col_avgs = np.nanmean(mat, axis=0)
#     overall_avg = np.nanmean(mat)

#     import matplotlib.gridspec as gridspec

#     fig = plt.figure(figsize=(max(10, len(levels) * 0.9), max(6, len(games) * 0.6)))
#     spec = gridspec.GridSpec(2, 2, width_ratios=[len(levels), 1], height_ratios=[len(games), 1], hspace=0.05, wspace=0.05)

#     ax_main = fig.add_subplot(spec[0, 0])
#     im = ax_main.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
#     ax_main.set_yticks(range(len(games)))
#     ax_main.set_yticklabels(games)
#     ax_main.set_xticks(range(len(levels)))
#     ax_main.set_xticklabels([str(x) for x in levels])
#     ax_main.set_xlabel("Level")
#     ax_main.set_title(title, pad=12)

#     ax_right = fig.add_subplot(spec[0, 1], sharey=ax_main)
#     ax_right.imshow(row_avgs[:, None], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
#     ax_right.set_xticks([])
#     ax_right.set_yticks(range(len(games)))
#     ax_right.set_yticklabels([])
#     ax_right.set_xlabel("Game Avg")

#     ax_bottom = fig.add_subplot(spec[1, 0], sharex=ax_main)
#     ax_bottom.imshow([col_avgs], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
#     ax_bottom.set_xticks(range(len(levels)))
#     ax_bottom.set_xticklabels([])
#     ax_bottom.set_yticks([])
#     ax_bottom.set_ylabel("Level Avg", labelpad=10)

#     ax_corner = fig.add_subplot(spec[1, 1])
#     ax_corner.imshow([[overall_avg]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
#     ax_corner.set_xticks([])
#     ax_corner.set_yticks([])
#     ax_corner.set_xlabel("Overall Avg")

#     fig.colorbar(im, ax=ax_main, fraction=0.04, pad=0.03)
#     _finalize_figure(fig, out_path)

def plot_heatmap_game_level_metric(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    title: str,
    cmap: str = "YlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    value_fmt: str = ".2f",
    gap: float = 0.3,
) -> None:
    """
    Plot a game x level heatmap with:
    - exact numeric annotations in every cell
    - row averages on the right
    - column averages at the bottom
    - overall average in the bottom-right corner
    - a small visual gap between the main heatmap and the averages
    - the colorbar placed after the averages

    Parameters
    ----------
    rows
        Input records.
    out_path
        Output file path.
    metric_key
        Metric to visualize.
    title
        Figure title.
    cmap
        Matplotlib colormap name.
    vmin, vmax
        Optional color scale limits.
    value_fmt
        Format string for cell annotations, e.g. ".2f", ".1f", ".0f".
    gap
        Small spacer size between the main heatmap and average panels.
    """
    summary = aggregate_means(rows, ["game", "level"])

    games = sorted({str(r["game"]) for r in summary})
    levels = sorted({_to_int(r["level"], 0) for r in summary})

    mat = np.full((len(games), len(levels)), np.nan, dtype=float)
    lookup = {
        (str(r["game"]), _to_int(r["level"], 0)): _to_float(r.get(metric_key, np.nan), np.nan)
        for r in summary
    }

    for i, game in enumerate(games):
        for j, level in enumerate(levels):
            if (game, level) in lookup:
                mat[i, j] = lookup[(game, level)]

    row_avgs = np.nanmean(mat, axis=1)
    col_avgs = np.nanmean(mat, axis=0)
    overall_avg = np.nanmean(mat)

    if vmin is None:
        vmin = float(np.nanmin(mat)) if np.isfinite(np.nanmin(mat)) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(mat)) if np.isfinite(np.nanmax(mat)) else 1.0

    fig = plt.figure(figsize=(max(10, len(levels) * 1.0), max(6, len(games) * 0.7)))

    # 4 columns: main | spacer | row avg | colorbar
    # 4 rows:    main | spacer | col avg | overall row
    gs = gridspec.GridSpec(
        4,
        4,
        width_ratios=[len(levels), gap, 1.2, 0.35],
        height_ratios=[len(games), gap, 1.2, 0.0],
        wspace=0.05,
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 2], sharey=ax_main)
    ax_col = fig.add_subplot(gs[2, 0], sharex=ax_main)
    ax_corner = fig.add_subplot(gs[2, 2])
    cax = fig.add_subplot(gs[0:3, 3])

    im_main = ax_main.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    im_row = ax_row.imshow(row_avgs[:, None], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    im_col = ax_col.imshow(col_avgs[None, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    im_corner = ax_corner.imshow([[overall_avg]], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    def _annotate(ax, data: np.ndarray) -> None:
            # Clear any existing text objects to be safe
            for t in list(ax.texts):
                t.remove()

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    text = "NA" if np.isnan(val) else format(val, value_fmt)
                    
                    ax.text(
                        j, i, text,
                        ha="center", va="center",
                        fontsize=20,
                        color="black" # Forced black regardless of background
                    )

    _annotate(ax_main, mat)
    _annotate(ax_row, row_avgs[:, None])
    _annotate(ax_col, col_avgs[None, :])
    _annotate(ax_corner, np.array([[overall_avg]]))

    # Main heatmap axes
    ax_main.set_xticks(range(len(levels)))
    ax_main.set_xticklabels([]) # str(x) for x in levels
    ax_main.set_yticks(range(len(games)))
    ax_main.set_yticklabels(games)
    ax_main.set_xlabel("Level")
    ax_main.set_ylabel("Game")
    ax_main.set_title(title, pad=12)

    # Row averages
    ax_row.set_xticks([0])
    ax_row.set_xticklabels([]) # Avg
    ax_row.tick_params(axis="y", left=False, labelleft=False)
    ax_row.set_xlabel("Game Avg")

    # Column averages
    ax_col.set_xticks(range(len(levels)))
    ax_col.set_xticklabels([]) # str(x) for x in levels
    ax_col.set_yticks([0])
    ax_col.set_yticklabels([]) # Avg
    ax_col.set_ylabel("Level Avg")

    # Overall average
    ax_corner.set_xticks([0])
    ax_corner.set_xticklabels([]) # "Avg"
    ax_corner.set_yticks([0])
    ax_corner.set_yticklabels([]) # "Avg"
    ax_corner.set_xlabel("Overall")
    ax_corner.set_ylabel("")

    # Draw light gridlines so cells are clearly separated
    def _style_grid(ax, nrows: int, ncols: int) -> None:
        ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        # ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        # ax.tick_params(which="minor", bottom=False, left=False) #
        ax.set_xticklabels([], minor=True)      #
        ax.set_yticklabels([], minor=True)  #
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1) #
        ax.tick_params(which="both", bottom=False, left=False) #
    _style_grid(ax_main, mat.shape[0], mat.shape[1])
    _style_grid(ax_row, row_avgs.shape[0], 1)
    _style_grid(ax_col, 1, col_avgs.shape[0])
    _style_grid(ax_corner, 1, 1)

    # Colorbar after averages
    cbar = fig.colorbar(im_main, cax=cax)
    cbar.set_label(metric_key)
    for _ax in (ax_main, ax_row, ax_col, ax_corner):
        _force_black_axis_text(_ax)
    _force_black_colorbar_text(cbar)

    _finalize_figure(fig, out_path, annotate=False)


def plot_time_per_step_by_game_thinking(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "qwen_thinking_mode"])
    games = sorted({str(r["game"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(think_modes) <= 2 else max(0.18, 0.8 / max(len(think_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 5))
    for i, thinking in enumerate(think_modes):
        vals = []
        for game in games:
            match = [r for r in summary if str(r["game"]) == game and str(r["qwen_thinking_mode"]) == thinking]
            vals.append(_to_float(match[0]["mean_time_per_step_seconds"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(think_modes) - 1) / 2) * width, vals, width, label=f"thinking={thinking}")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylabel("Mean Time per Step (s)")
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_time_per_step_by_game_causal(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["game", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    x = np.arange(len(games))
    width = 0.35 if len(causal_modes) <= 2 else max(0.18, 0.8 / max(len(causal_modes), 1))

    fig, ax = plt.subplots(figsize=(max(8, len(games) * 1.3), 5))
    for i, causal in enumerate(causal_modes):
        vals = []
        for game in games:
            match = [r for r in summary if str(r["game"]) == game and str(r["causal_mode"]) == causal]
            vals.append(_to_float(match[0]["mean_time_per_step_seconds"], 0.0) if match else 0.0)
        ax.bar(x + (i - (len(causal_modes) - 1) / 2) * width, vals, width, label=f"causal={causal}")

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=25, ha="right")
    ax.set_ylabel("Mean Time per Step (s)")
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_time_per_step_by_model_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile", "planning_horizon_x"])
    models = sorted({str(r["model_profile"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for model in models:
        subset = sorted(
            [r for r in summary if str(r["model_profile"]) == model],
            key=lambda r: _to_int(r["planning_horizon_x"], 1),
        )
        xs = [_to_int(r["planning_horizon_x"], 1) for r in subset]
        ys = [_to_float(r["mean_time_per_step_seconds"], 0.0) for r in subset]
        ax.plot(xs, ys, marker="o", linewidth=2, label=model)

    ax.set_xlabel("Planning Horizon")
    ax.set_xticks([1, 5, 10])
    ax.set_ylabel("Mean Time per Step (s)")
    ax.grid(alpha=0.25)
    _finalize_figure(fig, out_path)


def plot_metric_by_mode_bar(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    labels = [
        f"T{r['qwen_thinking_mode']}-C{r['causal_mode']}-H{_to_int(r['planning_horizon_x'],1)}"
        for r in summary
    ]
    vals = [_to_float(r.get(metric_key, 0.0), 0.0) for r in summary]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 5))
    ax.bar(x, vals, color="tab:blue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    _finalize_figure(fig, out_path)


def plot_mean_time_per_step_lines(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["causal_mode", "qwen_thinking_mode", "planning_horizon_x"])
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})

    fig, axes = plt.subplots(1, len(think_modes), figsize=(max(10, 5 * len(think_modes)), 4.8), squeeze=False)
    for idx, thinking in enumerate(think_modes):
        ax = axes[0][idx]
        for causal in causal_modes:
            vals = []
            for h in horizons:
                row = next(
                    (
                        r
                        for r in summary
                        if str(r["causal_mode"]) == causal
                        and str(r["qwen_thinking_mode"]) == thinking
                        and _to_int(r["planning_horizon_x"], 1) == h
                    ),
                    None,
                )
                vals.append(_to_float(row.get("mean_time_per_step_seconds", 0.0), 0.0) if row else float("nan"))
            ax.plot(horizons, vals, marker="o", linewidth=2, label=f"causal={causal}")
        ax.set_xlabel("Planning Horizon")
        ax.set_xticks([1, 5, 10])
        ax.set_title(f"Mean Time/Step (thinking={thinking})")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        if idx == 0:
            ax.set_ylabel("Seconds per Step")

    _finalize_figure(fig, out_path)


def plot_planning_activity_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["qwen_thinking_mode", "planning_horizon_x"])
    think_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for t in think_modes:
        vals_queries = []
        vals_queueexec = []
        for h in horizons:
            row = next(
                (
                    r
                    for r in summary
                    if str(r["qwen_thinking_mode"]) == t
                    and _to_int(r["planning_horizon_x"], 1) == h
                ),
                None,
            )
            vals_queries.append(_to_float(row.get("mean_plan_queries_count", 0.0), 0.0) if row else 0.0)
            vals_queueexec.append(_to_float(row.get("mean_queued_actions_executed", 0.0), 0.0) if row else 0.0)
        axes[0].plot(horizons, vals_queries, marker="o", label=f"thinking={t}")
        axes[1].plot(horizons, vals_queueexec, marker="o", label=f"thinking={t}")

    axes[0].set_title("Mean Plan Queries by Horizon")
    axes[0].set_ylabel("Plan Queries")
    axes[1].set_title("Mean Queued Actions Executed by Horizon")
    axes[1].set_ylabel("Queued Actions")
    for ax in axes:
        ax.set_xlabel("Planning Horizon")
        ax.set_xticks([1, 5, 10])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_scatter_reward_tokens_by_game(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    games = sorted({str(r["game"]) for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(games), 1)))
    cmap = {g: colors[i % len(colors)] for i, g in enumerate(games)}

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for g in games:
        subset = [r for r in rows if str(r["game"]) == g]
        xs = [_to_float(r["sum_total_tokens"], 0.0) for r in subset]
        ys = [_to_float(r["normalized_total_reward"], 0.0) for r in subset]
        ax.scatter(xs, ys, alpha=0.55, s=28, label=g, color=cmap[g])
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Normalized Reward")
    ax.set_title("Normalized Reward vs Tokens by Game")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_scatter_steps_runtime_by_winner(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    winner_groups = ["PLAYER_WINS", "PLAYER_LOSES", "NO_WINNER", "UNKNOWN"]
    colors = {
        "PLAYER_WINS": "tab:green",
        "PLAYER_LOSES": "tab:red",
        "NO_WINNER": "tab:orange",
        "UNKNOWN": "tab:gray",
    }
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for w in winner_groups:
        subset = [r for r in rows if str(r.get("winner", "UNKNOWN")) == w]
        if not subset:
            continue
        xs = [_to_float(r["steps"], 0.0) for r in subset]
        ys = [_to_float(r["runtime_seconds"], 0.0) for r in subset]
        ax.scatter(xs, ys, alpha=0.55, s=28, label=w, color=colors[w])
    ax.set_xlabel("Steps")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs Steps by Winner Label")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_pareto_winrate_runtime_by_config(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    points, pareto_ids = build_pareto_points(rows)
    if not points:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    pareto_points = sorted(
        [p for p in points if _to_int(p["id"], 0) in pareto_ids],
        key=lambda p: (_to_float(p["runtime_per_step"], 0.0), -_to_float(p["win_rate"], 0.0)),
    )

    fig, ax = plt.subplots(figsize=(16, 9))
    text_objects = []
    for p in points:
        t = ax.text(
            _to_float(p["runtime_per_step"], 0.0),
            _to_float(p["win_rate"], 0.0),
            str(_to_int(p["id"], 0)),
            fontsize=10,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85, "pad": 0.15},
        )
        text_objects.append(t)

    if pareto_points:
        xs = [_to_float(p["runtime_per_step"], 0.0) for p in pareto_points]
        ys = [_to_float(p["win_rate"], 0.0) for p in pareto_points]
        ax.plot(xs, ys, color="crimson", linewidth=2.0, linestyle="-", label="Pareto Front")
        ax.scatter(
            xs,
            ys,
            s=110,
            facecolors="none",
            edgecolors="crimson",
            linewidths=2.0,
            marker="o",
            label="Pareto Config",
            zorder=3,
        )

    adjust_text(
        text_objects,
        ax=ax,
        only_move={"points": "xy", "text": "xy"},
        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6, alpha=0.7),
    )

    ax.set_xlabel("Mean Runtime per Step (s)")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)

    # Left legend for Pareto styling.
    style_legend = ax.legend(loc="upper left", frameon=False)
    ax.add_artist(style_legend)

    # Right-side legend mapping point number -> full config.
    mapping_handles = []
    for p in points:
        cfg = (
            f"{_to_int(p['id'], 0)}: {p['model']} | "
            f"T={p['thinking']} C={p['causal']} H={_to_int(p['horizon'], 1)}"
        )
        if _to_int(p["id"], 0) in pareto_ids:
            cfg = f"* {cfg}"
        mapping_handles.append(
            Line2D([0], [0], linestyle="None", marker=None, color="none", label=cfg)
        )
    ax.legend(
        handles=mapping_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title="Config IDs (* = Pareto)",
        title_fontsize=10,
        ncol=2,
    )
    _finalize_figure(fig, out_path, annotate=False)


def build_pareto_points(rows: Sequence[Dict[str, object]]) -> Tuple[List[Dict[str, object]], set[int]]:
    summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    configs = sorted(
        summary,
        key=lambda r: (
            str(r["model_profile"]),
            str(r["qwen_thinking_mode"]),
            str(r["causal_mode"]),
            _to_int(r["planning_horizon_x"], 1),
        ),
    )
    points: List[Dict[str, object]] = []
    for idx, r in enumerate(configs, start=1):
        points.append(
            {
                "id": idx,
                "model": str(r["model_profile"]),
                "thinking": str(r["qwen_thinking_mode"]),
                "causal": str(r["causal_mode"]),
                "horizon": _to_int(r["planning_horizon_x"], 1),
                "runtime_per_step": _to_float(r.get("mean_time_per_step_seconds", 0.0), 0.0),
                "win_rate": _to_float(r.get("win_rate", 0.0), 0.0),
            }
        )

    pareto_ids: set[int] = set()
    for p in points:
        dominated = False
        for q in points:
            if q["id"] == p["id"]:
                continue
            better_or_equal = _to_float(q["runtime_per_step"], 0.0) <= _to_float(p["runtime_per_step"], 0.0) and _to_float(q["win_rate"], 0.0) >= _to_float(p["win_rate"], 0.0)
            strictly_better = _to_float(q["runtime_per_step"], 0.0) < _to_float(p["runtime_per_step"], 0.0) or _to_float(q["win_rate"], 0.0) > _to_float(p["win_rate"], 0.0)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto_ids.add(_to_int(p["id"], 0))
    return points, pareto_ids


def write_pareto_key_csv(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    points, pareto_ids = build_pareto_points(rows)
    key_rows: List[Dict[str, object]] = []
    for p in points:
        key_rows.append(
            {
                "id": _to_int(p["id"], 0),
                "is_pareto": 1 if _to_int(p["id"], 0) in pareto_ids else 0,
                "model_profile": str(p["model"]),
                "qwen_thinking_mode": str(p["thinking"]),
                "causal_mode": str(p["causal"]),
                "planning_horizon_x": _to_int(p["horizon"], 1),
                "win_rate": round(_to_float(p["win_rate"], 0.0), 6),
                "mean_runtime_per_step_seconds": round(_to_float(p["runtime_per_step"], 0.0), 6),
            }
        )
    write_csv(out_path, key_rows)


def plot_pareto_key_table(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    points, pareto_ids = build_pareto_points(rows)
    points = sorted(points, key=lambda p: _to_int(p["id"], 0))
    headers = ["ID", "Pareto", "Model", "Thinking", "Causal", "Horizon", "WinRate", "Runtime/Step(s)"]
    table_rows: List[List[str]] = []
    for p in points:
        table_rows.append(
            [
                str(_to_int(p["id"], 0)),
                "*" if _to_int(p["id"], 0) in pareto_ids else "",
                str(p["model"]),
                str(p["thinking"]),
                str(p["causal"]),
                str(_to_int(p["horizon"], 1)),
                f"{_to_float(p['win_rate'], 0.0):.3f}",
                f"{_to_float(p['runtime_per_step'], 0.0):.3f}",
            ]
        )
    fig_h = max(6.0, 0.34 * (len(table_rows) + 1))
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.1)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8eef9")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fb")
    _finalize_figure(fig, out_path, annotate=False)


def plot_correlation_heatmap(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    metric_keys = [
        "is_win",
        "steps",
        "total_reward",
        "runtime_seconds",
        "sum_total_tokens",
        "meaningful_step_ratio",
        "effective_step_ratio",
        "llm_action_parse_rate",
        "action_execution_match_rate",
        "plan_queries_count",
        "queued_actions_executed",
        "reward_per_1k_tokens",
        "steps_per_second",
    ]
    labels = [
        "Win",
        "Steps",
        "Reward",
        "Runtime",
        "Tokens",
        "Meaningful",
        "Effective",
        "ParseRate",
        "ExecMatch",
        "PlanQueries",
        "QueuedExec",
        "Reward/1kTok",
        "Steps/Sec",
    ]
    X = np.array([[float(_to_float(r.get(k, 0.0), 0.0)) for k in metric_keys] for r in rows], dtype=float)
    if X.shape[0] < 2:
        corr = np.zeros((len(metric_keys), len(metric_keys)), dtype=float)
    else:
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="YlGn")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Correlation Heatmap (Run-Level Metrics)")
    cbar = fig.colorbar(im, ax=ax)
    _force_black_axis_text(ax)
    _force_black_colorbar_text(cbar)
    _finalize_figure(fig, out_path)


def plot_cdf_by_game(rows: Sequence[Dict[str, object]], out_path: Path, value_key: str, title: str, xlabel: str) -> None:
    games = sorted({str(r["game"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for g in games:
        vals = sorted(_to_float(r.get(value_key, 0.0), 0.0) for r in rows if str(r["game"]) == g)
        if not vals:
            continue
        y = np.arange(1, len(vals) + 1) / float(len(vals))
        ax.plot(vals, y, label=g)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _finalize_figure(fig, out_path)


def plot_per_game_horizon_curves(rows: Sequence[Dict[str, object]], plots_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    summary = aggregate_means(rows, ["game", "qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    games = sorted({str(r["game"]) for r in summary})
    for game in games:
        subset = [r for r in summary if str(r["game"]) == game]
        causal_modes = sorted({str(r["causal_mode"]) for r in subset})
        think_modes = sorted({str(r["qwen_thinking_mode"]) for r in subset})
        horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in subset})

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for c in causal_modes:
            for t in think_modes:
                vals_win = []
                vals_reward = []
                for h in horizons:
                    row = next(
                        (
                            r
                            for r in subset
                            if str(r["causal_mode"]) == c
                            and str(r["qwen_thinking_mode"]) == t
                            and _to_int(r["planning_horizon_x"], 1) == h
                        ),
                        None,
                    )
                    vals_win.append(_to_float(row.get("win_rate", 0.0), 0.0) if row else 0.0)
                    vals_reward.append(_to_float(row.get("mean_normalized_total_reward", 0.0), 0.0) if row else 0.0)
                label = f"causal={c},thinking={t}"
                axes[0].plot(horizons, vals_win, marker="o", label=label)
                axes[1].plot(horizons, vals_reward, marker="o", label=label)
        axes[0].set_title(f"{game}: Win Rate vs Horizon")
        axes[1].set_title(f"{game}: Normalized Reward vs Horizon")
        axes[0].set_ylabel("Win Rate")
        axes[0].set_ylim(0, 1)
        axes[1].set_ylabel("Mean Normalized Reward")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.set_xlabel("Planning Horizon")
            ax.set_xticks([1, 5, 10])
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fname = f"per_game_horizon_{_safe_slug(game)}.png"
        fpath = plots_dir / fname
        _finalize_figure(fig, fpath)
        out[f"Per-Game Horizon Curves ({game})"] = fpath
    return out


def plot_per_game_level_profiles(rows: Sequence[Dict[str, object]], plots_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    summary = aggregate_means(rows, ["game", "level", "qwen_thinking_mode", "causal_mode"])
    games = sorted({str(r["game"]) for r in summary})
    for game in games:
        subset = [r for r in summary if str(r["game"]) == game]
        levels = sorted({_to_int(r["level"], 0) for r in subset})
        combos = sorted({(str(r["qwen_thinking_mode"]), str(r["causal_mode"])) for r in subset})
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for t, c in combos:
            win_vals = []
            rew_vals = []
            for lvl in levels:
                row = next(
                    (
                        r
                        for r in subset
                        if _to_int(r["level"], 0) == lvl
                        and str(r["qwen_thinking_mode"]) == t
                        and str(r["causal_mode"]) == c
                    ),
                    None,
                )
                win_vals.append(_to_float(row.get("win_rate", 0.0), 0.0) if row else 0.0)
                rew_vals.append(_to_float(row.get("mean_normalized_total_reward", 0.0), 0.0) if row else 0.0)
            label = f"T{t}-C{c}"
            axes[0].plot(levels, win_vals, marker="o", label=label)
            axes[1].plot(levels, rew_vals, marker="o", label=label)
        axes[0].set_title(f"{game}: Win Rate vs Level")
        axes[1].set_title(f"{game}: Normalized Reward vs Level")
        axes[0].set_ylabel("Win Rate")
        axes[0].set_ylim(0, 1)
        axes[1].set_ylabel("Mean Normalized Reward")
        axes[1].set_ylim(0, 1.05)
        for ax in axes:
            ax.set_xlabel("Level")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fname = f"per_game_level_{_safe_slug(game)}.png"
        fpath = plots_dir / fname
        _finalize_figure(fig, fpath)
        out[f"Per-Game Level Profiles ({game})"] = fpath
    return out


def plot_per_game_model_profiles(rows: Sequence[Dict[str, object]], plots_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    summary = aggregate_means(rows, ["game", "level", "model_profile"])
    games = sorted({str(r["game"]) for r in summary})
    if not games:
        return out

    # Keep known spatial game ordering when present.
    preferred_game_order = ["spatialgame1", "spatialgame2", "spatialgame3"]
    ordered_games = [g for g in preferred_game_order if g in games] + [g for g in games if g not in preferred_game_order]
    models = sorted({str(r["model_profile"]) for r in summary})

    fig, axes = plt.subplots(
        len(ordered_games),
        2,
        figsize=(12, max(3.2 * len(ordered_games), 8.5)),
        sharex="col",
        sharey="col",
        squeeze=False,
    )

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for row_idx, game in enumerate(ordered_games):
        subset = [r for r in summary if str(r["game"]) == game]
        levels = sorted({_to_int(r["level"], 0) for r in subset})
        ax_win = axes[row_idx][0]
        ax_rew = axes[row_idx][1]

        for model in models:
            win_vals = []
            rew_vals = []
            for lvl in levels:
                row = next(
                    (
                        r
                        for r in subset
                        if _to_int(r["level"], 0) == lvl and str(r["model_profile"]) == model
                    ),
                    None,
                )
                win_vals.append(_to_float(row.get("win_rate", 0.0), 0.0) if row else float("nan"))
                rew_vals.append(_to_float(row.get("mean_normalized_total_reward", 0.0), 0.0) if row else float("nan"))

            line_win = ax_win.plot(levels, win_vals, marker="o", linewidth=2)[0]
            ax_rew.plot(levels, rew_vals, marker="o", linewidth=2, color=line_win.get_color())

            if model not in legend_labels:
                legend_handles.append(Line2D([0], [0], color=line_win.get_color(), marker="o", linewidth=2))
                legend_labels.append(model)

        ax_win.set_ylim(0, 1.0)
        ax_rew.set_ylim(0, 1.05)
        ax_win.grid(alpha=0.25)
        ax_rew.grid(alpha=0.25)
        ax_rew.set_ylabel("Mean Normalized\nReward")

        ax_win.text(
            -0.22,
            0.5,
            game,
            transform=ax_win.transAxes,
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        ax_win.set_ylabel("Win Rate")

    axes[0][0].set_title("Win Rate vs Level")
    axes[0][1].set_title("Normalized Reward vs Level")
    axes[-1][0].set_xlabel("Level")
    axes[-1][1].set_xlabel("Level")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(4, max(1, len(legend_labels))),
        frameon=False,
        title="Model",
    )

    fpath = plots_dir / "per_game_model_combined.pdf"
    plt.tight_layout(rect=[0.04, 0.08, 1, 1])
    _finalize_figure(fig, fpath, annotate=False)
    out["Per-Game Model Profiles (Combined)"] = fpath
    return out


# def plot_per_model_win_rate_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
#     summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
#     if not summary:
#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.text(0.5, 0.5, "No data", ha="center", va="center")
#         ax.set_xticks([])
#         ax.set_yticks([])
#         _finalize_figure(fig, out_path, annotate=False)
#         return

#     ordered_model_fragments = ["8b", "4b", "1.7b", "0.6b"]
#     available_models = sorted({str(r["model_profile"]) for r in summary})
#     model_names: List[str] = []
#     for frag in ordered_model_fragments:
#         for model in available_models:
#             if frag in model.lower() and model not in model_names:
#                 model_names.append(model)
#                 break
#     # Fill remaining slots deterministically if any expected model is absent.
#     for model in available_models:
#         if model not in model_names:
#             model_names.append(model)
#         if len(model_names) >= 4:
#             break
#     while len(model_names) < 4:
#         model_names.append(f"no_data_{len(model_names)+1}")

#     thinking_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
#     causal_modes = sorted({str(r["causal_mode"]) for r in summary})
#     horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary if _to_int(r["planning_horizon_x"], 1) in {1, 5, 10}})
#     if not horizons:
#         horizons = [1, 5, 10]

#     style_by_thinking = {"on": "-", "off": ":"}
#     color_by_causal = {"off": "tab:blue", "on": "tab:orange"}

#     fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True, sharey=True)
#     for ax, model in zip(axes, model_names):
#         subset = [r for r in summary if str(r["model_profile"]) == model]
#         if not subset:
#             ax.text(0.5, 0.5, f"No data: {model}", transform=ax.transAxes, ha="center", va="center")
#             ax.set_ylim(0, 1)
#             ax.grid(alpha=0.25)
#             continue
#         series = [(causal, thinking) for causal in causal_modes for thinking in thinking_modes]
#         series_offsets = np.linspace(-0.03, 0.03, num=max(len(series), 1))
#         for series_idx, (causal, thinking) in enumerate(series):
#                 vals = []
#                 for h in horizons:
#                     row = next(
#                         (
#                             r
#                             for r in subset
#                             if str(r["causal_mode"]) == causal
#                             and str(r["qwen_thinking_mode"]) == thinking
#                             and _to_int(r["planning_horizon_x"], 1) == h
#                         ),
#                         None,
#                     )
#                     vals.append(_to_float(row.get("win_rate", 0.0), 0.0) if row else float("nan"))
#                 ax.plot(
#                     horizons,
#                     vals,
#                     marker="o",
#                     linewidth=2,
#                     linestyle=style_by_thinking.get(thinking, "-"),
#                     color=color_by_causal.get(causal, "tab:gray"),
#                     label="_nolegend_",
#                 )
#                 # Add readable value labels with deterministic per-series vertical offset.
#                 y_offset = float(series_offsets[series_idx]) if len(series_offsets) else 0.0
#                 for x, y in zip(horizons, vals):
#                     if not np.isfinite(y):
#                         continue
#                     y_text = min(0.99, max(0.01, float(y) + y_offset))
#                     ax.text(
#                         float(x),
#                         y_text,
#                         f"{float(y):.2f}",
#                         color=color_by_causal.get(causal, "tab:gray"),
#                         fontsize=8,
#                         ha="center",
#                         va="center",
#                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
#                     )
#         ax.set_ylabel("Win Rate")
#         ax.set_ylim(0, 1)
#         ax.grid(alpha=0.25)
#         ax.text(0.01, 0.98, model, transform=ax.transAxes, ha="left", va="top", fontsize=11)

#     axes[-1].set_xlabel("Planning Horizon")
#     axes[-1].set_xticks([1, 5, 10])
#     axes[-1].set_xlim(0.7, 10.3)
#     # Single global legend area:
#     # - Thinking legend encodes line style (solid/dotted)
#     # - Causal legend encodes color
#     thinking_handles = [
#         Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="thinking=on"),
#         Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="thinking=off"),
#     ]
#     causal_handles = [
#         Line2D([0], [0], color=color_by_causal.get("on", "tab:orange"), linestyle="-", linewidth=2, label="causal=on"),
#         Line2D([0], [0], color=color_by_causal.get("off", "tab:blue"), linestyle="-", linewidth=2, label="causal=off"),
#     ]
#     legend_thinking = fig.legend(
#         handles=thinking_handles,
#         loc="upper center",
#         bbox_to_anchor=(0.28, 0.02),
#         ncol=1,
#         frameon=False,
#         title="Thinking",
#     )
#     fig.add_artist(legend_thinking)
#     fig.legend(
#         handles=causal_handles,
#         loc="upper center",
#         bbox_to_anchor=(0.72, 0.02),
#         ncol=1,
#         frameon=False,
#         title="Causal",
#     )
#     _finalize_figure(fig, out_path, annotate=False)

def _ordered_model_names_from_summary(summary: Sequence[Dict[str, object]]) -> List[str]:
    ordered_model_fragments = ["8b", "4b", "1.7b", "0.6b"]
    available_models = sorted(
        {str(r["model_profile"]).replace("local-", "").replace("-vllm", "") for r in summary}
    )
    model_names: List[str] = []
    for frag in ordered_model_fragments:
        for model in available_models:
            if frag in model.lower() and model not in model_names:
                model_names.append(model)
                break
    for model in available_models:
        if model not in model_names:
            model_names.append(model)
        if len(model_names) >= 4:
            break
    while len(model_names) < 4:
        model_names.append(f"no_data_{len(model_names)+1}")
    return model_names


def _plot_per_model_metric_by_horizon(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.2f}",
    y_ticks: Optional[Sequence[float]] = None,
) -> None:
    summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    
    if not summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    model_names = _ordered_model_names_from_summary(summary)

    thinking_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary if _to_int(r["planning_horizon_x"], 1) in {1, 5, 10}})
    if not horizons:
        horizons = [1, 5, 10]

    style_by_thinking = {"on": "-", "off": ":"}
    color_by_causal = {"off": "tab:blue", "on": "tab:orange"}
    is_completion_metric = metric_key == "completion_rate"
    label_fontsize = 18
    force_points = 1.4 if is_completion_metric else 0.8
    force_text = 3.0 if is_completion_metric else 1.2
    expand_points = (2.8, 2.8) if is_completion_metric else (2.0, 2.0)
    expand_text = (2.2, 2.6) if is_completion_metric else (1.4, 1.6)

    def _reposition_texts_within_axes(fig: plt.Figure, ax: plt.Axes, texts: Sequence[plt.Text], pad_px: float = 6.0) -> None:
        """Gently nudge labels inside axes bounds in display-space, preserving relative placement."""
        if not texts:
            return
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer=renderer)
        x0 = ax_bbox.x0 + pad_px
        y0 = ax_bbox.y0 + pad_px
        x1 = ax_bbox.x1 - pad_px
        y1 = ax_bbox.y1 - pad_px
        inv = ax.transData.inverted()

        for t in texts:
            tb = t.get_window_extent(renderer=renderer)
            dx = 0.0
            dy = 0.0
            if tb.x0 < x0:
                dx = x0 - tb.x0
            elif tb.x1 > x1:
                dx = x1 - tb.x1
            if tb.y0 < y0:
                dy = y0 - tb.y0
            elif tb.y1 > y1:
                dy = y1 - tb.y1
            if dx == 0.0 and dy == 0.0:
                continue
            cx = (tb.x0 + tb.x1) * 0.5
            cy = (tb.y0 + tb.y1) * 0.5
            new_cx = cx + dx
            new_cy = cy + dy
            new_x, new_y = inv.transform((new_cx, new_cy))
            t.set_position((float(new_x), float(new_y)))

    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True, sharey=True)
    
    for ax, model in zip(axes, model_names):
        # Filter subset based on sanitized model name comparison
        subset = [
            r for r in summary 
            if str(r["model_profile"]).replace("local-", "").replace("-vllm", "") == model
        ]
        
        if not subset:
            ax.text(0.5, 0.5, f"No data: {model}", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            continue

        series = [(causal, thinking) for causal in causal_modes for thinking in thinking_modes]
        
        # List to collect text objects for adjust_text
        subplot_texts = []
        point_xs: List[float] = []
        point_ys: List[float] = []

        for series_idx, (causal, thinking) in enumerate(series):
            vals = []
            for h in horizons:
                row = next(
                    (r for r in subset 
                     if str(r["causal_mode"]) == causal 
                     and str(r["qwen_thinking_mode"]) == thinking 
                     and _to_int(r["planning_horizon_x"], 1) == h),
                    None,
                )
                vals.append(_to_float(row.get(metric_key, 0.0), 0.0) if row else float("nan"))
            
            # Plot the line and markers
            ax.plot(
                horizons,
                vals,
                marker="o",
                linewidth=2.5,
                linestyle=style_by_thinking.get(thinking, "-"),
                color=color_by_causal.get(causal, "tab:gray"),
                label="_nolegend_",
            )

            # Create text labels
            for x, y in zip(horizons, vals):
                if not np.isfinite(y):
                    continue
                point_xs.append(float(x))
                point_ys.append(float(y))
                
                # Plot text with increased fontsize (10)
                t = ax.text(
                    float(x),
                    float(y) + 0.01,
                    value_fmt.format(float(y)),
                    color=color_by_causal.get(causal, "tab:gray"),
                    fontsize=label_fontsize,
                    fontweight='bold',
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.3},
                    clip_on=False,
                )
                subplot_texts.append(t)

        # Apply adjust_text per subplot to resolve overlaps
        if subplot_texts:
            adjust_text(
                subplot_texts,
                x=point_xs,
                y=point_ys,
                ax=ax,
                only_move={'points': 'xy', 'text': 'xy'},
                force_points=force_points,
                force_text=force_text,
                expand_points=expand_points,
                expand_text=expand_text,
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3)
            )
            _reposition_texts_within_axes(fig, ax, subplot_texts, pad_px=8.0)

        ax.set_ylabel(ylabel, fontsize=16)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if y_ticks is not None:
            ax.set_yticks(list(y_ticks))
        ax.grid(alpha=0.25)
        ax.text(
            0.5,
            1.06,
            model,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    axes[-1].set_xlabel("Planning Horizon", fontsize=16)
    axes[-1].set_xticks([1, 5, 10])
    axes[-1].set_xlim(0.5, 10.5)

    # Global Legends
    thinking_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="thinking=on"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="thinking=off"),
    ]
    causal_handles = [
        Line2D([0], [0], color=color_by_causal.get("on", "tab:orange"), linestyle="-", linewidth=2, label="causal=on"),
        Line2D([0], [0], color=color_by_causal.get("off", "tab:blue"), linestyle="-", linewidth=2, label="causal=off"),
    ]
    
    legend_thinking = fig.legend(
        handles=thinking_handles,
        loc="upper center",
        bbox_to_anchor=(0.28, 0.03),
        ncol=1,
        frameon=False,
        title="Thinking",
    )
    fig.add_artist(legend_thinking)
    
    fig.legend(
        handles=causal_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.03),
        ncol=1,
        frameon=False,
        title="Causal",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Make room for the legends at the bottom
    _finalize_figure(fig, out_path, annotate=False)


def plot_per_model_win_rate_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    _plot_per_model_metric_by_horizon(
        rows=rows,
        out_path=out_path,
        metric_key="win_rate",
        ylabel="Win Rate",
        ylim=(-0.05, 1.05),
        value_fmt="{:.2f}",
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


def plot_per_model_completion_rate_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    _plot_per_model_metric_by_horizon(
        rows=rows,
        out_path=out_path,
        metric_key="completion_rate",
        ylabel="Completion Rate",
        ylim=(-0.05, 1.05),
        value_fmt="{:.2f}",
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


def plot_per_model_mean_time_per_step_by_horizon(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "planning_horizon_x"])
    if not summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    model_names = _ordered_model_names_from_summary(summary)
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    horizons = sorted({_to_int(r["planning_horizon_x"], 1) for r in summary if _to_int(r["planning_horizon_x"], 1) in {1, 5, 10}})
    if not horizons:
        horizons = [1, 5, 10]
    style_by_thinking = {"on": "-", "off": ":"}
    color_by_causal = {"off": "tab:blue", "on": "tab:orange"}

    fig, axes = plt.subplots(8, 1, figsize=(10, 13.5), sharex=True)
    for idx, model in enumerate(model_names):
        ax_top = axes[2 * idx]
        ax_bot = axes[2 * idx + 1]
        subset = [
            r
            for r in summary
            if str(r["model_profile"]).replace("local-", "").replace("-vllm", "") == model
        ]

        top_texts: List[plt.Text] = []
        bot_texts: List[plt.Text] = []
        top_xs: List[float] = []
        top_ys: List[float] = []
        bot_xs: List[float] = []
        bot_ys: List[float] = []
        for thinking, ax, text_bucket in [("on", ax_top, top_texts), ("off", ax_bot, bot_texts)]:
            for causal in causal_modes:
                vals = []
                for h in horizons:
                    row = next(
                        (
                            r
                            for r in subset
                            if str(r["qwen_thinking_mode"]) == thinking
                            and str(r["causal_mode"]) == causal
                            and _to_int(r["planning_horizon_x"], 1) == h
                        ),
                        None,
                    )
                    vals.append(_to_float(row.get("mean_time_per_step_seconds", 0.0), 0.0) if row else float("nan"))
                ax.plot(
                    horizons,
                    vals,
                    marker="o",
                    linewidth=2.5,
                    linestyle=style_by_thinking.get(thinking, "-"),
                    color=color_by_causal.get(causal, "tab:gray"),
                    label="_nolegend_",
                )
                for x, y in zip(horizons, vals):
                    if not np.isfinite(y):
                        continue
                    if thinking == "on":
                        top_xs.append(float(x))
                        top_ys.append(float(y))
                    else:
                        bot_xs.append(float(x))
                        bot_ys.append(float(y))
                    t = ax.text(
                        float(x),
                        float(y) + 0.01,
                        f"{float(y):.2f}",
                        color=color_by_causal.get(causal, "tab:gray"),
                        fontsize=16,
                        ha="center",
                        va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.25},
                        clip_on=False,
                    )
                    text_bucket.append(t)
            ax.grid(alpha=0.25)

        if top_texts:
            adjust_text(
                top_texts,
                x=top_xs,
                y=top_ys,
                ax=ax_top,
                only_move={"points": "xy", "text": "xy"},
                force_points=0.8,
                force_text=1.2,
                expand_points=(2.0, 2.0),
                expand_text=(1.4, 1.6),
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
            )
        if bot_texts:
            adjust_text(
                bot_texts,
                x=bot_xs,
                y=bot_ys,
                ax=ax_bot,
                only_move={"points": "xy", "text": "xy"},
                force_points=0.8,
                force_text=1.2,
                expand_points=(2.0, 2.0),
                expand_text=(1.4, 1.6),
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
            )

        # Simulated axis break: top pane for thinking=on, bottom pane for thinking=off
        top_vals = [
            _to_float(r.get("mean_time_per_step_seconds", 0.0), 0.0)
            for r in subset
            if str(r.get("qwen_thinking_mode", "")) == "on"
        ]
        bot_vals = [
            _to_float(r.get("mean_time_per_step_seconds", 0.0), 0.0)
            for r in subset
            if str(r.get("qwen_thinking_mode", "")) == "off"
        ]
        if top_vals:
            tmin, tmax = min(top_vals), max(top_vals)
            pad = max((tmax - tmin) * 0.2, 1e-4)
            ax_top.set_ylim(max(0.0, tmin - pad), tmax + pad)
        if bot_vals:
            bmin, bmax = min(bot_vals), max(bot_vals)
            pad = max((bmax - bmin) * 0.2, 1e-4)
            ax_bot.set_ylim(max(0.0, bmin - pad), bmax + pad)

        _reposition_texts_within_axes(fig, ax_top, top_texts, pad_px=8.0)
        _reposition_texts_within_axes(fig, ax_bot, bot_texts, pad_px=8.0)

        ax_top.set_ylabel("Mean Time/Step (s)", fontsize=16, labelpad=40)
        ax_top.yaxis.set_label_coords(-0.09, -0.5)
        ax_bot.set_ylabel("")
        ax_top.text(
            0.5,
            1.08,
            model,
            transform=ax_top.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        # Draw break marks between top/bottom sub-axes.
        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes[-1].set_xlabel("Planning Horizon", fontsize=16)
    axes[-1].set_xticks([1, 5, 10])
    axes[-1].set_xlim(0.5, 10.5)

    thinking_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="thinking=on"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="thinking=off"),
    ]
    causal_handles = [
        Line2D([0], [0], color=color_by_causal.get("on", "tab:orange"), linestyle="-", linewidth=2, label="causal=on"),
        Line2D([0], [0], color=color_by_causal.get("off", "tab:blue"), linestyle="-", linewidth=2, label="causal=off"),
    ]
    legend_thinking = fig.legend(
        handles=thinking_handles,
        loc="upper center",
        bbox_to_anchor=(0.28, 0.02),
        ncol=1,
        frameon=False,
        title="Thinking",
    )
    fig.add_artist(legend_thinking)
    fig.legend(
        handles=causal_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.02),
        ncol=1,
        frameon=False,
        title="Causal",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _finalize_figure(fig, out_path, annotate=False)


def _plot_per_model_metric_by_game(
    rows: Sequence[Dict[str, object]],
    out_path: Path,
    metric_key: str,
    ylabel: str,
    ylim: Optional[Tuple[float, float]] = None,
    value_fmt: str = "{:.2f}",
    y_ticks: Optional[Sequence[float]] = None,
) -> None:
    summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "game"])

    if not summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    model_names = _ordered_model_names_from_summary(summary)

    thinking_modes = sorted({str(r["qwen_thinking_mode"]) for r in summary})
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    games = ["spatialgame1", "spatialgame2", "spatialgame3"]
    x = np.arange(len(games))

    style_by_thinking = {"on": "-", "off": ":"}
    color_by_causal = {"off": "tab:blue", "on": "tab:orange"}
    is_completion_metric = metric_key == "completion_rate"
    label_fontsize = 14
    force_points = 1.4 if is_completion_metric else 0.8
    force_text = 3.0 if is_completion_metric else 1.2
    expand_points = (2.8, 2.8) if is_completion_metric else (2.0, 2.0)
    expand_text = (2.2, 2.6) if is_completion_metric else (1.4, 1.6)

    fig, axes = plt.subplots(4, 1, figsize=(10, 11.5), sharex=True, sharey=True)

    for ax, model in zip(axes, model_names):
        subset = [
            r for r in summary
            if str(r["model_profile"]).replace("local-", "").replace("-vllm", "") == model
        ]

        if not subset:
            ax.text(0.5, 0.5, f"No data: {model}", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            continue

        series = [(causal, thinking) for causal in causal_modes for thinking in thinking_modes]

        subplot_texts: List[plt.Text] = []
        point_xs: List[float] = []
        point_ys: List[float] = []
        for series_idx, (causal, thinking) in enumerate(series):
            vals = []
            for game in games:
                row = next(
                    (r for r in subset
                     if str(r["causal_mode"]) == causal
                     and str(r["qwen_thinking_mode"]) == thinking
                     and str(r["game"]) == game),
                    None,
                )
                vals.append(_to_float(row.get(metric_key, 0.0), 0.0) if row else float("nan"))

            ax.plot(
                x,
                vals,
                marker="o",
                linewidth=2.5,
                linestyle=style_by_thinking.get(thinking, "-"),
                color=color_by_causal.get(causal, "tab:gray"),
                label="_nolegend_",
            )

            for xi, y in zip(x, vals):
                if not np.isfinite(y):
                    continue
                point_xs.append(float(xi))
                point_ys.append(float(y))
                t = ax.text(
                    float(xi),
                    float(y) + 0.01,
                    value_fmt.format(float(y)),
                    color=color_by_causal.get(causal, "tab:gray"),
                    fontsize=label_fontsize,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.3},
                    clip_on=False,
                )
                subplot_texts.append(t)

        if subplot_texts:
            adjust_text(
                subplot_texts,
                x=point_xs,
                y=point_ys,
                ax=ax,
                only_move={'points': 'xy', 'text': 'xy'},
                force_points=force_points,
                force_text=force_text,
                expand_points=expand_points,
                expand_text=expand_text,
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3)
            )
            _reposition_texts_within_axes(fig, ax, subplot_texts, pad_px=8.0)

        ax.set_ylabel(ylabel, fontsize=16)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if y_ticks is not None:
            ax.set_yticks(list(y_ticks))
        ax.grid(alpha=0.25)
        ax.text(
            0.5,
            1.06,
            model,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    axes[-1].set_xlabel("Game", fontsize=12)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(games)
    axes[-1].set_xlim(-0.5, len(games) - 0.5)

    thinking_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="thinking=on"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="thinking=off"),
    ]
    causal_handles = [
        Line2D([0], [0], color=color_by_causal.get("on", "tab:orange"), linestyle="-", linewidth=2, label="causal=on"),
        Line2D([0], [0], color=color_by_causal.get("off", "tab:blue"), linestyle="-", linewidth=2, label="causal=off"),
    ]

    legend_thinking = fig.legend(
        handles=thinking_handles,
        loc="upper center",
        bbox_to_anchor=(0.28, 0.03),
        ncol=1,
        frameon=False,
        title="Thinking",
    )
    fig.add_artist(legend_thinking)

    fig.legend(
        handles=causal_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.03),
        ncol=1,
        frameon=False,
        title="Causal",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _finalize_figure(fig, out_path, annotate=False)


def plot_per_model_win_rate_by_game(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    _plot_per_model_metric_by_game(
        rows=rows,
        out_path=out_path,
        metric_key="win_rate",
        ylabel="Win Rate",
        ylim=(-0.05, 1.05),
        value_fmt="{:.2f}",
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


def plot_per_model_completion_rate_by_game(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    _plot_per_model_metric_by_game(
        rows=rows,
        out_path=out_path,
        metric_key="completion_rate",
        ylabel="Completion Rate",
        ylim=(-0.05, 1.05),
        value_fmt="{:.2f}",
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


def plot_per_model_mean_time_per_step_by_game(rows: Sequence[Dict[str, object]], out_path: Path) -> None:
    summary = aggregate_means(rows, ["model_profile", "qwen_thinking_mode", "causal_mode", "game"])
    if not summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        _finalize_figure(fig, out_path, annotate=False)
        return

    model_names = _ordered_model_names_from_summary(summary)
    causal_modes = sorted({str(r["causal_mode"]) for r in summary})
    games = ["spatialgame1", "spatialgame2", "spatialgame3"]
    x = np.arange(len(games))
    style_by_thinking = {"on": "-", "off": ":"}
    color_by_causal = {"off": "tab:blue", "on": "tab:orange"}

    fig, axes = plt.subplots(8, 1, figsize=(10, 13.5), sharex=True)
    for idx, model in enumerate(model_names):
        ax_top = axes[2 * idx]
        ax_bot = axes[2 * idx + 1]
        subset = [
            r
            for r in summary
            if str(r["model_profile"]).replace("local-", "").replace("-vllm", "") == model
        ]

        top_texts: List[plt.Text] = []
        bot_texts: List[plt.Text] = []
        top_xs: List[float] = []
        top_ys: List[float] = []
        bot_xs: List[float] = []
        bot_ys: List[float] = []
        for thinking, ax, text_bucket in [("on", ax_top, top_texts), ("off", ax_bot, bot_texts)]:
            for causal in causal_modes:
                vals = []
                for game in games:
                    row = next(
                        (
                            r
                            for r in subset
                            if str(r["qwen_thinking_mode"]) == thinking
                            and str(r["causal_mode"]) == causal
                            and str(r["game"]) == game
                        ),
                        None,
                    )
                    vals.append(_to_float(row.get("mean_time_per_step_seconds", 0.0), 0.0) if row else float("nan"))
                ax.plot(
                    x,
                    vals,
                    marker="o",
                    linewidth=2.5,
                    linestyle=style_by_thinking.get(thinking, "-"),
                    color=color_by_causal.get(causal, "tab:gray"),
                    label="_nolegend_",
                )
                for xi, y in zip(x, vals):
                    if not np.isfinite(y):
                        continue
                    if thinking == "on":
                        top_xs.append(float(xi))
                        top_ys.append(float(y))
                    else:
                        bot_xs.append(float(xi))
                        bot_ys.append(float(y))
                    t = ax.text(
                        float(xi),
                        float(y) + 0.01,
                        f"{float(y):.2f}",
                        color=color_by_causal.get(causal, "tab:gray"),
                        fontsize=16,
                        ha="center",
                        va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.25},
                        clip_on=False,
                    )
                    text_bucket.append(t)
            ax.grid(alpha=0.25)

        if top_texts:
            adjust_text(
                top_texts,
                x=top_xs,
                y=top_ys,
                ax=ax_top,
                only_move={"points": "xy", "text": "xy"},
                force_points=0.8,
                force_text=1.2,
                expand_points=(2.0, 2.0),
                expand_text=(1.4, 1.6),
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
            )
        if bot_texts:
            adjust_text(
                bot_texts,
                x=bot_xs,
                y=bot_ys,
                ax=ax_bot,
                only_move={"points": "xy", "text": "xy"},
                force_points=0.8,
                force_text=1.2,
                expand_points=(2.0, 2.0),
                expand_text=(1.4, 1.6),
                avoid_self=True,
                ensure_inside_axes=True,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3),
            )

        top_vals = [
            _to_float(r.get("mean_time_per_step_seconds", 0.0), 0.0)
            for r in subset
            if str(r.get("qwen_thinking_mode", "")) == "on"
        ]
        bot_vals = [
            _to_float(r.get("mean_time_per_step_seconds", 0.0), 0.0)
            for r in subset
            if str(r.get("qwen_thinking_mode", "")) == "off"
        ]
        if top_vals:
            tmin, tmax = min(top_vals), max(top_vals)
            pad = max((tmax - tmin) * 0.2, 1e-4)
            ax_top.set_ylim(max(0.0, tmin - pad), tmax + pad)
        if bot_vals:
            bmin, bmax = min(bot_vals), max(bot_vals)
            pad = max((bmax - bmin) * 0.2, 1e-4)
            ax_bot.set_ylim(max(0.0, bmin - pad), bmax + pad)

        _reposition_texts_within_axes(fig, ax_top, top_texts, pad_px=8.0)
        _reposition_texts_within_axes(fig, ax_bot, bot_texts, pad_px=8.0)

        ax_top.set_ylabel("Mean Time/Step (s)", fontsize=16, labelpad=40)
        ax_top.yaxis.set_label_coords(-0.09, -0.5)
        ax_bot.set_ylabel("")
        ax_top.text(
            0.5,
            1.08,
            model,
            transform=ax_top.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.1)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    axes[-1].set_xlabel("Game", fontsize=12)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(games)
    axes[-1].set_xlim(-0.5, len(games) - 0.5)

    thinking_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="thinking=on"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=2, label="thinking=off"),
    ]
    causal_handles = [
        Line2D([0], [0], color=color_by_causal.get("on", "tab:orange"), linestyle="-", linewidth=2, label="causal=on"),
        Line2D([0], [0], color=color_by_causal.get("off", "tab:blue"), linestyle="-", linewidth=2, label="causal=off"),
    ]
    legend_thinking = fig.legend(
        handles=thinking_handles,
        loc="upper center",
        bbox_to_anchor=(0.28, 0.02),
        ncol=1,
        frameon=False,
        title="Thinking",
    )
    fig.add_artist(legend_thinking)
    fig.legend(
        handles=causal_handles,
        loc="upper center",
        bbox_to_anchor=(0.72, 0.02),
        ncol=1,
        frameon=False,
        title="Causal",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    _finalize_figure(fig, out_path, annotate=False)


def write_plots_readme(plots_dir: Path, plot_files: Dict[str, Path]) -> None:
    explanation = {
        "Win Rate by Game x Thinking": "Compares win rate per game between thinking on/off.",
        "Completion Rate by Game x Thinking": "Compares completion rate per game between thinking on/off.",
        "Win Rate by Game x Causal": "Compares win rate per game between causal on/off.",
        "Completion Rate by Game x Causal": "Compares completion rate per game between causal on/off.",
        "Completion Rate by Model": "Compares completion rate across models.",
        "Mean Time per Step by Model": "Compares time-per-step across models.",
        "Mean Time per Step by Game x Thinking": "Grouped time-per-step comparison per game, split by thinking mode.",
        "Mean Time per Step by Game x Causal": "Grouped time-per-step comparison per game, split by causal mode.",
        "Mean Time per Step by Model x Horizon": "How time-per-step changes with planning horizon for each model.",
        "Completion Rate by Game x Model": "Compares completion rate per game across models.",
        "Reward vs Level": "Mean normalized reward trajectory as level increases (split by thinking mode).",
        "Spatialgame2 Causal-On Absolute Reward by Level x Thinking": "For spatialgame2 with causal=on only, mean absolute reward by level grouped by thinking mode.",
        "Reward by Thinking x Causal": "Mean normalized reward comparison for thinking and causal combinations.",
        "Mean Runtime by Thinking x Causal": "Average episode runtime for each thinking/causal combination.",
        "Causal On-Off Summary (Win/Runtime/Completion)": "Three bars plots comparing causal off vs on for win rate, mean runtime, and completion rate.",
        "Win Rate vs Planning Horizon": "How horizon length changes win rate (split by thinking mode).",
        "Completion Rate vs Planning Horizon": "Grouped bar comparison of completion rate across horizons (split by thinking mode).",
        "Causal x Horizon Win-Rate Heatmap": "Win-rate heatmaps over causal mode and horizon for each thinking mode.",
        "Causal Win-Rate Comparison by Level": "Direct line comparison of causal on/off win rate by level for each game.",
        "Causal Completion Comparison by Level": "Grouped bar comparison of completion rate by level for each game, split by thinking/causal mode.",
        "Causal Runtime Comparison by Level": "Direct line comparison of causal on/off runtime by level for each game.",
        "Runtime vs Tokens Scatter": "Run-level runtime vs token usage; color/marker separates outcomes and thinking.",
        "Reward vs Tokens by Game": "Normalized reward-token tradeoff grouped by game.",
        "Runtime vs Steps by Winner": "Episode duration scaling with step count, grouped by winner state.",
        "Dense Metrics Heatmap": "Compact matrix of key dense metrics across (thinking, causal, horizon).",
        "Win-Rate Heatmap (Game x Level)": "Win-rate landscape over game and level.",
        "Completion Heatmap (Game x Level)": "Completion-rate landscape over game and level.",
        "Reward Heatmap (Game x Level)": "Normalized reward landscape over game and level.",
        "Steps Heatmap (Game x Level)": "Average episode length over game and level.",
        "Execution Match by Mode": "Action execution-match rate across full mode combinations.",
        "Parse Rate by Mode": "LLM action-parse success rate across mode combinations.",
        "Token Efficiency by Mode": "Reward per 1k tokens across mode combinations.",
        "Mean Time per Step by Mode": "Line plot by horizon with separate causal on/off lines (faceted by thinking mode).",
        "Planning Activity by Horizon": "Planning queries and queued-action usage as horizon changes.",
        "Correlation Heatmap": "Pairwise correlation among core run-level metrics.",
        "Reward CDF by Game": "Distribution profile of normalized reward for each game.",
        "Steps CDF by Game": "Distribution profile of episode length for each game.",
        "Step Distribution": "Overall histogram of episode lengths.",
        "Per Model Win Rate by Horizon": "Four vertically stacked model panels: win rate vs horizon; solid/dotted lines for thinking and colors for causal mode.",
        "Per Model Completion Rate by Horizon": "Four vertically stacked model panels: completion rate vs horizon; solid/dotted lines for thinking and colors for causal mode.",
        "Per Model Mean Time/Step by Horizon": "Four vertically stacked model panels: mean time per step vs horizon; solid/dotted lines for thinking and colors for causal mode.",
        "Per Model Win Rate by Game": "Four vertically stacked model panels: win rate vs game; solid/dotted lines for thinking and colors for causal mode.",
        "Per Model Completion Rate by Game": "Four vertically stacked model panels: completion rate vs game; solid/dotted lines for thinking and colors for causal mode.",
        "Per Model Mean Time/Step by Game": "Four vertically stacked model panels: mean time per step vs game; solid/dotted lines for thinking and colors for causal mode.",
        "Summary Table: Causal": "Causal on/off summary tables for each model plus overall average.",
        "Summary Table: Thinking": "Thinking on/off summary tables for each model plus overall average.",
        "Summary Table: Horizon": "Horizon (1/5/10) summary tables for each model plus overall average.",
        "Pareto Front (Win Rate vs Runtime by Config)": "Win-rate vs runtime Pareto analysis across all model/config combinations with numbered config IDs and Pareto highlight.",
        "Pareto Config Key Table": "Readable table mapping Pareto-plot point IDs to full configs and metrics.",
        "Paper M1 Metrics by Mode": "M1.1-M1.4 shown together for each (thinking, causal, horizon) mode.",
        "Paper M2.1 Causal Delta (On-Off)": "Difference of M1.1-M1.4 between causal on/off, grouped by thinking and horizon.",
        "Paper M3.1 Thinking Delta (On-Off)": "Difference of M1.1-M1.4 between thinking on/off, grouped by causal and horizon.",
        "Paper M3.2 Horizon Comparison": "M1.1-M1.4 curves across planning horizons (1, 5, 10) for all mode combinations.",
    }

    lines = [
        "# Plot Guide",
        "",
        "This folder contains generated analysis figures for spatial experiments.",
        "",
        "| Plot File | What It Shows |",
        "|---|---|",
    ]

    for title, path in plot_files.items():
        rel = path.name
        desc = explanation.get(title, "Per-game diagnostic plot for detailed factor interactions.")
        lines.append(f"| `{rel}` | {desc} |")

    (plots_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_selected_plots_to_latex_repo(plot_files: Dict[str, Path]) -> List[Path]:
    exported: List[Path] = []
    if not LATEX_PAPER_ROOT.exists():
        return exported

    LATEX_SPATIAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    source_by_name = {path.name: path for path in plot_files.values()}
    for source_name, dest_name in LATEX_FIGURE_EXPORTS.items():
        src = source_by_name.get(source_name)
        if src is None or not src.exists():
            continue
        dest = LATEX_SPATIAL_FIGURES_DIR / dest_name
        shutil.copy2(src, dest)
        exported.append(dest)
    return exported


def build_media_manifest(rows: Sequence[Dict[str, object]], out_dir: Path, top_k: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    previews_dir = out_dir / "media_previews"

    with_gif = [r for r in rows if str(r.get("gif_path", ""))]
    best = sorted(with_gif, key=lambda r: (_to_int(r.get("is_win", 0), 0), _to_float(r.get("total_reward", 0.0), 0.0), -_to_float(r.get("runtime_seconds", 0.0), 0.0)), reverse=True)
    worst = sorted(with_gif, key=lambda r: (_to_int(r.get("is_win", 0), 0), _to_float(r.get("total_reward", 0.0), 0.0), _to_float(r.get("runtime_seconds", 0.0), 0.0)))

    selected: List[Dict[str, object]] = []
    seen = set()
    for collection in (best[:top_k], worst[:top_k]):
        for r in collection:
            key = (str(r.get("game_env_id", "")), str(r.get("mode", "")), _to_int(r.get("run_id", 0), 0))
            if key in seen:
                continue
            seen.add(key)
            selected.append(r)

    manifest_rows: List[Dict[str, object]] = []
    preview_rows: List[Dict[str, object]] = []

    for idx, r in enumerate(selected):
        gif_path = Path(str(r["gif_path"]))
        preview_rel = ""
        if gif_path.exists():
            preview_path = previews_dir / f"sample_{idx:03d}__{r['game']}_lvl{r['level']}_run{r['run_id']}.png"
            if maybe_extract_preview(gif_path, preview_path):
                preview_rel = str(preview_path.relative_to(out_dir))

        manifest_row = {
            "sample_idx": idx,
            "game": r["game"],
            "level": r["level"],
            "run_id": r["run_id"],
            "thinking": r["qwen_thinking_mode"],
            "causal": r["causal_mode"],
            "horizon": r["planning_horizon_x"],
            "winner": r["winner"],
            "steps": r["steps"],
            "reward": r["total_reward"],
            "runtime_seconds": r["runtime_seconds"],
            "mode": r["mode"],
            "gif_path": r["gif_path"],
            "preview_png": preview_rel,
            "run_summary_path": r.get("run_summary_path", ""),
        }
        manifest_rows.append(manifest_row)
        if preview_rel:
            preview_rows.append(
                {
                    "sample_idx": idx,
                    "game": r["game"],
                    "level": r["level"],
                    "run_id": r["run_id"],
                    "preview_png": preview_rel,
                    "gif_path": r["gif_path"],
                }
            )

    return manifest_rows, preview_rows


def _html_table(rows: Sequence[Dict[str, object]], columns: Sequence[str], max_rows: int = 50) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body_rows = []
    for r in list(rows)[:max_rows]:
        cells = "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_html_report(
    out_path: Path,
    summary: Dict[str, object],
    plots: Dict[str, str],
    top_modes: List[Dict[str, object]],
    bottom_modes: List[Dict[str, object]],
    media_rows: List[Dict[str, object]],
) -> None:
    overview = summary["overview"]
    html_parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Spatial Combined Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.35;}",
        "h1,h2{margin:0.6em 0 0.3em;}",
        "table{border-collapse:collapse;width:100%;margin:10px 0 22px;}",
        "th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px;}",
        "th{background:#f2f2f2;text-align:left;}",
        ".plots img{max-width:100%;height:auto;border:1px solid #ddd;margin:10px 0 18px;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;}",
        ".card{border:1px solid #ddd;padding:10px;}",
        ".muted{color:#666;font-size:12px;}",
        "</style></head><body>",
        "<h1>Spatial Tasks Combined Results</h1>",
        f"<p class='muted'>Generated at UTC {html.escape(str(summary.get('generated_at_utc', '')))}</p>",
        "<h2>Overview</h2>",
        "<ul>",
        f"<li>Total records discovered: {overview['records_discovered']}</li>",
        f"<li>Total runs normalized: {overview['runs_normalized']}</li>",
        f"<li>Games: {', '.join(overview['games'])}</li>",
        f"<li>Thinking modes: {', '.join(overview['thinking_modes'])}</li>",
        f"<li>Causal modes: {', '.join(overview['causal_modes'])}</li>",
        f"<li>Horizons: {', '.join(map(str, overview['planning_horizons']))}</li>",
        f"<li>Win rate: {overview['win_rate']:.4f}</li>",
        f"<li>Completion rate: {overview['completion_rate']:.4f}</li>",
        "</ul>",
        "<h2>Top Configurations</h2>",
        _html_table(
            top_modes,
            [
                "qwen_thinking_mode",
                "causal_mode",
                "planning_horizon_x",
                "n",
                "win_rate",
                "mean_total_reward",
                "mean_runtime_seconds",
                "mean_reward_per_1k_tokens",
                "mean_action_execution_match_rate",
            ],
            max_rows=25,
        ),
        "<h2>Bottom Configurations</h2>",
        _html_table(
            bottom_modes,
            [
                "qwen_thinking_mode",
                "causal_mode",
                "planning_horizon_x",
                "n",
                "win_rate",
                "mean_total_reward",
                "mean_runtime_seconds",
                "mean_reward_per_1k_tokens",
                "mean_action_execution_match_rate",
            ],
            max_rows=25,
        ),
        "<h2>Plots</h2>",
        "<div class='plots'>",
    ]

    for title, rel in plots.items():
        html_parts.append(f"<h3>{html.escape(title)}</h3><img src='{html.escape(rel)}' alt='{html.escape(title)}' />")
    html_parts.append("</div>")

    html_parts.append("<h2>Media Samples</h2>")
    html_parts.append("<div class='grid'>")
    for r in media_rows[:30]:
        preview = str(r.get("preview_png", ""))
        gif = str(r.get("gif_path", ""))
        html_parts.append("<div class='card'>")
        html_parts.append(
            f"<div><b>{html.escape(str(r.get('game','')))} lvl{html.escape(str(r.get('level','')))} run{html.escape(str(r.get('run_id','')))}</b></div>"
        )
        html_parts.append(
            f"<div class='muted'>thinking={html.escape(str(r.get('thinking','')))} causal={html.escape(str(r.get('causal','')))} horizon={html.escape(str(r.get('horizon','')))} | winner={html.escape(str(r.get('winner','')))} | reward={html.escape(str(r.get('reward','')))} | steps={html.escape(str(r.get('steps','')))}</div>"
        )
        if preview:
            html_parts.append(f"<img src='{html.escape(preview)}' style='width:100%;height:auto;border:1px solid #ddd;margin-top:8px;'/> ")
        html_parts.append(f"<div><a href='{html.escape(gif)}'>GIF</a></div>")
        html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append("</body></html>")
    out_path.write_text("\n".join(html_parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate and visualize spatial experiment results.")
    p.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    p.add_argument("--out_dir", type=Path, default=None, help="Optional explicit output directory.")
    p.add_argument("--include_backups", action="store_true", help="Also include records moved to backups/spatial_archive*.")
    p.add_argument("--top_k", type=int, default=20, help="Top-K rows for ranking tables and media sampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir.resolve() if args.out_dir else (experiment_root / "combined_report" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_paths = discover_record_paths(experiment_root=experiment_root, include_backups=args.include_backups)
    if not record_paths:
        raise SystemExit(f"No spatial records found under {experiment_root}/results_index/records")

    raw_records: List[Dict[str, object]] = []
    for rp in record_paths:
        payload = _safe_read_json(rp)
        if not payload:
            continue
        payload["_record_path"] = str(rp.resolve())
        raw_records.append(payload)

    rows = [normalize_record(r) for r in raw_records]
    allowed_models = infer_model_filter_for_root(experiment_root)
    if allowed_models is not None:
        rows = [r for r in rows if str(r.get("model_profile", "")) in allowed_models]

    # Deduplicate by a stable key. Keep latest timestamp_utc if duplicate.
    by_key: Dict[Tuple[str, int, str, int], Dict[str, object]] = {}
    for r in rows:
        key = (
            str(r.get("game_env_id", "")),
            str(r.get("model_profile", "")),
            _to_int(r.get("run_id", 0), 0),
            str(r.get("mode", "")),
            _to_int(r.get("planning_horizon_x", 1), 1),
        )
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
            continue
        if str(r.get("timestamp_utc", "")) >= str(prev.get("timestamp_utc", "")):
            by_key[key] = r
    rows = list(by_key.values())

    if not rows:
        raise SystemExit("No valid rows after normalization.")

    add_normalized_reward_metrics(rows)

    # Visuals (paper-only).
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_files = {
        "Win Rate by Game x Thinking": plots_dir / "win_rate_by_game_thinking.pdf",
        "Completion Rate by Game x Thinking": plots_dir / "completion_rate_by_game_thinking.pdf",
        "Win-Rate Heatmap (Game x Level)": plots_dir / "winrate_heatmap_game_level.pdf",
        "Completion Heatmap (Game x Level)": plots_dir / "completion_heatmap_game_level.pdf",
        "Win-Rate Heatmap (Model x Level)": plots_dir / "winrate_heatmap_model_level.pdf",
        "Causal x Horizon Win-Rate Heatmap": plots_dir / "causal_horizon_winrate_heatmap.pdf",
        "Per Model Win Rate by Horizon": plots_dir / "per_model_win_rate_by_horizon.pdf",
        "Per Model Completion Rate by Horizon": plots_dir / "per_model_completion_rate_by_horizon.pdf",
        "Per Model Mean Time/Step by Horizon": plots_dir / "per_model_mean_time_per_step_by_horizon.pdf",
        "Per Model Win Rate by Game": plots_dir / "per_model_win_rate_by_game.pdf",
        "Per Model Completion Rate by Game": plots_dir / "per_model_completion_rate_by_game.pdf",
        "Per Model Mean Time/Step by Game": plots_dir / "per_model_mean_time_per_step_by_game.pdf",
        "Summary Table: Causal": plots_dir / "summary_table_causal.pdf",
        "Summary Table: Thinking": plots_dir / "summary_table_thinking.pdf",
        "Summary Table: Horizon": plots_dir / "summary_table_horizon.pdf",
    }
    plot_win_rate_by_game_thinking(rows, plot_files["Win Rate by Game x Thinking"])
    plot_completion_rate_by_game_thinking(rows, plot_files["Completion Rate by Game x Thinking"])
    plot_heatmap_game_level_metric(
        rows,
        plot_files["Win-Rate Heatmap (Game x Level)"],
        metric_key="win_rate",
        title="Win Rate Heatmap (Game x Level)",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap_game_level_metric(
        rows,
        plot_files["Completion Heatmap (Game x Level)"],
        metric_key="completion_rate",
        title="Completion Rate Heatmap (Game x Level)",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_heatmap_model_level_metric(
        rows,
        plot_files["Win-Rate Heatmap (Model x Level)"],
        metric_key="win_rate",
        title="Win Rate Heatmap (Model x Level)",
        cmap="YlGn",
        vmin=0.0,
        vmax=1.0,
    )
    plot_causal_horizon_heatmap(rows, plot_files["Causal x Horizon Win-Rate Heatmap"])
    plot_per_model_win_rate_by_horizon(rows, plot_files["Per Model Win Rate by Horizon"])
    plot_per_model_completion_rate_by_horizon(rows, plot_files["Per Model Completion Rate by Horizon"])
    plot_per_model_mean_time_per_step_by_horizon(rows, plot_files["Per Model Mean Time/Step by Horizon"])
    plot_per_model_win_rate_by_game(rows, plot_files["Per Model Win Rate by Game"])
    plot_per_model_completion_rate_by_game(rows, plot_files["Per Model Completion Rate by Game"])
    plot_per_model_mean_time_per_step_by_game(rows, plot_files["Per Model Mean Time/Step by Game"])
    plot_summary_table_causal(rows, plot_files["Summary Table: Causal"])
    plot_summary_table_thinking(rows, plot_files["Summary Table: Thinking"])
    plot_summary_table_horizon(rows, plot_files["Summary Table: Horizon"])
    plot_files.update(plot_per_game_model_profiles(rows, plots_dir))
    write_three_summary_tables_latex(rows, out_dir / "tables.tex")

    print(f"[DONE] Paper plots written to: {plots_dir}")
    return

    # Global summary.
    overview = {
        "records_discovered": len(record_paths),
        "runs_normalized": len(rows),
        "games": sorted({str(r["game"]) for r in rows}),
        "thinking_modes": sorted({str(r["qwen_thinking_mode"]) for r in rows}),
        "causal_modes": sorted({str(r["causal_mode"]) for r in rows}),
        "planning_horizons": sorted({_to_int(r["planning_horizon_x"], 1) for r in rows}),
        "win_rate": float(sum(_to_int(r["is_win"], 0) for r in rows) / max(len(rows), 1)),
        "completion_rate": float(sum(_to_int(r["is_completed"], 0) for r in rows) / max(len(rows), 1)),
    }
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(experiment_root),
        "out_dir": str(out_dir),
        "include_backups": bool(args.include_backups),
        "overview": overview,
        "artifacts": {
            "all_runs_csv": str((out_dir / "all_runs_detailed.csv").resolve()),
            "summary_by_mode_csv": str((out_dir / "summary_by_mode.csv").resolve()),
            "summary_by_model_csv": str((out_dir / "summary_by_model.csv").resolve()),
            "media_manifest_csv": str((out_dir / "media_manifest.csv").resolve()),
            "latex_exported_figures": [str(p.resolve()) for p in exported_latex_figures],
        },
    }
    write_json(out_dir / "combined_summary.json", summary)

    # HTML report.
    rel_plots = {k: str(v.relative_to(out_dir)) for k, v in plot_files.items()}
    write_html_report(
        out_path=out_dir / "report.html",
        summary=summary,
        plots=rel_plots,
        top_modes=top_modes,
        bottom_modes=bottom_modes,
        media_rows=media_manifest,
    )

    readme_lines = [
        "# Spatial Combined Aggregation",
        "",
        f"- generated_at_utc: `{summary['generated_at_utc']}`",
        f"- experiment_root: `{experiment_root}`",
        f"- include_backups: `{args.include_backups}`",
        f"- records_discovered: `{overview['records_discovered']}`",
        f"- runs_normalized: `{overview['runs_normalized']}`",
        f"- games: `{', '.join(overview['games'])}`",
        f"- thinking_modes: `{', '.join(overview['thinking_modes'])}`",
        f"- causal_modes: `{', '.join(overview['causal_modes'])}`",
        f"- planning_horizons: `{', '.join(map(str, overview['planning_horizons']))}`",
        f"- win_rate: `{overview['win_rate']:.4f}`",
        f"- completion_rate: `{overview['completion_rate']:.4f}`",
        "",
        "## Main Outputs",
        "- `all_runs_detailed.csv`",
        "- `summary_by_mode.csv`",
        "- `summary_by_model.csv`",
        "- `summary_by_game.csv`",
        "- `summary_by_game_level.csv`",
        "- `summary_by_game_level_mode.csv`",
        "- `summary_by_thinking.csv`",
        "- `summary_by_causal.csv`",
        "- `summary_by_horizon.csv`",
        "- `top_modes_by_win_rate.csv`",
        "- `bottom_modes_by_win_rate.csv`",
        "- `media_manifest.csv`",
        "- `preview_manifest.csv`",
        "- `pareto_config_key.csv`",
        "- `paper_metrics_by_mode.csv`",
        "- `combined_summary.json`",
        "- `report.html`",
        "",
        "## Plots",
    ]
    for title, plot in rel_plots.items():
        readme_lines.append(f"- `{plot}` ({title})")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"[DONE] Combined report written to: {out_dir}")
    print(f"[DONE] Open HTML report: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
