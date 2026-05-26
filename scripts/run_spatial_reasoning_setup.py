#!/usr/bin/env python3
"""Isolated experiment setup for the spatial-reasoning additional side goal.

This script is intentionally isolated from existing experiment scripts:
- It never edits shared prompt template files in-place.
- It writes all artifacts under its own output root.
- It passes a per-run prompt template path via environment variable only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple



ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_TEMPLATE = ROOT / "project" / "llm" / "utils" / "prompt_templates" / "prompt.json"
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments" / "spatial_reasoning"


def resolve_spatial_paths(experiment_root: Path) -> Dict[str, Path]:
    root = experiment_root.resolve()
    return {
        "experiment_root": root,
        "prompt_variants_dir": root / "prompt_variants",
        "runs_root": root / "runs",
        "results_index_root": root / "results_index",
    }


def ensure_within_root(root: Path, path: Path, label: str) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise RuntimeError(
            f"[ISOLATION] {label} escaped experiment_root: {path_resolved} (root={root_resolved})"
        )


def build_instruction() -> str:
    rules: List[str] = [
        "=== Output Contract ===",
        "Return exactly 2 lines and nothing else.",
        "Line 1: Action:<integer_id>",
        "Line 2: Feedback:<max 14 words>",
        "",
        "=== Task Context ===",
        "You are solving a grid-based spatial reasoning task.",
        "Use the action legend, state map, and game objective to choose movement.",
        "Feedback should summarize immediate spatial progress toward the objective.",
        "",
        "=== Decision Rules ===",
        "- Action must be a valid numeric ID from Action Legend.",
        "- Treat avatar/nokey/withkey or any *avatar* alias as self.",
        "- Prioritize legal movement that improves shortest-path progress toward objective.",
    ]

    rules.append("- Feedback should not include coordinates.")

    rules.extend(
        [
            "- If previous action caused no movement and no reward, prefer a different action.",
            "- No markdown, no JSON, no reasoning, no extra lines.",
        ]
    )
    return "\n".join(rules)


def write_prompt_variant(prompt_variants_dir: Path) -> Path:
    cfg = json.loads(DEFAULT_PROMPT_TEMPLATE.read_text(encoding="utf-8"))
    cfg["instruction"] = build_instruction()

    prompt_variants_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_variants_dir / "prompt__spatial.json"
    prompt_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return prompt_path


def parse_env_id(env_id: str) -> Tuple[str, Optional[int]]:
    # gvgai-<game>-lvl<level>-v0
    if not env_id.startswith("gvgai-"):
        return env_id, None
    core = env_id[len("gvgai-") :]
    if "-lvl" not in core:
        return core, None
    game, tail = core.rsplit("-lvl", 1)
    lvl_s = ""
    for ch in tail:
        if ch.isdigit():
            lvl_s += ch
        else:
            break
    if not lvl_s:
        return game, None
    return game, int(lvl_s)


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_step_metrics(path_str: str) -> Dict[str, object]:
    path = Path(path_str) if path_str else None
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_llm_actions_from_run_dir(run_dir: str) -> List[Optional[int]]:
    if not run_dir:
        return []
    llm_dir = Path(run_dir) / "llm_io"
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
    actions: List[Optional[int]] = []
    for r in responses:
        text = str(r)
        # Exclude pure planning responses from action-parse metrics.
        has_plan = bool(re.search(r"\bPlanActions\s*:\s*\[[^\]]*\]", text, flags=re.IGNORECASE))
        has_action = bool(re.search(r"\bAction\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE))
        if has_plan and not has_action:
            continue
        m = re.search(r"\bAction\s*[:=]\s*(\d+)\b", text, flags=re.IGNORECASE)
        actions.append(int(m.group(1)) if m else None)
    return actions


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

    # In planning mode, action queries occur sparsely. Align against sampled step actions.
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


def normalize_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    norm: List[Dict[str, object]] = []
    for row in rows:
        env_id = str(row.get("game_env_id", ""))
        game_name, level = parse_env_id(env_id)
        status = str(row.get("status", ""))
        winner = str(row.get("winner", ""))

        run_dir = str(row.get("run_dir", ""))
        gif_path = ""
        if run_dir:
            run_path = Path(run_dir)
            candidates = [run_path / "gameplay.gif", run_path / "gameplay.gif.gif"]
            candidates.extend(sorted(run_path.glob("gameplay*.gif*")))
            for c in candidates:
                if c and c.exists():
                    gif_path = str(c)
                    break

        step_metrics_path = str(row.get("step_metrics_json_path", ""))
        step_metrics = load_step_metrics(step_metrics_path)
        advanced = step_metrics.get("advanced_step_metrics", {}) if isinstance(step_metrics, dict) else {}
        step_actions = [x.get("action") for x in step_metrics.get("actions_by_step", [])] if isinstance(step_metrics, dict) else []
        llm_actions = parse_llm_actions_from_run_dir(run_dir)
        exec_metrics = compute_action_execution_metrics(
            step_actions=step_actions,
            llm_actions=llm_actions,
            planning_horizon_x=to_int(row.get("planning_horizon_x"), 1),
        )

        effective_step_ratio = to_float(advanced.get("effective_step_ratio"), 0.0)
        action_entropy_windowed = to_float(advanced.get("action_entropy_windowed"), 0.0)
        non_progress_streak_p95 = to_float(advanced.get("non_progress_streak_p95"), 0.0)
        repeat_state_repeat_action_rate = to_float(advanced.get("repeat_state_repeat_action_rate"), 0.0)
        action_change_rate = to_float(advanced.get("action_change_rate"), 0.0)
        catastrophic_step_rate = to_float(advanced.get("catastrophic_step_rate"), 0.0)

        steps = to_int(row.get("steps"), 0)
        total_reward = to_float(row.get("total_reward"), 0.0)
        runtime_seconds = to_float(row.get("runtime_seconds"), 0.0)
        total_tokens = to_float(row.get("sum_total_tokens"), 0.0)
        meaningful = to_float(row.get("meaningful_step_ratio"), 0.0)
        token_efficiency_reward_per_1k = total_reward / max(total_tokens / 1000.0, 1e-9)
        steps_per_second = steps / max(runtime_seconds, 1e-9)
        effective_steps_per_second = (effective_step_ratio * steps) / max(runtime_seconds, 1e-9)

        norm.append(
            {
                "condition": str(row.get("condition", "unknown")),
                "game_env_id": env_id,
                "game": game_name,
                "level": level,
                "model_profile": str(row.get("model_profile", "")),
                "mode": str(row.get("mode", "")),
                "qwen_thinking_mode": str(row.get("qwen_thinking_mode", "")),
                "causal_mode": str(row.get("causal_mode", "")),
                "planning_mode": str(row.get("planning_mode", "")),
                "planning_horizon_x": to_int(row.get("planning_horizon_x"), 1),
                "run_id": to_int(row.get("run_id"), 0),
                "status": status,
                "winner": winner,
                "is_completed": 1 if status == "completed" else 0,
                "is_win": 1 if winner == "PLAYER_WINS" else 0,
                "steps": steps,
                "total_reward": total_reward,
                "runtime_seconds": runtime_seconds,
                "meaningful_step_ratio": meaningful,
                "sum_total_tokens": total_tokens,
                "mean_step_time_seconds": to_float(row.get("mean_step_time_seconds"), 0.0),
                "effective_step_ratio": effective_step_ratio,
                "action_entropy_windowed": action_entropy_windowed,
                "non_progress_streak_p95": non_progress_streak_p95,
                "repeat_state_repeat_action_rate": repeat_state_repeat_action_rate,
                "action_change_rate": action_change_rate,
                "catastrophic_step_rate": catastrophic_step_rate,
                "llm_action_parse_rate": round(float(exec_metrics["llm_action_parse_rate"]), 6),
                "action_execution_match_rate": round(float(exec_metrics["action_execution_match_rate"]), 6),
                "action_execution_coverage": round(float(exec_metrics["action_execution_coverage"]), 6),
                "action_parse_success_count": to_int(row.get("action_parse_success_count"), 0),
                "action_parse_fail_count": to_int(row.get("action_parse_fail_count"), 0),
                "action_fallback_count": to_int(row.get("action_fallback_count"), 0),
                "action_fallback_last_valid_count": to_int(row.get("action_fallback_last_valid_count"), 0),
                "action_fallback_nil_count": to_int(row.get("action_fallback_nil_count"), 0),
                "action_parse_tolerant_success_count": to_int(row.get("action_parse_tolerant_success_count"), 0),
                "action_parse_strict_success_count": to_int(row.get("action_parse_strict_success_count"), 0),
                "action_think_incomplete_retry_count": to_int(row.get("action_think_incomplete_retry_count"), 0),
                "action_think_incomplete_terminal_fail_count": to_int(row.get("action_think_incomplete_terminal_fail_count"), 0),
                "plan_think_incomplete_retry_count": to_int(row.get("plan_think_incomplete_retry_count"), 0),
                "plan_think_incomplete_terminal_fail_count": to_int(row.get("plan_think_incomplete_terminal_fail_count"), 0),
                "token_efficiency_reward_per_1k": round(token_efficiency_reward_per_1k, 6),
                "steps_per_second": round(steps_per_second, 6),
                "effective_steps_per_second": round(effective_steps_per_second, 6),
                "error_type": str(row.get("error_type", "")),
                "error_message": str(row.get("error_message", "")),
                "run_dir": run_dir,
                "run_summary_path": str(row.get("run_summary_path", "")),
                "step_metrics_json_path": step_metrics_path,
                "gif_path": gif_path,
            }
        )
    return norm


def group_mean(rows: List[Dict[str, object]], key_fields: List[str]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for r in rows:
        key = tuple(r[k] for k in key_fields)
        b = buckets.setdefault(
            key,
            {
                **{k: r[k] for k in key_fields},
                "n": 0,
                "completed": 0,
                "wins": 0,
                "steps_sum": 0.0,
                "reward_sum": 0.0,
                "runtime_sum": 0.0,
                "meaningful_sum": 0.0,
                "tokens_sum": 0.0,
                "step_time_sum": 0.0,
                "effective_sum": 0.0,
                "entropy_sum": 0.0,
                "non_progress_sum": 0.0,
                "repeat_action_sum": 0.0,
                "action_change_sum": 0.0,
                "catastrophic_sum": 0.0,
                "llm_parse_sum": 0.0,
                "exec_match_sum": 0.0,
                "exec_coverage_sum": 0.0,
                "action_parse_success_sum": 0.0,
                "action_parse_fail_sum": 0.0,
                "action_fallback_sum": 0.0,
                "action_fallback_last_valid_sum": 0.0,
                "action_fallback_nil_sum": 0.0,
                "action_parse_tolerant_success_sum": 0.0,
                "action_parse_strict_success_sum": 0.0,
                "action_think_retry_sum": 0.0,
                "action_think_terminal_fail_sum": 0.0,
                "plan_think_retry_sum": 0.0,
                "plan_think_terminal_fail_sum": 0.0,
                "token_eff_sum": 0.0,
                "steps_per_second_sum": 0.0,
                "effective_steps_per_second_sum": 0.0,
            },
        )
        b["n"] = int(b["n"]) + 1
        b["completed"] = int(b["completed"]) + int(r["is_completed"])
        b["wins"] = int(b["wins"]) + int(r["is_win"])
        b["steps_sum"] = float(b["steps_sum"]) + float(r["steps"])
        b["reward_sum"] = float(b["reward_sum"]) + float(r["total_reward"])
        b["runtime_sum"] = float(b["runtime_sum"]) + float(r["runtime_seconds"])
        b["meaningful_sum"] = float(b["meaningful_sum"]) + float(r["meaningful_step_ratio"])
        b["tokens_sum"] = float(b["tokens_sum"]) + float(r["sum_total_tokens"])
        b["step_time_sum"] = float(b["step_time_sum"]) + float(r["mean_step_time_seconds"])
        b["effective_sum"] = float(b["effective_sum"]) + float(r["effective_step_ratio"])
        b["entropy_sum"] = float(b["entropy_sum"]) + float(r["action_entropy_windowed"])
        b["non_progress_sum"] = float(b["non_progress_sum"]) + float(r["non_progress_streak_p95"])
        b["repeat_action_sum"] = float(b["repeat_action_sum"]) + float(r["repeat_state_repeat_action_rate"])
        b["action_change_sum"] = float(b["action_change_sum"]) + float(r["action_change_rate"])
        b["catastrophic_sum"] = float(b["catastrophic_sum"]) + float(r["catastrophic_step_rate"])
        b["llm_parse_sum"] = float(b["llm_parse_sum"]) + float(r["llm_action_parse_rate"])
        b["exec_match_sum"] = float(b["exec_match_sum"]) + float(r["action_execution_match_rate"])
        b["exec_coverage_sum"] = float(b["exec_coverage_sum"]) + float(r["action_execution_coverage"])
        b["action_parse_success_sum"] = float(b["action_parse_success_sum"]) + float(r["action_parse_success_count"])
        b["action_parse_fail_sum"] = float(b["action_parse_fail_sum"]) + float(r["action_parse_fail_count"])
        b["action_fallback_sum"] = float(b["action_fallback_sum"]) + float(r["action_fallback_count"])
        b["action_fallback_last_valid_sum"] = float(b["action_fallback_last_valid_sum"]) + float(r["action_fallback_last_valid_count"])
        b["action_fallback_nil_sum"] = float(b["action_fallback_nil_sum"]) + float(r["action_fallback_nil_count"])
        b["action_parse_tolerant_success_sum"] = float(b["action_parse_tolerant_success_sum"]) + float(r["action_parse_tolerant_success_count"])
        b["action_parse_strict_success_sum"] = float(b["action_parse_strict_success_sum"]) + float(r["action_parse_strict_success_count"])
        b["action_think_retry_sum"] = float(b["action_think_retry_sum"]) + float(r["action_think_incomplete_retry_count"])
        b["action_think_terminal_fail_sum"] = float(b["action_think_terminal_fail_sum"]) + float(r["action_think_incomplete_terminal_fail_count"])
        b["plan_think_retry_sum"] = float(b["plan_think_retry_sum"]) + float(r["plan_think_incomplete_retry_count"])
        b["plan_think_terminal_fail_sum"] = float(b["plan_think_terminal_fail_sum"]) + float(r["plan_think_incomplete_terminal_fail_count"])
        b["token_eff_sum"] = float(b["token_eff_sum"]) + float(r["token_efficiency_reward_per_1k"])
        b["steps_per_second_sum"] = float(b["steps_per_second_sum"]) + float(r["steps_per_second"])
        b["effective_steps_per_second_sum"] = float(b["effective_steps_per_second_sum"]) + float(r["effective_steps_per_second"])

    out: List[Dict[str, object]] = []
    for _, b in sorted(buckets.items(), key=lambda x: x[0]):
        n = max(int(b["n"]), 1)
        out.append(
            {
                **{k: b[k] for k in key_fields},
                "n": int(b["n"]),
                "completion_rate": round(float(b["completed"]) / n, 6),
                "win_rate": round(float(b["wins"]) / n, 6),
                "mean_steps": round(float(b["steps_sum"]) / n, 6),
                "mean_reward": round(float(b["reward_sum"]) / n, 6),
                "mean_runtime_seconds": round(float(b["runtime_sum"]) / n, 6),
                "mean_meaningful_step_ratio": round(float(b["meaningful_sum"]) / n, 6),
                "mean_total_tokens": round(float(b["tokens_sum"]) / n, 6),
                "mean_step_time_seconds": round(float(b["step_time_sum"]) / n, 6),
                "mean_effective_step_ratio": round(float(b["effective_sum"]) / n, 6),
                "mean_action_entropy_windowed": round(float(b["entropy_sum"]) / n, 6),
                "mean_non_progress_streak_p95": round(float(b["non_progress_sum"]) / n, 6),
                "mean_repeat_state_repeat_action_rate": round(float(b["repeat_action_sum"]) / n, 6),
                "mean_action_change_rate": round(float(b["action_change_sum"]) / n, 6),
                "mean_catastrophic_step_rate": round(float(b["catastrophic_sum"]) / n, 6),
                "mean_llm_action_parse_rate": round(float(b["llm_parse_sum"]) / n, 6),
                "mean_action_execution_match_rate": round(float(b["exec_match_sum"]) / n, 6),
                "mean_action_execution_coverage": round(float(b["exec_coverage_sum"]) / n, 6),
                "mean_action_parse_success_count": round(float(b["action_parse_success_sum"]) / n, 6),
                "mean_action_parse_fail_count": round(float(b["action_parse_fail_sum"]) / n, 6),
                "mean_action_fallback_count": round(float(b["action_fallback_sum"]) / n, 6),
                "mean_action_fallback_last_valid_count": round(float(b["action_fallback_last_valid_sum"]) / n, 6),
                "mean_action_fallback_nil_count": round(float(b["action_fallback_nil_sum"]) / n, 6),
                "mean_action_parse_tolerant_success_count": round(float(b["action_parse_tolerant_success_sum"]) / n, 6),
                "mean_action_parse_strict_success_count": round(float(b["action_parse_strict_success_sum"]) / n, 6),
                "mean_action_think_incomplete_retry_count": round(float(b["action_think_retry_sum"]) / n, 6),
                "mean_action_think_incomplete_terminal_fail_count": round(float(b["action_think_terminal_fail_sum"]) / n, 6),
                "mean_plan_think_incomplete_retry_count": round(float(b["plan_think_retry_sum"]) / n, 6),
                "mean_plan_think_incomplete_terminal_fail_count": round(float(b["plan_think_terminal_fail_sum"]) / n, 6),
                "mean_token_efficiency_reward_per_1k": round(float(b["token_eff_sum"]) / n, 6),
                "mean_steps_per_second": round(float(b["steps_per_second_sum"]) / n, 6),
                "mean_effective_steps_per_second": round(float(b["effective_steps_per_second_sum"]) / n, 6),
            }
        )
    return out




def run_condition(
    games: List[str],
    model: str,
    max_steps: int,
    num_runs: int,
    level: int,
    qwen_thinking: str,
    translator_mode: str,
    causal_mode: str,
    scm_bootstrap: str,
    planning_mode: str,
    planning_horizon_x: int,
    max_workers: int,
    shuffle_action_meanings: str,
    action_shuffle_seed: int,
    per_run_budget_nonthinking_seconds: int,
    per_run_budget_thinking_seconds: int,
    experiment_root: Path,
    prompt_variants_dir: Path,
    runs_root: Path,
    results_index_root: Path,
) -> None:
    prompt_path = write_prompt_variant(prompt_variants_dir=prompt_variants_dir)
    ensure_within_root(experiment_root, prompt_path, "prompt_variant_path")

    base_output_dir = runs_root
    results_index_dir = results_index_root
    base_output_dir.mkdir(parents=True, exist_ok=True)
    results_index_dir.mkdir(parents=True, exist_ok=True)
    ensure_within_root(experiment_root, base_output_dir, "base_output_dir")
    ensure_within_root(experiment_root, results_index_dir, "results_index_dir")

    cmd = [
        "python",
        "project/main.py",
        "--games",
        *games,
        "--specific_level",
        str(level),
        "--models",
        model,
        "--modes",
        "zero-shot",
        "--qwen_thinking",
        qwen_thinking,
        "--translator_mode",
        translator_mode,
        "--causal_mode",
        causal_mode,
        "--scm_bootstrap",
        scm_bootstrap,
        "--planning_mode",
        planning_mode,
        "--planning_horizon_x",
        str(planning_horizon_x),
        "--num_runs",
        str(num_runs),
        "--max_steps",
        str(max_steps),
        "--max_workers",
        str(max_workers),
        "--base_output_dir",
        str(base_output_dir),
        "--results_index_dir",
        str(results_index_dir),
        "--shuffle_action_meanings",
        shuffle_action_meanings,
        "--action_shuffle_seed",
        str(action_shuffle_seed),
        "--log_verbosity",
        "info",
    ]

    env = dict(os.environ)
    env["GVGAI_PROMPT_TEMPLATE_PATH"] = str(prompt_path)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ROOT}:{existing_pythonpath}" if existing_pythonpath else str(ROOT)

    per_run_budget_seconds = (
        per_run_budget_thinking_seconds if qwen_thinking == "on" else per_run_budget_nonthinking_seconds
    )
    timeout_seconds = max(
        per_run_budget_seconds * max(num_runs, 1) * max(len(games), 1),
        per_run_budget_seconds,
    )
    subprocess.run(cmd, check=True, cwd=ROOT, env=env, timeout=timeout_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games",
        nargs="+",
        default=["spatialgame1_v0", "spatialgame2_v0", "spatialgame3_v0"],
        help="Games to run for each condition.",
    )
    parser.add_argument("--model", default="local-qwen3-8b-vllm")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--level", type=int, default=0)

    # Runtime toggles forwarded to project/main.py
    parser.add_argument("--qwen_thinking", choices=["auto", "on", "off"], default="off")
    parser.add_argument("--translator_mode", choices=["on", "off", "vgdl"], default="on")
    parser.add_argument("--causal_mode", choices=["off", "on"], default="off")
    parser.add_argument("--scm_bootstrap", choices=["off", "on"], default="off")
    parser.add_argument("--planning_mode", choices=["off", "lookahead_actions"], default="off")
    parser.add_argument("--planning_horizon_x", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--shuffle_action_meanings", choices=["off", "on"], default="off")
    parser.add_argument("--action_shuffle_seed", type=int, default=42)
    parser.add_argument("--per_run_budget_nonthinking_seconds", type=int, default=1000)
    parser.add_argument("--per_run_budget_thinking_seconds", type=int, default=5000)
    parser.add_argument(
        "--experiment_root",
        type=str,
        default=str(DEFAULT_EXPERIMENT_ROOT),
        help="Root folder for all spatial experiment artifacts.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only create prompt variants and print planned runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spatial_paths = resolve_spatial_paths(Path(args.experiment_root))
    experiment_root = spatial_paths["experiment_root"]
    prompt_variants_dir = spatial_paths["prompt_variants_dir"]
    runs_root = spatial_paths["runs_root"]
    results_index_root = spatial_paths["results_index_root"]
    experiment_root.mkdir(parents=True, exist_ok=True)
    for p in [prompt_variants_dir, runs_root, results_index_root]:
        ensure_within_root(experiment_root, p, "resolved_path")

    print(
        "[SETUP] resolved_paths "
        f"experiment_root={experiment_root} "
        f"prompt_variants_dir={prompt_variants_dir} "
        f"runs_root={runs_root} "
        f"results_index_root={results_index_root}"
    )

    if args.max_steps > 100:
        print(f"[INFO] Capping max_steps from {args.max_steps} to 100 for spatial experiments.")
        args.max_steps = 100
    if args.planning_mode == "off":
        print("[INFO] planning_mode=off is deprecated for experiments; using lookahead_actions with horizon 1.")
        args.planning_mode = "lookahead_actions"
        args.planning_horizon_x = 1

    prompt_path = write_prompt_variant(prompt_variants_dir=prompt_variants_dir)
    print(f"[SETUP] prompt={prompt_path}")

    if args.dry_run:
        print(
            "[DRY-RUN] would run "
            f"games={args.games} model={args.model} max_steps={args.max_steps} num_runs={args.num_runs} level={args.level} "
            f"qwen_thinking={args.qwen_thinking} translator={args.translator_mode} causal={args.causal_mode} "
            f"scm_bootstrap={args.scm_bootstrap} planning={args.planning_mode} planx={args.planning_horizon_x}"
        )
    else:
        run_condition(
            games=args.games,
            model=args.model,
            max_steps=args.max_steps,
            num_runs=args.num_runs,
            level=args.level,
            qwen_thinking=args.qwen_thinking,
            translator_mode=args.translator_mode,
            causal_mode=args.causal_mode,
            scm_bootstrap=args.scm_bootstrap,
            planning_mode=args.planning_mode,
            planning_horizon_x=args.planning_horizon_x,
            max_workers=args.max_workers,
            shuffle_action_meanings=args.shuffle_action_meanings,
            action_shuffle_seed=args.action_shuffle_seed,
            per_run_budget_nonthinking_seconds=args.per_run_budget_nonthinking_seconds,
            per_run_budget_thinking_seconds=args.per_run_budget_thinking_seconds,
            experiment_root=experiment_root,
            prompt_variants_dir=prompt_variants_dir,
            runs_root=runs_root,
            results_index_root=results_index_root,
        )

    if args.dry_run:
        print(f"[DONE] Spatial side-goal experiment root: {experiment_root}")
        return

    print(f"[DONE] Spatial side-goal experiment root: {experiment_root}")


if __name__ == "__main__":
    main()
