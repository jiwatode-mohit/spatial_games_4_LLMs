import matplotlib.pyplot as plt
from collections import Counter
import json
import os
import csv
import math
import numpy as np
from typing import Optional, Dict, Any

def compute_token_metrics(step_log):
    input_tokens_by_step = []
    output_tokens_by_step = []
    total_tokens_by_step = []
    token_usage_source_by_step = []
    reasoning_tokens_by_step = []
    cached_input_tokens_by_step = []
    step_time_seconds_by_step = []

    sum_input = 0
    sum_output = 0
    sum_total = 0
    num_with_usage = 0
    num_estimated = 0
    num_unavailable = 0
    sum_step_time_seconds = 0.0

    for entry in step_log:
        source = entry.get("token_usage_source", "unavailable")
        input_tokens = int(entry.get("input_tokens", 0) or 0)
        output_tokens = int(entry.get("output_tokens", 0) or 0)
        total_tokens = int(entry.get("total_tokens", input_tokens + output_tokens) or 0)
        reasoning_tokens = entry.get("reasoning_tokens", None)
        cached_input_tokens = entry.get("cached_input_tokens", None)
        step_time_seconds = float(entry.get("step_time_seconds", 0.0) or 0.0)

        input_tokens_by_step.append(input_tokens)
        output_tokens_by_step.append(output_tokens)
        total_tokens_by_step.append(total_tokens)
        token_usage_source_by_step.append(source)
        reasoning_tokens_by_step.append(reasoning_tokens)
        cached_input_tokens_by_step.append(cached_input_tokens)
        step_time_seconds_by_step.append(step_time_seconds)

        if source != "unavailable":
            num_with_usage += 1
        if source == "estimated":
            num_estimated += 1
        if source == "unavailable":
            num_unavailable += 1

        sum_input += input_tokens
        sum_output += output_tokens
        sum_total += total_tokens
        sum_step_time_seconds += step_time_seconds

    mean_step_time_seconds = (sum_step_time_seconds / len(step_log)) if step_log else 0.0

    return {
        "input_tokens_by_step": input_tokens_by_step,
        "output_tokens_by_step": output_tokens_by_step,
        "total_tokens_by_step": total_tokens_by_step,
        "token_usage_source_by_step": token_usage_source_by_step,
        "reasoning_tokens_by_step": reasoning_tokens_by_step,
        "cached_input_tokens_by_step": cached_input_tokens_by_step,
        "step_time_seconds_by_step": step_time_seconds_by_step,
        "sum_input_tokens": sum_input,
        "sum_output_tokens": sum_output,
        "sum_total_tokens": sum_total,
        "sum_step_time_seconds": round(sum_step_time_seconds, 6),
        "mean_step_time_seconds": round(mean_step_time_seconds, 6),
        "num_steps_with_usage": num_with_usage,
        "num_steps_estimated": num_estimated,
        "num_steps_unavailable": num_unavailable,
    }


def _extract_avatar_pos(state: str):
    for y, row in enumerate((state or "").splitlines()):
        for x, ch in enumerate(row):
            if "a" in ch.lower() or "avatar" in ch.lower():
                return (y, x)
    return None


def _detect_entity_disappearance(s_t: str, s_tp1: str) -> bool:
    def flatten_exclude_avatar(state):
        return [ch for row in state for ch in row if not ("a" in ch.lower() or "avatar" in ch.lower())]

    count_t = Counter(flatten_exclude_avatar(s_t or ""))
    count_tp1 = Counter(flatten_exclude_avatar(s_tp1 or ""))
    return any(count_tp1.get(k, 0) < count_t[k] for k in count_t)


def _shannon_entropy(actions):
    if not actions:
        return 0.0
    c = Counter(actions)
    n = float(sum(c.values()))
    if n <= 0:
        return 0.0
    ent = 0.0
    for v in c.values():
        p = v / n
        if p > 0:
            ent -= p * math.log(p, 2)
    return float(ent)


def compute_advanced_step_metrics(
    states,
    step_log,
    catastrophic_threshold: float = -1.0,
    entropy_window: int = 20,
    run_metadata: Optional[Dict[str, Any]] = None,
):
    max_steps = min(len(step_log), len(states) - 1)
    if max_steps <= 0:
        return {
            "action_change_rate": 0.0,
            "action_entropy_windowed": 0.0,
            "non_progress_streak_p95": 0.0,
            "effective_step_ratio": 0.0,
            "decision_latency_p50": 0.0,
            "decision_latency_p95": 0.0,
            "replan_trigger_rate": 0.0,
            "scm_revision_intensity": 0.0,
            "scm_consistency_rate": None,
            "catastrophic_step_rate": 0.0,
            "repeat_state_repeat_action_rate": 0.0,
            "entropy_series_windowed": [],
        }

    actions = [step_log[t].get("action", 0) for t in range(max_steps)]
    rewards = [float(step_log[t].get("reward", 0.0) or 0.0) for t in range(max_steps)]
    step_times = [float(step_log[t].get("step_time_seconds", 0.0) or 0.0) for t in range(max_steps)]
    replan_flags = [bool(step_log[t].get("early_replan_triggered", False)) for t in range(max_steps)]
    scm_flags = [step_log[t].get("scm_consistent", None) for t in range(max_steps)]

    action_changes = 0
    for i in range(1, len(actions)):
        if actions[i] != actions[i - 1]:
            action_changes += 1
    action_change_rate = (action_changes / max(1, len(actions) - 1))

    entropy_series = []
    for i in range(max_steps):
        l = max(0, i - entropy_window + 1)
        entropy_series.append(_shannon_entropy(actions[l:i + 1]))
    action_entropy_windowed = float(np.mean(entropy_series)) if entropy_series else 0.0

    effective_flags = []
    non_progress_streaks = []
    cur_streak = 0
    repeat_visits = 0
    repeat_same_action = 0
    seen_state_to_action = {}

    for t in range(max_steps):
        s_t = states[t] or ""
        s_tp1 = states[t + 1] or ""
        pos_t = _extract_avatar_pos(s_t)
        pos_tp1 = _extract_avatar_pos(s_tp1)
        reward_triggered = rewards[t] != 0
        entity_triggered = _detect_entity_disappearance(s_t, s_tp1)
        moved = pos_t != pos_tp1
        effective = moved or reward_triggered or entity_triggered
        effective_flags.append(bool(effective))

        if effective:
            if cur_streak > 0:
                non_progress_streaks.append(cur_streak)
            cur_streak = 0
        else:
            cur_streak += 1

        state_key = s_t
        if state_key in seen_state_to_action:
            repeat_visits += 1
            if seen_state_to_action[state_key] == actions[t]:
                repeat_same_action += 1
        seen_state_to_action[state_key] = actions[t]

    if cur_streak > 0:
        non_progress_streaks.append(cur_streak)

    effective_step_ratio = float(np.mean(effective_flags)) if effective_flags else 0.0
    non_progress_streak_p95 = float(np.percentile(non_progress_streaks, 95)) if non_progress_streaks else 0.0
    decision_latency_p50 = float(np.percentile(step_times, 50)) if step_times else 0.0
    decision_latency_p95 = float(np.percentile(step_times, 95)) if step_times else 0.0
    replan_trigger_rate = float(np.mean(replan_flags)) if replan_flags else 0.0
    catastrophic_step_rate = float(np.mean([r <= catastrophic_threshold for r in rewards])) if rewards else 0.0
    repeat_state_repeat_action_rate = (repeat_same_action / repeat_visits) if repeat_visits > 0 else 0.0

    scm_consistency_vals = [v for v in scm_flags if isinstance(v, bool)]
    scm_consistency_rate = float(np.mean(scm_consistency_vals)) if scm_consistency_vals else None

    # Online SCM updates are removed; keep this metric for schema compatibility.
    scm_revision_intensity = 0.0

    return {
        "action_change_rate": round(action_change_rate, 6),
        "action_entropy_windowed": round(action_entropy_windowed, 6),
        "non_progress_streak_p95": round(non_progress_streak_p95, 6),
        "effective_step_ratio": round(effective_step_ratio, 6),
        "decision_latency_p50": round(decision_latency_p50, 6),
        "decision_latency_p95": round(decision_latency_p95, 6),
        "replan_trigger_rate": round(replan_trigger_rate, 6),
        "scm_revision_intensity": round(scm_revision_intensity, 6),
        "scm_consistency_rate": round(scm_consistency_rate, 6) if scm_consistency_rate is not None else None,
        "catastrophic_step_rate": round(catastrophic_step_rate, 6),
        "repeat_state_repeat_action_rate": round(repeat_state_repeat_action_rate, 6),
        "entropy_series_windowed": [round(v, 6) for v in entropy_series],
    }

def generate_reward_report(
    reflection_manager,
    output_dir,
    winner=None,
    action_meanings: Optional[Dict[int, str]] = None,
    default_action: int = 0
):
    """Generate and save reward trend and action distribution from reflection_manager."""
    reward_history = [entry["reward"] for entry in reflection_manager.step_log]
    action_history = [entry["action"] for entry in reflection_manager.step_log]

    print(f"\n=== Game analysis ===")
    print(f"Total steps: {len(reflection_manager.step_log)}")
    print(f"Total reward: {sum(reward_history)}")
    if winner is not None:
        print(f"Winner: {winner}")

    plt.figure(figsize=(12, 5))

    # Reward trend
    plt.subplot(121)
    plt.plot(reward_history)
    plt.title("Reward Trend")

    # Action distribution
    plt.subplot(122)
    action_dist = Counter(action_history)
    if action_meanings:
        ordered_actions = sorted(int(a) for a in action_meanings.keys())
    else:
        ordered_actions = sorted(action_dist.keys())

    counts = [action_dist.get(a, 0) for a in ordered_actions]
    labels = []
    for a in ordered_actions:
        meaning = None
        if action_meanings:
            meaning = action_meanings.get(a)
            if meaning is None:
                meaning = action_meanings.get(str(a))
        meaning = meaning or f"ACTION_{a}"
        suffix = " (default/fallback)" if a == default_action else ""
        labels.append(f"{a}: {meaning}{suffix}")

    plt.bar(range(len(ordered_actions)), counts)
    plt.xticks(range(len(ordered_actions)), labels, rotation=45, ha="right")
    plt.xlabel("Action ID and Meaning")
    plt.title("Action Distribution")

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "game_analysis.png"))
    plt.close()

def save_step_metrics_json(
    path,
    step_flags,
    positions,
    step_log,
    key="meaningful",
    winner=None,
    runtime_info: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
):
    """Save boolean list and avatar positions as step-wise metrics json."""
    if len(step_flags) == 0:
        step_ratio = 0.0
    else:
        step_ratio = sum(step_flags) / len(step_flags)
    metrics = {
        f"{key}_steps": step_flags,
        f"{key}_step_ratio": step_ratio,
        "avatar_positions": [[pos] for pos in positions], # Transform into n*1 shape
        "actions_by_step": [
            {
                "action": entry.get("action"),
                "action_meaning": entry.get("action_meaning", f"ACTION_{entry.get('action', 0)}"),
            }
            for entry in step_log
        ],
    }
    metrics["token_metrics"] = compute_token_metrics(step_log)
    metrics["advanced_step_metrics"] = compute_advanced_step_metrics(
        states=run_metadata.get("states_for_metrics", []) if isinstance(run_metadata, dict) else [],
        step_log=step_log,
        run_metadata=run_metadata,
    )
    if winner is not None:
        metrics["winner"] = winner
    if runtime_info:
        metrics["runtime"] = runtime_info
    if run_metadata:
        safe_metadata = dict(run_metadata)
        safe_metadata.pop("states_for_metrics", None)
        metrics["run_metadata"] = safe_metadata
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def analyze_meaningful_steps(states, step_log):
    def extract_avatar_pos(state):
        # Removed vertical flipping logic and debug print
        for y, row in enumerate(state.splitlines()): # Split lines here directly
            for x, ch in enumerate(row):
                if 'a' in ch.lower() or 'avatar' in ch.lower():
                    return (y, x) # Return raw y
        return None

    def detect_entity_disappearance(s_t, s_tp1):
        def flatten_exclude_avatar(state):
            return [ch for row in state for ch in row if not ('a' in ch.lower() or 'avatar' in ch.lower())]
        count_t = Counter(flatten_exclude_avatar(s_t))
        count_tp1 = Counter(flatten_exclude_avatar(s_tp1))
        return any(count_tp1.get(k, 0) < count_t[k] for k in count_t)

    flags = []
    pos_prev = None

    max_steps = min(len(step_log), len(states) - 1)

    for t in range(max_steps):
        entry = step_log[t]
        s_t = states[t]
        s_tp1 = states[t + 1]
        a_t = entry["action"]
        r_tp1 = entry["reward"]

        pos_t = extract_avatar_pos(s_t)
        pos_tp1 = extract_avatar_pos(s_tp1)

        reward_triggered = r_tp1 != 0
        entity_triggered = detect_entity_disappearance(s_t, s_tp1)
        canceling = (pos_prev == pos_tp1 and pos_t != pos_tp1)

        if a_t == 0 or (pos_t == pos_tp1 and not reward_triggered and not entity_triggered) or canceling:
            meaningful = False
        else:
            meaningful = True

        flags.append(meaningful)
        pos_prev = pos_t
        # Removed debug prints for individual step positions

    positions = [extract_avatar_pos(states[t]) for t in range(max_steps)] # Collect pos_t for each step
    # Removed debug prints for the final list
    return flags, sum(flags) / len(flags) if flags else 0.0, positions


def save_step_metrics_csv(
    states,
    step_log,
    output_path,
    winner=None,
    runtime_info: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
):
    """Save full step-by-step metrics to CSV for analysis."""

    def extract_avatar_pos(state):
        # Removed vertical flipping logic and debug print
        for y, row in enumerate(state.splitlines()): # Split lines here directly
            for x, ch in enumerate(row):
                if 'a' in ch.lower() or 'avatar' in ch.lower():
                    return (y, x) # Return raw y
        return None

    def detect_entity_disappearance(s_t, s_tp1):
        def flatten_exclude_avatar(state):
            return [ch for row in state for ch in row if not ('a' in ch.lower() or 'avatar' in ch.lower())]
        count_t = Counter(flatten_exclude_avatar(s_t))
        count_tp1 = Counter(flatten_exclude_avatar(s_tp1))
        return any(count_tp1.get(k, 0) < count_t[k] for k in count_t)

    rows = []
    pos_prev = None

    max_steps = min(len(step_log), len(states) - 1)
    
    # 确定所有可能的字段
    fieldnames = ["step", "action", "action_meaning", "reward", "avatar_pos_before", "avatar_pos_after", 
                 "reward_triggered", "entity_triggered", "meaningful"]
    if winner is not None:
        fieldnames.append("winner")
    if runtime_info is not None:
        fieldnames.extend([
            "run_started_at",
            "run_finished_at",
            "runtime_seconds",
            "runtime_minutes",
        ])
    fieldnames.extend([
        "step_time_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "token_usage_source",
        "reasoning_tokens",
        "cached_input_tokens",
        "sum_input_tokens",
        "sum_output_tokens",
        "sum_total_tokens",
        "sum_step_time_seconds",
        "mean_step_time_seconds",
        "num_steps_with_usage",
        "num_steps_estimated",
        "num_steps_unavailable",
        "action_change_rate",
        "action_entropy_windowed",
        "non_progress_streak_p95",
        "effective_step_ratio",
        "decision_latency_p50",
        "decision_latency_p95",
        "replan_trigger_rate",
        "scm_revision_intensity",
        "scm_consistency_rate",
        "catastrophic_step_rate",
        "repeat_state_repeat_action_rate",
    ])

    token_metrics = compute_token_metrics(step_log)
    advanced_metrics = compute_advanced_step_metrics(states=states, step_log=step_log, run_metadata=run_metadata)

    for t in range(max_steps):
        entry = step_log[t]
        s_t = states[t]
        s_tp1 = states[t + 1]
        a_t = entry["action"]
        r_tp1 = entry["reward"]

        pos_t = extract_avatar_pos(s_t)
        pos_tp1 = extract_avatar_pos(s_tp1)

        reward_triggered = r_tp1 != 0
        entity_triggered = detect_entity_disappearance(s_t, s_tp1)
        canceling = (pos_prev == pos_tp1 and pos_t != pos_tp1)

        meaningful = not (
            a_t == 0 or
            (pos_t == pos_tp1 and not reward_triggered and not entity_triggered) or
            canceling
        )

        row_data = {
            "step": t,
            "action": a_t,
            "action_meaning": entry.get("action_meaning", f"ACTION_{a_t}"),
            "reward": r_tp1,
            "avatar_pos_before": pos_t,
            "avatar_pos_after": pos_tp1,
            "reward_triggered": reward_triggered,
            "entity_triggered": entity_triggered,
            "meaningful": meaningful,
            "step_time_seconds": token_metrics["step_time_seconds_by_step"][t] if t < len(token_metrics["step_time_seconds_by_step"]) else 0.0,
            "input_tokens": token_metrics["input_tokens_by_step"][t] if t < len(token_metrics["input_tokens_by_step"]) else 0,
            "output_tokens": token_metrics["output_tokens_by_step"][t] if t < len(token_metrics["output_tokens_by_step"]) else 0,
            "total_tokens": token_metrics["total_tokens_by_step"][t] if t < len(token_metrics["total_tokens_by_step"]) else 0,
            "token_usage_source": token_metrics["token_usage_source_by_step"][t] if t < len(token_metrics["token_usage_source_by_step"]) else "unavailable",
            "reasoning_tokens": token_metrics["reasoning_tokens_by_step"][t] if t < len(token_metrics["reasoning_tokens_by_step"]) else None,
            "cached_input_tokens": token_metrics["cached_input_tokens_by_step"][t] if t < len(token_metrics["cached_input_tokens_by_step"]) else None,
            "sum_input_tokens": token_metrics["sum_input_tokens"],
            "sum_output_tokens": token_metrics["sum_output_tokens"],
            "sum_total_tokens": token_metrics["sum_total_tokens"],
            "sum_step_time_seconds": token_metrics["sum_step_time_seconds"],
            "mean_step_time_seconds": token_metrics["mean_step_time_seconds"],
            "num_steps_with_usage": token_metrics["num_steps_with_usage"],
            "num_steps_estimated": token_metrics["num_steps_estimated"],
            "num_steps_unavailable": token_metrics["num_steps_unavailable"],
            "action_change_rate": advanced_metrics["action_change_rate"],
            "action_entropy_windowed": advanced_metrics["action_entropy_windowed"],
            "non_progress_streak_p95": advanced_metrics["non_progress_streak_p95"],
            "effective_step_ratio": advanced_metrics["effective_step_ratio"],
            "decision_latency_p50": advanced_metrics["decision_latency_p50"],
            "decision_latency_p95": advanced_metrics["decision_latency_p95"],
            "replan_trigger_rate": advanced_metrics["replan_trigger_rate"],
            "scm_revision_intensity": advanced_metrics["scm_revision_intensity"],
            "scm_consistency_rate": advanced_metrics["scm_consistency_rate"],
            "catastrophic_step_rate": advanced_metrics["catastrophic_step_rate"],
            "repeat_state_repeat_action_rate": advanced_metrics["repeat_state_repeat_action_rate"],
        }
        
        # 如果有winner信息，添加到每一行
        if winner is not None:
            row_data["winner"] = winner if t == max_steps - 1 else ""
        if runtime_info is not None:
            row_data["run_started_at"] = runtime_info.get("run_started_at")
            row_data["run_finished_at"] = runtime_info.get("run_finished_at")
            row_data["runtime_seconds"] = runtime_info.get("runtime_seconds")
            row_data["runtime_minutes"] = runtime_info.get("runtime_minutes")
            
        rows.append(row_data)

        pos_prev = pos_t

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_full_analysis_report(
    reflection_manager,
    states,
    output_dir,
    winner=None,
    runtime_info: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    action_meanings: Optional[Dict[int, str]] = None,
    default_action: int = 0,
):
    """Master function to generate all outputs from reflection_manager + states."""
    os.makedirs(output_dir, exist_ok=True)

    generate_reward_report(
        reflection_manager,
        output_dir,
        winner,
        action_meanings=action_meanings,
        default_action=default_action
    )

    step_log = reflection_manager.step_log
    step_flags, _, positions = analyze_meaningful_steps(states, step_log) # Capture positions
    run_metadata_payload = dict(run_metadata or {})
    run_metadata_payload["states_for_metrics"] = states
    save_step_metrics_json(
        os.path.join(output_dir, "step_metrics.json"),
        step_flags,
        positions,
        step_log=step_log,
        winner=winner,
        runtime_info=runtime_info,
        run_metadata=run_metadata_payload,
    ) # Pass positions

    save_step_metrics_csv(
        states=states,
        step_log=step_log,
        output_path=os.path.join(output_dir, "step_metrics.csv"),
        winner=winner,
        runtime_info=runtime_info,
        run_metadata=run_metadata_payload,
    )
