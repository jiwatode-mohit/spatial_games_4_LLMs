#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROJECT_DIR = ROOT / "project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from llm.agent.llm_agent import LLMPlayer
from llm.agent.llm_translator import LLMTranslator
from llm.client import create_client_from_config
from llm.utils.agent_components import parse_action_from_response_with_meta, parse_vgdl
from llm.utils.build_prompt import build_dynamic_prompt, build_static_prompt
from llm.utils.config import get_profile_config
from llm.utils.vgdl_utils import load_level_map, load_vgdl_rules


DEFAULT_GAMES = ["spatialgame1_v0", "spatialgame2_v0", "spatialgame3_v0"]
ACTION_MAP = {
    0: "ACTION_NIL",
    1: "ACTION_LEFT",
    2: "ACTION_RIGHT",
    3: "ACTION_DOWN",
    4: "ACTION_UP",
}
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
MIN_CONTEXT_TOKENS = 16000
DEFAULT_MODEL_SUITE = [
    "local-qwen3-0.6b-vllm",
    "local-qwen3-1.7b-vllm",
    "local-qwen3-4b-vllm",
    "local-qwen3-8b-vllm",
]


@dataclass(frozen=True)
class CaseSpec:
    game: str
    level: int
    case_index: int
    prompt_state_ascii: str
    answer_row: int
    answer_col: int
    original_row: int
    original_col: int
    level_path: str


def sanitize_thinking_output(text: str) -> Tuple[str, bool, bool]:
    return LLMPlayer._sanitize_thinking_output(str(text or ""))


def configure_client_context_window(client: Any, min_context_tokens: int = MIN_CONTEXT_TOKENS) -> None:
    for attr in ("max_context_tokens", "max_model_len"):
        if hasattr(client, attr):
            try:
                current = int(getattr(client, attr) or 0)
            except Exception:
                current = 0
            try:
                setattr(client, attr, max(current, int(min_context_tokens)))
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run avatar-position extraction experiment for spatial games.")
    parser.add_argument("--model", default="local-qwen3-8b-vllm")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit list of model profiles to evaluate.")
    parser.add_argument("--model_suite", choices=["all", "local_vllm"], default=None, help="Run a predefined model suite.")
    parser.add_argument("--client_types", nargs="+", default=None, help="Optional client_type filter when expanding a suite.")
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    parser.add_argument("--thinking_modes", nargs="+", choices=["on", "off"], default=["off", "on"])
    parser.add_argument("--cases_per_level", type=int, default=10)
    parser.add_argument("--translator_mode", choices=["on", "off", "vgdl"], default="on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", default=str(ROOT / "positioning"))
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--print_live_io", action="store_true", help="Print prompt/raw/sanitized outputs live in the terminal.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--keep_going", action="store_true", help="Continue to the next model if one model run fails.")
    parser.add_argument("--max_cases", type=int, default=0, help="Debug cap after case generation; 0 means no cap.")
    return parser.parse_args()


def load_llm_profiles() -> Dict[str, Dict[str, Any]]:
    data = json.loads((ROOT / "project" / "llm_config.json").read_text(encoding="utf-8"))
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def resolve_model_list(args: argparse.Namespace) -> List[str]:
    if args.models:
        return list(dict.fromkeys(str(model) for model in args.models))

    profiles = load_llm_profiles()
    items = list(profiles.items())

    if args.model_suite == "local_vllm":
        items = [(name, cfg) for name, cfg in items if str(cfg.get("client_type", "")) == "vllm"]
    elif args.model_suite == "all":
        wanted = set(DEFAULT_MODEL_SUITE)
        items = [(name, cfg) for name, cfg in items if name in wanted]
    else:
        return [str(args.model)]

    if args.client_types:
        allowed = {str(x) for x in args.client_types}
        items = [(name, cfg) for name, cfg in items if str(cfg.get("client_type", "")) in allowed]

    models = [name for name, _ in items]
    if not models:
        raise RuntimeError("No models matched the requested suite/client_type filters.")
    return models


def resolve_qwen_sampling_overrides(model_name_full: str, qwen_thinking: str) -> Dict[str, Any]:
    try:
        profile = get_profile_config(model_name_full)
        client_type = profile.get("client_type")
        resolved_model = str(profile.get("model", "")).lower()
        is_qwen3_vllm = (client_type == "vllm") and ("qwen3" in resolved_model)
    except Exception:
        is_qwen3_vllm = False

    if not is_qwen3_vllm:
        return {}
    if qwen_thinking == "on":
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
    if qwen_thinking == "off":
        return {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}
    return {}


def discover_levels(game: str) -> List[int]:
    game_name = game.replace("_v0", "")
    game_dir = ROOT / "gym_gvgai" / "envs" / "games" / f"{game_name}_v0"
    levels: List[int] = []
    for path in sorted(game_dir.glob(f"{game_name}_lvl*.txt")):
        match = re.search(r"_lvl(\d+)\.txt$", path.name)
        if match:
            levels.append(int(match.group(1)))
    if not levels:
        raise RuntimeError(f"No level files found for {game}")
    return levels


def build_prompt_instruction() -> str:
    return (
        "=== Output Contract ===\n"
        "Return exactly 2 lines and nothing else.\n"
        "Line 1: Action:<integer_id>\n"
        "Line 2: Feedback: Position(row=<int>, col=<int>); Summary:<max 12 words, no action words left/right/up/down/use>\n\n"
        "=== Decision Rules ===\n"
        "- Action must be a valid numeric ID from Action Legend.\n"
        "- Avatar may appear as avatar/nokey/withkey or any *avatar* alias in Sprite Mapping; treat all as self.\n"
        "- Feedback must use exactly this structure: Position(row=<int>, col=<int>); Summary:<text>.\n"
        "- The position must be the avatar's current position in Current State, not the next position after the action.\n"
        "- Count rows and columns over the full grid, including wall tiles.\n"
        "- Coordinates are zero-based: top row is row 0 and leftmost column is col 0.\n"
        "- Summary should briefly state immediate progress toward the objective.\n"
        "- If previous action caused no movement and no reward, prefer a different action ID unless game mechanics require repeating direction.\n"
        "- If directional actions may rotate before moving, allow one retry of same direction only when consistent with last transition.\n"
        "- If uncertain about the action, still output the safest valid action ID, but keep the position exact.\n"
        "- No markdown, no extra text, no JSON, no reasoning."
    )


def write_prompt_variant(run_dir: Path) -> Path:
    source_path = ROOT / "project" / "llm" / "utils" / "prompt_templates" / "prompt.json"
    cfg = json.loads(source_path.read_text(encoding="utf-8"))
    cfg["instruction"] = build_prompt_instruction()
    prompt_dir = run_dir / "prompt_variants"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / "prompt__positioning.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return path


def parse_level_mapping(vgdl_rules: str) -> Dict[str, List[str]]:
    _, level_mapping = parse_vgdl(vgdl_rules)
    return {str(k): [str(x) for x in v] for k, v in level_mapping.items()}


def char_label_for_prompt(char: str, sprites: Sequence[str]) -> str:
    lowered = [s.lower() for s in sprites]
    for alias in ("nokey", "withkey", "avatar"):
        if alias in lowered:
            return alias
    for label in ("goal", "exit", "key", "redkey", "bluekey", "reddoor", "bluedoor", "door", "wall", "floor", "background"):
        if label in lowered:
            return label
    for sprite in sprites:
        if sprite.lower() not in {"floor", "background"}:
            return sprite
    return sprites[0] if sprites else char


def build_sprite_mapping(level_mapping: Dict[str, List[str]]) -> Dict[str, str]:
    sprite_mapping: Dict[str, str] = {}
    preferred = ["avatar", "nokey", "withkey", "goal", "exit", "key", "redkey", "bluekey", "reddoor", "bluedoor", "door", "wall", "floor", "background"]
    for sprite_name in preferred:
        for char, sprites in level_mapping.items():
            if sprite_name in [s.lower() for s in sprites] and sprite_name not in sprite_mapping:
                sprite_mapping[sprite_name] = char
    for char, sprites in level_mapping.items():
        label = char_label_for_prompt(char, sprites)
        sprite_mapping.setdefault(label, char)
    return sprite_mapping


def find_avatar_char(level_mapping: Dict[str, List[str]]) -> str:
    for char, sprites in level_mapping.items():
        lowered = [s.lower() for s in sprites]
        if any(alias in lowered for alias in ("avatar", "nokey", "withkey")):
            return char
    raise RuntimeError("Avatar character not found in level mapping.")


def find_empty_chars(level_mapping: Dict[str, List[str]]) -> List[str]:
    empty_chars: List[str] = []
    for char, sprites in level_mapping.items():
        lowered = {s.lower() for s in sprites}
        if lowered.issubset({"floor", "background"}):
            empty_chars.append(char)
    if not empty_chars:
        raise RuntimeError("No empty tile character found in level mapping.")
    return empty_chars


def level_to_lines(level_text: str) -> List[List[str]]:
    return [list(line.rstrip("\n")) for line in level_text.splitlines() if line.strip()]


def locate_avatar(lines: Sequence[Sequence[str]], avatar_char: str) -> Tuple[int, int]:
    for row_idx, row in enumerate(lines):
        for col_idx, char in enumerate(row):
            if char == avatar_char:
                return row_idx, col_idx
    raise RuntimeError(f"Avatar char {avatar_char!r} not found in level.")


def enumerate_empty_tiles(lines: Sequence[Sequence[str]], empty_chars: Sequence[str]) -> List[Tuple[int, int]]:
    allowed = set(empty_chars)
    coords: List[Tuple[int, int]] = []
    for row_idx, row in enumerate(lines):
        for col_idx, char in enumerate(row):
            if char in allowed:
                coords.append((row_idx, col_idx))
    return coords


def sample_case_positions(
    game: str,
    level: int,
    empty_tiles: Sequence[Tuple[int, int]],
    cases_per_level: int,
    seed: int,
) -> List[Tuple[int, int]]:
    if len(empty_tiles) < cases_per_level:
        raise RuntimeError(
            f"{game} level {level} only has {len(empty_tiles)} empty tiles; need {cases_per_level}."
        )
    key = f"{game}|{level}|{seed}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    coords = list(empty_tiles)
    rng.shuffle(coords)
    return coords[:cases_per_level]


def relocate_avatar(
    lines: Sequence[Sequence[str]],
    original_pos: Tuple[int, int],
    new_pos: Tuple[int, int],
    avatar_char: str,
    empty_char: str,
) -> str:
    grid = [list(row) for row in lines]
    grid[original_pos[0]][original_pos[1]] = empty_char
    grid[new_pos[0]][new_pos[1]] = avatar_char
    return "\n".join("".join(row) for row in grid)


def build_prompt_for_case(
    translated_rules: str,
    prompt_template_path: Path,
    prompt_state_ascii: str,
    sprite_mapping: Dict[str, str],
) -> str:
    static_prompt = build_static_prompt(
        vgdl_rules=translated_rules,
        action_map=ACTION_MAP,
        optional_prompt="",
        prompt_template_path=str(prompt_template_path),
    )
    dynamic_prompt = build_dynamic_prompt(
        current_ascii=prompt_state_ascii,
        last_ascii=None,
        current_image_path=None,
        last_image_path=None,
        avatar_position=None,
        last_position=None,
        action_map=ACTION_MAP,
        action_history=[],
        reflection_manager=None,
        prompt_template_path=str(prompt_template_path),
        logger=None,
        sprite_mapping=sprite_mapping,
        plan="",
        scm_context="",
        llm_model_name="gpt-3.5-turbo",
        rotate=False,
        expanded=False,
    )
    return f"{static_prompt}\n\n{dynamic_prompt}".strip()


def extract_feedback_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.lower().startswith("feedback:"):
            return line.strip()
    return ""


def parse_feedback_position(text: str) -> Optional[Tuple[int, int]]:
    search_spaces = [extract_feedback_line(text), str(text or "")]
    for candidate in search_spaces:
        if not candidate:
            continue
        match = POSITION_ROW_COL_PATTERN.search(candidate)
        if match:
            return int(match.group(1)), int(match.group(2))
        match = POSITION_X_Y_PATTERN.search(candidate)
        if match:
            # x/y convention maps to col/row, so convert to (row, col).
            return int(match.group(2)), int(match.group(1))
        match = BARE_POSITION_TUPLE_PATTERN.search(candidate)
        if match:
            # Bare tuples are treated as (x, y) -> (row, col) = (y, x).
            return int(match.group(2)), int(match.group(1))
    return None


def strict_contract_ok(text: str) -> bool:
    lines = [line.rstrip() for line in str(text or "").strip().splitlines() if line.strip()]
    if len(lines) != 2:
        return False
    if not re.fullmatch(r"Action:\d+", lines[0]):
        return False
    return bool(
        re.fullmatch(
            r"Feedback:\s*Position\(row=-?\d+,\s*col=-?\d+\);\s*Summary:.+",
            lines[1],
        )
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else 0.0


def aggregate_rows(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        items = grouped[key]
        summary: Dict[str, Any] = {field: value for field, value in zip(group_fields, key)}
        summary["cases"] = len(items)
        summary["raw_contract_rate"] = round(mean_or_zero(r["raw_contract_ok"] for r in items), 6)
        summary["sanitized_contract_rate"] = round(mean_or_zero(r["sanitized_contract_ok"] for r in items), 6)
        summary["action_parse_rate"] = round(mean_or_zero(r["action_parse_success"] for r in items), 6)
        summary["position_parse_rate"] = round(mean_or_zero(r["position_parse_success"] for r in items), 6)
        summary["row_match_rate"] = round(mean_or_zero(r["row_match"] for r in items), 6)
        summary["col_match_rate"] = round(mean_or_zero(r["col_match"] for r in items), 6)
        summary["exact_position_rate"] = round(mean_or_zero(r["exact_position_match"] for r in items), 6)
        summary["had_think_rate"] = round(mean_or_zero(r["had_think"] for r in items), 6)
        summary["residual_think_rate"] = round(mean_or_zero(r["residual_think"] for r in items), 6)
        summary["mean_latency_seconds"] = round(mean_or_zero(r["latency_seconds"] for r in items), 6)
        summary["mean_input_tokens"] = round(mean_or_zero(r["input_tokens"] for r in items), 3)
        summary["mean_output_tokens"] = round(mean_or_zero(r["output_tokens"] for r in items), 3)
        summary["mean_total_tokens"] = round(mean_or_zero(r["total_tokens"] for r in items), 3)
        summary["total_input_tokens"] = int(sum(int(r["input_tokens"]) for r in items))
        summary["total_output_tokens"] = int(sum(int(r["output_tokens"]) for r in items))
        summary["total_tokens"] = int(sum(int(r["total_tokens"]) for r in items))
        summary_rows.append(summary)
    return summary_rows


def normalize_run_rows(rows: Sequence[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["model"] = model_name
        normalized.append(out)
    return normalized


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def print_live_io_block(
    *,
    case: CaseSpec,
    thinking: str,
    state_ascii: str,
    response_raw: str,
    sanitized_response: str,
    predicted_position: Optional[Tuple[int, int]],
) -> None:
    header = (
        f"=== LIVE IO | game={case.game} lvl={case.level} case={case.case_index} "
        f"thinking={thinking} actual_position=({case.answer_row},{case.answer_col}) ==="
    )
    print(header, flush=True)
    print("=== STATE ===", flush=True)
    print(state_ascii, flush=True)
    print(f"=== SCRIPT POSITION (zero-based) === ({case.answer_row}, {case.answer_col})", flush=True)
    print("=== OUTPUT RAW ===", flush=True)
    print(response_raw if response_raw else "<empty>", flush=True)
    print("=== OUTPUT SANITIZED ===", flush=True)
    print(sanitized_response if sanitized_response else "<empty>", flush=True)
    if predicted_position is None:
        print("=== LLM PARSED POSITION === <none>", flush=True)
    else:
        print(f"=== LLM PARSED POSITION === ({predicted_position[0]}, {predicted_position[1]})", flush=True)
    print("=== END LIVE IO ===", flush=True)


def build_readme(
    args: argparse.Namespace,
    run_dir: Path,
    per_case_rows: Sequence[Dict[str, Any]],
    overall_summary: Sequence[Dict[str, Any]],
    levels_summary: Sequence[Dict[str, Any]],
) -> str:
    overall = overall_summary[0] if overall_summary else {}
    lines = [
        "# Spatial Positioning Experiment",
        "",
        f"- Timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Model: `{args.model}`",
        f"- Thinking modes: `{', '.join(args.thinking_modes)}`",
        f"- Translator mode: `{args.translator_mode}`",
        f"- Cases per level: `{args.cases_per_level}`",
        f"- Games: `{', '.join(args.games)}`",
        f"- Output dir: `{run_dir}`",
        "",
        "## Overall",
        "",
        f"- Cases: `{overall.get('cases', 0)}`",
        f"- Raw strict contract rate: `{overall.get('raw_contract_rate', 0.0)}`",
        f"- Sanitized strict contract rate: `{overall.get('sanitized_contract_rate', 0.0)}`",
        f"- Position parse rate: `{overall.get('position_parse_rate', 0.0)}`",
        f"- Exact position rate: `{overall.get('exact_position_rate', 0.0)}`",
        f"- Mean latency seconds: `{overall.get('mean_latency_seconds', 0.0)}`",
        f"- Total tokens: `{overall.get('total_tokens', 0)}`",
        "",
        "## Files",
        "",
        "- `per_case_results.csv`",
        "- `per_case_results.json`",
        "- `summary_overall.csv`",
        "- `summary_by_thinking.csv`",
        "- `summary_by_game.csv`",
        "- `summary_by_level.csv`",
        "- `summary_by_game_level.csv`",
        "- `summary_by_thinking_game_level.csv`",
        "",
        "## Best/Worst Level Buckets",
        "",
    ]
    if levels_summary:
        best = max(levels_summary, key=lambda row: float(row.get("exact_position_rate", 0.0)))
        worst = min(levels_summary, key=lambda row: float(row.get("exact_position_rate", 0.0)))
        lines.append(
            f"- Best exact-position bucket: `{best.get('game', '')} lvl{best.get('level', '')} thinking={best.get('thinking', '')}` => `{best.get('exact_position_rate', 0.0)}`"
        )
        lines.append(
            f"- Worst exact-position bucket: `{worst.get('game', '')} lvl{worst.get('level', '')} thinking={worst.get('thinking', '')}` => `{worst.get('exact_position_rate', 0.0)}`"
        )
    else:
        lines.append("- No level summaries generated.")

    if per_case_rows:
        misses = [row for row in per_case_rows if not row["exact_position_match"]]
        if misses:
            sample = misses[0]
            lines.extend(
                [
                    "",
                    "## Example Miss",
                    "",
                    f"- Game/level: `{sample['game']} lvl{sample['level']}`",
                    f"- Thinking: `{sample['thinking']}`",
                    f"- True position: `({sample['answer_row']}, {sample['answer_col']})`",
                    f"- Predicted position: `({sample['predicted_row']}, {sample['predicted_col']})`",
                    f"- Prompt hash: `{sample['prompt_hash']}`",
                ]
            )
    return "\n".join(lines) + "\n"


def build_batch_readme(
    batch_root: Path,
    models: Sequence[str],
    all_rows: Sequence[Dict[str, Any]],
    summary_by_model_thinking: Sequence[Dict[str, Any]],
) -> str:
    lines = [
        "# Spatial Positioning Batch Run",
        "",
        f"- Timestamp (UTC): `{datetime.now(timezone.utc).isoformat()}`",
        f"- Models: `{', '.join(models)}`",
        f"- Cases logged: `{len(all_rows)}`",
        "",
        "## Files",
        "",
        "- `all_runs_detailed.csv`",
        "- `summary_overall.csv`",
        "- `summary_by_model.csv`",
        "- `summary_by_model_thinking.csv`",
        "- `summary_by_model_game_level.csv`",
        "",
    ]
    if summary_by_model_thinking:
        best = max(summary_by_model_thinking, key=lambda row: float(row.get("exact_position_rate", 0.0)))
        worst = min(summary_by_model_thinking, key=lambda row: float(row.get("exact_position_rate", 0.0)))
        lines.extend(
            [
                "## Extremes",
                "",
                f"- Best model/thinking: `{best.get('model', '')} thinking={best.get('thinking', '')}` => `{best.get('exact_position_rate', 0.0)}`",
                f"- Worst model/thinking: `{worst.get('model', '')} thinking={worst.get('thinking', '')}` => `{worst.get('exact_position_rate', 0.0)}`",
            ]
        )
    return "\n".join(lines) + "\n"


def create_case_specs(args: argparse.Namespace) -> Tuple[List[CaseSpec], Dict[Tuple[str, int], Dict[str, Any]]]:
    cases: List[CaseSpec] = []
    context: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for game in args.games:
        env_prefix = game.replace("_v0", "")
        for level in discover_levels(game):
            env_name = f"gvgai-{env_prefix}-lvl{level}-v0"
            vgdl_rules = load_vgdl_rules(env_name)
            level_layout = load_level_map(env_name, level + 1)
            if not level_layout:
                raise RuntimeError(f"Failed to load level map for {env_name}")
            level_mapping = parse_level_mapping(vgdl_rules)
            sprite_mapping = build_sprite_mapping(level_mapping)
            avatar_char = find_avatar_char(level_mapping)
            empty_chars = find_empty_chars(level_mapping)
            original_lines = level_to_lines(level_layout)
            original_pos = locate_avatar(original_lines, avatar_char)
            empty_tiles = [coord for coord in enumerate_empty_tiles(original_lines, empty_chars) if coord != original_pos]
            sampled_positions = sample_case_positions(
                game=game,
                level=level,
                empty_tiles=empty_tiles,
                cases_per_level=args.cases_per_level,
                seed=args.seed,
            )
            empty_char = empty_chars[0]
            level_path = str(
                ROOT / "gym_gvgai" / "envs" / "games" / f"{env_prefix}_v0" / f"{env_prefix}_lvl{level}.txt"
            )
            for case_index, (row_idx, col_idx) in enumerate(sampled_positions):
                prompt_state_ascii = relocate_avatar(
                    lines=original_lines,
                    original_pos=original_pos,
                    new_pos=(row_idx, col_idx),
                    avatar_char=avatar_char,
                    empty_char=empty_char,
                )
                cases.append(
                    CaseSpec(
                        game=game,
                        level=level,
                        case_index=case_index,
                        prompt_state_ascii=prompt_state_ascii,
                        answer_row=row_idx,
                        answer_col=col_idx,
                        original_row=original_pos[0],
                        original_col=original_pos[1],
                        level_path=level_path,
                    )
                )
            context[(game, level)] = {
                "env_name": env_name,
                "vgdl_rules": vgdl_rules,
                "level_layout": level_layout,
                "sprite_mapping": sprite_mapping,
                "level_path": level_path,
            }
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    return cases, context


def run_queries(
    args: argparse.Namespace,
    run_dir: Path,
    prompt_template_path: Path,
    cases: Sequence[CaseSpec],
    context: Dict[Tuple[str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    translated_rules_cache: Dict[Tuple[str, int], str] = {}
    clients: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []
    llm_io_dir = run_dir / "llm_io"
    llm_io_jsonl = llm_io_dir / "queries.jsonl"

    try:
        if not args.dry_run:
            for thinking in args.thinking_modes:
                client = create_client_from_config(
                    args.model,
                    runtime_overrides={
                        "qwen_thinking_mode": thinking,
                        "qwen_sampling_overrides": resolve_qwen_sampling_overrides(args.model, thinking),
                    },
                )
                configure_client_context_window(client)
                if hasattr(client, "max_tokens"):
                    client.max_tokens = args.max_tokens
                clients[thinking] = client

        translator: Optional[LLMTranslator] = None
        if args.translator_mode == "on" and not args.dry_run:
            translator = LLMTranslator(model_name=args.model)
            configure_client_context_window(translator.llm)

        total = len(cases) * len(args.thinking_modes)
        done = 0
        for case in cases:
            key = (case.game, case.level)
            env_ctx = context[key]
            if key not in translated_rules_cache:
                if args.dry_run:
                    translated_rules_cache[key] = "Game rules in natural language:\n1) Genre\n2) Core Mechanics\n3) Objective\n4) Win Conditions\n5) Loss Conditions\n6) Strategy"
                elif args.translator_mode == "on":
                    if translator is None:
                        raise RuntimeError("Translator was not initialized.")
                    translation = translator.translate(
                        vgdl_rules=env_ctx["vgdl_rules"],
                        level_layout=env_ctx["level_layout"],
                    )
                    translated_rules_cache[key] = f"Game rules in natural language:\n{translation}"
                elif args.translator_mode == "vgdl":
                    translated_rules_cache[key] = f"Game rules (raw VGDL):\n{env_ctx['vgdl_rules']}"
                else:
                    translated_rules_cache[key] = ""

            prompt = build_prompt_for_case(
                translated_rules=translated_rules_cache[key],
                prompt_template_path=prompt_template_path,
                prompt_state_ascii=case.prompt_state_ascii,
                sprite_mapping=env_ctx["sprite_mapping"],
            )

            for thinking in args.thinking_modes:
                response_raw = ""
                latency_seconds = 0.0
                usage = {
                    "token_usage_source": "unavailable",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }

                if not args.dry_run:
                    client = clients[thinking]
                    client.clear_history()
                    started = time.perf_counter()
                    response_raw = client.query(prompt)
                    latency_seconds = time.perf_counter() - started
                    usage = client.get_last_usage() or usage

                sanitized_response, had_think, residual_think = sanitize_thinking_output(response_raw)
                raw_contract_ok = strict_contract_ok(response_raw)
                sanitized_contract_ok = strict_contract_ok(sanitized_response)
                action_meta = parse_action_from_response_with_meta(sanitized_response, ACTION_MAP)
                predicted_position = parse_feedback_position(sanitized_response)
                row_match = bool(predicted_position and predicted_position[0] == case.answer_row)
                col_match = bool(predicted_position and predicted_position[1] == case.answer_col)
                exact_match = bool(predicted_position == (case.answer_row, case.answer_col))

                done += 1
                print(
                    f"[{done}/{total}] game={case.game} lvl={case.level} case={case.case_index} thinking={thinking} "
                    f"raw_contract={int(raw_contract_ok)} exact_position={int(exact_match)}"
                )

                row = {
                    "game": case.game,
                    "level": case.level,
                    "thinking": thinking,
                    "case_index": case.case_index,
                    "level_path": case.level_path,
                    "original_row": case.original_row,
                    "original_col": case.original_col,
                    "answer_row": case.answer_row,
                    "answer_col": case.answer_col,
                    "predicted_row": predicted_position[0] if predicted_position else "",
                    "predicted_col": predicted_position[1] if predicted_position else "",
                    "raw_contract_ok": int(raw_contract_ok),
                    "sanitized_contract_ok": int(sanitized_contract_ok),
                    "action_parse_success": int(bool(action_meta.get("parse_success"))),
                    "action_parse_tier": str(action_meta.get("parse_tier", "")),
                    "position_parse_success": int(predicted_position is not None),
                    "row_match": int(row_match),
                    "col_match": int(col_match),
                    "exact_position_match": int(exact_match),
                    "had_think": int(had_think),
                    "residual_think": int(residual_think),
                    "latency_seconds": round(latency_seconds, 6),
                    "token_usage_source": str(usage.get("token_usage_source", "unavailable")),
                    "input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0),
                    "prompt_hash": prompt_hash(prompt),
                    "prompt": prompt,
                    "response_raw": response_raw,
                    "response_sanitized": sanitized_response,
                }
                results.append(row)
                if args.print_live_io:
                    print_live_io_block(
                        case=case,
                        thinking=thinking,
                        state_ascii=case.prompt_state_ascii,
                        response_raw=response_raw,
                        sanitized_response=sanitized_response,
                        predicted_position=predicted_position,
                    )
                append_jsonl(
                    llm_io_jsonl,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "game": case.game,
                        "level": case.level,
                        "thinking": thinking,
                        "case_index": case.case_index,
                        "actual_position": {
                            "row": case.answer_row,
                            "col": case.answer_col,
                        },
                        "original_position": {
                            "row": case.original_row,
                            "col": case.original_col,
                        },
                        "predicted_position": (
                            {"row": predicted_position[0], "col": predicted_position[1]}
                            if predicted_position
                            else None
                        ),
                        "raw_contract_ok": int(raw_contract_ok),
                        "sanitized_contract_ok": int(sanitized_contract_ok),
                        "position_parse_success": int(predicted_position is not None),
                        "exact_position_match": int(exact_match),
                        "token_usage": {
                            "source": str(usage.get("token_usage_source", "unavailable")),
                            "input_tokens": int(usage.get("input_tokens", 0) or 0),
                            "output_tokens": int(usage.get("output_tokens", 0) or 0),
                            "total_tokens": int(usage.get("total_tokens", 0) or 0),
                        },
                        "latency_seconds": round(latency_seconds, 6),
                        "input_prompt": prompt,
                        "output_raw": response_raw,
                        "output_sanitized": sanitized_response,
                    },
                )
    finally:
        for client in clients.values():
            try:
                client.shutdown()
            except Exception:
                pass
    return results


def summarize_run(run_dir: Path, args: argparse.Namespace, per_case_rows: Sequence[Dict[str, Any]]) -> None:
    summary_overall = aggregate_rows(per_case_rows, [])
    summary_by_thinking = aggregate_rows(per_case_rows, ["thinking"])
    summary_by_game = aggregate_rows(per_case_rows, ["game"])
    summary_by_level = aggregate_rows(per_case_rows, ["level"])
    summary_by_game_level = aggregate_rows(per_case_rows, ["game", "level"])
    summary_by_thinking_game_level = aggregate_rows(per_case_rows, ["thinking", "game", "level"])

    write_csv(run_dir / "per_case_results.csv", per_case_rows)
    write_json(run_dir / "per_case_results.json", per_case_rows)
    write_csv(run_dir / "summary_overall.csv", summary_overall)
    write_csv(run_dir / "summary_by_thinking.csv", summary_by_thinking)
    write_csv(run_dir / "summary_by_game.csv", summary_by_game)
    write_csv(run_dir / "summary_by_level.csv", summary_by_level)
    write_csv(run_dir / "summary_by_game_level.csv", summary_by_game_level)
    write_csv(run_dir / "summary_by_thinking_game_level.csv", summary_by_thinking_game_level)
    write_json(
        run_dir / "summary_bundle.json",
        {
            "overall": summary_overall,
            "by_thinking": summary_by_thinking,
            "by_game": summary_by_game,
            "by_level": summary_by_level,
            "by_game_level": summary_by_game_level,
            "by_thinking_game_level": summary_by_thinking_game_level,
        },
    )
    if per_case_rows:
        write_json(
            run_dir / "llm_io" / "tail_preview_5.json",
            [
                {
                    "game": row["game"],
                    "level": row["level"],
                    "thinking": row["thinking"],
                    "case_index": row["case_index"],
                    "actual_position": {"row": row["answer_row"], "col": row["answer_col"]},
                    "predicted_position": {"row": row["predicted_row"], "col": row["predicted_col"]},
                    "response_raw": row["response_raw"],
                    "response_sanitized": row["response_sanitized"],
                }
                for row in per_case_rows[-5:]
            ],
        )

    readme = build_readme(
        args=args,
        run_dir=run_dir,
        per_case_rows=per_case_rows,
        overall_summary=summary_overall,
        levels_summary=summary_by_thinking_game_level,
    )
    (run_dir / "README.md").write_text(readme, encoding="utf-8")


def write_batch_aggregate(batch_root: Path, all_rows: Sequence[Dict[str, Any]], models: Sequence[str]) -> None:
    if not all_rows:
        return
    summary_overall = aggregate_rows(all_rows, [])
    summary_by_model = aggregate_rows(all_rows, ["model"])
    summary_by_model_thinking = aggregate_rows(all_rows, ["model", "thinking"])
    summary_by_model_game_level = aggregate_rows(all_rows, ["model", "game", "level"])
    summary_by_model_thinking_game_level = aggregate_rows(all_rows, ["model", "thinking", "game", "level"])

    write_csv(batch_root / "all_runs_detailed.csv", all_rows)
    write_json(batch_root / "all_runs_detailed.json", all_rows)
    write_csv(batch_root / "summary_overall.csv", summary_overall)
    write_csv(batch_root / "summary_by_model.csv", summary_by_model)
    write_csv(batch_root / "summary_by_model_thinking.csv", summary_by_model_thinking)
    write_csv(batch_root / "summary_by_model_game_level.csv", summary_by_model_game_level)
    write_csv(batch_root / "summary_by_model_thinking_game_level.csv", summary_by_model_thinking_game_level)
    (batch_root / "README.md").write_text(
        build_batch_readme(
            batch_root=batch_root,
            models=models,
            all_rows=all_rows,
            summary_by_model_thinking=summary_by_model_thinking,
        ),
        encoding="utf-8",
    )


def run_single_model(
    args: argparse.Namespace,
    *,
    output_root: Path,
    run_name: Optional[str] = None,
) -> Tuple[Path, List[Dict[str, Any]]]:
    run_name = run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    run_dir = output_root / f"{run_name}__model_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_template_path = write_prompt_variant(run_dir)
    write_json(run_dir / "run_config.json", vars(args))

    cases, context = create_case_specs(args)
    case_manifest = [
        {
            "model": args.model,
            "game": case.game,
            "level": case.level,
            "case_index": case.case_index,
            "answer_row": case.answer_row,
            "answer_col": case.answer_col,
            "original_row": case.original_row,
            "original_col": case.original_col,
            "level_path": case.level_path,
            "prompt_state_ascii": case.prompt_state_ascii,
        }
        for case in cases
    ]
    write_json(run_dir / "case_manifest.json", case_manifest)

    per_case_rows = run_queries(
        args=args,
        run_dir=run_dir,
        prompt_template_path=prompt_template_path,
        cases=cases,
        context=context,
    )
    per_case_rows = normalize_run_rows(per_case_rows, args.model)
    summarize_run(run_dir, args, per_case_rows)
    print(f"Results written to {run_dir}")
    return run_dir, per_case_rows


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    models = resolve_model_list(args)

    if len(models) == 1:
        args.model = models[0]
        run_single_model(args, output_root=output_root, run_name=run_name)
        return

    batch_root = output_root / f"{run_name}__suite"
    batch_root.mkdir(parents=True, exist_ok=True)
    write_json(
        batch_root / "batch_config.json",
        {
            **vars(args),
            "resolved_models": models,
        },
    )

    all_rows: List[Dict[str, Any]] = []
    run_manifest: List[Dict[str, Any]] = []
    for model_name in models:
        model_args = deepcopy(args)
        model_args.model = model_name
        try:
            run_dir, rows = run_single_model(model_args, output_root=batch_root, run_name=run_name)
            all_rows.extend(rows)
            run_manifest.append({"model": model_name, "run_dir": str(run_dir), "status": "ok"})
        except Exception as exc:
            run_manifest.append({"model": model_name, "status": "failed", "error": str(exc)})
            if not args.keep_going:
                write_json(batch_root / "run_manifest.json", run_manifest)
                raise

    write_json(batch_root / "run_manifest.json", run_manifest)
    write_batch_aggregate(batch_root, all_rows, models)
    print(f"Batch results written to {batch_root}")


if __name__ == "__main__":
    main()
