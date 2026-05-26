import json
import os
from typing import Optional, Union, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import tiktoken

from typing import Any, DefaultDict

DEFAULT_PROMPT_PATH = os.environ.get(
    "GVGAI_PROMPT_TEMPLATE_PATH",
    os.path.join(os.path.dirname(__file__), "prompt_templates", "prompt.json"),
)

def _normalize_action_meaning(name: str) -> str:
    if not name:
        return "unknown"
    base = name.replace("ACTION_", "").strip()
    if not base:
        return "unknown"
    return base.lower()

def _format_action_legend(action_map: dict) -> str:
    lines = []
    for action_id, action_name in action_map.items():
        meaning = _normalize_action_meaning(action_name)
        lines.append(f"Action {action_id} => {action_name} | meaning: {meaning}")
    lines.append("Rule: Use numeric ID from this table. If words and numbers conflict, follow this table.")
    return "\n".join(lines)

def _format_action_reminder(action_map: dict) -> str:
    parts = []
    for action_id, action_name in action_map.items():
        parts.append(f"{action_id}:{action_name}")
    return ", ".join(parts)

def _format_action_history_with_ids(action_history: List[int], action_map: dict) -> str:
    parts = []
    for action_id in action_history or []:
        action_name = action_map.get(action_id, f"ACTION_{action_id}")
        parts.append(f"{action_id}:{action_name}")
    return ",".join(parts)

def _build_char_to_entity_mapping(sprite_mapping: Optional[dict]) -> Dict[str, str]:
    """Convert sprite->char mapping into char->entity label mapping."""
    char_to_entity: Dict[str, str] = {}
    if not sprite_mapping:
        return char_to_entity

    for sprite_name, symbol in sprite_mapping.items():
        if not isinstance(symbol, str) or len(symbol) != 1:
            continue
        if symbol in char_to_entity:
            continue
        entity = str(sprite_name).strip() if sprite_name is not None else symbol
        if not entity:
            entity = symbol
        char_to_entity[symbol] = entity.replace(" ", "+")
    return char_to_entity

def _render_mapped_ascii_state(ascii_map: str, sprite_mapping: Optional[dict]) -> str:
    """Render ASCII grid with each symbol replaced by entity labels."""
    if not ascii_map:
        return ascii_map

    char_to_entity = _build_char_to_entity_mapping(sprite_mapping)
    if not char_to_entity:
        return ascii_map

    rendered_lines = []
    for line in ascii_map.splitlines():
        rendered_lines.append(" ".join(char_to_entity.get(ch, ch) for ch in line))
    return "\n".join(rendered_lines)

def ascii_to_position_mapping(ascii_map: str, sprite_mapping: dict) -> List[str]:
    expanded_lines = []
    char_to_entity = _build_char_to_entity_mapping(sprite_mapping)
    lines = ascii_map.strip().splitlines()
    for row_idx, line in enumerate(lines):
        for col_idx, char in enumerate(line):
            entity = char_to_entity.get(char, char)
            expanded_lines.append(f"row={row_idx}, col={col_idx} -> {entity}")
    return expanded_lines


def rotate_ascii_left(ascii_map: str) -> str:
    lines = [list(line) for line in ascii_map.strip().splitlines()]
    if not lines:
        return ascii_map
    rotated = list(zip(*lines[::-1]))
    return "\n".join("".join(row) for row in rotated)

class ReflectionManager:
    def __init__(self, max_history=3):
        self.history: List[Dict[str, Any]] = []
        self.step_log: List[Dict[str, Any]] = []  # 每步行为记录
        self.max_history = max_history

    def add_reflection(self, reflection: str, step: Optional[int] = None, reason: Optional[str] = None):
        if reflection:
            entry = {
                "step": step,
                "reason": reason,
                "content": reflection
            }
            self.history.append(entry)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def log_step(self, step: int, action: int, reward: float, **extras):
        self.step_log.append({
            "step": step,
            "action": action,
            "reward": reward,
            **extras
        })

    def get_last_reward(self) -> Optional[float]:
        if self.step_log:
            return self.step_log[-1]["reward"]
        return None

    def get_formatted_history(self) -> str:
        lines = []
        for i, entry in enumerate(self.history):
            prefix = f"[Reflection {i + 1}]"
            step_str = f" (step {entry['step']})" if entry.get("step") is not None else ""
            reason_str = f" [{entry['reason']}]" if entry.get("reason") else ""
            lines.append(f"{prefix}{step_str}{reason_str}: {entry['content']}")
        return "\n".join(lines)


class PromptLogger:
    def __init__(self, model_name: str, game_name: Optional[str] = None, log_dir: Optional[str] = None):
        self.model_name = model_name
        self.game_name = game_name
        self.use_direct_log_dir = bool(log_dir)
        if log_dir:
            self.base_log_dir = Path(log_dir)
        else:
            self.base_log_dir = Path(__file__).parent.parent / "log"
        self.logs: List[Dict[str, str]] = []
        self.sampled_step_inputs: List[Dict[str, Any]] = []
        self.sampled_input_frequency: Optional[int] = None

    def log(self, role: str, content: str):
        self.logs.append({"role": role, "content": content})

    def log_sampled_step_input(self, step: int, prompt: str, frequency: int = 50):
        if frequency <= 0:
            return
        if step % frequency != 0:
            return
        self.sampled_input_frequency = frequency
        self.sampled_step_inputs.append({
            "step": step,
            "frequency": frequency,
            "prompt": prompt,
        })

    def save(self):
        try:
            self.base_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # 如果指定的目录创建失败，回退到默认目录
            self.base_log_dir = Path(__file__).parent / "log"
            self.base_log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{timestamp}.json"
        if self.use_direct_log_dir:
            subdir = self.base_log_dir / "llm_io"
        else:
            subdir = self.base_log_dir / self.model_name
            if self.game_name:
                subdir = subdir / self.game_name
        subdir.mkdir(parents=True, exist_ok=True)
        filepath = subdir / filename

        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=2)

        print(f"[{self.model_name}] Prompt log saved to {filepath}")
        if self.sampled_step_inputs:
            frequency = self.sampled_input_frequency or 50
            sampled_path = subdir / f"{timestamp}.sampled_inputs_every_{frequency}_steps.json"
            with open(sampled_path, "w") as f:
                json.dump(self.sampled_step_inputs, f, indent=2)
            print(f"[{self.model_name}] Sampled input log saved to {sampled_path}")

    def log_response(self, response: str):
        self.log("response", response)

def truncate_prompt_by_token(prompt: str, max_tokens: int = 4000, model: str = "gpt-3.5-turbo") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(prompt)
    truncated = enc.decode(tokens[:max_tokens])
    return truncated

def build_static_prompt(
    vgdl_rules: Optional[str] = None,
    action_map: Optional[dict] = None,
    optional_prompt: Optional[str] = '',
    prompt_template_path: Optional[str] = None
) -> str:
    if prompt_template_path is None:
        prompt_template_path = DEFAULT_PROMPT_PATH

    with open(prompt_template_path, 'r') as f:
        config = json.load(f)

    sections = []

    if vgdl_rules and config.get("include_rules", True):
        sections.append("=== Game Rules ===\n" + vgdl_rules)

    if 'system' in config and config['system'].strip():
        sections.append(config['system'])

    # if sprite_mapping and config.get("include_mapping", True):
    #     sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()]
    #     sections.append("=== Sprite Mapping ===\n" + "\n".join(sprite_lines))

    if action_map and config.get("include_actions", True):
        sections.append("=== Action Legend (Primary Decision Table) ===\n" + _format_action_legend(action_map))

    if optional_prompt and optional_prompt.strip():
        sections.append("=== Feedback === \n")
        sections.append(optional_prompt.strip())

    if 'instruction' in config and config['instruction'].strip():
        sections.append(config['instruction'])

    return "\n\n".join(sections)

def build_dynamic_prompt(
    current_ascii: str,
    last_ascii: Optional[str] = None,
    current_image_path: Optional[str] = None,
    last_image_path: Optional[str] = None,
    avatar_position: Optional[Tuple[int, int]] = None,
    last_position: Optional[Tuple[int,int]] = None,
    action_map: Optional[dict] = None,
    action_history: Optional[List[int]] = None,
    reflection_manager: Optional['ReflectionManager'] = None,
    prompt_template_path: Optional[str] = None,
    logger: Optional[PromptLogger] = None,
    sprite_mapping: Optional[dict] = None,
    plan: Optional[str] = '',
    scm_context: Optional[str] = '',
    llm_model_name: str = "gpt-3.5-turbo",
    rotate: bool = False,
    expanded: bool = False
) -> str:
    if prompt_template_path is None:
        prompt_template_path = DEFAULT_PROMPT_PATH

    with open(prompt_template_path, 'r') as f:
        config = json.load(f)

    sections = []

    if last_ascii and config.get("include_last_state", True):
        sections.append("=== Last State ===\n" + _render_mapped_ascii_state(last_ascii, sprite_mapping))

    if last_image_path and config.get("include_last_image", False):
        sections.append(f"=== Last Image Path ===\n{last_image_path}")

    if current_ascii and config.get("include_current_state", True):
        expanded_map_prompt = ''
        current_ascii_prompt = _render_mapped_ascii_state(current_ascii, sprite_mapping)
        if rotate:
            rotated = rotate_ascii_left(current_ascii)
            current_ascii_prompt = (
                f"=== Original State ===\n{_render_mapped_ascii_state(current_ascii, sprite_mapping)}\n\n"
                f"=== Rotated State (Y-axis emphasis) ===\n{_render_mapped_ascii_state(rotated, sprite_mapping)}\n"
                "Note: In the original state, each row corresponds to horizontal (left-right) layout in the game.\n"
                "In the rotated version, each row corresponds to vertical (up-down) layout."
            )
        if expanded:
            expanded_map_lines = ascii_to_position_mapping(current_ascii, sprite_mapping)
            expanded_map_prompt = (
                "\n\n=== Map in Natural language ===\n"
                "Each line shows entity at (row, col).\n"
                "In rotated version, row emphasizes vertical (up-down).\n"
                + "\n".join(expanded_map_lines)
            )


        sections.append("=== Current State ===\n" + current_ascii_prompt + expanded_map_prompt)

    if current_image_path and config.get("include_current_image", False):
        sections.append(f"=== Current Image Path ===\n{current_image_path}")

    if reflection_manager and config.get("include_reward", True):
        last_reward = reflection_manager.get_last_reward()
        if last_reward is not None:
            sections.append(f"=== Last Reward ===\n{last_reward}")

    if reflection_manager and config.get("include_reflection", True):
        current_reflection = reflection_manager.get_formatted_history()
        if current_reflection:
            sections.append("=== Reflection ===\n" + current_reflection)

    if avatar_position and last_position and config.get("include_avatar_position", True):
        sections.append(
            f"=== Current Position ===\n"
            f"(Left is decreasing x, Right is increasing x, Up is decreasing y, Down is increasing y)\n"
            f"(row = {avatar_position[0]}, col = {avatar_position[1]})\n"
            f"=== Last Position ===\n"
            f"(row = {last_position[0]}, col = {last_position[1]})"
        )

    if action_map and config.get("include_actions", True):
        sections.append("=== Action Reminder ===\n" + _format_action_reminder(action_map))
        s = _format_action_history_with_ids(action_history or [], action_map)
        if s:
            sections.append("=== Action History ===\n" + s)
    if config.get('include_plan',True) and plan:
        sections.append('===Long-term Plan===\n'+plan)
    if scm_context:
        sections.append("=== Current Belief SCM ===\n" + scm_context)
    full_prompt = "\n\n".join(sections)

    if logger:
        logger.log("prompt", full_prompt)

    return full_prompt

if __name__ == "__main__":
    print("Static Prompt Test:\n")
    print(build_static_prompt(vgdl_rules="Use keys to open doors.",  action_map={0: "LEFT", 1: "RIGHT"}))

    print("\nDynamic Prompt Test:\n")
    print(build_dynamic_prompt(current_ascii="..........\n....A.....\n..........", rotate=True, sprite_mapping={"avatar": "A"},))
