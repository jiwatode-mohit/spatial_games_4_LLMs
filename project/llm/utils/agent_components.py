import os
import imageio
from collections import defaultdict
import re
import string
import csv
import string
from io import StringIO

from typing import List, Dict, Tuple, Optional, Union, Any




class show_state_gif:
    def __init__(self):
        self.frames = []

    def __call__(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, 'GIF', duration=0.1)


def create_directory(base_dir='imgs'):
    if os.path.exists(base_dir):
        index = 1
        while True:
            new_dir = f"{base_dir}_{index}"
            if not os.path.exists(new_dir):
                base_dir = new_dir
                break
            index += 1
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# VGDL State Parsing


# VGDL SpriteSet + LevelMapping Parser
def parse_vgdl(vgdl_text: Union[str, List[str]]) -> Tuple[set, dict]:
    sprite_names = set()
    level_mapping = {}

    if isinstance(vgdl_text, list):
        vgdl_lines = vgdl_text
        vgdl_string = '\n'.join(vgdl_text)
    else:
        vgdl_string = vgdl_text
        vgdl_lines = vgdl_text.split('\n')

    # === Parse SpriteSet ===
    sprite_section = re.search(r"SpriteSet(.*?)LevelMapping", vgdl_string, re.DOTALL)
    if sprite_section:
        lines = sprite_section.group(1).split('\n')
        parent_stack = []
        for line in lines:
            line = line.rstrip()
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            while parent_stack and parent_stack[-1][0] >= indent:
                parent_stack.pop()
            parts = line.strip().split('>')
            name = parts[0].strip()
            if parent_stack:
                full_name = parent_stack[-1][1] + "." + name
            else:
                full_name = name
            sprite_names.add(full_name)
            parent_stack.append((indent, full_name))

    # === Parse LevelMapping ===
    in_level_mapping = False
    for line in vgdl_lines:
        line = line.strip()
        if line.startswith('LevelMapping'):
            in_level_mapping = True
            continue
        if line.startswith('InteractionSet') or line.startswith('TerminationSet'):
            break
        if in_level_mapping and '>' in line:
            parts = line.split('>', 1)
            char = parts[0].strip()
            sprite_list = parts[1].strip().split()
            level_mapping[char] = sprite_list
            sprite_names.update(sprite_list)

    return sprite_names, level_mapping


def convert_state(state, sprite_to_char, debug: bool = False):
    result = []
    for row_idx, row in enumerate(state):
        line = ''
        for col_idx, cell in enumerate(row):
            chosen = '.'
            sprite = cell.strip()
            reason = ''

            if sprite in sprite_to_char:
                chosen = sprite_to_char[sprite]
                reason = 'direct match'
            else:
                for part in sprite.split():
                    if part in sprite_to_char:
                        chosen = sprite_to_char[part]
                        reason = f'fallback to "{part}"'
                        break
            if debug:
                print(f"[{row_idx:02d},{col_idx:02d}] '{sprite}' → '{chosen}' ({reason})")
            line += chosen
        result.append(line)
    return '\n'.join(result)

def get_available_chars(sprite_to_char: Dict[str, str], sprite_names: set) -> list:
    used = set(sprite_to_char.values())

    # Step 1: symbols
    symbols = "@#$%&*"
    symbol_pool = [c for c in symbols if c not in used]

    # Step 2: first-letter of each sprite
    letters = []
    seen_letters = set()
    for name in sorted(sprite_names):
        if not name:
            continue
        first = name.strip()[0].lower()
        if first not in used and first not in seen_letters and first.isalpha():
            letters.append(first)
            seen_letters.add(first)

    # Step 3: fallback: remaining lowercase, uppercase, digits
    fallback_pool = [
        c for c in (string.ascii_lowercase + string.ascii_uppercase + string.digits)
        if c not in used and c not in letters
    ]

    return symbol_pool + letters + fallback_pool
    
def check_unique_mapping(sprite_to_char: dict):
    inverse = {}
    for sprite, char in sprite_to_char.items():
        if char in inverse:
            print(f"[DUPLICATE] Character '{char}' used for both '{inverse[char]}' and '{sprite}'")
        inverse[char] = sprite



def normalize_sprite(sprite: str) -> str:
    """
    Normalize a sprite string:
    - Remove leading/trailing spaces
    - Deduplicate tokens
    - Sort tokens alphabetically
    """
    tokens = sprite.strip().split()
    unique_tokens = sorted(set(tokens))
    return ' '.join(unique_tokens)




def detect_input_type(state_str: str) -> str:
    lines = state_str.strip().splitlines()
    csv_like = sum(',' in line for line in lines)
    ascii_like = sum(all(c in string.printable for c in line) for line in lines)

    if csv_like >= len(lines) // 2:
        return "csv"
    elif ascii_like and csv_like == 0:
        return "ascii"
    return "unknown"


def ascii_to_pseudo_grid(state_str: str) -> str:
    lines = state_str.strip().splitlines()
    csv_grid = [','.join(list(line)) for line in lines]
    return '\n'.join(csv_grid)


def generate_mapping_and_ascii(
    state_str: str,
    vgdl_text: str,
    existing_mapping: Optional[dict] = None,
    debug: bool = False
) -> Tuple[dict, str]:
    # Step 0: Detect and convert input format
    # if detect_input_type(state_str) == "ascii":
    #     state_str = ascii_to_pseudo_grid(state_str)

    # Step 1: Parse VGDL rules
    sprite_names, level_mapping = parse_vgdl(vgdl_text)

    # Step 2: Parse CSV into grid
    reader = csv.reader(StringIO(state_str))
    state_grid = []
    all_leaf_sprites = set()

    for row in reader:
        row_data = []
        for cell in row:
            raw_sprite = cell.strip()
            sprite = normalize_sprite(raw_sprite) if raw_sprite else ''
            row_data.append(sprite)
            if sprite:
                all_leaf_sprites.add(sprite)
        state_grid.append(row_data)

    # Step 3: Build sprite-to-char mapping
    sprite_to_char = dict(existing_mapping) if existing_mapping else {}
    sprite_to_char.setdefault('avatar', 'a')
    if 'background' in all_leaf_sprites:
        sprite_to_char.setdefault('background', '.')
    elif 'floor' in all_leaf_sprites:
        sprite_to_char.setdefault('floor', '.')

    available_chars = get_available_chars(sprite_to_char, all_leaf_sprites)

    for char, sprite_list in level_mapping.items():
        key = ' '.join(sprite_list)
        if key in sprite_to_char:
            continue
        if 'avatar' in sprite_list and sprite_to_char.get('avatar') == 'a':
            continue
        if char not in sprite_to_char.values():
            sprite_to_char[key] = char

    for sprite in sorted(all_leaf_sprites):
        if sprite in sprite_to_char:
            continue
        if 'avatar' in sprite and sprite_to_char.get('avatar') == 'a':
            continue
        if available_chars:
            sprite_to_char[sprite] = available_chars.pop(0)
        else:
            raise ValueError("Ran out of characters to assign.")

    if debug:
        check_unique_mapping(sprite_to_char)

    ascii_level = convert_state(state_grid, sprite_to_char, debug=debug)
    ascii_flipped_y ='\n'.join(reversed(ascii_level.splitlines()))

    return sprite_to_char, ascii_level, ascii_flipped_y


def extract_avatar_position_from_state(
    ascii_lines: Union[str, List[str]],
    sprite_to_char: Dict[str, str],
    flip_vertical: bool = False
) -> Optional[Tuple[int, int]]:
    # 自动处理字符串输入
    if isinstance(ascii_lines, str):
        ascii_lines = ascii_lines.splitlines()

    # 防止误传 list of characters
    if isinstance(ascii_lines, list) and all(isinstance(x, str) and len(x) == 1 for x in ascii_lines):
        ascii_lines = ''.join(ascii_lines).splitlines()

    candidate_chars = []
    if 'avatar' in sprite_to_char and isinstance(sprite_to_char.get('avatar'), str):
        candidate_chars.append(sprite_to_char.get('avatar'))

    # Accept common avatar aliases used in VGDL games.
    for alias in ('nokey', 'withkey'):
        ch = sprite_to_char.get(alias)
        if isinstance(ch, str):
            candidate_chars.append(ch)

    # Accept any sprite key containing "avatar" as an additional alias.
    for sprite_name, ch in sprite_to_char.items():
        if isinstance(ch, str) and 'avatar' in str(sprite_name).lower():
            candidate_chars.append(ch)

    # Preserve order, remove duplicates, and keep a safe fallback.
    deduped = []
    seen = set()
    for ch in candidate_chars:
        if len(ch) == 1 and ch not in seen:
            deduped.append(ch)
            seen.add(ch)
    if not deduped:
        deduped = ['a']

    height = len(ascii_lines)

    for y, row in enumerate(ascii_lines):
        for avatar_char in deduped:
            if avatar_char in row:
                x = row.index(avatar_char)
                actual_y = (height - 1 - y) if flip_vertical else y
                return (actual_y, x)

    return None


import re
import json


def _extract_numeric_action_from_line(text: str, action_map: dict):
    m = re.search(r"\baction\s*[:=]\s*(\d+)\b", text, re.IGNORECASE)
    if not m:
        return None
    val = int(m.group(1))
    if val in action_map:
        return val, action_map[val]
    return None


def _normalize_action_aliases(action_map: dict) -> Dict[str, int]:
    aliases: Dict[str, int] = {}
    for aid, aname in action_map.items():
        try:
            action_id = int(aid)
        except Exception:
            continue
        raw_name = str(aname or "").strip()
        if not raw_name:
            continue
        aliases[raw_name.lower()] = action_id
        upper = raw_name.upper()
        aliases[upper.lower()] = action_id
        core = upper.replace("ACTION_", "").lower()
        if core:
            aliases[core] = action_id
            for tok in core.split("_"):
                if tok:
                    aliases[tok] = action_id

    nil_id = None
    for aid, aname in action_map.items():
        if str(aname).upper() == "ACTION_NIL":
            nil_id = int(aid)
            break
    if nil_id is not None:
        for k in ("nil", "nothing", "none", "noop", "stay", "wait"):
            aliases[k] = nil_id
    return aliases


def parse_action_from_response_with_meta(response: str, action_map: dict) -> Dict[str, Any]:
    """
    Tiered action parsing:
      1) strict Action:<id> / Action=<id>
      2) tolerant action_id:<id>
      3) tolerant JSON {"action": ...}
      4) tolerant first-line integer
      5) tolerant aliases (left/right/up/down/use/nil/ACTION_LEFT)
    """
    text = str(response or "")
    meta: Dict[str, Any] = {
        "action_id": None,
        "action_name": None,
        "parse_success": False,
        "parse_tier": "failed",
        "parse_reason": "no_pattern_matched",
    }
    if not action_map:
        meta["parse_reason"] = "empty_action_map"
        return meta

    # Tier 1: strict Action:<id>/Action=<id>.
    m = re.search(r"\baction\s*[:=]\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        aid = int(m.group(1))
        if aid in action_map:
            meta.update(
                {
                    "action_id": aid,
                    "action_name": action_map[aid],
                    "parse_success": True,
                    "parse_tier": "strict",
                    "parse_reason": "matched_action_colon_equals",
                }
            )
            return meta
        meta["parse_reason"] = "strict_invalid_action_id"

    # Tier 2: tolerant "action_id: <id>" or "action id: <id>".
    m = re.search(r"\baction(?:[_\s-]*id)\s*[:=]\s*(\d+)\b", text, re.IGNORECASE)
    if m:
        aid = int(m.group(1))
        if aid in action_map:
            meta.update(
                {
                    "action_id": aid,
                    "action_name": action_map[aid],
                    "parse_success": True,
                    "parse_tier": "tolerant_action_id",
                    "parse_reason": "matched_action_id_field",
                }
            )
            return meta
        meta["parse_reason"] = "tolerant_action_id_invalid"

    # Tier 3: tolerant JSON {"action": 3} / {"action":"ACTION_LEFT"} / {"action":"left"}.
    json_candidates = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        json_candidates.append(stripped)
    json_candidates.extend(m.group(0) for m in re.finditer(r"\{[^{}]*\}", text))
    aliases = _normalize_action_aliases(action_map)
    for cand in json_candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict) or "action" not in obj:
            continue
        aval = obj.get("action")
        if isinstance(aval, int) and aval in action_map:
            meta.update(
                {
                    "action_id": int(aval),
                    "action_name": action_map[int(aval)],
                    "parse_success": True,
                    "parse_tier": "tolerant_json",
                    "parse_reason": "json_numeric_action",
                }
            )
            return meta
        if isinstance(aval, str):
            norm = aval.strip().lower()
            if norm in aliases and aliases[norm] in action_map:
                aid = int(aliases[norm])
                meta.update(
                    {
                        "action_id": aid,
                        "action_name": action_map[aid],
                        "parse_success": True,
                        "parse_tier": "tolerant_json",
                        "parse_reason": "json_string_action_alias",
                    }
                )
                return meta
    if json_candidates:
        meta["parse_reason"] = "json_action_invalid_or_unmapped"

    # Tier 4: bare valid integer on first non-empty line.
    first_line = ""
    for ln in text.splitlines():
        s = ln.strip().strip("`")
        if s:
            first_line = s
            break
    if first_line and re.fullmatch(r"-?\d+", first_line):
        aid = int(first_line)
        if aid in action_map:
            meta.update(
                {
                    "action_id": aid,
                    "action_name": action_map[aid],
                    "parse_success": True,
                    "parse_tier": "tolerant_first_line_int",
                    "parse_reason": "first_line_integer",
                }
            )
            return meta
        meta["parse_reason"] = "first_line_integer_invalid"

    # Tier 5: alias mapping from canonical names or directional words.
    canonical_match = re.search(r"\bACTION_[A-Z_]+\b", text)
    if canonical_match:
        norm = canonical_match.group(0).strip().lower()
        if norm in aliases and aliases[norm] in action_map:
            aid = int(aliases[norm])
            meta.update(
                {
                    "action_id": aid,
                    "action_name": action_map[aid],
                    "parse_success": True,
                    "parse_tier": "tolerant_alias",
                    "parse_reason": "canonical_action_alias",
                }
            )
            return meta
        meta["parse_reason"] = "canonical_alias_unmapped"

    token_order = re.findall(r"[A-Za-z_]+", text.lower())
    for tok in token_order:
        if tok in aliases and aliases[tok] in action_map:
            aid = int(aliases[tok])
            meta.update(
                {
                    "action_id": aid,
                    "action_name": action_map[aid],
                    "parse_success": True,
                    "parse_tier": "tolerant_alias",
                    "parse_reason": f"token_alias:{tok}",
                }
            )
            return meta

    return meta


def parse_action_sequence_from_response(response: str, action_map: dict, horizon: int) -> List[int]:
    """
    Parse a sequence of actions for lookahead planning.
    Accepts either:
    - PlanActions:[1,2,3]
    - JSON {"plan_actions": [1,2,3]}
    - repeated Action:<id> lines
    Returns up to `horizon` valid actions from action_map.
    """
    if horizon <= 0:
        return []

    plan = []
    text = str(response or "")

    # 1) JSON object path
    try:
        m = re.search(r'\{[\s\S]*?"plan_actions"\s*:\s*\[[^\]]*\][\s\S]*?\}', text, re.IGNORECASE)
        if m:
            data = json.loads(m.group(0))
            seq = data.get("plan_actions", [])
            if isinstance(seq, list):
                for x in seq:
                    if isinstance(x, int) and x in action_map:
                        plan.append(int(x))
    except Exception:
        pass

    # 2) PlanActions:[...] inline format
    if not plan:
        m = re.search(r"\bPlanActions\s*:\s*\[([^\]]*)\]", text, re.IGNORECASE)
        if m:
            for tok in re.findall(r"-?\d+", m.group(1)):
                aid = int(tok)
                if aid in action_map:
                    plan.append(aid)

    # 3) Multiple Action:<id> lines fallback
    if not plan:
        for line in text.splitlines():
            parsed = _extract_numeric_action_from_line(line, action_map)
            if parsed is not None:
                plan.append(int(parsed[0]))

    return plan[:horizon]

def parse_action_from_response(response: str, action_map: dict):
    meta = parse_action_from_response_with_meta(response, action_map)
    if meta.get("parse_success"):
        aid = int(meta["action_id"])
        return aid, action_map[aid]
    raise ValueError(
        "Unable to parse action: expected explicit 'Action:<id>' with a valid ID from Action Legend."
    )



if __name__ == "__main__":
   
    action_map = {
    0: "ACTION_NIL",
    1: "ACTION_USE",
    2: "ACTION_UP",
    3: "ACTION_DOWN",
    4: "ACTION_LEFT",
    5: "ACTION_RIGHT"
    }
    test_responses = [
        '''```Action:1```
Feedback: I am using the fire action to launch a missile upward at column 17 to destroy the alien at (0, 17), as I have a clear path and there are no immediate threats in my current column.
''',
        'ACTION_DOWN (4)',
        '{"action": "use"}',
        'I will move left now.',
        'The action is 3.',
        'Let me **2** this time.',
        'Response: Action:1'
    ]
    for resp in test_responses:
        aid, aname = parse_action_from_response(resp, action_map)
        print(f"Response: {resp}  -->  Parsed Action: {aid} ({aname})")
