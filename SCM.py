import json
import os
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from llm.client import create_client_from_config
except Exception:  # pragma: no cover - import fallback for test/runtime path variants
    from project.llm.client import create_client_from_config
from scm_blueprint import SCMBlueprint, bp


REQUIRED_SCM_KEYS = {
    "meta",
    "blueprint_nodes",
    "blueprint_edges",
    "static_nodes",
    "dynamic_variables",
    "equations",
    "action_effects",
    "interaction_rules",
    "reward_rules",
    "failure_conditions",
    "termination_rules",
    "confidence_notes",
}


def _coerce_to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        # Preserve semantic content as a single structured item.
        return [value]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        # Accept model outputs that stringify arrays/objects.
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, (dict, str, int, float, bool)):
                return [parsed]
        except Exception:
            pass
        # Fallback: split coarse bullet/line strings into entries.
        lines = [line.strip("-* \t") for line in txt.splitlines() if line.strip()]
        return lines if lines else [txt]
    return [value]


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Try direct JSON parse first.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Strip fenced markdown blocks and retry.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # Find first object-like payload.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def _scm_validation_errors(scm_obj: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(scm_obj, dict):
        return ["SCM object is not a JSON object"]

    missing = REQUIRED_SCM_KEYS.difference(set(scm_obj.keys()))
    if missing:
        errors.append(f"Missing required keys: {sorted(missing)}")

    if not isinstance(scm_obj.get("meta", {}), dict):
        errors.append("'meta' must be an object")
    if not isinstance(scm_obj.get("equations", []), list):
        errors.append("'equations' must be an array")
    if not isinstance(scm_obj.get("action_effects", []), list):
        errors.append("'action_effects' must be an array")
    if not isinstance(scm_obj.get("interaction_rules", []), list):
        errors.append("'interaction_rules' must be an array")
    return errors


def _load_blueprint_from_path(blueprint_path: Optional[str]) -> SCMBlueprint:
    # Default blueprint is imported from scm_blueprint.py as `bp`.
    if not blueprint_path:
        return bp

    path = Path(blueprint_path)
    if not path.exists():
        return bp

    # If path points to scm_blueprint.py, use imported object.
    if path.name == "scm_blueprint.py":
        return bp

    # Optional JSON blueprint support.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes", {})
        edges = payload.get("edges", [])
        notes = payload.get("scm_notes", [])
        allow_extra = bool(payload.get("allow_extra", True))
        return SCMBlueprint(nodes=nodes, edges=edges, scm_notes=notes, allow_extra=allow_extra)
    except Exception:
        return bp


class SCMManager:
    def __init__(
        self,
        model_name: str,
        run_dir: str,
        run_id: int,
        game_env_id: str,
        blueprint_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.run_dir = Path(run_dir).resolve()
        self.run_id = int(run_id)
        self.game_env_id = game_env_id
        self.blueprint = _load_blueprint_from_path(blueprint_path)
        # SCM bootstrap/update should stay structured and short; disable thinking mode.
        self.client = create_client_from_config(
            model_name,
            runtime_overrides={"qwen_thinking_mode": "off"},
        )
        if hasattr(self.client, "max_tokens"):
            self.client.max_tokens = 2048

        self.scm_dir = self.run_dir / "scm"
        self.scm_dir.mkdir(parents=True, exist_ok=True)

        self.initial_path = self.scm_dir / f"run_{self.run_id}__initial.json"
        self.current_path = self.scm_dir / f"run_{self.run_id}__current_belief.json"
        self.manifest_path = self.scm_dir / f"run_{self.run_id}__update_manifest.json"
        self.bootstrap_raw_response_path = self.scm_dir / f"run_{self.run_id}__bootstrap_raw_response.txt"

        self.current_belief: Optional[Dict[str, Any]] = None
        self.update_manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "game_env_id": self.game_env_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updates_disabled": True,
            "bootstrap": None,
        }
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.update_manifest, indent=2), encoding="utf-8")

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_initial_prompt(self, vgdl_rules: str, level_layout: Optional[str]) -> str:
        return (
            "Build a machine-readable SCM JSON from raw VGDL rules and level layout.\n"
            "Do not use chain-of-thought. Return JSON only.\n"
            "Schema keys required exactly:\n"
            f"{sorted(REQUIRED_SCM_KEYS)}\n\n"
            "Important: equations, action_effects, interaction_rules, reward_rules, "
            "failure_conditions, termination_rules, confidence_notes MUST all be JSON arrays.\n\n"
            "SCM blueprint:\n"
            f"{self.blueprint.to_text()}\n\n"
            "Raw VGDL:\n"
            f"{vgdl_rules or ''}\n\n"
            "Level layout:\n"
            f"{level_layout or ''}\n"
        )

    def _finalize_scm(self, scm_obj: Dict[str, Any], meta_overrides: Dict[str, Any]) -> Dict[str, Any]:
        out = deepcopy(scm_obj)
        out.setdefault("meta", {})
        out["meta"].update(meta_overrides)
        out.setdefault("blueprint_nodes", self.blueprint.nodes)
        out.setdefault("blueprint_edges", [list(edge) for edge in self.blueprint.edges])
        out["static_nodes"] = _coerce_to_list(out.get("static_nodes", []))
        out["dynamic_variables"] = _coerce_to_list(out.get("dynamic_variables", []))
        out["equations"] = _coerce_to_list(out.get("equations", []))
        out["action_effects"] = _coerce_to_list(out.get("action_effects", []))
        out["interaction_rules"] = _coerce_to_list(out.get("interaction_rules", []))
        out["reward_rules"] = _coerce_to_list(out.get("reward_rules", []))
        out["failure_conditions"] = _coerce_to_list(out.get("failure_conditions", []))
        out["termination_rules"] = _coerce_to_list(out.get("termination_rules", []))
        out["confidence_notes"] = _coerce_to_list(out.get("confidence_notes", []))
        return out

    def bootstrap(self, vgdl_rules: str, level_layout: Optional[str]) -> Tuple[bool, Optional[str]]:
        prompt = self._build_initial_prompt(vgdl_rules=vgdl_rules, level_layout=level_layout)
        response = self.client.query(prompt)
        try:
            self.bootstrap_raw_response_path.write_text(response or "", encoding="utf-8")
        except Exception:
            pass
        parsed = _safe_json_extract(response)
        if parsed is None:
            self.update_manifest["bootstrap"] = {
                "status": "failed",
                "reason": "parse_error",
                "raw_response_path": str(self.bootstrap_raw_response_path.resolve()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self._flush_manifest()
            return False, "Could not parse SCM bootstrap response as JSON."

        finalized = self._finalize_scm(
            parsed,
            meta_overrides={
                "source": "bootstrap",
                "game_env_id": self.game_env_id,
                "run_id": self.run_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )
        errors = _scm_validation_errors(finalized)
        if errors:
            self.update_manifest["bootstrap"] = {
                "status": "failed",
                "reason": "validation_error",
                "errors": errors,
                "raw_response_path": str(self.bootstrap_raw_response_path.resolve()),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self._flush_manifest()
            return False, "; ".join(errors)

        self.current_belief = finalized
        self._write_json(self.initial_path, finalized)
        self._write_json(self.current_path, finalized)
        self.update_manifest["bootstrap"] = {
            "status": "ok",
            "path": str(self.initial_path.resolve()),
            "raw_response_path": str(self.bootstrap_raw_response_path.resolve()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self._flush_manifest()
        return True, None

    def render_prompt_context(self, max_items: int = 12) -> str:
        if not self.current_belief:
            return ""
        belief = self.current_belief
        equations = belief.get("equations", [])[:max_items]
        action_effects = belief.get("action_effects", [])[:max_items]
        interactions = belief.get("interaction_rules", [])[:max_items]
        reward_rules = belief.get("reward_rules", [])[:max_items]
        failures = belief.get("failure_conditions", [])[:max_items]
        terminations = belief.get("termination_rules", [])[:max_items]
        return (
            "Belief SCM summary:\n"
            f"- Equations: {json.dumps(equations, ensure_ascii=True)}\n"
            f"- Action effects: {json.dumps(action_effects, ensure_ascii=True)}\n"
            f"- Interactions: {json.dumps(interactions, ensure_ascii=True)}\n"
            f"- Reward rules: {json.dumps(reward_rules, ensure_ascii=True)}\n"
            f"- Failure conditions: {json.dumps(failures, ensure_ascii=True)}\n"
            f"- Termination rules: {json.dumps(terminations, ensure_ascii=True)}"
        )

    def get_paths(self) -> Dict[str, str]:
        return {
            "scm_initial_path": str(self.initial_path.resolve()),
            "scm_current_belief_path": str(self.current_path.resolve()),
            "scm_manifest_path": str(self.manifest_path.resolve()),
        }

    def get_summary(self) -> Dict[str, Any]:
        return {
            **self.get_paths(),
            "scm_bootstrap_status": (self.update_manifest.get("bootstrap") or {}).get("status"),
            "scm_manifest": self.update_manifest,
        }


def mapped_ascii_for_scm(ascii_state: str, sprite_map: Optional[Dict[str, str]]) -> str:
    if not ascii_state:
        return ""
    if not sprite_map:
        return ascii_state

    char_to_sprite: Dict[str, str] = {}
    for sprite, ch in sprite_map.items():
        if isinstance(ch, str) and len(ch) == 1 and ch not in char_to_sprite:
            char_to_sprite[ch] = str(sprite)

    mapped_lines: List[str] = []
    for line in ascii_state.splitlines():
        mapped_lines.append(" ".join(char_to_sprite.get(ch, ch) for ch in line))
    return "\n".join(mapped_lines)
