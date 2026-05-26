import os
import re
import time
import random
from collections import Counter
from typing import Optional, Any, Tuple, Literal, Dict
from llm.client import create_client_from_config
from llm.utils.agent_components import (
    parse_action_from_response_with_meta,
    parse_action_sequence_from_response,
)
from llm.utils.build_prompt import build_static_prompt, build_dynamic_prompt, PromptLogger, ReflectionManager
from llm.utils.config import get_profile_config
from llm.utils.game_analysis import generate_full_analysis_report

_VERBOSITY_ORDER = {"quiet": 0, "info": 1, "debug": 2}
_THINK_FLAG_PATTERN = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_THINK_BLOCK_PATTERN = re.compile(r"<\s*think\b[^>]*>[\s\S]*?<\s*/\s*think\s*>", re.IGNORECASE)

def _current_verbosity() -> str:
    v = os.getenv("GVGAI_LOG_VERBOSITY", "info").strip().lower()
    return v if v in _VERBOSITY_ORDER else "info"

def _log(level: str, message: str) -> None:
    current = _current_verbosity()
    if _VERBOSITY_ORDER.get(current, 1) >= _VERBOSITY_ORDER.get(level, 1):
        print(message)

def rotate_ascii_left(ascii_map: str) -> str:
    lines = [list(line) for line in ascii_map.strip().splitlines()]
    if not lines:
        return ascii_map
    rotated = list(zip(*lines[::-1]))
    return "\n".join("".join(row) for row in rotated)

class LLMPlayer:
    def __init__(
        self,
        model_name: str,
        env: Any,
        vgdl_rules: str,
        initial_state: Optional[str] = None,
        extra_prompt: Optional[str] = '',
        mode: Literal["contextual", "zero-shot"] = "contextual",
        rotate_state: bool = False,
        expand_state: bool = False,
        log_dir: Optional[str] = None,
        max_action_history_len: Optional[int] = 3,  # New parameter for history length
        qwen_thinking: Literal["auto", "on", "off"] = "auto",
        qwen_sampling_overrides: Optional[Dict[str, Any]] = None,
        planning_mode: Literal["off", "lookahead_actions"] = "off",
        planning_horizon_x: int = 1,
        shuffle_action_meanings: bool = False,
        action_shuffle_seed: int = 42,
    ):
        self.model_name = model_name
        self.env = env
        self.vgdl_rules = vgdl_rules
        self.mode = mode
        self.extra_prompt = extra_prompt
        self.rotate_state = rotate_state
        self.expand_state = expand_state
        self.max_action_history_len = max_action_history_len # Store the max length
        self.qwen_thinking = qwen_thinking
        self.qwen_sampling_overrides = qwen_sampling_overrides or {}
        self.planning_mode = planning_mode
        self.planning_horizon_x = max(1, int(planning_horizon_x))
        self.shuffle_action_meanings = bool(shuffle_action_meanings)
        self.action_shuffle_seed = int(action_shuffle_seed)
        self.llm_client = create_client_from_config(
            model_name,
            runtime_overrides={
                "qwen_thinking_mode": self.qwen_thinking,
                "qwen_sampling_overrides": self.qwen_sampling_overrides,
            }
        )
        # Allow long outputs for thinking-mode runs to avoid truncating action/planning tails.
        if hasattr(self.llm_client, "max_tokens"):
            try:
                _ = int(getattr(self.llm_client, "max_tokens", 16000) or 16000)
                self.llm_client.max_tokens = 16000
            except Exception:
                self.llm_client.max_tokens = 16000
        self.reflection_mgr = ReflectionManager()
        self.position_history = []
        self.sprite_map = {}
        self.total_reward = 0.0
        self.state_history = []
        self.action_history = []
        self.last_state = initial_state
        self.winner = None
        self.last_token_usage = {}
        self.last_step_time_seconds = 0.0
        self.last_action_overridden = False
        self.action_plan_queue = []
        self.last_action_from_plan_queue = False
        self.last_early_replan_triggered = False
        self.plan_queries_count = 0
        self.queued_actions_executed = 0
        self.planning_parse_fail_count = 0
        self.early_replan_count = 0
        self._planned_actions_generated = 0
        self.think_sanitized_count = 0
        self.think_incomplete_retry_count = 0
        self.think_incomplete_terminal_fail_count = 0
        self.action_think_incomplete_retry_count = 0
        self.action_think_incomplete_terminal_fail_count = 0
        self.plan_think_incomplete_retry_count = 0
        self.plan_think_incomplete_terminal_fail_count = 0
        self.token_rung_max_used = 0
        self.action_parse_success_count = 0
        self.action_parse_fail_count = 0
        self.action_parse_tolerant_success_count = 0
        self.action_parse_strict_success_count = 0
        self.action_parse_tier_counts = Counter()
        self.action_fallback_count = 0
        self.action_fallback_last_valid_count = 0
        self.action_fallback_nil_count = 0
        self.plan_parse_success_count = 0
        self.plan_parse_partial_count = 0
        self.queued_no_progress_streak = 0
        self.last_fallback_used = False
        self.last_fallback_type = "none"
        self.last_parse_tier = "none"
        self.last_parse_reason = ""
        self.last_think_retry_used_action = False
        self.last_think_retry_used_plan = False

        try:
            self.action_map = {
                i: env.unwrapped.get_action_meanings()[i]
                for i in range(env.action_space.n)
            }
        except AttributeError:
            self.action_map = {i: f"ACTION_{i}" for i in range(env.action_space.n)}
        self.action_map_prompt = dict(self.action_map)
        if self.shuffle_action_meanings:
            shuffled_items = list(self.action_map_prompt.items())
            random.Random(self.action_shuffle_seed).shuffle(shuffled_items)
            self.action_map_prompt = dict(shuffled_items)
        self.action_prompt_order_ids = [int(aid) for aid in self.action_map_prompt.keys()]
        self.last_mentioned_action_total_count = 0
        self.last_mentioned_action_match_count = 0
        self.last_prompt_last_mentioned_action_id = None
        self.last_action_matches_last_mentioned = False

        game_name = getattr(env.unwrapped.spec, "id", "UnknownEnv")
        self.logger = PromptLogger(model_name=self.model_name, game_name=game_name, log_dir=log_dir)

        if self.mode == "contextual":
            self.llm_client.clear_history()
            static_prompt = build_static_prompt(
                vgdl_rules=self.vgdl_rules,
                action_map=self.action_map_prompt,
                optional_prompt=self.extra_prompt
            )


            self.llm_client.set_system_prompt(static_prompt)

    @staticmethod
    def _has_think_flags(response: str) -> bool:
        return bool(_THINK_FLAG_PATTERN.search(str(response or "")))

    @staticmethod
    def _sanitize_thinking_output(response: str) -> Tuple[str, bool, bool]:
        raw = str(response or "")
        had_think = bool(_THINK_FLAG_PATTERN.search(raw))
        cleaned = _THINK_BLOCK_PATTERN.sub("", raw).strip()
        residual_think = bool(_THINK_FLAG_PATTERN.search(cleaned))
        return cleaned, had_think, residual_think

    def _select_fallback_action(self) -> Tuple[int, str, str]:
        if self.action_history:
            aid = int(self.action_history[-1])
            return aid, self.action_map.get(aid, f"ACTION_{aid}"), "last_valid"
        if 0 in self.action_map:
            return 0, self.action_map[0], "nil"
        fallback_id = min(int(k) for k in self.action_map.keys())
        return fallback_id, self.action_map[fallback_id], "nil"

    def _query_sanitized_with_single_think_retry(
        self,
        prompt: str,
        image_path: Optional[str],
        label: str,
        count_as_action: bool = True,
    ) -> Tuple[str, Dict[str, Any], bool, bool]:
        """
        Query once; if response has incomplete think tags after sanitization, retry once.
        Returns: (sanitized_response, usage_dict, success, retry_used)
        """
        retry_used = False
        usage: Dict[str, Any] = {}
        for attempt in range(2):
            if attempt > 0 and hasattr(self.llm_client, "messages"):
                # Keep system/older context, but drop the immediate failed turn before retrying.
                msgs = getattr(self.llm_client, "messages", None)
                if isinstance(msgs, list) and msgs:
                    if isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant":
                        msgs.pop()
                    if (
                        msgs
                        and isinstance(msgs[-1], dict)
                        and msgs[-1].get("role") == "user"
                        and str(msgs[-1].get("content", "")) == str(prompt)
                    ):
                        msgs.pop()

            response = self.llm_client.query(prompt, image_path=image_path)
            self.logger.log_response(response)
            u = self.llm_client.get_last_usage()
            usage = u if isinstance(u, dict) else {}

            if not response or not str(response).strip():
                return "", usage, False, retry_used
            if str(response).strip().startswith("Error:"):
                return str(response), usage, False, retry_used

            sanitized, had_think, residual_think = self._sanitize_thinking_output(response)
            if had_think:
                self.think_sanitized_count += 1
            if not residual_think:
                return sanitized, usage, True, retry_used

            retry_used = True
            self.think_incomplete_retry_count += 1
            if count_as_action:
                self.action_think_incomplete_retry_count += 1
            else:
                self.plan_think_incomplete_retry_count += 1
            _log("info", f"[THINK-INCOMPLETE] {label}: retrying once with same prompt.")

        # Second attempt still had incomplete think tags.
        self.think_incomplete_terminal_fail_count += 1
        if count_as_action:
            self.action_think_incomplete_terminal_fail_count += 1
        else:
            self.plan_think_incomplete_terminal_fail_count += 1
        return "", usage, False, retry_used

    def _resolve_action_with_fallback(self, response: str) -> Tuple[int, str]:
        meta = parse_action_from_response_with_meta(response, self.action_map)
        self.last_parse_tier = str(meta.get("parse_tier", "failed"))
        self.last_parse_reason = str(meta.get("parse_reason", ""))
        if meta.get("parse_success"):
            aid = int(meta["action_id"])
            aname = self.action_map.get(aid, f"ACTION_{aid}")
            self.action_parse_success_count += 1
            tier = str(meta.get("parse_tier", ""))
            self.action_parse_tier_counts[tier] += 1
            if tier == "strict":
                self.action_parse_strict_success_count += 1
            else:
                self.action_parse_tolerant_success_count += 1
            self.last_fallback_used = False
            self.last_fallback_type = "none"
            self.last_action_overridden = False
            return aid, aname

        # Controlled fallback: one deterministic fallback action per parse failure.
        self.action_parse_fail_count += 1
        self.action_fallback_count += 1
        fallback_action, fallback_name, fallback_type = self._select_fallback_action()
        if fallback_type == "last_valid":
            self.action_fallback_last_valid_count += 1
        else:
            self.action_fallback_nil_count += 1
        self.last_fallback_used = True
        self.last_fallback_type = fallback_type
        self.last_action_overridden = True
        self.last_parse_tier = "failed"
        self.last_parse_reason = str(meta.get("parse_reason", "parse_failed"))
        return int(fallback_action), fallback_name

    def _query_with_adaptive_thinking_retry(
        self,
        prompt: str,
        image_path: Optional[str],
        token_rungs: Tuple[int, ...],
        api_retries_per_rung: int,
        label: str,
    ) -> Tuple[str, Dict[str, Any], bool]:
        """
        Returns: (sanitized_response, token_usage, success)
        success=False means all rungs exhausted with API/empty/incomplete-think failures.
        """
        base_wait_time = 5
        last_usage = {}

        for rung_idx, rung_max_tokens in enumerate(token_rungs):
            self.token_rung_max_used = max(self.token_rung_max_used, int(rung_max_tokens))
            _log("info", f"[THINK-RETRY] {label}: using max_tokens={rung_max_tokens} (rung {rung_idx + 1}/{len(token_rungs)}).")

            for api_attempt in range(api_retries_per_rung):
                if self.mode == "zero-shot":
                    # Prevent retry attempts from bloating chat history in zero-shot flow.
                    self.llm_client.clear_history()

                try:
                    response = self.llm_client.query(
                        prompt,
                        image_path=image_path,
                        max_tokens_override=int(rung_max_tokens),
                    )
                except TypeError:
                    # Backward-compatible path for clients without per-call overrides.
                    previous_max = getattr(self.llm_client, "max_tokens", None)
                    if hasattr(self.llm_client, "max_tokens"):
                        self.llm_client.max_tokens = int(rung_max_tokens)
                    response = self.llm_client.query(prompt, image_path=image_path)
                    if hasattr(self.llm_client, "max_tokens"):
                        self.llm_client.max_tokens = previous_max
                self.logger.log_response(response)
                last_usage = self.llm_client.get_last_usage()

                if response and response.strip().startswith("Error:"):
                    _log("info", f"[API ERROR] {label} attempt {api_attempt + 1}/{api_retries_per_rung} at max_tokens={rung_max_tokens}: {response.strip()}")
                    if api_attempt < api_retries_per_rung - 1:
                        wait_time = base_wait_time * (2 ** api_attempt)
                        time.sleep(wait_time)
                        continue
                    break

                if not response or not response.strip():
                    _log("info", f"[WARNING] Empty response ({label}) attempt {api_attempt + 1}/{api_retries_per_rung} at max_tokens={rung_max_tokens}.")
                    if api_attempt < api_retries_per_rung - 1:
                        time.sleep(base_wait_time)
                        continue
                    break

                sanitized, had_think, residual_think = self._sanitize_thinking_output(response)
                if had_think:
                    self.think_sanitized_count += 1
                if residual_think:
                    self.think_incomplete_retry_count += 1
                    _log("info", f"[THINK-INCOMPLETE] {label}: residual think flags after sanitization at max_tokens={rung_max_tokens}.")
                    break

                _log("debug", f"[THINK-SANITIZED] {label}: sanitized response accepted.")
                return sanitized, (last_usage if isinstance(last_usage, dict) else {}), True

        self.think_incomplete_terminal_fail_count += 1
        return "", (last_usage if isinstance(last_usage, dict) else {}), False

    def select_action(
        self,
        current_state: str,
        current_position: Optional[str] = None,
        current_image_path: Optional[str] = None,
        last_image_path: Optional[str] = None,
        sprite_map: Optional[dict] = None,
        extra_prompt: Optional[str] = None,
        plan: Optional[str] = None,
        scm_context: Optional[str] = None,
    ) -> int:
        # if self.rotate_state:
        #     rotated_state = rotate_ascii_left(current_state)
        #     current_state = f"=== Original State ===\n{current_state}\n\n=== Rotated State (Y-axis emphasis) ===\n{rotated_state}"
        self.last_early_replan_triggered = False
        self.last_action_from_plan_queue = False
        self.last_fallback_used = False
        self.last_fallback_type = "none"
        self.last_parse_tier = "none"
        self.last_parse_reason = ""
        self.last_think_retry_used_action = False
        self.last_think_retry_used_plan = False

        self.state_history.append(current_state)
        self.position_history.append(current_position)

        config = get_profile_config(self.model_name)
        # The model name for tokenizer/logging is now resolved by get_profile_config.
        model_for_tokenizer = config.get("model")
        if not model_for_tokenizer:
            # Fallback if 'model' is not in the profile,
            # though this should ideally be caught by config validation earlier.
            print(f"Warning: 'model' not found in profile '{self.model_name}'. Using profile name for tokenizer.")
            model_for_tokenizer = self.model_name # Or a default like "gpt-3.5-turbo" if that's safer for tiktoken

        if self.mode == "zero-shot":
            self.llm_client.clear_history()
            prompt = build_static_prompt(
                vgdl_rules=self.vgdl_rules,
                action_map=self.action_map_prompt,
                optional_prompt=extra_prompt
            )
            prompt += "\n\n" + build_dynamic_prompt(
                current_ascii=current_state,
                current_image_path=current_image_path,
                avatar_position=current_position,
                action_map=self.action_map_prompt,
                sprite_mapping=sprite_map,
                rotate=self.rotate_state,
                expanded=self.expand_state,
                plan=plan,
                scm_context=scm_context,
            )
        else:
            previous_state = self.state_history[-2] if len(self.state_history) >= 2 else None
            last_position = self.position_history[-2] if len(self.position_history) >= 2 else current_position
            
            # Truncate action history before passing to build_dynamic_prompt
            effective_action_history = self.action_history
            if self.max_action_history_len is not None and len(self.action_history) > self.max_action_history_len:
                effective_action_history = self.action_history[-self.max_action_history_len:]

            prompt = build_dynamic_prompt(
                current_ascii=current_state,
                last_ascii=previous_state,
                current_image_path=current_image_path,
                avatar_position=current_position,
                last_position=last_position,
                action_map=self.action_map_prompt,
                action_history=effective_action_history, # Use truncated history
                reflection_manager=self.reflection_mgr,
                logger=self.logger,
                llm_model_name=model_for_tokenizer, # Use the resolved model name
                sprite_mapping=sprite_map,
                rotate=self.rotate_state,
                expanded=self.expand_state,
                plan=plan,
                scm_context=scm_context,
            )

        step_idx = len(self.action_history) + 1
        # Persist every step prompt so action/planning branches are fully auditable.
        self.logger.log("prompt", prompt)
        self.logger.log_sampled_step_input(step=step_idx, prompt=prompt, frequency=10)
        self.last_prompt_last_mentioned_action_id = self._extract_last_mentioned_action_id(prompt)
        self.last_action_matches_last_mentioned = False

        step_token_usage = {}
        step_started_at = time.time()

        # Use queued plan actions between planning queries.
        if (
            self.planning_mode == "lookahead_actions"
            and self.planning_horizon_x > 1
            and self.action_plan_queue
        ):
            action = int(self.action_plan_queue.pop(0))
            self.last_action_from_plan_queue = True
            self.queued_actions_executed += 1
            step_token_usage = self.llm_client._usage_unavailable() if hasattr(self.llm_client, "_usage_unavailable") else {}
            self.last_token_usage = step_token_usage if isinstance(step_token_usage, dict) else {}
            self.last_step_time_seconds = max(0.0, time.time() - step_started_at)
            self.action_history.append(action)
            self.last_state = current_state
            self.last_parse_tier = "plan_queue"
            self.last_parse_reason = "queued_plan_action"
            if self.last_prompt_last_mentioned_action_id is not None:
                self.last_mentioned_action_total_count += 1
                self.last_action_matches_last_mentioned = int(action) == int(self.last_prompt_last_mentioned_action_id)
                if self.last_action_matches_last_mentioned:
                    self.last_mentioned_action_match_count += 1
            return action
        
        response, step_token_usage, action_query_ok, action_retry_used = self._query_sanitized_with_single_think_retry(
            prompt=prompt,
            image_path=current_image_path,
            label="action_query",
            count_as_action=True,
        )
        self.last_think_retry_used_action = bool(action_retry_used)
        if not action_query_ok and not response:
            raise RuntimeError("LLM returned empty action response; no fallback action allowed in strict experiment mode.")
        if not action_query_ok and str(response).strip().startswith("Error:"):
            raise RuntimeError(f"LLM action query error: {str(response).strip()}")
        if not action_query_ok:
            raise RuntimeError("LLM action response has incomplete <think> tags; no fallback action allowed in strict experiment mode.")
        self.token_rung_max_used = 0

        if (
            self.planning_mode == "lookahead_actions"
            and self.planning_horizon_x > 1
        ):
            self.last_action_from_plan_queue = False
            planning_prompt = (
                prompt
                + "\n\n=== Planning Task ===\n"
                + f"Plan exactly the next {self.planning_horizon_x} actions.\n"
                + "Return one line only in this exact format:\n"
                + f"PlanActions:[a1,a2,...,a{self.planning_horizon_x}]\n"
                + "Use only valid numeric IDs from the Action Legend."
            )
            plan_response, plan_usage, plan_query_ok, plan_retry_used = self._query_sanitized_with_single_think_retry(
                prompt=planning_prompt,
                image_path=current_image_path,
                label="planning_query",
                count_as_action=False,
            )
            self.last_think_retry_used_plan = bool(plan_retry_used)
            if isinstance(plan_usage, dict) and plan_usage:
                step_token_usage = plan_usage
            plan_actions = []
            if plan_query_ok:
                plan_actions = parse_action_sequence_from_response(
                    plan_response,
                    self.action_map,
                    horizon=self.planning_horizon_x,
                )
            if plan_actions:
                self.plan_queries_count += 1
                self._planned_actions_generated += len(plan_actions)
                self.plan_parse_success_count += 1
                self.queued_no_progress_streak = 0
                if len(plan_actions) < self.planning_horizon_x:
                    self.plan_parse_partial_count += 1
                action = int(plan_actions[0])
                action_name = self.action_map.get(action, f"ACTION_{action}")
                self.action_plan_queue = [int(a) for a in plan_actions[1:]]
                self.last_parse_tier = "plan_head"
                self.last_parse_reason = "parsed_plan_actions"
            else:
                self.planning_parse_fail_count += 1
                action, action_name = self._resolve_action_with_fallback(response)
        else:
            self.last_action_from_plan_queue = False
            action, action_name = self._resolve_action_with_fallback(response)
        _log("debug", f"Sanitized response: {response}")
        _log("info", f"Action: {action} ({action_name})")
        self.last_token_usage = step_token_usage if isinstance(step_token_usage, dict) else {}
        self.last_step_time_seconds = max(0.0, time.time() - step_started_at)
        self.action_history.append(action)
        self.last_state = current_state
        if self.last_prompt_last_mentioned_action_id is not None:
            self.last_mentioned_action_total_count += 1
            self.last_action_matches_last_mentioned = int(action) == int(self.last_prompt_last_mentioned_action_id)
            if self.last_action_matches_last_mentioned:
                self.last_mentioned_action_match_count += 1
        return action

    @staticmethod
    def _extract_last_mentioned_action_id(prompt: str) -> Optional[int]:
        matches = re.findall(r"\bAction\s+(\d+)\s*=>", str(prompt or ""), re.IGNORECASE)
        if not matches:
            return None
        try:
            return int(matches[-1])
        except Exception:
            return None

    def update(self, action: int, reward: float, winner=None):
        self.total_reward += reward
        step = len(self.reflection_mgr.step_log)
        usage = self.last_token_usage if isinstance(self.last_token_usage, dict) else {}
        self.reflection_mgr.log_step(
            step=step,
            action=action,
            action_meaning=self.action_map.get(action, f"ACTION_{action}"),
            reward=reward,
            step_time_seconds=round(float(self.last_step_time_seconds), 6),
            token_usage_source=usage.get("token_usage_source", "unavailable"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            raw_usage=usage.get("raw_usage"),
            action_overridden=bool(self.last_action_overridden),
            fallback_used=bool(self.last_fallback_used),
            fallback_type=str(self.last_fallback_type),
            parse_tier=str(self.last_parse_tier),
            parse_reason=str(self.last_parse_reason),
            think_retry_used_action=bool(self.last_think_retry_used_action),
            think_retry_used_plan=bool(self.last_think_retry_used_plan),
            action_from_plan_queue=bool(self.last_action_from_plan_queue),
            queued_plan_action_consumed=bool(self.last_action_from_plan_queue),
            early_replan_triggered=bool(self.last_early_replan_triggered),
            last_mentioned_action_id=self.last_prompt_last_mentioned_action_id,
            action_matches_last_mentioned=bool(self.last_action_matches_last_mentioned),
        )
        self.last_action_from_plan_queue = False
        self.last_early_replan_triggered = False
        if winner is not None:
            self.winner = winner

    def get_planning_summary(self) -> Dict[str, Any]:
        avg_actions_per_query = 0.0
        if self.plan_queries_count > 0:
            avg_actions_per_query = self._planned_actions_generated / float(self.plan_queries_count)
        return {
            "planning_mode": self.planning_mode,
            "planning_horizon_x": self.planning_horizon_x,
            "plan_queries_count": int(self.plan_queries_count),
            "queued_actions_executed": int(self.queued_actions_executed),
            "early_replan_count": int(self.early_replan_count),
            "planning_parse_fail_count": int(self.planning_parse_fail_count),
            "plan_parse_success_count": int(self.plan_parse_success_count),
            "plan_parse_partial_count": int(self.plan_parse_partial_count),
            "action_parse_success_count": int(self.action_parse_success_count),
            "action_parse_fail_count": int(self.action_parse_fail_count),
            "action_parse_tolerant_success_count": int(self.action_parse_tolerant_success_count),
            "action_parse_strict_success_count": int(self.action_parse_strict_success_count),
            "action_parse_tier_counts": dict(self.action_parse_tier_counts),
            "action_fallback_count": int(self.action_fallback_count),
            "action_fallback_last_valid_count": int(self.action_fallback_last_valid_count),
            "action_fallback_nil_count": int(self.action_fallback_nil_count),
            "think_sanitized_count": int(self.think_sanitized_count),
            "think_incomplete_retry_count": int(self.think_incomplete_retry_count),
            "think_incomplete_terminal_fail_count": int(self.think_incomplete_terminal_fail_count),
            "action_think_incomplete_retry_count": int(self.action_think_incomplete_retry_count),
            "action_think_incomplete_terminal_fail_count": int(self.action_think_incomplete_terminal_fail_count),
            "plan_think_incomplete_retry_count": int(self.plan_think_incomplete_retry_count),
            "plan_think_incomplete_terminal_fail_count": int(self.plan_think_incomplete_terminal_fail_count),
            "token_rung_max_used": int(self.token_rung_max_used),
            "avg_actions_per_plan_query": round(avg_actions_per_query, 6),
        }

    def get_last_mentioned_action_summary(self) -> Dict[str, Any]:
        total = int(self.last_mentioned_action_total_count)
        matches = int(self.last_mentioned_action_match_count)
        match_rate = (matches / float(total)) if total > 0 else 0.0
        return {
            "shuffle_action_meanings": bool(self.shuffle_action_meanings),
            "action_shuffle_seed": int(self.action_shuffle_seed),
            "action_prompt_order_ids": list(self.action_prompt_order_ids),
            "last_mentioned_action_total_count": total,
            "last_mentioned_action_match_count": matches,
            "last_mentioned_action_match_rate": round(match_rate, 6),
        }

    def save_logs(self):
        self.logger.save()
        self.llm_client.save_history(self.logger.game_name)

    def export_analysis(
        self,
        output_dir: str,
        runtime_info: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ):
        generate_full_analysis_report(
            reflection_manager=self.reflection_mgr,
            states=self.state_history,
            output_dir=output_dir,
            winner=self.winner,
            runtime_info=runtime_info,
            run_metadata=run_metadata,
            action_meanings=self.action_map,
            default_action=0,
        )

    def clear_history(self):
        self.llm_client.clear_history()

class LLMPlanner:
    def __init__(self, model_name: str, vgdl_rules: str):
        self.model_name = model_name
        self.llm_client = create_client_from_config(model_name)
        self.env = None
        self.action_map = {}
        self.vgdl = vgdl_rules
        self.state_history = []
        self.strategy_history = ''

    def initialize(self, env) -> None:
        self.env = env
        if hasattr(env.unwrapped, "get_action_meanings"):
            self.action_map = {
                i: env.unwrapped.get_action_meanings()[i]
                for i in range(env.action_space.n)
            }
        else:
            self.action_map = {
                i: f"ACTION_{i}"
                for i in range(env.action_space.n)
            }

    def clear_history(self):
        self.llm_client.clear_history()

    def query(self, image_path: Optional[str] = None,
              current_state: Optional[str] = '', action_history: Optional[int] = None,
              current_position: Optional[Tuple[int, int]] = None, sprite_mapping: Optional[dict] = None, prompt: Optional[str] = '') -> str:

        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(current_state)

        current_location=''

        base_prompt = (
            "Generate ABSTRACT OBJECTIVE-ORIENTED strategies using this framework:\n"
            "1. SYMBOL SEMANTICS: Use ONLY symbol names from the sprite mapping (e.g. 'door' not '%')\n"
            "2. MECHANICAL PURPOSE: Focus on how symbol types interact (e.g. 'keys open doors')\n"
            "3. ZONE PROGRESSION: Describe objectives by area features (e.g. 'eastern laser zone')\n"
            "4. FAILURE RECOVERY: If stuck, switch symbol type priorities with **Alert**\n\n"
            "Forbidden in responses:\n"
            "- Coordinates (x=.../y=...)\n"
            "- Directional commands (left/right/up/down)\n"
            "- Explicit positions (column/row)\n\n"
            "Required structure:\n"
            "1. CURRENT CAPABILITY: What symbol interactions are possible now?\n"
            "2. STRATEGIC CHOICE: Which symbol type best enables progression?\n"
            "3. EXECUTION PRINCIPLE: How should interactions be performed?\n"
            "   Example: 'Batch-process all nearby keys before approaching doors'"
        )
        last_strategy = self.strategy_history
        format_prompt = (
            "\nFormat response as: "
            "```**<symbol_type> strategy: <mechanism>** with/without **Alert**```\n"
            "Example: **door strategy: Collect keys to bypass gate** \n If you recieved a BAD feedback, you MUST revise your strategy."
        )

        sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()]
        sprite_mapping_prompt = ("=== Sprite Mapping ===\n" + "\n".join(sprite_lines))
        prev_state_prompt = '\n======Previous State=======\n' + prev_state
        current_state_prompt = '\n=======Current State========\n' + current_state
        action_history_text = '\n=====Action Sequence=====\n' + f'{action_history}\n'
        if current_position:
            current_location = (
                '\n======Current Location=======\n'
                f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
                "Coordinate system: X+ → Right, Y+ → Down\n"
                "Walls block movement\n"
            )
        evaluation = '\n======Evaluator Feedback=======\n' + prompt

        full_prompt = (
            sprite_mapping_prompt + '\n' +
            prev_state_prompt + '\n' +
            current_state_prompt +
            current_location +
            action_history_text + '\n' +
            base_prompt +'\n' +
            evaluation +'\n' +
            last_strategy +'\n' +
            format_prompt
        )
        self.strategy_history = ''

        response = self.llm_client.query(full_prompt, image_path=image_path)
        return response
        # matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        # for match in matches:
        #     self.strategy_history += match
        #     print(match.strip())
        # return match if matches else ""

    def save_logs(self):
        self.logger.save()
        self.llm_client.save_history(self.logger.game_name)


class LLMEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm_client = create_client_from_config(model_name)
        self.state_history = []
        self.reward_history = []

    def clear_history(self):
        self.llm_client.clear_history()

    def query(
        self,
        current_state: str,
        # last_state: Optional[str] = None,
        action_taken: Optional[int] = None,
        reward: Optional[float] = None,
        done: Optional[bool] = False,
        current_position: Optional[Tuple[int, int]] = None,
        sprite_mapping: Optional[dict] = None,
        image_path: Optional[str] = None
    ) -> str:
        prev_state = self.state_history[-1] if self.state_history else 'None'
        self.state_history.append(current_state)

        sprite_lines = [f"{k} -> '{v}'" for k, v in sprite_mapping.items()] if sprite_mapping else []
        sprite_mapping_prompt = ("=== Sprite Mapping ===\n" + "\n".join(sprite_lines)) if sprite_lines else ""

        last_state_prompt = '\n======Previous State=======\n' + (prev_state or 'None')
        current_state_prompt = '\n=======Current State========\n' + current_state

        current_location_prompt = (
            '\n======Current Location=======\n'
            f"Avatar 'a' at (row = {current_position[0]}, col = {current_position[1]})\n"
            "Coordinate system: X+ → Right, Y+ → Down\n"
            "Walls block movement\n"
        ) if current_position else ""

        action_info = f"Action Taken: {action_taken}" if action_taken is not None else "Action Taken: None"
        reward_info = f"Reward Received: {reward}" if reward is not None else "Reward: None"
        self.reward_history.append(reward)
        done_info = f"Game Done: {done}"

        action_summary = (
            '\n=====Action Summary=====\n'
            f"{action_info}\n"
            f"{reward_info}\n"
            f"{done_info}"
        )

        # Base evaluation instruction
        base_prompt = (
                        "\nEvaluate the agent's last action with STRICT classification:\n"
            "First, decide if the action was GOOD or BAD based on:\n"
            "- EFFECTIVENESS: Did it progress toward winning?\n"
            "- RISK: Did it expose the agent to danger?\n"
            "- STRATEGIC FIT: Was it aligned with objectives?\n\n"
            "Rules:\n"
            "- You MUST classify as either GOOD or BAD.\n"
            "- Then briefly explain WHY such as blocking by a wall.\n"
            "- Format your entire response as:\n"
            f"```Evaluation: <GOOD or BAD> \nFeedback: <your reasoning and strategy> with a reward of {reward} from the environment```"
        )

        full_prompt = (
            sprite_mapping_prompt + '\n' +
            last_state_prompt + '\n' +
            current_state_prompt + '\n' +
            current_location_prompt + '\n' +
            action_summary + '\n' +
            base_prompt
        )

        response = self.llm_client.query(full_prompt, image_path=image_path)
        matches = re.findall(r"```(.*?)```", response, re.DOTALL)
        for match in matches:
            print(match.strip())
        return matches[-1] if matches else ""

    def save_logs(self):
        # Placeholder for now
        pass
