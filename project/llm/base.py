from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import json
import os
from datetime import datetime
from pathlib import Path


class LLMClientBase(ABC):
    """
    Base class for all LLM clients. Supports multi-turn chat by default.
    """

    def __init__(self, model_name: str, model: str):
        self.model_name = model_name
        self.default_model = model
        self.api_key = None
        self.messages: List[Dict[str, str]] = []  # full chat history
        self.last_usage: Dict[str, Any] = self._usage_unavailable()

    @abstractmethod
    def query(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        Submit a query to the LLM. If `image_path` is provided and supported,
        the implementation should handle it accordingly.
        """
        pass
    def set_system_prompt(self, system_prompt: str):
        """Optionally set system prompt message."""
        self.clear_history()
        self.messages.append({
            "role": "system",
            "content": system_prompt
        })

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "model": self.model_name  # optional metadata
        })

    def clear_history(self):
        """Clear all chat history."""
        self.messages.clear()

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    def _usage_unavailable(self) -> Dict[str, Any]:
        return {
            "token_usage_source": "unavailable",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
            "raw_usage": None,
        }

    def _normalize_provider_usage(self, raw_usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not raw_usage or not isinstance(raw_usage, dict):
            return self._usage_unavailable()

        input_tokens = self._safe_int(raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)))
        output_tokens = self._safe_int(raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)))
        total_tokens = self._safe_int(raw_usage.get("total_tokens", input_tokens + output_tokens))

        completion_details = raw_usage.get("completion_tokens_details", {})
        prompt_details = raw_usage.get("prompt_tokens_details", {})
        reasoning_tokens = None
        cached_input_tokens = None

        if isinstance(completion_details, dict):
            rt = completion_details.get("reasoning_tokens")
            if rt is not None:
                reasoning_tokens = self._safe_int(rt, default=0)
        if isinstance(prompt_details, dict):
            cit = prompt_details.get("cached_tokens")
            if cit is not None:
                cached_input_tokens = self._safe_int(cit, default=0)

        return {
            "token_usage_source": "provider_reported",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_input_tokens": cached_input_tokens,
            "raw_usage": raw_usage,
        }

    def _count_tokens_text(self, text: str, model: Optional[str] = None) -> int:
        if not text:
            return 0
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model or self.default_model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(0, len(text) // 4)

    def _count_tokens_messages(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> int:
        total = 0
        for msg in messages or []:
            total += 3  # OpenAI-style chat overhead approximation
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self._count_tokens_text(content, model=model)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        total += self._count_tokens_text(part.get("text", ""), model=model)
                    elif part.get("type") == "image_url":
                        total += 256  # coarse multimodal payload estimate
        return total

    def _estimate_usage(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        response_text: str = "",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_tokens = self._count_tokens_messages(messages or self.messages, model=model)
        output_tokens = self._count_tokens_text(response_text or "", model=model)
        return {
            "token_usage_source": "estimated",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "reasoning_tokens": None,
            "cached_input_tokens": None,
            "raw_usage": None,
        }

    def _set_last_usage(self, usage: Optional[Dict[str, Any]]):
        self.last_usage = usage if isinstance(usage, dict) else self._usage_unavailable()

    def get_last_usage(self) -> Dict[str, Any]:
        return self.last_usage if isinstance(self.last_usage, dict) else self._usage_unavailable()

    def save_history(self, game_name: Optional[str] = None, filepath: Optional[str] = None):
        """
        Save chat history to a JSON file.
        If no filepath is provided, a log directory structure will be created as: log/{model_name}/{game_name}/
        and a timestamped filename will be used.
        """
        base_log_dir = Path(__file__).parent.parent / "log"
        try:
            base_log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            base_log_dir = Path(__file__).parent / "log"
            base_log_dir.mkdir(parents=True, exist_ok=True)

        if not filepath:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"{timestamp}.json"
            subdir = base_log_dir / self.model_name
            if game_name:
                subdir = subdir / game_name
            subdir.mkdir(parents=True, exist_ok=True)
            filepath = subdir / filename

        with open(filepath, "w") as f:
            json.dump(self.messages, f, indent=2)

        print(f"[{self.model_name}] Chat history saved to {filepath}")

    def load_history(self, filepath: str):
        """Load chat history from a file."""
        with open(filepath, "r") as f:
            self.messages = json.load(f)

    def shutdown(self):
        """Optional resource cleanup."""
        pass
