import base64
import fcntl
import os
import subprocess
import time
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

import requests
import tiktoken

from ..base import LLMClientBase
from llm.utils.config import truncate_messages_by_token


class VLLMClient(LLMClientBase):
    """
    Local vLLM client using the OpenAI-compatible HTTP API.
    If configured, it can auto-start a local vLLM server.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_start_server: bool = True,
        start_timeout_seconds: int = 300,
        server_args: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,
        fail_fast_on_empty: bool = True,
        log_server_output: bool = True,
        server_log_dir: Optional[str] = None,
        qwen_thinking_mode: str = "auto",
    ):
        super().__init__(model_name="vllm", model=model or "Qwen/Qwen3-8B")

        self.default_model = model or "Qwen/Qwen3-8B"
        self.base_url = (base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.chat_endpoint = "/v1/chat/completions"
        self.models_endpoint = "/v1/models"
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.temperature = 0.0
        self.max_tokens = 2000
        self.top_p = 1.0
        self.top_k = None
        self.min_p = None
        explicit_max_context_tokens = (
            int(max_context_tokens) if max_context_tokens and int(max_context_tokens) > 0 else None
        )
        self.max_context_tokens = explicit_max_context_tokens or 8000
        self.max_model_len = self._resolve_max_model_len(
            configured_max_model_len=max_model_len,
            server_args=server_args,
            fallback_max_context_tokens=self.max_context_tokens,
            explicit_max_context_tokens=explicit_max_context_tokens,
        )

        self.auto_start_server = auto_start_server
        self.start_timeout_seconds = start_timeout_seconds
        self.server_args = server_args or []
        self.vllm_process = None
        self.started_by_me = False
        self.fail_fast_on_empty = fail_fast_on_empty
        self.log_server_output = log_server_output
        self.server_log_dir = Path(server_log_dir) if server_log_dir else (
            Path(__file__).resolve().parents[2] / "log" / "vllm_server"
        )
        self.server_stdout_log = None
        self.server_stderr_log = None
        self.server_stdout_handle = None
        self.server_stderr_handle = None
        self.vllm_executable = self._resolve_vllm_executable()
        self.qwen_thinking_mode = (qwen_thinking_mode or "auto").strip().lower()
        if self.qwen_thinking_mode not in {"auto", "on", "off"}:
            self.qwen_thinking_mode = "auto"

        if system_prompt:
            self.set_system_prompt(system_prompt)

        if self.auto_start_server:
            self.ensure_server_running()

    def _resolve_max_model_len(
        self,
        configured_max_model_len: Optional[int],
        server_args: Optional[List[str]],
        fallback_max_context_tokens: int,
        explicit_max_context_tokens: Optional[int] = None,
    ) -> int:
        resolved = None
        if configured_max_model_len and configured_max_model_len > 0:
            resolved = int(configured_max_model_len)

        if resolved is None:
            args = server_args or []
            for idx, arg in enumerate(args):
                if arg == "--max-model-len" and idx + 1 < len(args):
                    try:
                        parsed = int(args[idx + 1])
                        if parsed > 0:
                            resolved = parsed
                            break
                    except (TypeError, ValueError):
                        pass
                elif arg.startswith("--max-model-len="):
                    try:
                        parsed = int(arg.split("=", 1)[1])
                        if parsed > 0:
                            resolved = parsed
                            break
                    except (TypeError, ValueError):
                        pass

        if resolved is None:
            resolved = int(fallback_max_context_tokens)

        if explicit_max_context_tokens and explicit_max_context_tokens > 0:
            if explicit_max_context_tokens < resolved:
                print(
                    "[VLLMClient] Capping max_model_len to explicit max_context_tokens: "
                    f"{resolved} -> {explicit_max_context_tokens}"
                )
            resolved = min(resolved, explicit_max_context_tokens)

        return int(resolved)

    def _count_prompt_tokens(self, messages: List[dict]) -> int:
        try:
            enc = tiktoken.encoding_for_model(self.default_model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            total += 3  # role/message overhead approximation
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(enc.encode(content))
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            total += len(enc.encode(part.get("text", "")))
                        elif part.get("type") == "image_url":
                            # Account for multimodal payload with a coarse constant.
                            total += 256
        return total

    def _compute_effective_max_tokens(self, messages: List[dict], configured_max_tokens: int):
        prompt_tokens = self._count_prompt_tokens(messages)
        remaining = self.max_model_len - prompt_tokens
        safety_buffer = max(64, int(math.ceil(self.max_model_len * 0.02)))
        safe_output_max = remaining - safety_buffer

        if safe_output_max <= 0:
            return None, (
                f"Error: [VLLMClient] Context overflow before generation: "
                f"prompt_tokens={prompt_tokens}, max_model_len={self.max_model_len}, "
                f"safety_buffer={safety_buffer}, remaining={remaining}."
            )

        effective = min(configured_max_tokens, safe_output_max)
        effective = max(1, effective)
        print(
            "[VLLMClient] Token budget: "
            f"prompt_tokens={prompt_tokens}, max_model_len={self.max_model_len}, "
            f"configured_max_tokens={configured_max_tokens}, effective_max_tokens={effective}"
        )
        return effective, None

    def _adjust_max_model_len_from_error(self, error_body: str) -> bool:
        """
        Parse vLLM 400 error body and update local max_model_len if server reports
        a lower authoritative context length.
        """
        if not error_body:
            return False
        match = re.search(r"maximum context length is\s+(\d+)", error_body)
        if not match:
            return False
        try:
            server_ctx = int(match.group(1))
        except (TypeError, ValueError):
            return False
        if server_ctx > 0 and server_ctx != self.max_model_len:
            old = self.max_model_len
            self.max_model_len = server_ctx
            print(
                f"[VLLMClient] Adjusted max_model_len from {old} to {self.max_model_len} "
                "based on server error."
            )
            return True
        return False

    def _healthcheck(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}{self.models_endpoint}", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _parse_host_port(self):
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return host, port

    def _wait_until_ready(self):
        deadline = time.time() + self.start_timeout_seconds
        while time.time() < deadline:
            if self.vllm_process is not None and self.vllm_process.poll() is not None:
                stderr_tail = self._tail_log(self.server_stderr_log, num_lines=80)
                raise RuntimeError(
                    "vLLM server process terminated during startup."
                    + (f"\n--- vLLM stderr tail ---\n{stderr_tail}" if stderr_tail else "")
                )
            if self._healthcheck():
                return
            time.sleep(2)

        stderr_tail = self._tail_log(self.server_stderr_log, num_lines=80)
        raise RuntimeError(
            f"Timed out waiting for vLLM server at {self.base_url} "
            f"after {self.start_timeout_seconds} seconds."
            + (f"\n--- vLLM stderr tail ---\n{stderr_tail}" if stderr_tail else "")
        )

    def _tail_log(self, log_path: Optional[Path], num_lines: int = 60) -> str:
        if not log_path or not log_path.exists():
            return ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-num_lines:]).strip()
        except Exception:
            return ""

    def _prepare_server_log_files(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.server_log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = self.server_log_dir / f"vllm_server_{timestamp}_{pid}.stdout.log"
        stderr_log = self.server_log_dir / f"vllm_server_{timestamp}_{pid}.stderr.log"
        self.server_stdout_log = stdout_log
        self.server_stderr_log = stderr_log
        return stdout_log, stderr_log

    def _start_server(self):
        host, port = self._parse_host_port()
        cmd = [self.vllm_executable, "serve", self.default_model, "--host", host, "--port", str(port)] + self.server_args
        if self.log_server_output:
            stdout_log, stderr_log = self._prepare_server_log_files()
            print(f"[VLLMClient] Starting local vLLM server with command: {' '.join(cmd)}")
            print(f"[VLLMClient] Server stdout log: {stdout_log}")
            print(f"[VLLMClient] Server stderr log: {stderr_log}")
            self.server_stdout_handle = open(stdout_log, "a", encoding="utf-8")
            self.server_stderr_handle = open(stderr_log, "a", encoding="utf-8")
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=self.server_stdout_handle,
                stderr=self.server_stderr_handle,
                text=True,
            )
        else:
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        self.started_by_me = True
        self._wait_until_ready()

    def ensure_server_running(self):
        if self._healthcheck():
            return

        host, port = self._parse_host_port()
        lock_path = f"/tmp/vllm_autostart_{host}_{port}.lock"
        os.makedirs("/tmp", exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            if self._healthcheck():
                return
            try:
                self._start_server()
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "vLLM executable not found. Install vllm in this environment first."
                ) from exc
            except Exception:
                # Another process may have just started it.
                if not self._healthcheck():
                    raise

    def _resolve_vllm_executable(self) -> str:
        # Highest priority: explicit environment override.
        explicit = os.getenv("VLLM_EXECUTABLE", "").strip()
        if explicit:
            return explicit

        # Next: lookup in PATH.
        in_path = shutil.which("vllm")
        if in_path:
            return in_path

        # Final fallback: sibling binary next to current Python interpreter.
        py_bin = os.path.dirname(sys.executable)
        candidate = os.path.join(py_bin, "vllm")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

        # Let subprocess raise FileNotFoundError with this token.
        return "vllm"

    def shutdown(self):
        if self.vllm_process and self.started_by_me:
            self.vllm_process.terminate()
            try:
                self.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
            self.vllm_process = None
            self.started_by_me = False
        if self.server_stdout_handle:
            self.server_stdout_handle.close()
            self.server_stdout_handle = None
        if self.server_stderr_handle:
            self.server_stderr_handle.close()
            self.server_stderr_handle = None

    def _headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _is_qwen3_model(self) -> bool:
        model_name = (self.default_model or "").lower()
        return "qwen3" in model_name

    def _apply_qwen_thinking_mode(self, payload: dict) -> None:
        if not self._is_qwen3_model():
            return

        print(f"[VLLMClient] Qwen thinking mode: {self.qwen_thinking_mode}")
        if self.qwen_thinking_mode == "auto":
            return

        payload["chat_template_kwargs"] = {
            "enable_thinking": self.qwen_thinking_mode == "on"
        }

    def query(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        if self.auto_start_server:
            self.ensure_server_running()

        self.add_message("user", prompt)
        if image_path:
            response_text = self._query_multimodal(prompt, image_path, max_tokens_override=max_tokens_override)
        else:
            response_text = self._query_text_only(max_tokens_override=max_tokens_override)
        self.add_message("assistant", response_text)
        return response_text

    def _query_text_only(self, max_tokens_override: Optional[int] = None) -> str:
        messages = truncate_messages_by_token(
            self.messages, self.max_context_tokens, self.default_model
        )
        configured_max_tokens = (
            int(max_tokens_override)
            if (max_tokens_override is not None and int(max_tokens_override) > 0)
            else (self.max_tokens if self.max_tokens is not None else 2000)
        )
        effective_max_tokens, overflow_error = self._compute_effective_max_tokens(
            messages=messages,
            configured_max_tokens=configured_max_tokens,
        )
        if overflow_error:
            self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
            return overflow_error

        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": self.temperature if self.temperature is not None else 0.0,
            "max_tokens": effective_max_tokens,
            "top_p": self.top_p if self.top_p is not None else 1.0,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        self._apply_qwen_thinking_mode(payload)

        attempt = 0
        corrective_retries_remaining = 1
        while attempt < 3:
            attempt += 1
            try:
                response = requests.post(
                    f"{self.base_url}{self.chat_endpoint}",
                    headers=self._headers(),
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                self._set_last_usage(self._normalize_provider_usage(data.get("usage")))
                content = data["choices"][0]["message"]["content"].strip()
                if self.get_last_usage().get("token_usage_source") != "provider_reported":
                    self._set_last_usage(self._estimate_usage(messages=messages, response_text=content, model=self.default_model))
                return content
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = response.text[:400] if response is not None else ""
                except Exception:
                    detail = ""
                if response is not None and response.status_code == 400 and self._adjust_max_model_len_from_error(detail):
                    # Recompute max_tokens with corrected server-reported context and retry immediately.
                    effective_max_tokens, overflow_error = self._compute_effective_max_tokens(
                        messages=messages,
                        configured_max_tokens=configured_max_tokens,
                    )
                    if overflow_error:
                        self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
                        return overflow_error
                    payload["max_tokens"] = effective_max_tokens
                    if attempt == 3 and corrective_retries_remaining > 0:
                        corrective_retries_remaining -= 1
                        attempt -= 1
                    continue
                err = (
                    f"Error: [VLLMClient] HTTPError status="
                    f"{response.status_code if response is not None else 'unknown'} "
                    f"attempt={attempt}/3 msg={str(e)}"
                )
                if detail:
                    err += f" body={detail}"
                if attempt < 3:
                    time.sleep(2)
                    continue
                self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
                return err
            except Exception as e:
                err = f"Error: [VLLMClient] {type(e).__name__} attempt={attempt}/3 msg={str(e)}"
                if attempt < 3:
                    time.sleep(2)
                    continue
                self._set_last_usage(self._usage_unavailable())
                return err
        self._set_last_usage(self._usage_unavailable())
        return "Error: [VLLMClient] Request failed after retries."

    def _query_multimodal(
        self,
        prompt: str,
        image_path: str,
        max_tokens_override: Optional[int] = None,
    ) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ]
        configured_max_tokens = (
            int(max_tokens_override)
            if (max_tokens_override is not None and int(max_tokens_override) > 0)
            else (self.max_tokens if self.max_tokens is not None else 2000)
        )
        effective_max_tokens, overflow_error = self._compute_effective_max_tokens(
            messages=messages,
            configured_max_tokens=configured_max_tokens,
        )
        if overflow_error:
            self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
            return overflow_error

        payload = {
            "model": self.default_model,
            "messages": messages,
            "temperature": self.temperature if self.temperature is not None else 0.0,
            "max_tokens": effective_max_tokens,
            "top_p": self.top_p if self.top_p is not None else 1.0,
        }
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        self._apply_qwen_thinking_mode(payload)

        attempt = 0
        corrective_retries_remaining = 1
        while attempt < 3:
            attempt += 1
            try:
                response = requests.post(
                    f"{self.base_url}{self.chat_endpoint}",
                    headers=self._headers(),
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()
                data = response.json()
                self._set_last_usage(self._normalize_provider_usage(data.get("usage")))
                content = data["choices"][0]["message"]["content"].strip()
                if self.get_last_usage().get("token_usage_source") != "provider_reported":
                    self._set_last_usage(self._estimate_usage(messages=messages, response_text=content, model=self.default_model))
                return content
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = response.text[:400] if response is not None else ""
                except Exception:
                    detail = ""
                if response is not None and response.status_code == 400 and self._adjust_max_model_len_from_error(detail):
                    # Recompute max_tokens with corrected server-reported context and retry immediately.
                    effective_max_tokens, overflow_error = self._compute_effective_max_tokens(
                        messages=messages,
                        configured_max_tokens=configured_max_tokens,
                    )
                    if overflow_error:
                        self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
                        return overflow_error
                    payload["max_tokens"] = effective_max_tokens
                    if attempt == 3 and corrective_retries_remaining > 0:
                        corrective_retries_remaining -= 1
                        attempt -= 1
                    continue
                err = (
                    f"Error: [VLLMClient] HTTPError status="
                    f"{response.status_code if response is not None else 'unknown'} "
                    f"attempt={attempt}/3 msg={str(e)}"
                )
                if detail:
                    err += f" body={detail}"
                if attempt < 3:
                    time.sleep(2)
                    continue
                self._set_last_usage(self._estimate_usage(messages=messages, response_text="", model=self.default_model))
                return err
            except Exception as e:
                err = f"Error: [VLLMClient] {type(e).__name__} attempt={attempt}/3 msg={str(e)}"
                if attempt < 3:
                    time.sleep(2)
                    continue
                self._set_last_usage(self._usage_unavailable())
                return err
        self._set_last_usage(self._usage_unavailable())
        return "Error: [VLLMClient] Request failed after retries."
