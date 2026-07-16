"""Simple OpenAI-compatible LLMClient implementation.

Reads provider configuration from ``.env`` (or environment variables):

- ``LLM_PROVIDER`` — provider label (default: deepseek)
- ``LLM_API_KEY`` — API key (required)
- ``LLM_BASE_URL`` — API base URL (default: https://api.deepseek.com)
- ``LLM_MODEL`` — model name (default: deepseek-chat)

Usage:

    from generic_agent.simple_llm import SimpleLLM

    llm = SimpleLLM()
    agent = create_log_analysis_agent(llm)
    result = agent.run("check logs for errors")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from openai import OpenAI

from loop import LLMClient, LLMError, LLMResponse, ToolCall


def _load_env() -> None:
    """Load ``.env`` from the same directory as this file."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        if load_dotenv is not None:
            load_dotenv(env_path, override=False)
        else:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip("\"'")
                    if key not in os.environ:
                        os.environ[key] = val


_load_env()


class SimpleLLM(LLMClient):
    """OpenAI-compatible LLM client that reads config from ``.env``.

    Attributes:
        provider: Provider label (e.g. ``"deepseek"``, ``"openai"``).
        model_name: Model deployment name.
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
        api_key = api_key or os.getenv("LLM_API_KEY", "")
        base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.model_name = model or os.getenv("LLM_MODEL", "deepseek-chat")

        if not api_key:
            raise LLMError(
                f"LLM_API_KEY not set for provider '{self.provider}'. "
                f"Set it in {Path(__file__).resolve().parent / '.env'} "
                f"or via the LLM_API_KEY environment variable.",
                provider=self.provider,
                model=self.model_name,
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ── internal helpers ────────────────────────────────────────────────────

    def _build_kwargs(self, tools: list | None) -> dict:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            # Most OpenAI-compatible providers expect tool_choice = "auto"
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _response_from_raw(self, raw: Any) -> LLMResponse:
        """Convert an OpenAI ``chat.completions`` object to ``LLMResponse``."""
        choice = raw.choices[0] if raw.choices else None

        content = ""
        tool_calls: list[ToolCall] = []
        reasoning_content: str | None = None

        if choice and choice.message:
            msg = choice.message
            content = msg.content or ""

            # reasoning / thinking content (DeepSeek R1, etc.)
            reasoning_content = getattr(msg, "reasoning_content", None)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args: dict = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {"_raw": tc.function.arguments}
                    tool_calls.append(
                        ToolCall(
                            id=tc.id or "",
                            name=tc.function.name or "",
                            arguments=args,
                        )
                    )

        finish_reason = getattr(choice, "finish_reason", None) if choice else None
        content_filter_triggered = finish_reason == "content_filter"

        usage = None
        if hasattr(raw, "usage") and raw.usage:
            u = raw.usage
            usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                "total_tokens": getattr(u, "total_tokens", 0) or 0,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage,
            content_filter_triggered=content_filter_triggered,
            reasoning_content=reasoning_content,
        )

    # ── streaming ───────────────────────────────────────────────────────────

    def stream_chat(
        self,
        messages: list,
        tools: list | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(tools)
        accumulated_content: list[str] = []
        accumulated_reasoning: list[str] = []
        tool_call_deltas: dict[int, dict] = {}
        finish_reason: str | None = None
        usage: dict | None = None

        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        for chunk in stream:
            if should_cancel and should_cancel():
                break

            delta = chunk.choices[0] if chunk.choices else None

            # --- usage (last chunk with usage but no content) ---
            if not delta and chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens or 0,
                    "completion_tokens": chunk.usage.completion_tokens or 0,
                    "total_tokens": chunk.usage.total_tokens or 0,
                }
                continue

            if delta is None:
                continue

            # finish reason
            if delta.finish_reason:
                finish_reason = delta.finish_reason

            # text content
            if delta.delta and delta.delta.content:
                accumulated_content.append(delta.delta.content)
                if on_text_chunk:
                    on_text_chunk(delta.delta.content)

            # reasoning content
            reasoning = getattr(delta.delta, "reasoning_content", None)
            if reasoning:
                accumulated_reasoning.append(reasoning)
                if on_reasoning_chunk:
                    on_reasoning_chunk(reasoning)

            # tool calls
            if delta.delta and delta.delta.tool_calls:
                for tc_delta in delta.delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": [],
                        }
                    d = tool_call_deltas[idx]
                    if tc_delta.id:
                        d["id"] += tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            d["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            d["arguments"].append(tc_delta.function.arguments)

        # --- build final response ---
        content = "".join(accumulated_content)
        reasoning = "".join(accumulated_reasoning) or None

        tool_calls: list[ToolCall] = []
        for idx in sorted(tool_call_deltas):
            d = tool_call_deltas[idx]
            args: dict = {}
            arg_str = "".join(d["arguments"])
            if arg_str:
                try:
                    args = json.loads(arg_str)
                except json.JSONDecodeError:
                    args = {"_raw": arg_str}
            tool_calls.append(ToolCall(id=d["id"], name=d["name"], arguments=args))

        content_filter_triggered = finish_reason == "content_filter"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=usage,
            content_filter_triggered=content_filter_triggered,
            reasoning_content=reasoning,
        )

    # ── non-streaming ───────────────────────────────────────────────────────

    def chat(self, messages: list, tools: list | None = None) -> LLMResponse:
        raw = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._build_kwargs(tools),
        )
        return self._response_from_raw(raw)
