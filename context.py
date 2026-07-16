"""Generic ContextBuilder: builds LLM message context for a ReAct AgentLoop.

Domain-neutral — accepts a configurable system prompt template so any
agent (trading, log analysis, DevOps, etc.) can supply its own persona
and routing instructions.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from memory import WorkspaceMemory
from tools import ToolRegistry

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI agent with {tool_count} tools available.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Guidelines

- Use tools when you need information or to perform actions.
- When you have enough information, provide your answer directly.
- All file paths are relative to run_dir (auto-injected).
- Respond in the same language the user used.

## Current Date & Time

Today is {current_datetime}.
"""


class ContextBuilder:
    """Builds message context for AgentLoop.

    Attributes:
        registry: Tool registry.
        memory: Workspace memory.
        system_prompt_template: Configurable system prompt template.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        memory: WorkspaceMemory,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        tool_descriptions_formatter: Optional[Callable[[], str]] = None,
        persistent_memory: Optional[Any] = None,
    ) -> None:
        """Initialize ContextBuilder.

        Args:
            registry: Tool registry.
            memory: Workspace memory.
            system_prompt_template: String template for the system prompt.
                Available format keys: tool_count, tool_descriptions,
                memory_summary, current_datetime.
            tool_descriptions_formatter: Optional callable that returns
                formatted tool descriptions.  Defaults to the built-in
                ``_format_tool_descriptions``.
            persistent_memory: Optional cross-session memory for auto-recall.
        """
        self.registry = registry
        self.memory = memory
        self.system_prompt_template = system_prompt_template
        self._tool_descriptions_formatter = (
            tool_descriptions_formatter or self._format_tool_descriptions
        )
        self._persistent_memory = persistent_memory

    def build_system_prompt(self, user_message: str = "") -> str:
        """Build system prompt.

        Args:
            user_message: User message (kept for API compatibility with
                subclasses).

        Returns:
            System prompt text.
        """
        _ = user_message
        now = datetime.now()
        return self.system_prompt_template.format(
            tool_count=len(self.registry._tools),
            tool_descriptions=self._tool_descriptions_formatter(),
            memory_summary=self.memory.to_summary(),
            current_datetime=now.strftime("%A, %B %d, %Y %H:%M (local)"),
        )

    def build_messages(
        self,
        user_message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build full message list.

        Optionally enriches the user message with relevant persistent
        memory recalls.

        Args:
            user_message: User message.
            history: Prior conversation messages.

        Returns:
            OpenAI-format message list.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt(user_message)},
        ]
        if history:
            messages.extend(history)

        enriched = user_message
        if self._persistent_memory:
            try:
                recalls = self._persistent_memory.find_relevant(
                    user_message, max_results=3
                )
                if recalls:
                    lines = [
                        f"- **{r.title}** ({r.memory_type}): {r.body[:500]}"
                        for r in recalls
                    ]
                    recall_block = "\n".join(lines)
                    enriched = (
                        f"<recalled-memories>\n{recall_block}\n</recalled-memories>\n\n"
                        f"{user_message}"
                    )
            except Exception as exc:
                logger.debug("Auto-recall failed: %s", exc)

        messages.append({"role": "user", "content": enriched})
        return messages

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions in Markdown."""
        lines = []
        for tool in self.registry._tools.values():
            params = tool.parameters.get("properties", {})
            required = tool.parameters.get("required", [])
            param_parts = []
            for pname, pschema in params.items():
                req = " (required)" if pname in required else ""
                description = pschema.get(
                    "description", pschema.get("type", "")
                )
                param_parts.append(f"    - {pname}: {description}{req}")
            param_text = (
                "\n".join(param_parts) if param_parts else "    (no params)"
            )
            lines.append(
                f"### {tool.name}\n{tool.description}\n  Params:\n{param_text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def format_tool_result(
        tool_call_id: str, tool_name: str, result: str
    ) -> Dict[str, Any]:
        """Format a tool execution result as a message."""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }

    @staticmethod
    def format_assistant_tool_calls(
        tool_calls: list,
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format an assistant tool_calls message, preserving thinking text.

        Args:
            tool_calls: List of tool call objects.
            content: Final assistant text.
            reasoning_content: Provider-specific reasoning field.

        Returns:
            OpenAI-format assistant message.
        """
        formatted_tool_calls = []
        has_extra_content = False
        for tc in tool_calls:
            tool_call = {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            extra_content = getattr(tc, "extra_content", None)
            if extra_content:
                tool_call["extra_content"] = dict(extra_content)
                has_extra_content = True
            formatted_tool_calls.append(tool_call)

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "tool_calls": formatted_tool_calls,
        }
        if has_extra_content:
            message["additional_kwargs"] = {
                "tool_calls": copy.deepcopy(formatted_tool_calls),
            }
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        return message
