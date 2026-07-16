"""Generic agent package — reusable ReAct agent loop for any domain.

Provides domain-neutral AgentLoop, ContextBuilder, ToolRegistry, and
progress/trace utilities that can be shared across agents (trading,
log analysis, DevOps, etc.) without modification.
"""

from .context import ContextBuilder
from .loop import AgentLoop, LLMClient, LLMError, LLMResponse, ToolCall
from .memory import WorkspaceMemory
from .progress import HeartbeatTimer, ProgressEvent
from .simple_llm import SimpleLLM
from .tools import BaseTool, ToolRegistry
from .trace import TraceWriter

__all__ = [
    "AgentLoop",
    "BaseTool",
    "ContextBuilder",
    "HeartbeatTimer",
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "ProgressEvent",
    "SimpleLLM",
    "ToolCall",
    "ToolRegistry",
    "TraceWriter",
    "WorkspaceMemory",
]
