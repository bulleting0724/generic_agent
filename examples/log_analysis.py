"""Example: Log Analysis Agent built on generic_agent.

Shows how to define custom tools, a domain-specific system prompt, and
wire up AgentLoop — all without touching the original trading agent code.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from loop import AgentLoop, ContextBuilder, ToolRegistry
from memory import WorkspaceMemory
from tools import BaseTool


# ── Custom tools ────────────────────────────────────────────────────────────────


class GrepLogs(BaseTool):
    """Search for a regex pattern in log files."""

    name = "grep_logs"
    description = "Search for a regex pattern in log files under a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory or file path to search",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "glob": {
                "type": "string",
                "description": "File glob pattern (default: *.log)",
                "default": "*.log",
            },
        },
        "required": ["path", "pattern"],
    }
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        path = Path(kwargs["path"])
        pattern = re.compile(kwargs["pattern"])
        glob_pat = kwargs.get("glob", "*.log")
        if not path.exists():
            return json.dumps(
                {"status": "error", "error": f"Path not found: {path}"}
            )
        files = list(path.rglob(glob_pat)) if path.is_dir() else [path]
        results = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if pattern.search(line):
                        results.append(
                            {
                                "file": str(f),
                                "line": i,
                                "content": line.strip(),
                            }
                        )
            except Exception as e:
                results.append({"file": str(f), "error": str(e)})
        return json.dumps(
            {
                "status": "ok",
                "matches": len(results),
                "results": results[:200],
            },
            ensure_ascii=False,
        )


class CountLogLevels(BaseTool):
    """Count ERROR / WARN / INFO / DEBUG log levels in files."""

    name = "count_log_levels"
    description = "Count ERROR/WARN/INFO/DEBUG log levels in log files."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory or file path",
            },
            "glob": {
                "type": "string",
                "description": "File glob pattern (default: *.log)",
                "default": "*.log",
            },
        },
        "required": ["path"],
    }
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        path = Path(kwargs["path"])
        glob_pat = kwargs.get("glob", "*.log")
        if not path.exists():
            return json.dumps(
                {"status": "error", "error": f"Path not found: {path}"}
            )
        files = list(path.rglob(glob_pat)) if path.is_dir() else [path]
        total: Counter[str] = Counter()
        per_file: dict[str, dict[str, int]] = {}
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                counts: Counter[str] = Counter()
                for line in text.splitlines():
                    for level in ("ERROR", "WARN", "INFO", "DEBUG"):
                        if level in line:
                            counts[level] += 1
                if counts:
                    per_file[str(f)] = dict(counts)
                    total.update(counts)
            except Exception as e:
                per_file[str(f)] = {"error": str(e)}
        return json.dumps(
            {
                "status": "ok",
                "total": dict(total),
                "per_file": per_file,
                "file_count": len(files),
            },
            ensure_ascii=False,
        )


class TailLogs(BaseTool):
    """Show the last N lines from a log file."""

    name = "tail_logs"
    description = "Show the last N lines from a log file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Log file path",
            },
            "n": {
                "type": "integer",
                "description": "Number of lines (default: 50)",
                "default": 50,
            },
        },
        "required": ["path"],
    }
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        path = Path(kwargs["path"])
        n = int(kwargs.get("n", 50))
        if not path.exists():
            return json.dumps(
                {"status": "error", "error": f"File not found: {path}"}
            )
        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        tail = lines[-n:]
        return json.dumps(
            {
                "status": "ok",
                "file": str(path),
                "total_lines": len(lines),
                "lines": tail,
            },
            ensure_ascii=False,
        )


class ListLogFiles(BaseTool):
    """List log files in a directory."""

    name = "list_log_files"
    description = "List log files in a directory with sizes."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path",
            },
            "glob": {
                "type": "string",
                "description": "File glob pattern (default: *.log)",
                "default": "*.log",
            },
        },
        "required": ["path"],
    }
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        path = Path(kwargs["path"])
        glob_pat = kwargs.get("glob", "*.log")
        if not path.is_dir():
            return json.dumps(
                {"status": "error", "error": f"Not a directory: {path}"}
            )
        files = sorted(path.rglob(glob_pat))
        entries = [
            {
                "file": str(f.relative_to(path)),
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in files
        ]
        return json.dumps(
            {
                "status": "ok",
                "directory": str(path),
                "file_count": len(entries),
                "files": entries,
            },
            ensure_ascii=False,
        )


# ── Domain-specific system prompt ───────────────────────────────────────────────

LOG_ANALYSIS_SYSTEM_PROMPT = """\
You are a log analysis agent. Investigate log files, identify errors and
anomalies, and produce clear quantitative summaries.
must response in Chinese.

## Tools

{tool_descriptions}

## State

{memory_summary}

## Guidelines

- Use ``grep_logs`` to search for error patterns and stack traces.
- Use ``count_log_levels`` to get the overall health picture.
- Use ``tail_logs`` to inspect recent activity.
- Use ``list_log_files`` to discover which files exist and their sizes.
- Report specific metrics: counts, frequencies, file paths, line numbers.
- Distinguish transient errors from recurring patterns.
- All file paths are relative to run_dir (auto-injected).

## Current Date & Time

Today is {current_datetime}.
"""


# ── Agent factory ───────────────────────────────────────────────────────────────

def create_log_analysis_agent(
    llm: Any, **kwargs: Any
) -> AgentLoop:
    """Create a log analysis AgentLoop.

    Args:
        llm: ChatLLM instance.
        **kwargs: Passed through to AgentLoop (e.g. max_iterations,
            event_callback).

    Returns:
        Configured AgentLoop.
    """
    registry = ToolRegistry()
    registry.register(GrepLogs())
    registry.register(CountLogLevels())
    registry.register(TailLogs())
    registry.register(ListLogFiles())

    context = ContextBuilder(
        registry=registry,
        memory=WorkspaceMemory(),
        system_prompt_template=LOG_ANALYSIS_SYSTEM_PROMPT,
    )

    return AgentLoop(
        registry=registry,
        llm=llm,
        context_builder=context,
        **kwargs,
    )


# ── Runnable entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from simple_llm import SimpleLLM

    llm = SimpleLLM()
    agent = create_log_analysis_agent(llm)

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    result = agent.run(f"Inspect '{target}' for errors and anomalies.")
    print(result.get("content", ""))
