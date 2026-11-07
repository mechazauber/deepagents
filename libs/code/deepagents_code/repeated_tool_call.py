"""Detect repeated identical tool calls and nudge the agent.

Drop this file into ``deepagents_code/`` as ``repeated_tool_call.py``.

The check is deterministic — no extra model call. It hashes each tool
call's name and arguments, and when the same one appears ``threshold``
times consecutively in the executed tool-call history, it appends a
``HumanMessage`` naming the repetition and asking the agent to change
approach. The agent keeps control; nothing is cancelled and no work is
handed off.

Design:
    The middleware inspects the message history in ``after_model``. It
    walks backward through ``ToolMessage`` entries (which carry the tool
    name and the result of an *executed* call) and ``AIMessage`` entries
    (which carry the tool calls that *triggered* them). It pairs them by
    ``tool_call_id`` to build a rolling window of (name, args_hash) pairs
    for the most recent executed tool calls. If the model's *new* tool
    call (the one in the just-produced ``AIMessage``) matches the last
    ``threshold - 1`` executed calls, a nudge is injected and the loop
    jumps back to the model node.

    This catches the pattern:
        turn 1: model calls grep("foo")  → tool runs, returns "no match"
        turn 2: model calls grep("foo")  → tool runs, returns "no match"
        turn 3: model calls grep("foo")  → nudge injected:
                "You've called grep('foo') 3 times with the same result.
                 Try a different approach."

Usage:
    from deepagents_code.repeated_tool_call import RepeatedToolCallMiddleware

    agent = create_agent(
        model="...",
        middleware=[RepeatedToolCallMiddleware(threshold=3)],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ResponseT,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import NotRequired, override

logger = logging.getLogger(__name__)

_NUDGE_SOURCE = "repeated_tool_call"


def _tool_call_fingerprint(name: str, args: dict[str, Any]) -> str:
    """Stable hash of a tool call's name and arguments.

    Args are JSON-serialized with sorted keys so that semantically
    identical calls (differing only in key order) produce the same
    fingerprint.
    """
    payload = json.dumps({"name": name, "args": args}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _describe_tool_call(name: str, args: dict[str, Any]) -> str:
    """Human-readable description of a tool call for the nudge message."""
    arg_summary: list[str] = []
    for key, value in sorted(args.items()):
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        arg_summary.append(f"{key}={value_str!r}")
    args_str = ", ".join(arg_summary) if arg_summary else ""
    return f"{name}({args_str})" if args_str else name


class RepeatedToolCallState(AgentState[ResponseT]):
    """State schema for ``RepeatedToolCallMiddleware``.

    Extends ``AgentState`` with a private channel for the rolling window
    of recent tool-call fingerprints.
    """

    _recent_tool_calls: NotRequired[list[str]]


class RepeatedToolCallMiddleware(
    AgentMiddleware[RepeatedToolCallState[ResponseT], ContextT, ResponseT]
):
    """Detects when an agent repeats the same tool call and nudges it.

    Monitors the ``after_model`` hook. When the model produces an
    ``AIMessage`` with tool calls, each call's (name, args) fingerprint is
    compared against the rolling window of previously *executed* tool
    calls (derived from ``ToolMessage`` entries in the history). If the
    new call matches the last ``threshold - 1`` executed calls, a
    ``HumanMessage`` is injected telling the agent to try a different
    approach, and the loop jumps back to the model node.

    Example:
        ```python
        from deepagents_code.repeated_tool_call import RepeatedToolCallMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[RepeatedToolCallMiddleware(threshold=3)],
        )
        ```
    """

    state_schema = RepeatedToolCallState  # type: ignore[assignment]

    def __init__(self, *, threshold: int = 3, window: int = 10) -> None:
        """Initialize the repeated-tool-call detector.

        Args:
            threshold: Number of times the same tool call must appear
                (including the current one) before a nudge is injected.
                Must be >= 2. For example, ``threshold=3`` means: if the
                model calls ``grep("foo")`` and ``grep("foo")`` was already
                executed twice in the recent history, a nudge fires.
            window: Size of the rolling window of recent executed
                tool-call fingerprints to track.
        """
        super().__init__()
        if threshold < 2:
            msg = "threshold must be >= 2"
            raise ValueError(msg)
        self.threshold = threshold
        self.window = window

    def _executed_tool_calls(
        self, messages: list[Any]
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Walk the message history and extract executed tool calls.

        A tool call is "executed" when there is a ``ToolMessage`` whose
        ``tool_call_id`` matches a preceding ``AIMessage``'s tool call
        with the same ``id``.

        Returns:
            List of (fingerprint, name, args) tuples in chronological
            order, most recent last.
        """
        pending: dict[str, tuple[str, dict[str, Any]]] = {}
        executed: list[tuple[str, str, dict[str, Any]]] = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                for call in msg.tool_calls:
                    call_id = call.get("id", "")
                    name = call.get("name", "")
                    args = call.get("args", {})
                    if call_id and name:
                        pending[call_id] = (name, args)
            elif isinstance(msg, ToolMessage):
                call_id = msg.tool_call_id
                if call_id and call_id in pending:
                    name, args = pending.pop(call_id)
                    fp = _tool_call_fingerprint(name, args)
                    executed.append((fp, name, args))

        return executed

    def _detect_repetition(
        self,
        new_call_fp: str,
        executed: list[tuple[str, str, dict[str, Any]]],
    ) -> bool:
        """Check if the new tool call matches the last ``threshold - 1``
        executed calls.

        This catches the pattern where the model is about to make the
        Nth identical call (N >= threshold).
        """
        needed = self.threshold - 1
        if len(executed) < needed:
            return False
        recent = executed[-needed:]
        return all(fp == new_call_fp for fp, _, _ in recent)

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(
        self,
        state: RepeatedToolCallState[ResponseT],
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check for repeated tool calls after a model response.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            If a repetition is detected, a ``HumanMessage`` nudge and a
            ``jump_to: "model"`` directive. Otherwise ``None``.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return None

        executed = self._executed_tool_calls(messages)

        for call in last_msg.tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            if not name:
                continue
            fp = _tool_call_fingerprint(name, args)
            if self._detect_repetition(fp, executed):
                description = _describe_tool_call(name, args)
                nudge = (
                    f"You have called {description} {self.threshold} times "
                    f"with the same result. This approach is not working. "
                    f"Try a different strategy, or ask the user for "
                    f"clarification."
                )
                logger.info(
                    "RepeatedToolCallMiddleware: injecting nudge for "
                    "repeated call: %s",
                    description,
                )
                return {
                    "messages": [
                        HumanMessage(
                            content=nudge,
                            additional_kwargs={"lc_source": _NUDGE_SOURCE},
                        )
                    ],
                    "jump_to": "model",
                }

        return None

    async def aafter_model(
        self,
        state: RepeatedToolCallState[ResponseT],
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async version of ``after_model``."""
        return self.after_model(state, runtime)
