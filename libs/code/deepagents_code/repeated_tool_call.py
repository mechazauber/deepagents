"""Detect repeated tool-call sequences and nudge the agent.

Drop this file into ``deepagents_code/`` as ``repeated_tool_call.py``.

The check is deterministic — no extra model call. It fingerprints each
executed tool call (name + args) and watches the rolling window of the
most recent ``window`` executed calls for two kinds of loops:

1. **Exact repetition** — the same call appearing ``threshold`` times
   in a row (the original behavior, kept as a fast path).
2. **Sequence loops** — a pattern of up to ``sequence_length``
   consecutive calls (default 5) that has already occurred
   ``sequence_repeats`` times (default 3) in the recent history.
   A pattern counts as "occurred" when it appears as a contiguous
   block somewhere in the window; the pattern may be shorter than
   ``sequence_length`` (minimum 2) so that tight two- or
   three-call cycles are caught even when the 5-call window contains
   other calls. The current turn's new calls are checked against the
   history *before* them, so a pattern is flagged when the model is
   about to complete its 3rd (or later) occurrence.

When a loop is detected, a ``HumanMessage`` naming the repeated calls
is appended and the loop jumps back to the model node. The agent keeps
control; nothing is cancelled and no work is handed off.

Design:
    The middleware inspects the message history in ``after_model``. It
    walks through ``ToolMessage`` entries (which carry the tool name
    and the result of an *executed* call) and ``AIMessage`` entries
    (which carry the tool calls that *triggered* them). It pairs them
    by ``tool_call_id`` to build a rolling window of (name,
    args_hash) pairs for the most recent executed tool calls. If the
    model's *new* tool call (the one in the just-produced
    ``AIMessage``) completes an exact repetition or a recurring
    sequence, a nudge is injected and the loop jumps back to the model
    node.

    This catches patterns like:
        turn 1: model calls grep("foo")  → tool runs, returns "no match"
        turn 2: model calls grep("foo")  → tool runs, returns "no match"
        turn 3: model calls grep("foo")  → nudge injected:
                "You've called grep('foo') 3 times with the same result.
                 Try a different approach."

    and cycling loops, e.g.:
        A read_file(a) → B read_file(b) → C read_file(a) → B → C → A
        → B → C → nudge injected: the 5-call pattern has recurred.

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


def _count_occurrences(
    history: tuple[str, ...], pattern: tuple[str, ...]
) -> int:
    """Count occurrences of ``pattern`` in ``history``.

    Scans left to right; each match advances the scan position by one,
    so overlapping occurrences (e.g. "ab" at positions 0 and 2 of
    "ababab") are all counted.
    """
    if not pattern or len(history) < len(pattern):
        return 0
    count = 0
    n = len(pattern)
    for i in range(len(history) - n + 1):
        if history[i : i + n] == pattern:
            count += 1
    return count


def _detect_sequence_loop(
    history: tuple[str, ...],
    new_fps: tuple[str, ...],
    sequence_length: int,
    sequence_repeats: int,
) -> tuple[str, ...] | None:
    """Find a recurring short pattern that the new calls would repeat.

    For every contiguous suffix of the new calls (length 2 to
    ``sequence_length``), check how many times that suffix already
    appears as a contiguous block in the history. If it has appeared
    at least ``sequence_repeats`` times, the new calls would complete
    another occurrence and the pattern is returned (longest suffix
    first, so the most specific match wins).

    This catches cycling loops like A-B-C-A-B-C-A-B-C (the 3-call
    pattern A-B-C has occurred 3 times; a new A-B-C is the 4th) and
    two-call cycles like A-B-A-B-A-B (the 2-call pattern A-B has
    occurred 3 times; a new A-B is the 4th).

    Returns:
        The repeated pattern as a tuple of fingerprints, or ``None``.
    """
    if len(new_fps) < 2:
        return None
    max_suffix = min(len(new_fps), sequence_length)
    for suffix_len in range(max_suffix, 1, -1):
        suffix = new_fps[-suffix_len:]
        if _count_occurrences(history, suffix) >= sequence_repeats:
            return suffix
    return None


class RepeatedToolCallState(AgentState[ResponseT]):
    """State schema for ``RepeatedToolCallMiddleware``.

    Extends ``AgentState`` with a private channel for the rolling window
    of recent tool-call fingerprints.
    """

    _recent_tool_calls: NotRequired[list[str]]


class RepeatedToolCallMiddleware(
    AgentMiddleware[RepeatedToolCallState[ResponseT], ContextT, ResponseT]
):
    """Detects repeated tool calls and recurring call sequences.

    Monitors the ``after_model`` hook. When the model produces an
    ``AIMessage`` with tool calls, each call's (name, args) fingerprint is
    compared against the rolling window of previously *executed* tool
    calls (derived from ``ToolMessage`` entries in the history). A nudge
    fires when either:

    - the new call matches the last ``threshold - 1`` executed calls
      (exact repetition, e.g. ``grep("foo")`` three times in a row), or
    - the new calls would complete a pattern of up to ``sequence_length``
      consecutive calls (default 5) that has already occurred
      ``sequence_repeats`` times (default 3) in the recent window
      (cycling loops, e.g. A-B-C-A-B-C-A-B-C).

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

    def __init__(
        self,
        *,
        threshold: int = 3,
        window: int = 10,
        sequence_length: int = 5,
        sequence_repeats: int = 3,
    ) -> None:
        """Initialize the repeated-tool-call detector.

        Args:
            threshold: Number of times the same tool call must appear
                (including the current one) before a nudge is injected.
                Must be >= 2. For example, ``threshold=3`` means: if the
                model calls ``grep("foo")`` and ``grep("foo")`` was already
                executed twice in the recent history, a nudge fires.
            window: Size of the rolling window of recent executed
                tool-call fingerprints to track.
            sequence_length: Maximum length of a recurring call pattern
                to look for. Must be >= 2. With the default 5, patterns
                of 2 to 5 consecutive calls are considered.
            sequence_repeats: Number of times a pattern must have
                already occurred in the window before a nudge fires.
                Must be >= 2. With the default 3, a pattern is flagged
                when the model is about to complete its 3rd occurrence.
        """
        super().__init__()
        if threshold < 2:
            msg = "threshold must be >= 2"
            raise ValueError(msg)
        if sequence_length < 2:
            msg = "sequence_length must be >= 2"
            raise ValueError(msg)
        if sequence_repeats < 2:
            msg = "sequence_repeats must be >= 2"
            raise ValueError(msg)
        self.threshold = threshold
        self.window = window
        self.sequence_length = sequence_length
        self.sequence_repeats = sequence_repeats

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

    def _detect_sequence(
        self,
        new_fps: tuple[str, ...],
        executed: list[tuple[str, str, dict[str, Any]]],
    ) -> tuple[str, ...] | None:
        """Check if the new calls would complete a recurring pattern.

        Returns:
            The repeated pattern as a tuple of fingerprints, or ``None``.
        """
        if not new_fps:
            return None
        if len(executed) < self.sequence_length:
            # Not enough history yet for a meaningful pattern check.
            return None
        history = tuple(fp for fp, _, _ in executed[-self.window :])
        return _detect_sequence_loop(
            history, new_fps, self.sequence_length, self.sequence_repeats
        )

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
            If a repetition or recurring sequence is detected, a
            ``HumanMessage`` nudge and a ``jump_to: "model"`` directive.
            Otherwise ``None``.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
            return None

        executed = self._executed_tool_calls(messages)

        new_calls: list[tuple[str, dict[str, Any]]] = []
        for call in last_msg.tool_calls:
            name = call.get("name", "")
            args = call.get("args", {})
            if not name:
                continue
            new_calls.append((name, args))
            if self._detect_repetition(_tool_call_fingerprint(name, args), executed):
                return self._nudge(
                    f"You have called {_describe_tool_call(name, args)} "
                    f"{self.threshold} times with the same result. This "
                    f"approach is not working. Try a different strategy, "
                    f"or ask the user for clarification."
                )

        if new_calls:
            new_fps = tuple(_tool_call_fingerprint(n, a) for n, a in new_calls)
            pattern = self._detect_sequence(new_fps, executed)
            if pattern is not None:
                # Describe the pattern from the history: find its most
                # recent occurrence and use the (name, args) of the calls
                # that produced those fingerprints.
                desc = self._describe_pattern(pattern, executed)
                nudge = (
                    f"You are repeating a sequence of {len(pattern)} tool "
                    f"calls that has already occurred {self.sequence_repeats} "
                    f"times: {desc}. This approach is not working. Try a "
                    f"different strategy, or ask the user for clarification."
                )
                return self._nudge(nudge)

        return None

    def _nudge(self, content: str) -> dict[str, Any]:
        """Build the nudge response with a jump back to the model node."""
        logger.info("RepeatedToolCallMiddleware: injecting nudge: %s", content)
        return {
            "messages": [
                HumanMessage(
                    content=content,
                    additional_kwargs={"lc_source": _NUDGE_SOURCE},
                )
            ],
            "jump_to": "model",
        }

    def _describe_pattern(
        self,
        pattern: tuple[str, ...],
        executed: list[tuple[str, str, dict[str, Any]]],
    ) -> str:
        """Human-readable description of a repeated fingerprint pattern.

        Finds the most recent occurrence of the pattern in the executed
        history and describes the calls that produced it.
        """
        n = len(pattern)
        idx = -1
        for i in range(len(executed) - n, -1, -1):
            if tuple(fp for fp, _, _ in executed[i : i + n]) == pattern:
                idx = i
                break
        if idx < 0:
            return ", ".join(f[:8] for f in pattern)
        parts: list[str] = []
        for _, name, args in executed[idx : idx + n]:
            parts.append(_describe_tool_call(name, args))
        return " -> ".join(parts)

    async def aafter_model(
        self,
        state: RepeatedToolCallState[ResponseT],
        runtime: Any,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async version of ``after_model``."""
        return self.after_model(state, runtime)
