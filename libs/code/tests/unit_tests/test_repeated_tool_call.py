"""Unit tests for RepeatedToolCallMiddleware."""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents_code.repeated_tool_call import (
    RepeatedToolCallMiddleware,
    _count_occurrences,
    _describe_tool_call,
    _detect_sequence_loop,
    _tool_call_fingerprint,
)


def _ai(
    call_id: str,
    name: str,
    args: dict[str, object],
) -> AIMessage:
    """Build an AIMessage with a single tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
    )


def _tm(call_id: str, name: str, content: str = "result") -> ToolMessage:
    """Build a ToolMessage paired with a tool call by id."""
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


class TestFingerprint:
    """Tests for the tool-call fingerprinting helper."""

    def test_stable_for_same_args(self) -> None:
        fp1 = _tool_call_fingerprint("grep", {"pattern": "foo", "path": "/tmp"})
        fp2 = _tool_call_fingerprint("grep", {"pattern": "foo", "path": "/tmp"})
        assert fp1 == fp2

    def test_key_order_independent(self) -> None:
        fp1 = _tool_call_fingerprint("grep", {"pattern": "foo", "path": "/tmp"})
        fp2 = _tool_call_fingerprint("grep", {"path": "/tmp", "pattern": "foo"})
        assert fp1 == fp2

    def test_different_args_differ(self) -> None:
        fp1 = _tool_call_fingerprint("grep", {"pattern": "foo", "path": "/tmp"})
        fp2 = _tool_call_fingerprint("grep", {"pattern": "bar", "path": "/tmp"})
        assert fp1 != fp2

    def test_different_names_differ(self) -> None:
        fp1 = _tool_call_fingerprint("grep", {"pattern": "foo"})
        fp2 = _tool_call_fingerprint("read_file", {"pattern": "foo"})
        assert fp1 != fp2

    def test_empty_args(self) -> None:
        fp1 = _tool_call_fingerprint("ls", {})
        fp2 = _tool_call_fingerprint("ls", {})
        assert fp1 == fp2


class TestDescribeToolCall:
    """Tests for the human-readable description helper."""

    def test_simple(self) -> None:
        assert _describe_tool_call("ls", {}) == "ls"

    def test_with_args(self) -> None:
        desc = _describe_tool_call("grep", {"pattern": "foo", "path": "/tmp"})
        assert "grep" in desc
        assert "pattern" in desc
        assert "foo" in desc

    def test_truncates_long_values(self) -> None:
        long_value = "x" * 200
        desc = _describe_tool_call("execute", {"command": long_value})
        assert "..." in desc
        assert len(desc) < 120


class TestConstructor:
    """Tests for middleware construction."""

    def test_default_threshold(self) -> None:
        mw = RepeatedToolCallMiddleware()
        assert mw.threshold == 3
        assert mw.window == 10
        assert mw.sequence_length == 5
        assert mw.sequence_repeats == 3

    def test_custom_threshold(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=5, window=20)
        assert mw.threshold == 5
        assert mw.window == 20

    def test_custom_sequence_params(self) -> None:
        mw = RepeatedToolCallMiddleware(sequence_length=4, sequence_repeats=4)
        assert mw.sequence_length == 4
        assert mw.sequence_repeats == 4

    def test_sequence_length_below_2_raises(self) -> None:
        try:
            RepeatedToolCallMiddleware(sequence_length=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError as exc:
            assert "sequence_length" in str(exc)

    def test_sequence_repeats_below_2_raises(self) -> None:
        try:
            RepeatedToolCallMiddleware(sequence_repeats=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError as exc:
            assert "sequence_repeats" in str(exc)

    def test_threshold_below_2_raises(self) -> None:
        try:
            RepeatedToolCallMiddleware(threshold=1)
            raise AssertionError("Should have raised ValueError")
        except ValueError as exc:
            assert "threshold" in str(exc)

    def test_threshold_zero_raises(self) -> None:
        try:
            RepeatedToolCallMiddleware(threshold=0)
            raise AssertionError("Should have raised ValueError")
        except ValueError as exc:
            assert "threshold" in str(exc)


class TestAfterModel:
    """Tests for the after_model hook."""

    def test_no_messages_returns_none(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        result = mw.after_model({"messages": []}, runtime=None)
        assert result is None

    def test_last_message_not_ai_returns_none(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        state = {"messages": [HumanMessage(content="hello")]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_ai_without_tool_calls_returns_none(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        state = {"messages": [AIMessage(content="I think we should...")]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_single_call_no_repetition(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai = _ai("c1", "grep", {"pattern": "foo"})
        state = {"messages": [ai]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_two_executions_triggers_at_threshold_3(self) -> None:
        """Two prior identical executions + new call = 3 total → nudge."""
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo", "path": "/tmp"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo", "path": "/tmp"})
        tm2 = _tm("c2", "grep", "no matches")
        ai3 = _ai("c3", "grep", {"pattern": "foo", "path": "/tmp"})

        state = {"messages": [ai1, tm1, ai2, tm2, ai3]}
        result = mw.after_model(state, runtime=None)
        assert result is not None
        assert result["jump_to"] == "model"
        messages = result["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert "grep" in messages[0].content
        assert "3 times" in messages[0].content
        assert messages[0].additional_kwargs["lc_source"] == "repeated_tool_call"

    def test_one_execution_no_nudge_at_threshold_3(self) -> None:
        """One prior execution + new call = 2 total → no nudge at threshold 3."""
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo", "path": "/tmp"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo", "path": "/tmp"})

        state = {"messages": [ai1, tm1, ai2]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_different_calls_no_nudge(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo", "path": "/tmp"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo", "path": "/tmp"})
        tm2 = _tm("c2", "grep", "no matches")
        # New call is different
        ai3 = _ai("c3", "read_file", {"file_path": "/tmp/foo.txt"})

        state = {"messages": [ai1, tm1, ai2, tm2, ai3]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_different_args_no_nudge(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo", "path": "/tmp"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo", "path": "/tmp"})
        tm2 = _tm("c2", "grep", "no matches")
        # Same tool, different args
        ai3 = _ai("c3", "grep", {"pattern": "bar", "path": "/tmp"})

        state = {"messages": [ai1, tm1, ai2, tm2, ai3]}
        result = mw.after_model(state, runtime=None)
        assert result is None

    def test_key_order_independent_detection(self) -> None:
        """Args with different key order but same values should match."""
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo", "path": "/tmp"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"path": "/tmp", "pattern": "foo"})
        tm2 = _tm("c2", "grep", "no matches")
        ai3 = _ai("c3", "grep", {"pattern": "foo", "path": "/tmp"})

        state = {"messages": [ai1, tm1, ai2, tm2, ai3]}
        result = mw.after_model(state, runtime=None)
        assert result is not None

    def test_threshold_2_triggers_after_one_prior(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=2)
        ai1 = _ai("c1", "execute", {"command": "ls"})
        tm1 = _tm("c1", "execute", "file1")
        ai2 = _ai("c2", "execute", {"command": "ls"})

        state = {"messages": [ai1, tm1, ai2]}
        result = mw.after_model(state, runtime=None)
        assert result is not None
        assert "2 times" in result["messages"][0].content

    def test_multi_tool_call_one_repeated(self) -> None:
        """One of several tool calls in a message is repeated → nudge."""
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "execute", {"command": "ls"})
        tm1 = _tm("c1", "execute", "file1")
        ai2 = _ai("c2", "execute", {"command": "ls"})
        tm2 = _tm("c2", "execute", "file1")
        # New message: one new call + one repeated call
        ai3 = AIMessage(
            content="",
            tool_calls=[
                {"id": "c3", "name": "read_file", "args": {"file_path": "/a.txt"}},
                {"id": "c4", "name": "execute", "args": {"command": "ls"}},
            ],
        )

        state = {"messages": [ai1, tm1, ai2, tm2, ai3]}
        result = mw.after_model(state, runtime=None)
        assert result is not None
        assert "execute" in result["messages"][0].content

    def test_unpaired_tool_call_ignored(self) -> None:
        """A ToolMessage with no matching AIMessage tool call is ignored.

        The orphan ToolMessage does not contribute to the executed history,
        so only the 1 paired execution (ai1+tm1) counts. With threshold=3,
        1 prior execution is not enough to trigger a nudge.
        """
        mw = RepeatedToolCallMiddleware(threshold=3)
        # Orphan ToolMessage (no matching AIMessage)
        tm_orphan = _tm("orphan", "grep", "no matches")
        ai1 = _ai("c1", "grep", {"pattern": "foo"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo"})

        state = {"messages": [tm_orphan, ai1, tm1, ai2]}
        result = mw.after_model(state, runtime=None)
        # Only 1 paired execution → no nudge at threshold 3
        assert result is None

    def test_nudge_message_has_correct_lc_source(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=2)
        ai1 = _ai("c1", "grep", {"pattern": "foo"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo"})

        state = {"messages": [ai1, tm1, ai2]}
        result = mw.after_model(state, runtime=None)
        assert result is not None
        msg = result["messages"][0]
        assert msg.additional_kwargs["lc_source"] == "repeated_tool_call"


class TestAsyncAfterModel:
    """Tests for the async after_model hook."""

    def test_async_matches_sync(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=2)
        ai1 = _ai("c1", "grep", {"pattern": "foo"})
        tm1 = _tm("c1", "grep", "no matches")
        ai2 = _ai("c2", "grep", {"pattern": "foo"})

        state = {"messages": [ai1, tm1, ai2]}
        sync_result = mw.after_model(state, runtime=None)
        async_result = asyncio.run(mw.aafter_model(state, runtime=None))
        assert sync_result is not None
        assert async_result is not None
        assert sync_result["jump_to"] == async_result["jump_to"]
        assert (
            sync_result["messages"][0].content
            == async_result["messages"][0].content
        )

    def test_async_no_repetition(self) -> None:
        mw = RepeatedToolCallMiddleware(threshold=3)
        ai1 = _ai("c1", "grep", {"pattern": "foo"})
        state = {"messages": [ai1]}
        result = asyncio.run(mw.aafter_model(state, runtime=None))
        assert result is None


class TestCountOccurrences:
    """Tests for the non-overlapping pattern counter."""

    def test_no_pattern(self) -> None:
        assert _count_occurrences(("a", "b", "c"), ("a", "b")) == 1
        assert _count_occurrences(("a", "b", "c"), ("x",)) == 0

    def test_overlapping_counted(self) -> None:
        # "ababab" contains "ab" at positions 0, 2 and 4.
        assert _count_occurrences(("a", "b", "a", "b", "a", "b"), ("a", "b")) == 3

    def test_overlapping_triplet(self) -> None:
        # "aaa" contains "aa" at positions 0 and 1.
        assert _count_occurrences(("a", "a", "a"), ("a", "a")) == 2

    def test_empty_pattern(self) -> None:
        assert _count_occurrences(("a", "b"), ()) == 0

    def test_pattern_longer_than_history(self) -> None:
        assert _count_occurrences(("a",), ("a", "b")) == 0


class TestDetectSequenceLoop:
    """Tests for the recurring-sequence detector (pure function)."""

    def test_cycling_pattern_detected(self) -> None:
        # A-B-C repeated 3 times in the window; new A-B-C completes a 4th.
        a, b, c = "fp_a", "fp_b", "fp_c"
        history = (a, b, c, a, b, c, a, b, c)
        result = _detect_sequence_loop(history, (a, b, c), 5, 3)
        assert result is not None
        assert result == (a, b, c)  # the 3-call pattern itself

    def test_shorter_pattern_within_window(self) -> None:
        # A-B-A-B-A-B (6 calls) with unrelated calls around it; a new
        # A-B completes another occurrence of the 2-call pattern.
        a, b, x = "fp_a", "fp_b", "fp_x"
        history = (x, a, b, a, b, a, b, x)
        result = _detect_sequence_loop(history, (a, b), 5, 3)
        assert result == (a, b)  # the 2-call pattern itself

    def test_single_call_new_not_enough(self) -> None:
        # A single new call can never complete a sequence loop.
        a = "fp_a"
        history = (a, a, a, a, a)
        assert _detect_sequence_loop(history, (a,), 5, 3) is None

    def test_below_repeat_count(self) -> None:
        # Pattern occurred only twice; 3rd occurrence not yet completed.
        a, b, c = "fp_a", "fp_b", "fp_c"
        history = (a, b, c, a, b, c)
        assert _detect_sequence_loop(history, (a, b, c), 5, 3) is None

    def test_exact_repetition_handled_by_threshold_not_sequence(self) -> None:
        # 4 identical calls in a row: the 5-call pattern "aaaaa" has
        # occurred only once, so the sequence detector must NOT fire —
        # the exact-repetition threshold owns this case.
        a = "fp_a"
        history = (a, a, a, a)
        assert _detect_sequence_loop(history, (a,), 5, 3) is None

    def test_no_repetition(self) -> None:
        history = tuple(f"fp_{i}" for i in range(10))
        assert _detect_sequence_loop(history, ("fp_new", "fp_new2"), 5, 3) is None

    def test_insufficient_history(self) -> None:
        a, b = "fp_a", "fp_b"
        assert _detect_sequence_loop((a, b), (a, b), 5, 3) is None


def _sequence_messages(
    plan: list[tuple[str, dict[str, object]]],
) -> list[object]:
    """Build AIMessage/ToolMessage pairs for a list of (name, args) calls."""
    messages: list[object] = []
    for i, (name, args) in enumerate(plan):
        call_id = f"s{i}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"id": call_id, "name": name, "args": args}],
            )
        )
        messages.append(ToolMessage(content="ok", tool_call_id=call_id, name=name))
    return messages


class TestAfterModelSequence:
    """Tests for sequence-loop detection through after_model."""

    def test_cycling_loop_nudged(self) -> None:
        """A-B-C repeated 3 times, then A-B-C again → nudge on 3rd turn."""
        mw = RepeatedToolCallMiddleware()
        a = ("read_file", {"file_path": "/a.txt"})
        b = ("read_file", {"file_path": "/b.txt"})
        c = ("read_file", {"file_path": "/c.txt"})
        # 9 executed calls: A-B-C three times.
        messages = _sequence_messages([a, b, c, a, b, c, a, b, c])
        # New turn: A-B-C again.
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": a[0], "args": a[1]},
                    {"id": "n2", "name": b[0], "args": b[1]},
                    {"id": "n3", "name": c[0], "args": c[1]},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is not None
        assert result["jump_to"] == "model"
        content = result["messages"][0].content
        assert "sequence" in content
        assert "3 times" in content
        # The nudge describes the pattern with readable call names.
        assert "read_file" in content

    def test_cycling_loop_not_nudged_before_third_occurrence(self) -> None:
        """A-B-C twice + new A-B-C → only 2nd completion, no nudge."""
        mw = RepeatedToolCallMiddleware()
        a = ("read_file", {"file_path": "/a.txt"})
        b = ("read_file", {"file_path": "/b.txt"})
        c = ("read_file", {"file_path": "/c.txt"})
        messages = _sequence_messages([a, b, c, a, b, c])
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": a[0], "args": a[1]},
                    {"id": "n2", "name": b[0], "args": b[1]},
                    {"id": "n3", "name": c[0], "args": c[1]},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is None

    def test_two_call_cycle_nudged(self) -> None:
        """A-B repeated 3 times, then A-B again → nudge."""
        mw = RepeatedToolCallMiddleware()
        a = ("ls", {"path": "/x"})
        b = ("ls", {"path": "/y"})
        messages = _sequence_messages([a, b, a, b, a, b])
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": a[0], "args": a[1]},
                    {"id": "n2", "name": b[0], "args": b[1]},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is not None
        assert "sequence of 2" in result["messages"][0].content

    def test_distinct_calls_no_nudge(self) -> None:
        """10 distinct calls followed by 2 more distinct → no nudge."""
        mw = RepeatedToolCallMiddleware()
        plan = [("read_file", {"file_path": f"/f{i}.txt"}) for i in range(10)]
        messages = _sequence_messages(plan)
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": "read_file", "args": {"file_path": "/g1.txt"}},
                    {"id": "n2", "name": "read_file", "args": {"file_path": "/g2.txt"}},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is None

    def test_sequence_detection_needs_enough_history(self) -> None:
        """Fewer than `sequence_length` executed calls → no sequence check."""
        mw = RepeatedToolCallMiddleware()
        a = ("ls", {"path": "/x"})
        b = ("ls", {"path": "/y"})
        # 4 executed calls (below sequence_length=5): A-B-A-B.
        messages = _sequence_messages([a, b, a, b])
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": a[0], "args": a[1]},
                    {"id": "n2", "name": b[0], "args": b[1]},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is None

    def test_nudge_lc_source(self) -> None:
        mw = RepeatedToolCallMiddleware()
        a = ("ls", {"path": "/x"})
        b = ("ls", {"path": "/y"})
        c = ("ls", {"path": "/z"})
        messages = _sequence_messages([a, b, c, a, b, c, a, b, c])
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "n1", "name": a[0], "args": a[1]},
                    {"id": "n2", "name": b[0], "args": b[1]},
                    {"id": "n3", "name": c[0], "args": c[1]},
                ],
            )
        )
        result = mw.after_model({"messages": messages}, runtime=None)
        assert result is not None
        assert result["messages"][0].additional_kwargs["lc_source"] == "repeated_tool_call"
