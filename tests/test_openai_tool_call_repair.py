"""
Tests for OpenAI and Anthropic tool call repair functions.

This module tests the auto-repair functionality that detects and fixes:
1. Orphaned tool_calls (tool_call without corresponding tool response) - repaired by injecting synthetic responses
2. Orphaned tool_results (tool response without matching tool_call) - repaired by removing the orphaned response

The errors being fixed:
    For orphaned tool_calls:
    "tool_use ids were found without tool_result blocks immediately after: 
     toolu_xxx. Each tool_use block must have a corresponding tool_result 
     block in the next message."

    For orphaned tool_results:
    "unexpected `tool_use_id` found in `tool_result` blocks: toolu_xxx.
     Each `tool_result` block must have a corresponding `tool_use` block
     in the previous message."
"""

import pytest

from letta.llm_api.openai_client import (
    validate_and_repair_openai_tool_call_pairing,
    validate_and_repair_responses_api_tool_call_pairing,
)
from letta.llm_api.anthropic_client import validate_and_repair_tool_use_pairing


class TestValidateAndRepairOpenAIToolCallPairing:
    """Test suite for validate_and_repair_openai_tool_call_pairing function."""

    def test_empty_messages_returns_empty(self):
        """Empty message list should return empty list."""
        result = validate_and_repair_openai_tool_call_pairing([])
        assert result == []

    def test_none_messages_returns_none(self):
        """None should be handled gracefully."""
        result = validate_and_repair_openai_tool_call_pairing(None)
        assert result is None

    def test_single_user_message_unchanged(self):
        """A single user message should pass through unchanged."""
        messages = [{"role": "user", "content": "Hello"}]
        result = validate_and_repair_openai_tool_call_pairing(messages)
        assert result == messages

    def test_valid_tool_call_with_response_unchanged(self):
        """Valid tool_call followed by tool response should pass through unchanged."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me check that.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "Sunny, 72°F",
            },
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"

    def test_orphaned_tool_call_at_end_gets_synthetic_response(self):
        """Tool_call at the end of messages (no following message) should get synthetic tool response."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_orphan",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                    }
                ],
            },
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should now have 3 messages (synthetic tool response added)
        assert len(result) == 3
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_orphan"
        assert "Error" in result[2]["content"]
        assert "get_weather" in result[2]["content"]

    def test_orphaned_tool_call_followed_by_user_message(self):
        """Tool_call followed by user message (not tool) should get synthetic tool response."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"query": "test"}'},
                    }
                ],
            },
            {"role": "user", "content": "What happened?"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 3 messages: assistant, synthetic tool response, user
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"
        assert result[2]["role"] == "user"

    def test_orphaned_tool_call_followed_by_system_message(self):
        """Tool_call followed by system message should get synthetic tool response before system message."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_sys",
                        "type": "function",
                        "function": {"name": "analyze", "arguments": "{}"},
                    }
                ],
            },
            {"role": "system", "content": "System update: new context"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 3 messages: assistant, synthetic tool response, system
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_sys"
        assert result[2]["role"] == "system"

    def test_orphaned_tool_call_followed_by_developer_message(self):
        """Tool_call followed by developer message should get synthetic tool response before developer message."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_dev",
                        "type": "function",
                        "function": {"name": "process", "arguments": "{}"},
                    }
                ],
            },
            {"role": "developer", "content": "Developer instruction"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 3 messages: assistant, synthetic tool response, developer
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_dev"
        assert result[2]["role"] == "developer"

    def test_multiple_tool_calls_all_repaired(self):
        """Multiple tool_calls in one assistant message should all get synthetic responses if missing."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": "tool_c", "arguments": "{}"}},
                ],
            },
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 4 messages: 1 assistant + 3 synthetic tool responses
        assert len(result) == 4
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 3
        tool_call_ids = {tr["tool_call_id"] for tr in tool_responses}
        assert tool_call_ids == {"call_1", "call_2", "call_3"}

    def test_partial_tool_responses_only_missing_repaired(self):
        """If some tool responses exist but others are missing, only missing ones get synthetic responses."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_has_result", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "call_missing", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_has_result", "content": "Result A"},
            # call_missing has no response!
            {"role": "user", "content": "Thanks!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 4 messages: assistant, real tool response, synthetic tool response, user
        assert len(result) == 4
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2

        # One should be original, one should be synthetic
        by_id = {tr["tool_call_id"]: tr for tr in tool_responses}
        assert by_id["call_has_result"]["content"] == "Result A"
        assert "Error" in by_id["call_missing"]["content"]

    def test_complex_conversation_with_multiple_tool_turns(self):
        """Test a realistic conversation with multiple tool call turns."""
        messages = [
            {"role": "user", "content": "Search and calculate"},
            {
                "role": "assistant",
                "content": "I'll search first.",
                "tool_calls": [
                    {"id": "call_search", "type": "function", "function": {"name": "search", "arguments": '{"q": "test"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_search", "content": "Found: 42"},
            {
                "role": "assistant",
                "content": "Now I'll calculate.",
                "tool_calls": [
                    {"id": "call_calc", "type": "function", "function": {"name": "calculate", "arguments": '{"x": 42}'}},
                ],
            },
            # Missing tool response for call_calc!
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 5 messages (synthetic tool response added at end)
        assert len(result) == 5
        assert result[4]["role"] == "tool"
        assert result[4]["tool_call_id"] == "call_calc"

    def test_no_tool_calls_in_assistant_message_unchanged(self):
        """Assistant message without tool_calls should pass through unchanged."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)
        assert result == messages

    def test_string_content_handled(self):
        """String content (not dict/list) should be handled gracefully."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)
        assert len(result) == 2

    def test_tool_call_without_id_skipped(self):
        """Tool_calls without id should be skipped (not cause errors)."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"type": "function", "function": {"name": "test"}},  # Missing id
                ],
            },
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)
        # Should pass through unchanged since there's no id to track
        assert len(result) == 1

    def test_system_message_between_tool_call_and_response(self):
        """System message between tool_call and tool response should trigger synthetic response insertion.
        
        The late/mispositioned tool response should be REMOVED because Anthropic requires
        tool_results to be in the user message immediately after the assistant with tool_use.
        Keeping both synthetic and late responses would cause duplicate tool_result blocks.
        """
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_xyz", "type": "function", "function": {"name": "do_something", "arguments": "{}"}},
                ],
            },
            {"role": "system", "content": "New system prompt injected"},
            {"role": "tool", "tool_call_id": "call_xyz", "content": "This comes too late"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The synthetic response should be inserted BEFORE the system message
        # The late/mispositioned tool response should be REMOVED (not kept)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_xyz"
        assert "Error" in result[1]["content"]  # This is the synthetic one
        assert result[2]["role"] == "system"
        # The late tool response is removed as it's mispositioned

    def test_developer_message_between_tool_call_and_response(self):
        """Developer message between tool_call and tool response should trigger synthetic response insertion.
        
        The late/mispositioned tool response should be REMOVED because Anthropic requires
        tool_results to be in the user message immediately after the assistant with tool_use.
        Keeping both synthetic and late responses would cause duplicate tool_result blocks.
        """
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_abc", "type": "function", "function": {"name": "fetch_data", "arguments": "{}"}},
                ],
            },
            {"role": "developer", "content": "Developer override"},
            {"role": "tool", "tool_call_id": "call_abc", "content": "Late response"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The synthetic response should be inserted BEFORE the developer message
        # The late/mispositioned tool response should be REMOVED (not kept)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_abc"
        assert "Error" in result[1]["content"]
        assert result[2]["role"] == "developer"
        # The late tool response is removed as it's mispositioned

    def test_unknown_role_triggers_boundary(self):
        """Unknown role should also trigger boundary detection."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_unk", "type": "function", "function": {"name": "test_func", "arguments": "{}"}},
                ],
            },
            {"role": "custom_role", "content": "Some custom message"},  # Unknown role
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should insert synthetic response before the unknown role message
        assert len(result) == 3
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_unk"
        assert result[2]["role"] == "custom_role"


class TestOrphanedToolResultRemoval:
    """Test suite for orphaned tool_result removal in validate_and_repair_openai_tool_call_pairing.

    These tests verify the SECOND PASS of the repair function that removes tool response
    messages (role='tool') that reference tool_call_ids which don't exist in any
    assistant message. This can happen when summarization deletes assistant messages
    but leaves behind their corresponding tool responses.

    The Anthropic API error for orphaned tool_results:
        "unexpected `tool_use_id` found in `tool_result` blocks: toolu_xxx.
         Each `tool_result` block must have a corresponding `tool_use` block
         in the previous message."
    """

    def test_orphaned_tool_result_removed(self):
        """Tool response without matching tool_call should be removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "call_nonexistent", "content": "Some result"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The orphaned tool response should be removed
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_valid_tool_result_kept_orphaned_removed(self):
        """Valid tool results should be kept while orphaned ones are removed."""
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_valid", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_valid", "content": "Valid result"},
            {"role": "tool", "tool_call_id": "call_orphan", "content": "Orphaned result"},  # No matching tool_call
            {"role": "user", "content": "Thanks!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 3 messages: assistant, valid tool response, user
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_valid"
        assert result[2]["role"] == "user"

    def test_multiple_orphaned_tool_results_removed(self):
        """Multiple orphaned tool results should all be removed."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "call_orphan_1", "content": "Result 1"},
            {"role": "tool", "tool_call_id": "call_orphan_2", "content": "Result 2"},
            {"role": "tool", "tool_call_id": "call_orphan_3", "content": "Result 3"},
            {"role": "assistant", "content": "Processing..."},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # All orphaned tool responses should be removed
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_orphaned_tool_results_after_summarization(self):
        """Simulate summarization removing assistant with tool_calls but leaving tool response."""
        # This is the exact scenario from the bug report: summarization deleted the assistant
        # message containing tool_calls but left the tool response message behind
        messages = [
            {"role": "user", "content": "What's the weather?"},
            # Assistant with tool_call was deleted by summarization
            {"role": "tool", "tool_call_id": "toolu_01BiPnYwiQjfTjKWnyYUCfZ", "content": "Sunny, 72°F"},
            {"role": "assistant", "content": "The weather is sunny and 72°F."},
            {"role": "user", "content": "Thanks!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Orphaned tool response should be removed
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_tool_result_without_tool_call_id_kept(self):
        """Tool response without tool_call_id field should not be removed (malformed but not orphaned)."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "content": "Some result"},  # No tool_call_id field
            {"role": "assistant", "content": "Hi!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The malformed tool message should be kept (we only remove orphaned ones with invalid IDs)
        assert len(result) == 3

    def test_mixed_valid_and_orphaned_in_sequence(self):
        """Test a complex sequence with interleaved valid and orphaned tool results."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_A", "type": "function", "function": {"name": "task_a", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_A", "content": "Result A"},
            {"role": "tool", "tool_call_id": "call_deleted", "content": "Orphaned from deleted assistant"},  # Orphaned
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_B", "type": "function", "function": {"name": "task_b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_B", "content": "Result B"},
            {"role": "assistant", "content": "Done!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 6 messages (orphaned one removed)
        assert len(result) == 6
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2
        tool_call_ids = {tr["tool_call_id"] for tr in tool_responses}
        assert tool_call_ids == {"call_A", "call_B"}

    def test_tool_result_before_its_assistant_message_removed(self):
        """Tool response appearing BEFORE its assistant message should be removed.
        
        This is a critical edge case for Anthropic compatibility. The error:
            "Each `tool_result` block must have a corresponding `tool_use` block
             in the previous message."
        
        If a tool_result references a tool_use that appears LATER in the conversation,
        it will cause a 400 error when converted to Anthropic format.
        """
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "tool", "tool_call_id": "call_future", "content": "Result for future call"},  # BEFORE its assistant
            {"role": "assistant", "tool_calls": [
                {"id": "call_future", "type": "function", "function": {"name": "get_data", "arguments": "{}"}},
            ]},
            {"role": "user", "content": "Thanks"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The tool response appearing BEFORE its assistant should be removed
        # A synthetic response should be injected AFTER the assistant
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_future"
        assert "Error" in result[2]["content"]  # This is the synthetic one
        assert result[3]["role"] == "user"

    def test_tool_result_for_earlier_assistant_removed(self):
        """Tool response referencing an earlier assistant (not immediately preceding) should be removed.
        
        This tests the Anthropic constraint that tool_results must reference tool_uses
        in the IMMEDIATELY PRECEDING assistant message.
        """
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_A", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_A", "content": "Result A"},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_B", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_B", "content": "Result B"},
            {"role": "tool", "tool_call_id": "call_A", "content": "Late duplicate result for A"},  # WRONG: A is not in immediately preceding assistant
            {"role": "user", "content": "Done"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # The late duplicate tool response for call_A should be removed
        # because it appears after assistant B, not after assistant A
        assert len(result) == 6
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2
        tool_call_ids = [tr["tool_call_id"] for tr in tool_responses]
        assert tool_call_ids == ["call_A", "call_B"]  # Only one response per call

    def test_synthetic_response_not_removed(self):
        """Synthetic tool responses injected by first pass should not be removed by second pass."""
        # This tests that synthetic responses added for orphaned tool_calls are not
        # then removed as orphaned tool_results
        messages = [
            {"role": "user", "content": "Search for something"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_orphan_call", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            # Missing tool response - will be synthesized in first pass
            {"role": "user", "content": "What happened?"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 4 messages: user, assistant, synthetic tool response, user
        assert len(result) == 4
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_orphan_call"
        assert "Error" in result[2]["content"]

    def test_synthetic_insertion_does_not_remove_existing_tool_response(self):
        """Regression test: synthetic insertion must not cause existing tool responses to be removed.
        
        This tests the fix for a bug where the correctly_positioned_indices calculation was off-by-one,
        causing existing tool responses to be incorrectly removed after synthetic ones were inserted.
        
        The bug scenario:
        1. Assistant has tool_calls=[call_1, call_2]
        2. Tool response for call_1 exists at correct position
        3. Tool response for call_2 is missing (orphaned)
        4. Synthetic response for call_2 is inserted
        5. BUG: The existing response for call_1 was incorrectly removed
        
        The error manifested as Anthropic 400: "unexpected tool_use_id found in tool_result blocks"
        because the synthetic response was kept but the original response was removed, leaving
        mismatched tool_use/tool_result pairs.
        """
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "calculate", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Search result: 42"},
            # call_2 is orphaned - no tool response!
            {"role": "user", "content": "Thanks!"},
        ]
        result = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have 4 messages: assistant, synthetic for call_2, existing for call_1, user
        # Note: synthetic is inserted at position 1, which shifts existing call_1 response to position 2
        assert len(result) == 4
        
        # Both tool responses must be present
        tool_responses = [m for m in result if m.get("role") == "tool"]
        assert len(tool_responses) == 2
        
        # Verify both tool_call_ids are represented
        tool_call_ids = {tr["tool_call_id"] for tr in tool_responses}
        assert tool_call_ids == {"call_1", "call_2"}
        
        # Verify the order and content
        by_id = {tr["tool_call_id"]: tr for tr in tool_responses}
        assert by_id["call_1"]["content"] == "Search result: 42"  # Original
        assert "Error" in by_id["call_2"]["content"]  # Synthetic


class TestValidateAndRepairResponsesAPIToolCallPairing:
    """Test suite for validate_and_repair_responses_api_tool_call_pairing function."""

    def test_empty_items_returns_empty(self):
        """Empty item list should return empty list."""
        result = validate_and_repair_responses_api_tool_call_pairing([])
        assert result == []

    def test_none_items_returns_none(self):
        """None should be handled gracefully."""
        result = validate_and_repair_responses_api_tool_call_pairing(None)
        assert result is None

    def test_valid_function_call_with_output_unchanged(self):
        """Valid function_call followed by function_call_output should pass through unchanged."""
        items = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "Sunny, 72°F",
            },
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)
        assert len(result) == 2
        assert result[0]["type"] == "function_call"
        assert result[1]["type"] == "function_call_output"
        assert result[1]["call_id"] == "call_123"

    def test_orphaned_function_call_gets_synthetic_output(self):
        """Function_call without matching function_call_output should get synthetic output."""
        items = [
            {
                "type": "function_call",
                "call_id": "call_orphan",
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have 2 items: function_call + synthetic output
        assert len(result) == 2
        assert result[0]["type"] == "function_call"
        assert result[1]["type"] == "function_call_output"
        assert result[1]["call_id"] == "call_orphan"
        assert "Error" in result[1]["output"]
        assert "get_weather" in result[1]["output"]

    def test_multiple_function_calls_all_repaired(self):
        """Multiple function_calls without outputs should all get synthetic outputs."""
        items = [
            {"type": "function_call", "call_id": "call_1", "name": "tool_a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_2", "name": "tool_b", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_3", "name": "tool_c", "arguments": "{}"},
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have 6 items: 3 function_calls + 3 synthetic outputs
        assert len(result) == 6
        outputs = [item for item in result if item.get("type") == "function_call_output"]
        assert len(outputs) == 3
        call_ids = {o["call_id"] for o in outputs}
        assert call_ids == {"call_1", "call_2", "call_3"}

    def test_partial_outputs_only_missing_repaired(self):
        """If some outputs exist but others are missing, only missing ones get synthetic outputs."""
        items = [
            {"type": "function_call", "call_id": "call_has_output", "name": "tool_a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_missing", "name": "tool_b", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_has_output", "output": "Result A"},
            # call_missing has no output!
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have 4 items: 2 function_calls + 2 outputs (1 real, 1 synthetic)
        assert len(result) == 4
        outputs = [item for item in result if item.get("type") == "function_call_output"]
        assert len(outputs) == 2

        # Check both outputs
        by_id = {o["call_id"]: o for o in outputs}
        assert by_id["call_has_output"]["output"] == "Result A"
        assert "Error" in by_id["call_missing"]["output"]

    def test_orphaned_function_call_output_removed(self):
        """function_call_output without matching function_call should be removed."""
        items = [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "function_call_output", "call_id": "call_orphan", "output": "orphan"},
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        assert len(result) == 2
        assert all(item.get("type") != "function_call_output" for item in result)

    def test_function_call_without_call_id_skipped(self):
        """Function_call without call_id should be skipped (not cause errors)."""
        items = [
            {"type": "function_call", "name": "test"},  # Missing call_id
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)
        # Should pass through unchanged since there's no call_id to track
        assert len(result) == 1

    def test_mixed_items_handled(self):
        """Mix of function_calls, outputs, and other item types should be handled."""
        items = [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "function_call", "call_id": "call_1", "name": "greet", "arguments": "{}"},
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            # call_1 has no output!
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have 4 items: message, function_call, synthetic output, message
        assert len(result) == 4
        # The synthetic output should be inserted after the function_call
        assert result[0]["type"] == "message"
        assert result[1]["type"] == "function_call"
        assert result[2]["type"] == "function_call_output"
        assert result[2]["call_id"] == "call_1"
        assert result[3]["type"] == "message"

    def test_output_before_call_is_removed_and_repaired(self):
        """Output before its function_call should be removed and replaced with synthetic output."""
        items = [
            {"type": "function_call_output", "call_id": "call_early", "output": "Early output"},
            {"type": "function_call", "call_id": "call_early", "name": "tool", "arguments": "{}"},
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Mispositioned output is removed; missing output is repaired immediately after the call.
        assert len(result) == 2
        assert result[0]["type"] == "function_call"
        assert result[1]["type"] == "function_call_output"
        assert result[1]["call_id"] == "call_early"
        assert "Error" in result[1]["output"]

    def test_duplicate_function_calls_each_get_one_output(self):
        """Multiple function_calls with the same call_id should not cause duplicate synthetic outputs."""
        items = [
            {"type": "function_call", "call_id": "call_dup", "name": "tool_a", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_dup", "name": "tool_a", "arguments": "{}"},  # Duplicate
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have 3 items: 2 function_calls + 1 synthetic output (deduped)
        # First function_call gets synthetic output, second sees it exists
        outputs = [item for item in result if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call_dup"


class TestIntegrationWithFullPipeline:
    """Integration tests simulating the full message processing pipeline."""

    def test_full_pipeline_with_orphaned_tool_call_openai_format(self):
        """Test the full pipeline: message with orphaned tool call in OpenAI format."""
        # Simulate a corrupted state: assistant made tool call, but response was never saved
        messages = [
            {"role": "user", "content": "Search for something"},
            {
                "role": "assistant",
                "content": "Searching...",
                "tool_calls": [
                    {"id": "call_search_123", "type": "function", "function": {"name": "web_search", "arguments": '{"query": "test"}'}},
                ],
            },
            # No tool response! This simulates a crash/timeout
        ]

        # Run repair
        repaired = validate_and_repair_openai_tool_call_pairing(messages)

        # Should have synthetic tool response inserted
        assert len(repaired) == 3
        assert repaired[2]["role"] == "tool"
        assert repaired[2]["tool_call_id"] == "call_search_123"
        assert "Error" in repaired[2]["content"]

    def test_full_pipeline_with_orphaned_function_call_responses_format(self):
        """Test the full pipeline: items with orphaned function_call in Responses API format."""
        items = [
            {"type": "message", "role": "user", "content": [{"type": "text", "text": "Search for something"}]},
            {
                "type": "function_call",
                "call_id": "call_search_456",
                "name": "web_search",
                "arguments": '{"query": "test"}',
            },
            # No function_call_output! This simulates a crash/timeout
        ]

        # Run repair
        repaired = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should have synthetic output inserted
        assert len(repaired) == 3
        outputs = [item for item in repaired if item.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert outputs[0]["call_id"] == "call_search_456"
        assert "Error" in outputs[0]["output"]


class TestIsAnthropicBackedProxy:
    """Test suite for is_anthropic_backed_proxy helper function.

    This helper detects Anthropic-backed proxies like CLIProxy so that
    we can apply appropriate constraints (e.g., disabling parallel tool calls).
    """

    def test_cliproxy_with_claude_model_returns_true(self):
        """CLIProxy with Claude model should be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="claude-sonnet-4-20250514",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True

    def test_cliproxy_with_opus_model_returns_true(self):
        """CLIProxy with Opus model should be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="opus-4.5",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True

    def test_cliproxy_with_sonnet_model_returns_true(self):
        """CLIProxy with Sonnet model should be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="sonnet-4-20250514",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True

    def test_cliproxy_with_haiku_model_returns_true(self):
        """CLIProxy with Haiku model should be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="haiku-3.5",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True

    def test_cliproxy_with_handle_containing_claude_returns_true(self):
        """CLIProxy with handle containing Claude should be detected."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="some-model",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            handle="cliproxy/claude-sonnet-4",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True

    def test_cliproxy_with_gpt_model_returns_false(self):
        """CLIProxy with GPT model should NOT be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="gpt-5.2-medium",
            model_endpoint_type="openai",
            provider_name="cliproxy",
            context_window=272000,
        )
        assert is_anthropic_backed_proxy(llm_config) is False

    def test_openai_provider_returns_false(self):
        """Regular OpenAI provider should not be detected as Anthropic-backed."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="gpt-4o",
            model_endpoint_type="openai",
            provider_name="openai",
            context_window=128000,
        )
        assert is_anthropic_backed_proxy(llm_config) is False

    def test_anthropic_provider_returns_false(self):
        """Direct Anthropic provider should not be detected (it uses AnthropicClient directly)."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="claude-sonnet-4",
            model_endpoint_type="anthropic",
            provider_name="anthropic",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is False

    def test_none_provider_name_returns_false(self):
        """Config with None provider_name should return False."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="gpt-4o",
            model_endpoint_type="openai",
            context_window=128000,
        )
        # provider_name defaults to None
        assert is_anthropic_backed_proxy(llm_config) is False

    def test_cliproxy_claude_in_handle_not_model_returns_true(self):
        """CLIProxy with Claude in handle (but not model name) should be detected."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="fast-tier",  # Generic model name
            model_endpoint_type="openai",
            provider_name="cliproxy",
            handle="cliproxy/claude-haiku-3",
            context_window=200000,
        )
        assert is_anthropic_backed_proxy(llm_config) is True


class TestAnthropicValidateAndRepairToolUsePairing:
    """Test suite for validate_and_repair_tool_use_pairing function in anthropic_client.py.

    This tests the Anthropic-format equivalent of the OpenAI repair function.
    Anthropic format uses content blocks with type='tool_use' and type='tool_result'
    rather than separate messages.
    """

    def test_empty_messages_returns_empty(self):
        """Empty message list should return empty list."""
        result = validate_and_repair_tool_use_pairing([])
        assert result == []

    def test_none_messages_returns_none(self):
        """None should be handled gracefully."""
        result = validate_and_repair_tool_use_pairing(None)
        assert result is None

    def test_valid_tool_use_with_result_unchanged(self):
        """Valid tool_use followed by tool_result in next user message should pass through."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "NYC"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Sunny, 72°F"},
                ],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"

    def test_orphaned_tool_use_at_end_gets_synthetic_result(self):
        """Tool_use at end of messages should get synthetic user message with tool_result."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_orphan", "name": "get_weather", "input": {"location": "NYC"}},
                ],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # Should have 3 messages (synthetic user message with tool_result added)
        assert len(result) == 3
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["tool_use_id"] == "toolu_orphan"
        assert "Error" in result[2]["content"][0]["content"]


class TestAnthropicOrphanedToolResultRemoval:
    """Test suite for orphaned tool_result removal in validate_and_repair_tool_use_pairing.

    These tests verify the SECOND PASS of the Anthropic repair function that removes
    tool_result blocks that reference tool_use_ids which don't exist in any assistant message.
    """

    def test_orphaned_tool_result_removed_from_user_message(self):
        """Tool_result without matching tool_use should be removed from user message content."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_nonexistent", "content": "Some result"},
                    {"type": "text", "text": "What happened?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # The orphaned tool_result should be removed, but the text should remain
        assert len(result) == 3
        user_msg = result[1]
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "What happened?"

    def test_valid_tool_result_kept_orphaned_removed(self):
        """Valid tool_results should be kept while orphaned ones are removed."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_valid", "name": "search", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_valid", "content": "Valid result"},
                    {"type": "tool_result", "tool_use_id": "toolu_orphan", "content": "Orphaned result"},
                ],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # Should have 2 messages, but user message should only have valid tool_result
        assert len(result) == 2
        user_content = result[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["tool_use_id"] == "toolu_valid"

    def test_all_tool_results_orphaned_gets_placeholder(self):
        """If all tool_results in a user message are orphaned, add placeholder text."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_orphan1", "content": "Result 1"},
                    {"type": "tool_result", "tool_use_id": "toolu_orphan2", "content": "Result 2"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Processing..."}]},
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # The user message should have a placeholder since all content was removed
        assert len(result) == 3
        user_content = result[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"
        assert "removed" in user_content[0]["text"].lower()

    def test_orphaned_tool_results_after_summarization(self):
        """Simulate summarization removing assistant with tool_use but leaving tool_result."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
            # Assistant with tool_use was deleted by summarization
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01BiPnYwiQjfTjKWnyYUCfZ", "content": "Sunny, 72°F"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "The weather is sunny."}]},
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # Orphaned tool_result should be replaced with placeholder
        assert len(result) == 3
        user_content = result[1]["content"]
        assert len(user_content) == 1
        assert user_content[0]["type"] == "text"

    def test_synthetic_result_not_removed(self):
        """Synthetic tool_results injected by first pass should not be removed by second pass."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search for something"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_orphan_use", "name": "search", "input": {}},
                ],
            },
            # Missing tool_result - will be synthesized
            {"role": "assistant", "content": [{"type": "text", "text": "Results..."}]},
        ]
        result = validate_and_repair_tool_use_pairing(messages)

        # Should have 4 messages: user, assistant with tool_use, synthetic user with tool_result, assistant
        assert len(result) == 4
        synthetic_user = result[2]
        assert synthetic_user["role"] == "user"
        assert synthetic_user["content"][0]["type"] == "tool_result"
        assert synthetic_user["content"][0]["tool_use_id"] == "toolu_orphan_use"
        assert "Error" in synthetic_user["content"][0]["content"]

