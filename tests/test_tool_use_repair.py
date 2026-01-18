"""
Tests for validate_and_repair_tool_use_pairing function in anthropic_client.py

This module tests the auto-repair functionality that detects and fixes orphaned
tool_use blocks (tool_use without corresponding tool_result) before sending
requests to the Anthropic API.

The error being fixed:
    "tool_use ids were found without tool_result blocks immediately after: 
     toolu_xxx. Each tool_use block must have a corresponding tool_result 
     block in the next message."
"""

import pytest

from letta.llm_api.anthropic_client import (
    validate_and_repair_tool_use_pairing,
    merge_tool_results_into_user_messages,
    dedupe_tool_results_in_user_messages,
)


class TestValidateAndRepairToolUsePairing:
    """Test suite for validate_and_repair_tool_use_pairing function."""

    def test_empty_messages_returns_empty(self):
        """Empty message list should return empty list."""
        result = validate_and_repair_tool_use_pairing([])
        assert result == []

    def test_none_messages_returns_empty(self):
        """None should be handled gracefully."""
        result = validate_and_repair_tool_use_pairing(None)
        assert result == [] or result is None

    def test_single_user_message_unchanged(self):
        """A single user message should pass through unchanged."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        result = validate_and_repair_tool_use_pairing(messages)
        assert result == messages

    def test_valid_tool_use_with_result_unchanged(self):
        """Valid tool_use followed by tool_result should pass through unchanged."""
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
        # Should be unchanged (no repairs needed)
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"
        # Verify tool_result is present
        tool_results = [b for b in result[1]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_123"

    def test_orphaned_tool_use_at_end_gets_synthetic_result(self):
        """Tool_use at the end of messages (no following message) should get synthetic tool_result."""
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
        
        # Should now have 3 messages (synthetic user message with tool_result added)
        assert len(result) == 3
        assert result[2]["role"] == "user"
        
        # Verify synthetic tool_result was added
        tool_results = [b for b in result[2]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_orphan"
        assert tool_results[0]["is_error"] is True
        assert "interrupted" in tool_results[0]["content"].lower()

    def test_orphaned_tool_use_followed_by_non_user_message(self):
        """Tool_use followed by assistant (not user) message should get synthetic tool_result."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_123", "name": "search", "input": {"query": "test"}},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I found something."}],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        
        # Should have 3 messages now (synthetic user message inserted)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "user"  # Synthetic
        assert result[2]["role"] == "assistant"
        
        # Verify synthetic tool_result
        tool_results = [b for b in result[1]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_123"
        assert tool_results[0]["is_error"] is True

    def test_missing_tool_result_in_user_message(self):
        """User message exists but is missing tool_result for a tool_use should get synthetic result injected."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_missing", "name": "calculator", "input": {"expr": "2+2"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Thanks!"}],  # No tool_result!
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        
        # Should still have 2 messages, but user message should have synthetic tool_result prepended
        assert len(result) == 2
        assert result[1]["role"] == "user"
        
        # Verify synthetic tool_result was added
        tool_results = [b for b in result[1]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_missing"
        assert tool_results[0]["is_error"] is True
        
        # Original text content should still be there
        text_blocks = [b for b in result[1]["content"] if b.get("type") == "text"]
        assert len(text_blocks) == 1

    def test_multiple_tool_uses_all_repaired(self):
        """Multiple tool_use blocks in one message should all get synthetic results if missing."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "toolu_2", "name": "tool_b", "input": {}},
                    {"type": "tool_use", "id": "toolu_3", "name": "tool_c", "input": {}},
                ],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        
        # Should have 2 messages now
        assert len(result) == 2
        assert result[1]["role"] == "user"
        
        # Verify all 3 tool_results were added
        tool_results = [b for b in result[1]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 3
        tool_use_ids = {tr["tool_use_id"] for tr in tool_results}
        assert tool_use_ids == {"toolu_1", "toolu_2", "toolu_3"}

    def test_partial_tool_results_only_missing_repaired(self):
        """If some tool_results exist but others are missing, only missing ones get synthetic results."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_has_result", "name": "tool_a", "input": {}},
                    {"type": "tool_use", "id": "toolu_missing", "name": "tool_b", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_has_result", "content": "Result A"},
                    # toolu_missing has no result!
                ],
            },
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        
        assert len(result) == 2
        
        # Verify we have 2 tool_results now
        tool_results = [b for b in result[1]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 2
        
        # One should be original, one should be synthetic
        by_id = {tr["tool_use_id"]: tr for tr in tool_results}
        assert by_id["toolu_has_result"]["content"] == "Result A"
        assert by_id["toolu_has_result"].get("is_error") is not True
        assert by_id["toolu_missing"]["is_error"] is True

    def test_complex_conversation_with_multiple_tool_turns(self):
        """Test a realistic conversation with multiple tool call turns."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search and calculate"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll search first."},
                    {"type": "tool_use", "id": "toolu_search", "name": "search", "input": {"q": "test"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_search", "content": "Found: 42"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Now I'll calculate."},
                    {"type": "tool_use", "id": "toolu_calc", "name": "calculate", "input": {"x": 42}},
                ],
            },
            # Missing tool_result for toolu_calc!
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        
        # Should have 5 messages (synthetic user message added at end)
        assert len(result) == 5
        assert result[4]["role"] == "user"
        
        # Verify synthetic tool_result for toolu_calc
        tool_results = [b for b in result[4]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_calc"

    def test_no_tool_use_in_assistant_message_unchanged(self):
        """Assistant message without tool_use should pass through unchanged."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        assert result == messages

    def test_string_content_handled(self):
        """String content (not list) should be handled gracefully."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = validate_and_repair_tool_use_pairing(messages)
        assert len(result) == 2

    def test_tool_use_without_id_skipped(self):
        """Tool_use blocks without id should be skipped (not cause errors)."""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "name": "no_id_tool", "input": {}},  # No id!
                ],
            },
        ]
        # Should not raise, should return messages (possibly with synthetic result if needed)
        result = validate_and_repair_tool_use_pairing(messages)
        assert result is not None


class TestMergeToolResultsIntoUserMessages:
    """Test suite for merge_tool_results_into_user_messages function."""

    def test_consecutive_user_messages_merged(self):
        """Two consecutive user messages should be merged into one."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "First"}]},
            {"role": "user", "content": [{"type": "text", "text": "Second"}]},
        ]
        result = merge_tool_results_into_user_messages(messages)
        
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2

    def test_alternating_roles_unchanged(self):
        """Alternating user/assistant messages should be unchanged."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ]
        result = merge_tool_results_into_user_messages(messages)
        assert len(result) == 3


class TestDedupeToolResultsInUserMessages:
    """Test suite for dedupe_tool_results_in_user_messages function."""

    def test_duplicate_tool_results_merged(self):
        """Duplicate tool_results with same tool_use_id should be merged."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Result 1"},
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "Result 2"},
                ],
            },
        ]
        result = dedupe_tool_results_in_user_messages(messages)
        
        tool_results = [b for b in result[0]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        # Content should be merged
        assert "Result 1" in tool_results[0]["content"]
        assert "Result 2" in tool_results[0]["content"]

    def test_different_tool_use_ids_kept_separate(self):
        """Tool_results with different tool_use_ids should remain separate."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Result 1"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "Result 2"},
                ],
            },
        ]
        result = dedupe_tool_results_in_user_messages(messages)
        
        tool_results = [b for b in result[0]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 2


class TestIntegrationWithFullPipeline:
    """Integration tests simulating the full message processing pipeline."""

    def test_full_pipeline_with_orphaned_tool_use(self):
        """Test the full pipeline: merge -> dedupe -> validate_repair."""
        # Simulate a corrupted state: assistant made tool call, but result was never saved
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search for something"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Searching..."},
                    {"type": "tool_use", "id": "toolu_search_123", "name": "web_search", "input": {"query": "test"}},
                ],
            },
            # No tool_result message! This simulates a crash/timeout
        ]
        
        # Run through full pipeline
        merged = merge_tool_results_into_user_messages(messages)
        deduped = dedupe_tool_results_in_user_messages(merged)
        repaired = validate_and_repair_tool_use_pairing(deduped)
        
        # Should now be valid for Anthropic
        assert len(repaired) == 3
        assert repaired[2]["role"] == "user"
        
        tool_results = [b for b in repaired[2]["content"] if b.get("type") == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "toolu_search_123"

    def test_already_valid_conversation_unchanged(self):
        """A valid conversation should pass through the pipeline unchanged (except order)."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Calculate 2+2"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_calc", "name": "calculator", "input": {"expr": "2+2"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_calc", "content": "4"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The answer is 4."}],
            },
        ]
        
        merged = merge_tool_results_into_user_messages(messages)
        deduped = dedupe_tool_results_in_user_messages(merged)
        repaired = validate_and_repair_tool_use_pairing(deduped)
        
        # Should have same structure
        assert len(repaired) == 4
        assert repaired[0]["role"] == "user"
        assert repaired[1]["role"] == "assistant"
        assert repaired[2]["role"] == "user"
        assert repaired[3]["role"] == "assistant"
