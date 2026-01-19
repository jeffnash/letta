"""
Tests for OpenAI tool call repair functions in openai_client.py

This module tests the auto-repair functionality that detects and fixes orphaned
tool_calls (tool_call without corresponding tool response) before sending
requests to the OpenAI API or OpenAI-compatible proxies.

The error being fixed:
    For proxies that convert to Anthropic format:
    "tool_use ids were found without tool_result blocks immediately after: 
     toolu_xxx. Each tool_use block must have a corresponding tool_result 
     block in the next message."
"""

import pytest

from letta.llm_api.openai_client import (
    validate_and_repair_openai_tool_call_pairing,
    validate_and_repair_responses_api_tool_call_pairing,
)


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
        """System message between tool_call and tool response should trigger synthetic response insertion."""
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
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_xyz"
        assert "Error" in result[1]["content"]  # This is the synthetic one
        assert result[2]["role"] == "system"
        assert result[3]["role"] == "tool"  # The original (late) tool response

    def test_developer_message_between_tool_call_and_response(self):
        """Developer message between tool_call and tool response should trigger synthetic response insertion."""
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
        assert len(result) == 4
        assert result[0]["role"] == "assistant"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_abc"
        assert "Error" in result[1]["content"]
        assert result[2]["role"] == "developer"
        assert result[3]["role"] == "tool"

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

    def test_output_before_call_still_matches(self):
        """Output appearing before its function_call in the list should still be considered matched."""
        items = [
            {"type": "function_call_output", "call_id": "call_early", "output": "Early output"},
            {"type": "function_call", "call_id": "call_early", "name": "tool", "arguments": "{}"},
        ]
        result = validate_and_repair_responses_api_tool_call_pairing(items)

        # Should pass through unchanged - output exists (even if ordered weirdly)
        assert len(result) == 2

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
        )
        assert is_anthropic_backed_proxy(llm_config) is False

    def test_none_provider_name_returns_false(self):
        """Config with None provider_name should return False."""
        from letta.llm_api.openai_client import is_anthropic_backed_proxy
        from letta.schemas.llm_config import LLMConfig

        llm_config = LLMConfig(
            model="gpt-4o",
            model_endpoint_type="openai",
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
        )
        assert is_anthropic_backed_proxy(llm_config) is True

