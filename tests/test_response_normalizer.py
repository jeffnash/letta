"""
Unit tests for the response normalizer module.

Tests the normalization of OpenAI-compatible proxy responses to ensure
they can be parsed by Pydantic without validation errors.
"""

import pytest

from letta.llm_api.response_normalizer import (
    normalize_chat_completion_response,
    redact_sensitive_content,
    truncate_tool_output,
    validate_and_normalize_response,
)
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse


class TestNormalizeChatCompletionResponse:
    """Tests for normalize_chat_completion_response function."""

    def test_normalize_null_object(self):
        """Test that null object field is normalized to 'chat.completion'."""
        response_data = {
            "id": "chatcmpl-123",
            "object": None,  # Proxy bug: null instead of "chat.completion"
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["object"] == "chat.completion"
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_index(self):
        """Test that null choice index is normalized to 0."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": None,  # Proxy bug: null instead of 0
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["choices"][0]["index"] == 0
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_usage(self):
        """Test that null usage is normalized to zeros."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": None,  # Proxy bug: null usage
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["usage"]["prompt_tokens"] == 0
        assert normalized["usage"]["completion_tokens"] == 0
        assert normalized["usage"]["total_tokens"] == 0
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_partial_usage(self):
        """Test that partial usage fields are normalized."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": None,  # Missing
                # total_tokens missing entirely
            },
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["usage"]["prompt_tokens"] == 10
        assert normalized["usage"]["completion_tokens"] == 0
        assert normalized["usage"]["total_tokens"] == 10  # Computed from prompt + completion
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_finish_reason(self):
        """Test that null finish_reason is normalized to 'stop'."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,  # Proxy bug
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["choices"][0]["finish_reason"] == "stop"
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_id(self):
        """Test that null id is generated."""
        response_data = {
            "id": None,
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        assert normalized["id"] is not None
        assert normalized["id"].startswith("chatcmpl-")
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_tool_call_id(self):
        """Test that null tool call id is generated."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": None,  # Proxy bug: null tool call id
                                "type": "function",
                                "function": {"name": "my_tool", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        tool_call = normalized["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"] is not None
        assert tool_call["id"].startswith("call_")
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_null_choices(self):
        """Test that null choices creates empty assistant message."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": None,  # Proxy bug: null choices
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = normalize_chat_completion_response(response_data)

        assert len(normalized["choices"]) == 1
        assert normalized["choices"][0]["message"]["role"] == "assistant"
        # Should parse without error
        ChatCompletionResponse(**normalized)

    def test_normalize_multiple_quirks(self):
        """Test that multiple quirks are normalized together."""
        response_data = {
            "id": None,
            "object": None,
            "created": None,
            "model": None,
            "choices": [
                {
                    "index": None,
                    "message": {
                        "role": None,
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": None,
                                "type": None,
                                "function": {
                                    "name": None,
                                    "arguments": None,
                                },
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }

        normalized = normalize_chat_completion_response(
            response_data, model="fallback-model"
        )

        # All fields should be normalized
        assert normalized["id"].startswith("chatcmpl-")
        assert normalized["object"] == "chat.completion"
        assert normalized["created"] is not None
        assert normalized["model"] == "fallback-model"
        assert normalized["choices"][0]["index"] == 0
        assert normalized["choices"][0]["finish_reason"] == "stop"
        assert normalized["choices"][0]["message"]["role"] == "assistant"
        assert normalized["usage"]["prompt_tokens"] == 0

        tool_call = normalized["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"].startswith("call_")
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "unknown"
        assert tool_call["function"]["arguments"] == "{}"

        # Should parse without error
        ChatCompletionResponse(**normalized)


class TestValidateAndNormalizeResponse:
    """Tests for validate_and_normalize_response function."""

    def test_returns_validated_response(self):
        """Test that function returns a validated ChatCompletionResponse."""
        response_data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        result = validate_and_normalize_response(response_data)

        assert isinstance(result, ChatCompletionResponse)
        assert result.id == "chatcmpl-123"
        assert result.choices[0].message.content == "Hello"


class TestRedactSensitiveContent:
    """Tests for redact_sensitive_content function."""

    def test_redact_api_key(self):
        """Test that API keys are redacted."""
        text = 'config = {"api_key": "sk-1234567890abcdefghij"}'
        result = redact_sensitive_content(text)
        assert "sk-1234567890" not in result
        assert "REDACTED" in result

    def test_redact_password(self):
        """Test that passwords are redacted."""
        text = 'password="mysecretpassword123"'
        result = redact_sensitive_content(text)
        assert "mysecretpassword" not in result
        assert "REDACTED" in result

    def test_redact_bearer_token(self):
        """Test that bearer tokens are redacted."""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = redact_sensitive_content(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "REDACTED" in result

    def test_truncate_long_text(self):
        """Test that long text is truncated."""
        text = "a" * 2000
        result = redact_sensitive_content(text, max_length=100)
        assert len(result) < 200  # Truncated + suffix
        assert "truncated" in result.lower() or "more chars" in result.lower()

    def test_preserve_short_text(self):
        """Test that short text without secrets is preserved."""
        text = "Hello, world!"
        result = redact_sensitive_content(text)
        assert result == text


class TestTruncateToolOutput:
    """Tests for truncate_tool_output function."""

    def test_preserve_short_output(self):
        """Test that short output is preserved."""
        output = "Short output"
        result = truncate_tool_output(output, max_chars=1000)
        assert result == output

    def test_truncate_long_output(self):
        """Test that long output is truncated."""
        output = "a" * 10000
        result = truncate_tool_output(output, max_chars=1000)
        assert len(result) < 2000  # Should be significantly shorter
        assert "truncated" in result.lower()

    def test_truncate_json_message_field(self):
        """Test that JSON with message field truncates the message."""
        import json

        data = {"status": "ok", "message": "x" * 10000, "code": 200}
        output = json.dumps(data)
        result = truncate_tool_output(output, max_chars=1000, preserve_json_structure=True)

        # Should still be valid JSON
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["code"] == 200
        assert "truncated" in parsed["message"].lower()
        assert len(parsed["message"]) < 1000

    def test_middle_truncation_for_non_json(self):
        """Test that non-JSON uses middle truncation."""
        output = "START" + "x" * 10000 + "END"
        result = truncate_tool_output(output, max_chars=1000, preserve_json_structure=False)

        # Should preserve start and end
        assert result.startswith("START")
        assert result.endswith("END")
        assert "truncated" in result.lower()
