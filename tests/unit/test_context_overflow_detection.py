"""
Unit tests for context window overflow error message detection.

This test suite verifies that `is_context_window_overflow_message()` correctly
identifies context overflow errors from all supported LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-5, o1, o3, Codex)
- Anthropic (Claude Sonnet, Opus, Haiku)
- Google Gemini/Vertex AI
- GitHub Copilot
- xAI Grok (Grok-1, Grok-2, Grok-3, Grok-4)
- Zhipu AI / GLM (GLM-4, GLM-4.6)
- Alibaba Qwen / DashScope
- DeepSeek (DeepSeek-V3, DeepSeek-R1)
- Mistral AI
- Moonshot AI / Kimi (Kimi K2)
- MiniMax (MiniMax-M2)
- Proxy services (CLIProxyAPI)

Error patterns are derived from official API documentation, SDK implementations,
and community-reported error messages.
"""

import pytest

from letta.llm_api.error_utils import is_context_window_overflow_message


class TestOpenAIPatterns:
    """Test OpenAI context overflow error patterns."""

    def test_standard_context_length_message(self):
        """Standard OpenAI context length exceeded message."""
        msg = "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens (8000 in the messages, 1000 in the completion). Please reduce the length of the messages or completion."
        assert is_context_window_overflow_message(msg) is True

    def test_context_length_exceeded_code(self):
        """OpenAI error code in message."""
        msg = '{"error": {"code": "context_length_exceeded", "message": "..."}}'
        assert is_context_window_overflow_message(msg) is True

    def test_exceeds_context_window(self):
        """OpenAI streaming error message."""
        msg = "Your input exceeds the context window of this model. Please adjust your input and try again."
        assert is_context_window_overflow_message(msg) is True

    def test_reduce_length_guidance(self):
        """OpenAI guidance message."""
        msg = "Please reduce the length of the messages or completion."
        assert is_context_window_overflow_message(msg) is True

    def test_maximum_prompt_length(self):
        """OpenAI/xAI prompt length message."""
        msg = "This model's maximum prompt length is 131072 but the request contains 136973 tokens."
        assert is_context_window_overflow_message(msg) is True

    def test_string_above_max_length(self):
        """OpenAI schema overflow error."""
        msg = "Invalid 'tools[0].function.description': string too long. Expected a string with maximum length 1048576, but got a string with length 2778531 instead."
        assert is_context_window_overflow_message(msg) is True


class TestAnthropicPatterns:
    """Test Anthropic Claude context overflow error patterns."""

    def test_prompt_too_long_standard(self):
        """Standard Anthropic prompt too long message."""
        msg = "prompt is too long: 201266 tokens > 200000 maximum"
        assert is_context_window_overflow_message(msg) is True

    def test_prompt_too_long_variant(self):
        """Anthropic prompt too long with different numbers."""
        msg = "prompt is too long: 103078 tokens > 102398 maximum"
        assert is_context_window_overflow_message(msg) is True

    def test_exceed_context_limit_combined(self):
        """Anthropic combined input + max_tokens error."""
        msg = "input length and `max_tokens` exceed context limit: 198157 + 21333 > 200000, decrease input length or max_tokens and try again"
        assert is_context_window_overflow_message(msg) is True

    def test_input_too_long(self):
        """Anthropic input too long message."""
        msg = "Input is too long for requested model."
        assert is_context_window_overflow_message(msg) is True

    def test_too_many_total_text_bytes(self):
        """Anthropic byte limit error."""
        msg = "too many total text bytes"
        assert is_context_window_overflow_message(msg) is True

    def test_request_too_large_code(self):
        """Anthropic request_too_large error code."""
        msg = '{"error": {"type": "request_too_large"}}'
        assert is_context_window_overflow_message(msg) is True


class TestGoogleGeminiPatterns:
    """Test Google Gemini/Vertex AI context overflow error patterns."""

    def test_input_token_count_exceeds(self):
        """Gemini input token count exceeds message."""
        msg = "The input token count (35868) exceeds the maximum number of tokens allowed (32768)."
        assert is_context_window_overflow_message(msg) is True

    def test_request_contains_tokens_exceeds(self):
        """Vertex AI request contains tokens message."""
        msg = "Request contains 35000 tokens, which exceeds the limit of 32768."
        assert is_context_window_overflow_message(msg) is True

    def test_input_context_too_long(self):
        """Gemini input context too long message."""
        msg = "Your input context is too long. Reduce your input context or temporarily switch to another model."
        assert is_context_window_overflow_message(msg) is True

    def test_context_too_long_unexpected_error(self):
        """Gemini 500 error with context too long."""
        msg = "An unexpected error occurred on Google's side. Your input context is too long."
        assert is_context_window_overflow_message(msg) is True

    def test_request_payload_size_exceeds(self):
        """Gemini payload size error."""
        msg = "Request payload size exceeds the limit"
        assert is_context_window_overflow_message(msg) is True

    def test_context_window_exceeded_code(self):
        """Gemini CONTEXT_WINDOW_EXCEEDED error code."""
        msg = "CONTEXT_WINDOW_EXCEEDED"
        assert is_context_window_overflow_message(msg) is True


class TestGitHubCopilotPatterns:
    """Test GitHub Copilot context overflow error patterns."""

    def test_prompt_token_count_exceeds(self):
        """Copilot prompt token count exceeds message."""
        msg = "prompt token count of 51808 exceeds the limit of 12288"
        assert is_context_window_overflow_message(msg) is True

    def test_model_max_prompt_tokens_exceeded(self):
        """Copilot-specific error code."""
        msg = '{"error": {"code": "model_max_prompt_tokens_exceeded"}}'
        assert is_context_window_overflow_message(msg) is True

    def test_copilot_openai_compatible(self):
        """Copilot using OpenAI-compatible error."""
        msg = "This model's maximum context length is 8192 tokens. However, you requested 8221 tokens (4221 in the messages, 4000 in the completion)."
        assert is_context_window_overflow_message(msg) is True


class TestXAIGrokPatterns:
    """Test xAI Grok context overflow error patterns."""

    def test_maximum_prompt_length(self):
        """Grok maximum prompt length message."""
        msg = "This model's maximum prompt length is 131072 but the request contains 136973 tokens."
        assert is_context_window_overflow_message(msg) is True


class TestZhipuGLMPatterns:
    """Test Zhipu AI / GLM context overflow error patterns."""

    def test_prompt_too_long(self):
        """GLM prompt too long message."""
        msg = "Prompt too long"
        assert is_context_window_overflow_message(msg) is True

    def test_input_length_too_long(self):
        """GLM input length too long message."""
        msg = "Input length is too long"
        assert is_context_window_overflow_message(msg) is True

    def test_error_code_1261(self):
        """GLM error code 1261."""
        msg = '{"code": "1261", "msg": "Prompt too long", "success": false}'
        assert is_context_window_overflow_message(msg) is True

    def test_error_code_1261_no_spaces(self):
        """GLM error code 1261 without spaces."""
        msg = '{"code":"1261","msg":"Prompt too long"}'
        assert is_context_window_overflow_message(msg) is True

    def test_glm_openai_compatible(self):
        """GLM using OpenAI-compatible format."""
        msg = "This model's maximum context length is 128000 tokens. However, your messages resulted in 504999 tokens. Please reduce the length of the messages."
        assert is_context_window_overflow_message(msg) is True


class TestQwenDashScopePatterns:
    """Test Alibaba Qwen / DashScope context overflow error patterns."""

    def test_range_of_input_length(self):
        """Qwen range of input length message."""
        msg = "Range of input length should be [1, 8192]"
        assert is_context_window_overflow_message(msg) is True

    def test_range_of_max_tokens(self):
        """Qwen range of max_tokens message."""
        msg = "Range of max_tokens should be [1, 32768]"
        assert is_context_window_overflow_message(msg) is True

    def test_invalid_parameter_with_length(self):
        """Qwen InvalidParameter with length."""
        msg = '{"code": "InvalidParameter", "message": "length validation failed"}'
        assert is_context_window_overflow_message(msg) is True


class TestDeepSeekPatterns:
    """Test DeepSeek context overflow error patterns."""

    def test_deepseek_openai_compatible(self):
        """DeepSeek using OpenAI-compatible format."""
        msg = "This model's maximum context length is 65536 tokens. However, you requested 73149 tokens (73149 in the messages, 0 in the completion)."
        assert is_context_window_overflow_message(msg) is True

    def test_deepseek_large_context(self):
        """DeepSeek with larger context model."""
        msg = "This model's maximum context length is 131072 tokens. However, you requested 134142 tokens (70142 in the messages, 64000 in the completion)."
        assert is_context_window_overflow_message(msg) is True


class TestMistralPatterns:
    """Test Mistral AI context overflow error patterns."""

    def test_prompt_contains_too_large(self):
        """Mistral prompt contains too large message."""
        msg = "Prompt contains 65673 tokens, too large for model with 32768 maximum context length"
        assert is_context_window_overflow_message(msg) is True

    def test_exceeds_model_max_context(self):
        """Mistral exceeds model's maximum context length."""
        msg = "The number of tokens in the prompt exceeds the model's maximum context length of 32768. Please use a shorter prompt."
        assert is_context_window_overflow_message(msg) is True

    def test_model_cap_exceeded_code(self):
        """Mistral model_cap_exceeded error code."""
        msg = '{"code": "model_cap_exceeded"}'
        assert is_context_window_overflow_message(msg) is True

    def test_error_code_3051(self):
        """Mistral numeric error code 3051."""
        msg = '{"code": 3051, "message": "Context length exceeded"}'
        assert is_context_window_overflow_message(msg) is True


class TestMoonshotKimiPatterns:
    """Test Moonshot AI / Kimi context overflow error patterns."""

    def test_input_token_length_too_long(self):
        """Kimi input token length too long message."""
        msg = "Input token length too long"
        assert is_context_window_overflow_message(msg) is True

    def test_exceeded_model_token_limit(self):
        """Kimi exceeded model token limit message."""
        msg = "Your request exceeded model token limit"
        assert is_context_window_overflow_message(msg) is True

    def test_length_of_tokens_too_long(self):
        """Kimi length of tokens too long message."""
        msg = "The length of tokens in the request is too long. Do not exceed the model's maximum token limit."
        assert is_context_window_overflow_message(msg) is True

    def test_messages_too_long(self):
        """Kimi $.messages too long message."""
        msg = "$.messages' is too long. Maximum length is 2048, but got 4970 items."
        assert is_context_window_overflow_message(msg) is True

    def test_conversation_length_exceeded(self):
        """Kimi conversation length exceeded message."""
        msg = "Conversation length exceeded. Please start a new session."
        assert is_context_window_overflow_message(msg) is True

    def test_reached_conversation_limit(self):
        """Kimi reached conversation limit message."""
        msg = "The current model has reached its conversation limit. Please switch to another model to continue."
        assert is_context_window_overflow_message(msg) is True


class TestMiniMaxPatterns:
    """Test MiniMax context overflow error patterns."""

    def test_status_code_1039(self):
        """MiniMax status_code 1039."""
        msg = '{"base_resp": {"status_code": 1039, "status_msg": "token limit"}}'
        assert is_context_window_overflow_message(msg) is True

    def test_status_code_1039_no_spaces(self):
        """MiniMax status_code 1039 without spaces."""
        msg = '{"base_resp":{"status_code":1039,"status_msg":"token limit"}}'
        assert is_context_window_overflow_message(msg) is True

    def test_status_msg_token_limit(self):
        """MiniMax status_msg token limit."""
        msg = '"status_msg": "token limit"'
        assert is_context_window_overflow_message(msg) is True

    def test_context_window_exceeds_limit(self):
        """MiniMax context window exceeds limit."""
        msg = "invalid params, context window exceeds limit"
        assert is_context_window_overflow_message(msg) is True

    def test_error_code_2013(self):
        """MiniMax error code 2013."""
        msg = '{"error": {"code": 2013, "message": "invalid params, context window exceeds limit"}}'
        assert is_context_window_overflow_message(msg) is True


class TestChinesePatterns:
    """Test Chinese language error message patterns."""

    def test_input_length_exceeds(self):
        """Chinese: input length exceeds limit."""
        msg = "输入长度超过限制"
        assert is_context_window_overflow_message(msg) is True

    def test_context_length(self):
        """Chinese: context length."""
        msg = "上下文长度超出"
        assert is_context_window_overflow_message(msg) is True

    def test_token_count_exceeded(self):
        """Chinese: token count exceeded."""
        msg = "token数量超过限制"
        assert is_context_window_overflow_message(msg) is True

    def test_exceeds_context(self):
        """Chinese: exceeds context."""
        msg = "超出上下文限制"
        assert is_context_window_overflow_message(msg) is True

    def test_length_limit(self):
        """Chinese: length limit."""
        msg = "长度限制"
        assert is_context_window_overflow_message(msg) is True

    def test_exceeds_maximum_length(self):
        """Chinese: exceeds maximum length."""
        msg = "超过最大长度"
        assert is_context_window_overflow_message(msg) is True

    def test_request_too_large(self):
        """Chinese: request too large."""
        msg = "请求过大"
        assert is_context_window_overflow_message(msg) is True

    def test_content_too_long(self):
        """Chinese: content too long."""
        msg = "内容过长"
        assert is_context_window_overflow_message(msg) is True


class TestCLIProxyAPIPatterns:
    """Test CLIProxyAPI wrapped error patterns."""

    def test_wrapped_anthropic_error(self):
        """CLIProxyAPI wrapped Anthropic error (the original issue)."""
        msg = 'INVALID_ARGUMENT: Bad request to OpenAI: Error code: 400 - {\'error\': {\'code\': 400, \'message\': \'{"type":"error","error":{"type":"invalid_request_error","message":"Prompt is too long"}}\', \'status\': \'FAILED_PRECONDITION\'}}'
        assert is_context_window_overflow_message(msg) is True

    def test_nested_json_prompt_too_long(self):
        """Nested JSON with prompt too long."""
        msg = '{"type":"error","error":{"type":"invalid_request_error","message":"Prompt is too long"}}'
        assert is_context_window_overflow_message(msg) is True


class TestGenericPatterns:
    """Test generic context overflow patterns."""

    def test_request_too_large(self):
        """Generic request too large."""
        msg = "request too large"
        assert is_context_window_overflow_message(msg) is True

    def test_input_too_long(self):
        """Generic input too long."""
        msg = "input too long"
        assert is_context_window_overflow_message(msg) is True

    def test_content_too_large(self):
        """Generic content too large."""
        msg = "content too large"
        assert is_context_window_overflow_message(msg) is True

    def test_payload_too_large(self):
        """Generic payload too large."""
        msg = "payload too large"
        assert is_context_window_overflow_message(msg) is True

    def test_too_many_tokens(self):
        """Generic too many tokens."""
        msg = "too many tokens"
        assert is_context_window_overflow_message(msg) is True

    def test_token_limit_exceeded(self):
        """Generic token limit exceeded."""
        msg = "token limit exceeded"
        assert is_context_window_overflow_message(msg) is True

    def test_context_window_exceeded(self):
        """Generic context window exceeded."""
        msg = "context window exceeded"
        assert is_context_window_overflow_message(msg) is True

    def test_length_exceeded(self):
        """Generic length exceeded."""
        msg = "length exceeded"
        assert is_context_window_overflow_message(msg) is True


class TestNegativeCases:
    """Test that non-context-overflow errors are NOT matched."""

    def test_connection_timeout(self):
        """Connection timeout should not match."""
        msg = "Connection timeout"
        assert is_context_window_overflow_message(msg) is False

    def test_rate_limit_exceeded(self):
        """Rate limit exceeded should not match."""
        msg = "Rate limit exceeded"
        assert is_context_window_overflow_message(msg) is False

    def test_invalid_api_key(self):
        """Invalid API key should not match."""
        msg = "Invalid API key"
        assert is_context_window_overflow_message(msg) is False

    def test_model_not_found(self):
        """Model not found should not match."""
        msg = "Model not found"
        assert is_context_window_overflow_message(msg) is False

    def test_authentication_failed(self):
        """Authentication failed should not match."""
        msg = "Authentication failed"
        assert is_context_window_overflow_message(msg) is False

    def test_permission_denied(self):
        """Permission denied should not match."""
        msg = "Permission denied"
        assert is_context_window_overflow_message(msg) is False

    def test_internal_server_error(self):
        """Internal server error should not match."""
        msg = "Internal server error"
        assert is_context_window_overflow_message(msg) is False

    def test_bad_gateway(self):
        """Bad gateway should not match."""
        msg = "Bad gateway"
        assert is_context_window_overflow_message(msg) is False

    def test_service_unavailable(self):
        """Service unavailable should not match."""
        msg = "Service unavailable"
        assert is_context_window_overflow_message(msg) is False

    def test_quota_exceeded(self):
        """Quota exceeded (billing) should not match."""
        msg = "Quota exceeded for this billing period"
        assert is_context_window_overflow_message(msg) is False

    def test_empty_string(self):
        """Empty string should not match."""
        msg = ""
        assert is_context_window_overflow_message(msg) is False

    def test_random_text(self):
        """Random text should not match."""
        msg = "The quick brown fox jumps over the lazy dog"
        assert is_context_window_overflow_message(msg) is False
