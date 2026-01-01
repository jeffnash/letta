"""
Keep these utilities free of heavy imports to avoid circular dependencies between
LLM clients (provider-specific) and streaming interfaces.
"""


def is_context_window_overflow_message(msg: str) -> bool:
    """Best-effort detection for context window overflow errors.

    Different providers (and even different API surfaces within the same provider)
    may phrase context-window errors differently. We centralize the heuristic so
    all layers (clients, streaming interfaces, agent loops) behave consistently.

    This function must handle errors from:
    - OpenAI direct API (GPT-4, GPT-4o, GPT-5, o1, o3, Codex, etc.)
    - Anthropic direct API (Claude Sonnet, Opus, Haiku)
    - Google Gemini/Vertex API
    - GitHub Copilot API
    - xAI Grok API (Grok-1, Grok-2, Grok-3, Grok-4)
    - ZAI/GLM API (Zhipu AI, ChatGLM)
    - Qwen/DashScope API (Alibaba)
    - DeepSeek API (DeepSeek-V3, DeepSeek-R1)
    - Mistral AI API
    - Moonshot/Kimi API (Kimi K2)
    - MiniMax API (MiniMax-M2)
    - Proxy services (e.g., CLIProxyAPI) that forward errors from any of the above

    The check is case-insensitive to catch variations in error message casing.

    Error patterns are derived from:
    - Official API documentation
    - SDK error handling code
    - Community-reported error messages
    - GitHub issues and Stack Overflow

    References:
    - OpenAI: https://platform.openai.com/docs/guides/error-codes
    - Anthropic: https://docs.anthropic.com/en/api/errors
    - Google: https://ai.google.dev/gemini-api/docs/troubleshooting
    - xAI: https://docs.x.ai/docs
    - DeepSeek: https://api-docs.deepseek.com
    """
    msg_lower = msg.lower()

    # ==========================================================================
    # OpenAI patterns (GPT-4, GPT-4o, GPT-5, o1, o3, Codex, etc.)
    # ==========================================================================
    # Error code: context_length_exceeded
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens"
    # - "This model's maximum context length is 128000 tokens. However, your messages resulted in 249114 tokens"
    # - "This model's maximum prompt length is 131072 but the request contains 136973 tokens"
    # - "Your input exceeds the context window of this model"
    # - "Invalid 'tools[0].function.description': string too long" (schema overflow)
    # - "Please reduce the length of the messages or completion"
    if (
        "maximum context length" in msg_lower
        or "context_length_exceeded" in msg_lower
        or "exceeds the context window" in msg_lower
        or "input tokens exceed the configured limit" in msg_lower
        or "reduce the length of the messages" in msg_lower
        or "maximum prompt length" in msg_lower
        or "string_above_max_length" in msg_lower
        or "string too long" in msg_lower
        or "max_output_tokens_reached" in msg_lower
    ):
        return True

    # ==========================================================================
    # Anthropic patterns (Claude Sonnet, Opus, Haiku)
    # ==========================================================================
    # Error type: invalid_request_error
    # HTTP Status: 400 Bad Request, 413 Request Too Large
    #
    # Examples:
    # - "prompt is too long: 201266 tokens > 200000 maximum"
    # - "prompt is too long: 103078 tokens > 102398 maximum"
    # - "input length and `max_tokens` exceed context limit: 198157 + 21333 > 200000"
    # - "Input is too long for requested model"
    # - "too many total text bytes"
    # - "This request would exceed your organization's rate limit of N input tokens per minute"
    if (
        "prompt is too long" in msg_lower
        or "exceed context limit" in msg_lower
        or "exceeds context limit" in msg_lower
        or "too many total text bytes" in msg_lower
        or "total text bytes" in msg_lower
        or "input is too long" in msg_lower
        or "request_too_large" in msg_lower
    ):
        return True

    # ==========================================================================
    # Google Gemini/Vertex AI patterns
    # ==========================================================================
    # Error codes: INVALID_ARGUMENT, INTERNAL, DEADLINE_EXCEEDED, RESOURCE_EXHAUSTED
    # HTTP Status: 400, 500, 504, 429
    #
    # Examples:
    # - "The input token count (35868) exceeds the maximum number of tokens allowed (32768)"
    # - "Request contains 35000 tokens, which exceeds the limit of 32768"
    # - "Your input context is too long"
    # - "An unexpected error occurred on Google's side. Your input context is too long"
    # - "Your prompt (or context) is too large to be processed in time"
    # - "Request payload size exceeds the limit"
    # - "CONTEXT_WINDOW_EXCEEDED"
    if (
        "input token count" in msg_lower and "exceeds" in msg_lower
        or "exceeds the maximum number of tokens" in msg_lower
        or "input context is too long" in msg_lower
        or "context is too long" in msg_lower
        or "request payload size exceeds" in msg_lower
        or "context_window_exceeded" in msg_lower
        or "input_token_limit" in msg_lower
        or "output_token_limit" in msg_lower
        or ("request contains" in msg_lower and "tokens" in msg_lower and "exceeds the limit" in msg_lower)
    ):
        return True

    # ==========================================================================
    # GitHub Copilot patterns
    # ==========================================================================
    # Error code: model_max_prompt_tokens_exceeded (Copilot-specific)
    # Also uses OpenAI-compatible: context_length_exceeded
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "prompt token count of 51808 exceeds the limit of 12288"
    # - "This model's maximum context length is 8192 tokens. However, you requested 8221 tokens"
    if (
        "model_max_prompt_tokens_exceeded" in msg_lower
        or "prompt token count" in msg_lower and "exceeds the limit" in msg_lower
    ):
        return True

    # ==========================================================================
    # xAI Grok patterns (Grok-1, Grok-2, Grok-3, Grok-4)
    # ==========================================================================
    # Error type: invalid_request_error
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "This model's maximum prompt length is 131072 but the request contains 136973 tokens"
    if "maximum prompt length" in msg_lower and "request contains" in msg_lower:
        return True

    # ==========================================================================
    # Zhipu AI / GLM patterns (GLM-4, GLM-4.6)
    # ==========================================================================
    # Internal error code: 1261 ("Prompt too long")
    # Note: Z.AI may silently truncate instead of erroring
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "Prompt too long"
    # - "Input length is too long"
    # - "This model's maximum context length is 128000 tokens. However, your messages resulted in 504999 tokens"
    # - Error code 1261 in response
    if (
        "prompt too long" in msg_lower
        or "input length is too long" in msg_lower
        or '"code": "1261"' in msg_lower
        or '"code":"1261"' in msg_lower
        or "'code': '1261'" in msg_lower
    ):
        return True

    # ==========================================================================
    # Alibaba Qwen / DashScope patterns
    # ==========================================================================
    # Error code: InvalidParameter
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "Range of input length should be [1, 8192]"
    # - "Range of max_tokens should be [1, X]"
    if (
        "range of input length should be" in msg_lower
        or "range of max_tokens should be" in msg_lower
        or ("invalidparameter" in msg_lower and ("length" in msg_lower or "token" in msg_lower))
    ):
        return True

    # ==========================================================================
    # DeepSeek patterns (DeepSeek-V3, DeepSeek-R1)
    # ==========================================================================
    # Error type: invalid_request_error (OpenAI-compatible)
    # HTTP Status: 400 Bad Request, 422
    #
    # Examples:
    # - "This model's maximum context length is 65536 tokens. However, you requested 73149 tokens"
    # - "This model's maximum context length is 131072 tokens. However, you requested 134142 tokens"
    # Note: Uses same pattern as OpenAI, already covered above

    # ==========================================================================
    # Mistral AI patterns
    # ==========================================================================
    # Error code: model_cap_exceeded, or numeric code 3051
    # HTTP Status: 400 Bad Request, 413
    #
    # Examples:
    # - "Prompt contains 65673 tokens, too large for model with 32768 maximum context length"
    # - "The number of tokens in the prompt exceeds the model's maximum context length of 32768"
    # - "Context length exceeded"
    if (
        "model_cap_exceeded" in msg_lower
        or '"code": 3051' in msg_lower
        or '"code":3051' in msg_lower
        or ("prompt contains" in msg_lower and "tokens" in msg_lower and "too large for model" in msg_lower)
        or "exceeds the model's maximum context length" in msg_lower
    ):
        return True

    # ==========================================================================
    # Moonshot AI / Kimi patterns (Kimi K2)
    # ==========================================================================
    # Error code: 40001, invalid_request_error
    # HTTP Status: 400 Bad Request, 413 Content Too Large
    #
    # Examples:
    # - "Input token length too long"
    # - "Your request exceeded model token limit"
    # - "The length of tokens in the request is too long"
    # - "$.messages' is too long. Maximum length is 2048"
    # - "Conversation length exceeded. Please start a new session"
    # - "The current model has reached its conversation limit"
    if (
        "input token length too long" in msg_lower
        or "exceeded model token limit" in msg_lower
        or "length of tokens in the request is too long" in msg_lower
        or "conversation length exceeded" in msg_lower
        or "reached its conversation limit" in msg_lower
        or ("$.messages" in msg_lower and "too long" in msg_lower)
    ):
        return True

    # ==========================================================================
    # MiniMax patterns (MiniMax-M2, MiniMax-Text-01)
    # ==========================================================================
    # Error code: 1039 ("token limit"), 2013 ("invalid params, context window exceeds limit")
    # HTTP Status: 400 Bad Request
    #
    # Examples:
    # - "token limit" (status_msg field)
    # - "invalid params, context window exceeds limit"
    # - Error code 1039 or 2013 in response
    if (
        '"status_code": 1039' in msg_lower
        or '"status_code":1039' in msg_lower
        or '"status_msg": "token limit"' in msg_lower
        or '"status_msg":"token limit"' in msg_lower
        or "context window exceeds limit" in msg_lower
        or '"code": 2013' in msg_lower
        or '"code":2013' in msg_lower
    ):
        return True

    # ==========================================================================
    # Chinese language error messages
    # ==========================================================================
    # Various Chinese LLM providers may return errors in Chinese
    #
    # Examples:
    # - "输入长度超过限制" (input length exceeds limit)
    # - "上下文长度超出" (context length exceeded)
    # - "token数量超过" (token count exceeded)
    # - "超出上下文限制" (exceeds context limit)
    # - "长度限制" (length limit)
    # - "超过最大长度" (exceeds maximum length)
    # - "请求过大" (request too large)
    if (
        "输入长度超过" in msg  # Input length exceeds (don't lowercase Chinese)
        or "上下文长度" in msg  # Context length
        or "token数量超过" in msg  # Token count exceeded
        or "超出上下文" in msg  # Exceeds context
        or "长度限制" in msg  # Length limit
        or "超过最大长度" in msg  # Exceeds maximum length
        or "请求过大" in msg  # Request too large
        or "内容过长" in msg  # Content too long
        or "超出限制" in msg  # Exceeds limit
    ):
        return True

    # ==========================================================================
    # Generic patterns that indicate context overflow
    # ==========================================================================
    # These catch edge cases, future providers, and proxy-forwarded errors
    if (
        # Size/length patterns
        "request too large" in msg_lower
        or "input too long" in msg_lower
        or "content too large" in msg_lower
        or "payload too large" in msg_lower
        or "message too long" in msg_lower
        or "text too long" in msg_lower
        # Token patterns
        or "tokens exceed" in msg_lower
        or "too many tokens" in msg_lower
        or "token budget exceeded" in msg_lower
        or "token limit" in msg_lower and ("exceed" in msg_lower or "reached" in msg_lower)
        # Context patterns
        or "context window exceeded" in msg_lower
        or "context overflow" in msg_lower
        or "context limit" in msg_lower and "exceed" in msg_lower
        # Generic exceed patterns
        or "exceeds the limit" in msg_lower
        or "exceeds maximum" in msg_lower
        or "length exceeded" in msg_lower
        or "size limit" in msg_lower and "exceed" in msg_lower
        # Max patterns
        or "max context" in msg_lower and "exceeded" in msg_lower
        or ("max_tokens" in msg_lower and "exceed" in msg_lower)
    ):
        return True

    return False
