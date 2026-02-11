"""
Response Normalizer for OpenAI-Compatible Proxies

This module provides a centralized normalization layer for LLM responses from
various OpenAI-compatible providers. It handles common schema quirks and
missing/null fields that would otherwise cause Pydantic validation failures.

Key failure modes addressed:
- F5: choices[0].index = None (proxy returns null instead of 0)
- F6: object = None (proxy returns null instead of "chat.completion")
- F7: usage = None (proxy doesn't return usage statistics)
- Additional: finish_reason = None, model = None, id = None
"""

import json
import uuid
from typing import Any, Dict, Optional

from letta.log import get_logger
from letta.schemas.openai.chat_completion_response import (
    ChatCompletionResponse,
    Choice,
    Message,
    UsageStatistics,
)

logger = get_logger(__name__)


def normalize_chat_completion_response(
    response_data: Dict[str, Any],
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Normalize a raw OpenAI-compatible response dict before Pydantic validation.
    
    This function mutates the response_data dict in-place and returns it.
    All normalizations are logged with run_id/step_id for correlation.
    
    Args:
        response_data: Raw response dict from LLM provider
        run_id: Optional run ID for log correlation
        step_id: Optional step ID for log correlation
        provider: Optional provider name for logging
        model: Optional model name for logging
        
    Returns:
        Normalized response dict ready for Pydantic validation
    """
    log_context = _build_log_context(run_id, step_id, provider, model)
    normalizations_applied = []
    
    # Normalize top-level fields
    if response_data.get("object") is None:
        response_data["object"] = "chat.completion"
        normalizations_applied.append("object=null->chat.completion")
    
    if response_data.get("id") is None:
        response_data["id"] = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        normalizations_applied.append("id=null->generated")
    
    if response_data.get("model") is None and model:
        response_data["model"] = model
        normalizations_applied.append(f"model=null->{model}")
    
    if response_data.get("created") is None:
        import time
        response_data["created"] = int(time.time())
        normalizations_applied.append("created=null->now")
    
    # Normalize usage statistics
    usage = response_data.get("usage")
    if usage is None:
        response_data["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        normalizations_applied.append("usage=null->zeros")
    elif isinstance(usage, dict):
        # Ensure required fields exist
        if usage.get("prompt_tokens") is None:
            usage["prompt_tokens"] = 0
            normalizations_applied.append("usage.prompt_tokens=null->0")
        if usage.get("completion_tokens") is None:
            usage["completion_tokens"] = 0
            normalizations_applied.append("usage.completion_tokens=null->0")
        if usage.get("total_tokens") is None:
            usage["total_tokens"] = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
            normalizations_applied.append("usage.total_tokens=null->computed")
    
    # Normalize choices array
    choices = response_data.get("choices")
    if choices is None:
        # This is a serious issue - create a minimal valid structure
        response_data["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
            },
            "finish_reason": "stop",
        }]
        normalizations_applied.append("choices=null->empty_assistant")
        logger.warning(
            "Response had no choices, created empty assistant message",
            extra={"run_id": run_id, "step_id": step_id, "provider": provider},
        )
    elif isinstance(choices, list):
        for i, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            
            # Normalize index
            if choice.get("index") is None:
                choice["index"] = i
                normalizations_applied.append(f"choices[{i}].index=null->{i}")
            
            # Normalize finish_reason
            if choice.get("finish_reason") is None:
                choice["finish_reason"] = "stop"
                normalizations_applied.append(f"choices[{i}].finish_reason=null->stop")
            
            # Normalize message
            message = choice.get("message")
            if message is None:
                choice["message"] = {
                    "role": "assistant",
                    "content": None,
                }
                normalizations_applied.append(f"choices[{i}].message=null->empty")
            elif isinstance(message, dict):
                # Ensure role exists
                if message.get("role") is None:
                    message["role"] = "assistant"
                    normalizations_applied.append(f"choices[{i}].message.role=null->assistant")
                
                # Normalize tool_calls if present
                tool_calls = message.get("tool_calls")
                if tool_calls is not None and isinstance(tool_calls, list):
                    for j, tc in enumerate(tool_calls):
                        if not isinstance(tc, dict):
                            continue
                        
                        # Ensure tool call has an ID
                        if tc.get("id") is None:
                            tc["id"] = f"call_{uuid.uuid4().hex[:8]}"
                            normalizations_applied.append(f"choices[{i}].tool_calls[{j}].id=null->generated")
                        
                        # Ensure type is set
                        if tc.get("type") is None:
                            tc["type"] = "function"
                            normalizations_applied.append(f"choices[{i}].tool_calls[{j}].type=null->function")
                        
                        # Normalize function object
                        func = tc.get("function")
                        if func is None:
                            tc["function"] = {"name": "unknown", "arguments": "{}"}
                            normalizations_applied.append(f"choices[{i}].tool_calls[{j}].function=null->unknown")
                        elif isinstance(func, dict):
                            if func.get("name") is None:
                                func["name"] = "unknown"
                                normalizations_applied.append(f"choices[{i}].tool_calls[{j}].function.name=null->unknown")
                            if func.get("arguments") is None:
                                func["arguments"] = "{}"
                                normalizations_applied.append(f"choices[{i}].tool_calls[{j}].function.arguments=null->{{}}")
                            else:
                                # Validate that arguments is valid JSON
                                # Truncated LLM responses (e.g. from stream cutoff due to context window overflow)
                                # can produce incomplete JSON that poisons message history if persisted
                                try:
                                    json.loads(func["arguments"])
                                except (json.JSONDecodeError, TypeError):
                                    logger.error(
                                        "Truncated/invalid JSON in tool call arguments, replacing with empty dict. "
                                        "Original (truncated to 200 chars): %s",
                                        str(func["arguments"])[:200],
                                        extra={"run_id": run_id, "step_id": step_id, "provider": provider},
                                    )
                                    func["arguments"] = "{}"
                                    normalizations_applied.append(
                                        f"choices[{i}].tool_calls[{j}].function.arguments=invalid_json->{{}}"
                                    )
    
    # Log normalizations if any were applied
    if normalizations_applied:
        logger.info(
            "Normalized LLM response: %s",
            ", ".join(normalizations_applied[:10]),  # Limit to first 10 for readability
            extra={
                "run_id": run_id,
                "step_id": step_id,
                "provider": provider,
                "model": model,
                "normalization_count": len(normalizations_applied),
            },
        )
        if len(normalizations_applied) > 10:
            logger.debug(
                "Full normalization list: %s",
                normalizations_applied,
                extra={"run_id": run_id, "step_id": step_id},
            )
    
    return response_data


def _build_log_context(
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Build a context string for log messages."""
    parts = []
    if run_id:
        parts.append(f"run={run_id[:8]}...")
    if step_id:
        parts.append(f"step={step_id[:8]}...")
    if provider:
        parts.append(f"provider={provider}")
    if model:
        parts.append(f"model={model}")
    return " ".join(parts) if parts else "no_context"


def validate_and_normalize_response(
    response_data: Dict[str, Any],
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> ChatCompletionResponse:
    """
    Normalize response data and validate it into a ChatCompletionResponse.
    
    This is a convenience function that combines normalization and validation.
    
    Args:
        response_data: Raw response dict from LLM provider
        run_id: Optional run ID for log correlation
        step_id: Optional step ID for log correlation
        provider: Optional provider name for logging
        model: Optional model name for logging
        
    Returns:
        Validated ChatCompletionResponse object
        
    Raises:
        ValidationError: If response cannot be normalized to valid structure
    """
    normalized = normalize_chat_completion_response(
        response_data,
        run_id=run_id,
        step_id=step_id,
        provider=provider,
        model=model,
    )
    return ChatCompletionResponse(**normalized)


def redact_sensitive_content(text: str, max_length: int = 1000) -> str:
    """
    Redact potentially sensitive content from tool outputs for logging.
    
    Patterns redacted:
    - API keys (various formats)
    - Passwords and secrets
    - Bearer tokens
    - AWS credentials
    - Private keys
    
    Args:
        text: Text to redact
        max_length: Maximum length before truncation
        
    Returns:
        Redacted and possibly truncated text
    """
    import re
    
    if not text:
        return text
    
    # Patterns to redact (case-insensitive)
    patterns = [
        # API keys (generic)
        (r'(?i)(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', r'\1=***REDACTED***'),
        # Bearer tokens (Authorization: Bearer xxx or bearer=xxx)
        (r'(?i)(bearer)\s+([a-zA-Z0-9_\-\.]{20,})', r'\1 ***REDACTED***'),
        (r'(?i)(authorization)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', r'\1=***REDACTED***'),
        # Passwords
        (r'(?i)(password|passwd|pwd|secret)["\']?\s*[:=]\s*["\']?([^\s"\']{8,})["\']?', r'\1=***REDACTED***'),
        # AWS access keys
        (r'(?i)(AKIA|ASIA)[A-Z0-9]{16}', '***AWS_KEY_REDACTED***'),
        # Private keys
        (r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----', '***PRIVATE_KEY_REDACTED***'),
        # .env style assignments
        (r'(?m)^([A-Z_]+(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL|AUTH)[A-Z_]*)\s*=\s*(.+)$', r'\1=***REDACTED***'),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length] + f"... [truncated {len(text) - max_length} chars]"
    
    return result


def truncate_tool_output(
    output: str,
    max_chars: int = 50000,
    preserve_json_structure: bool = True,
) -> str:
    """
    Truncate tool output while preserving useful information.
    
    For JSON outputs, attempts to truncate the 'message' field specifically
    to preserve metadata. For plain text, uses middle truncation to keep
    both beginning and end context.
    
    Args:
        output: Tool output string
        max_chars: Maximum characters to keep
        preserve_json_structure: If True, try to preserve JSON structure
        
    Returns:
        Truncated output string
    """
    import json
    
    if not output or len(output) <= max_chars:
        return output
    
    original_len = len(output)
    
    if preserve_json_structure:
        try:
            parsed = json.loads(output)
            if isinstance(parsed, dict) and "message" in parsed:
                msg = parsed["message"]
                if isinstance(msg, str) and len(msg) > max_chars // 2:
                    # Truncate the message field specifically
                    truncated_msg = msg[:max_chars // 2] + f"... [truncated {len(msg) - max_chars // 2} chars]"
                    parsed["message"] = truncated_msg
                    return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
    
    # Fall back to middle truncation
    head_chars = max_chars // 3
    tail_chars = max_chars // 3
    middle_note = f"\n\n... [truncated {original_len - head_chars - tail_chars} chars] ...\n\n"
    
    return output[:head_chars] + middle_note + output[-tail_chars:]
