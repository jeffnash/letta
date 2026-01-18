"""
Structured Logging Helpers for Letta Server

This module provides helpers for consistent structured logging with
run_id, step_id, and other correlation fields across the codebase.

Usage:
    from letta.log_helpers import log_with_context, LogContext

    # Create a context for the current run/step
    ctx = LogContext(run_id="run-123", step_id="step-456", agent_id="agent-789")
    
    # Log with context
    log_with_context(logger, "info", "Processing tool call", ctx, tool_name="my_tool")
    
    # Or use the context manager
    with ctx:
        logger.info("This will include context", extra=ctx.to_extra())
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LogContext:
    """
    Structured logging context for run/step correlation.
    
    This class holds the common context fields that should be included
    in all log messages related to a specific agent run/step.
    """
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    agent_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tool_rules_active: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_extra(self) -> Dict[str, Any]:
        """Convert context to a dict suitable for logger.info(..., extra=...)"""
        result = {}
        if self.run_id:
            result["run_id"] = self.run_id
        if self.step_id:
            result["step_id"] = self.step_id
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        if self.tool_rules_active:
            result["tool_rules_active"] = True
        result.update(self.extra)
        return result
    
    def short_ids(self) -> str:
        """Return shortened IDs for inline log messages."""
        parts = []
        if self.run_id:
            parts.append(f"run={self.run_id[:8]}...")
        if self.step_id:
            parts.append(f"step={self.step_id[:8]}...")
        return " ".join(parts) if parts else "no_context"
    
    def with_extra(self, **kwargs) -> "LogContext":
        """Return a new context with additional extra fields."""
        new_extra = {**self.extra, **kwargs}
        return LogContext(
            run_id=self.run_id,
            step_id=self.step_id,
            agent_id=self.agent_id,
            provider=self.provider,
            model=self.model,
            tool_rules_active=self.tool_rules_active,
            extra=new_extra,
        )


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    ctx: Optional[LogContext] = None,
    **extra_fields,
) -> None:
    """
    Log a message with structured context fields.
    
    Args:
        logger: The logger instance to use
        level: Log level ("debug", "info", "warning", "error", "critical")
        message: The log message
        ctx: Optional LogContext with run/step correlation
        **extra_fields: Additional fields to include in the log
    """
    extra = {}
    if ctx:
        extra.update(ctx.to_extra())
    extra.update(extra_fields)
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message, extra=extra if extra else None)


def format_tool_call_log(
    tool_name: str,
    tool_call_id: str,
    status: str,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    output_size: Optional[int] = None,
) -> str:
    """
    Format a standardized tool call log message.
    
    Args:
        tool_name: Name of the tool being called
        tool_call_id: Unique ID of the tool call
        status: Status of the call ("started", "completed", "error", "denied")
        duration_ms: Optional duration in milliseconds
        error: Optional error message
        output_size: Optional size of the output in characters
        
    Returns:
        Formatted log message string
    """
    parts = [f"tool_call={tool_name}", f"id={tool_call_id[:12]}...", f"status={status}"]
    
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.1f}ms")
    if output_size is not None:
        parts.append(f"output_size={output_size}")
    if error:
        # Truncate error to avoid log bloat
        truncated_error = error[:200] + "..." if len(error) > 200 else error
        parts.append(f"error={truncated_error}")
    
    return " | ".join(parts)


def format_llm_request_log(
    provider: str,
    model: str,
    endpoint_type: str,
    request_mode: str,
    tool_count: int = 0,
    tool_rules_active: bool = False,
) -> str:
    """
    Format a standardized LLM request log message.
    
    Args:
        provider: LLM provider name
        model: Model identifier
        endpoint_type: Type of endpoint (chat, responses, etc.)
        request_mode: Request mode (streaming, blocking)
        tool_count: Number of tools available
        tool_rules_active: Whether tool rules are active
        
    Returns:
        Formatted log message string
    """
    parts = [
        f"provider={provider}",
        f"model={model}",
        f"endpoint={endpoint_type}",
        f"mode={request_mode}",
        f"tools={tool_count}",
    ]
    
    if tool_rules_active:
        parts.append("tool_rules=active")
    
    return " | ".join(parts)


def redact_for_logging(text: str, max_length: int = 500) -> str:
    """
    Redact and truncate text for safe logging.
    
    Removes potential secrets and truncates to a reasonable length.
    
    Args:
        text: Text to redact
        max_length: Maximum length before truncation
        
    Returns:
        Redacted and truncated text
    """
    import re
    
    if not text:
        return text
    
    # Simple patterns to redact
    patterns = [
        (r'(?i)(api[_-]?key|apikey|password|secret|token)["\']?\s*[:=]\s*["\']?[^\s"\']+', r'\1=***'),
        (r'(?i)bearer\s+[a-zA-Z0-9_\-\.]+', 'bearer ***'),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    if len(result) > max_length:
        result = result[:max_length] + f"... [{len(text) - max_length} more chars]"
    
    return result


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation_name: str,
    ctx: Optional[LogContext] = None,
    log_start: bool = True,
    log_end: bool = True,
):
    """
    Context manager for logging the start and end of an operation.
    
    Usage:
        with log_operation(logger, "tool_execution", ctx):
            # do work
            pass
    
    Args:
        logger: Logger instance
        operation_name: Name of the operation being performed
        ctx: Optional logging context
        log_start: Whether to log when entering
        log_end: Whether to log when exiting
    """
    import time
    
    extra = ctx.to_extra() if ctx else {}
    start_time = time.time()
    
    if log_start:
        logger.debug(f"Starting {operation_name}", extra=extra)
    
    try:
        yield
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"{operation_name} failed after {duration_ms:.1f}ms: {type(e).__name__}: {str(e)[:200]}",
            extra={**extra, "duration_ms": duration_ms, "error_type": type(e).__name__},
        )
        raise
    else:
        if log_end:
            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"Completed {operation_name} in {duration_ms:.1f}ms",
                extra={**extra, "duration_ms": duration_ms},
            )
