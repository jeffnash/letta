"""Utilities for identifying and handling memory context messages during compaction.

Memory-as-message mode sends memory blocks as a separate message (developer or user role)
instead of embedding them in the system prompt. This module provides helpers to identify
these messages during compaction so they can be handled specially:
- Memory messages should not be included in summarization
- After compaction, a single canonical memory message should be preserved
"""

import re
from typing import List, Optional

from letta.schemas.message import Message

# Constant for tagging memory messages via the `name` field
MEMORY_MESSAGE_NAME = "__letta_memory__"

# Regex patterns for content-based fallback detection
# These are used when messages don't have the name tag (e.g., pre-migration messages)
MEMORY_BLOCKS_PATTERN = re.compile(r"<memory_blocks>")
MEMORY_UPDATE_PATTERN = re.compile(r"<memory_update\s+")
MEMORY_METADATA_PATTERN = re.compile(r"<memory_metadata>")


def is_memory_message(msg: Message) -> bool:
    """Check if a message is a memory context message or memory update message.
    
    Detection strategy:
    1. Primary: Check if msg.name == MEMORY_MESSAGE_NAME (fast, reliable)
    2. Fallback: Content-based detection for pre-migration messages
    
    Args:
        msg: The message to check
        
    Returns:
        True if this is a memory-related message, False otherwise
    """
    # Primary detection: check name field
    if msg.name == MEMORY_MESSAGE_NAME:
        return True
    
    # Fallback: content-based detection for legacy messages
    content_text = _extract_text_content(msg)
    if not content_text:
        return False
    
    # Strip optional <system-reminder> wrapper
    content_text = _strip_system_reminder_wrapper(content_text)
    
    # Check for memory content patterns
    if MEMORY_BLOCKS_PATTERN.search(content_text):
        return True
    if MEMORY_UPDATE_PATTERN.search(content_text):
        return True
    if MEMORY_METADATA_PATTERN.search(content_text):
        return True
    
    return False


def filter_memory_messages(messages: List[Message]) -> tuple[List[Message], List[Message]]:
    """Separate memory messages from conversation messages.
    
    Args:
        messages: List of messages to filter
        
    Returns:
        Tuple of (conversation_messages, memory_messages)
        - conversation_messages: Messages that are NOT memory-related
        - memory_messages: Messages that ARE memory-related (full context or deltas)
    """
    conversation_messages = []
    memory_messages = []
    
    for msg in messages:
        if is_memory_message(msg):
            memory_messages.append(msg)
        else:
            conversation_messages.append(msg)
    
    return conversation_messages, memory_messages


def _extract_text_content(msg: Message) -> Optional[str]:
    """Extract text content from a message for pattern matching.
    
    Args:
        msg: The message to extract content from
        
    Returns:
        The text content as a string, or None if no text content
    """
    if not msg.content:
        return None
    
    # Handle list of content items (TextContent, etc.)
    if isinstance(msg.content, list):
        text_parts = []
        for item in msg.content:
            if isinstance(item, dict):
                # Content item as dict
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                # TextContent object
                text_parts.append(item.text)
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    
    # Handle string content directly
    if isinstance(msg.content, str):
        return msg.content
    
    return None


def _strip_system_reminder_wrapper(content: str) -> str:
    """Strip <system-reminder> wrapper if present.
    
    Args:
        content: The content string, possibly wrapped
        
    Returns:
        The content with the wrapper stripped
    """
    # Pattern to match <system-reminder>...</system-reminder>
    pattern = re.compile(r"<system-reminder>\s*(.*?)\s*</system-reminder>", re.DOTALL)
    match = pattern.search(content)
    if match:
        return match.group(1)
    return content
