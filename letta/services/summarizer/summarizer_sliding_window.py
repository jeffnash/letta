from typing import List, Optional, Set, Tuple

from letta.helpers.message_helper import convert_message_creates_to_messages
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.agent import AgentState
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageCreate
from letta.schemas.user import User
from letta.services.context_window_calculator.token_counter import create_token_counter
from letta.services.message_manager import MessageManager
from letta.services.summarizer.memory_message_utils import filter_memory_messages
from letta.services.summarizer.summarizer import simple_summary
from letta.services.summarizer.summarizer_config import CompactionSettings
from letta.system import package_summarize_message_no_counts

logger = get_logger(__name__)


# Safety margin for approximate token counting.
# The bytes/4 heuristic underestimates by ~25-35% for JSON-serialized messages
# due to structural overhead (brackets, quotes, colons) each becoming tokens.
APPROX_TOKEN_SAFETY_MARGIN = 1.3


def _build_tool_call_id_to_assistant_index(messages: List[Message]) -> dict:
    """Build a mapping from tool_call_id to the index of the assistant message that made the call.
    
    This is used to ensure we don't evict tool response messages without their corresponding
    assistant message (which contains the tool_use block).
    
    Note: Tool calls without an `id` are ignored since they cannot be paired with tool responses.
    """
    tool_call_id_to_assistant_idx = {}
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.assistant and msg.tool_calls:
            for tc in msg.tool_calls:
                # Only include tool calls that have a valid id
                if tc.id:
                    tool_call_id_to_assistant_idx[tc.id] = i
    return tool_call_id_to_assistant_idx


def _build_assistant_index_to_tool_response_indices(messages: List[Message], tool_call_id_to_assistant_idx: dict) -> dict:
    """Build a mapping from assistant message index to the indices of all tool response messages.
    
    This is used to ensure we don't evict an assistant message (with tool_use) without also
    evicting all its corresponding tool response messages.
    
    Note: Tool response messages without a `tool_call_id` are ignored since they cannot be
    paired with assistant messages.
    """
    assistant_idx_to_tool_indices = {}
    for i, msg in enumerate(messages):
        # Only include tool responses that have a valid tool_call_id
        if msg.role == MessageRole.tool and msg.tool_call_id:
            assistant_idx = tool_call_id_to_assistant_idx.get(msg.tool_call_id)
            if assistant_idx is not None:
                if assistant_idx not in assistant_idx_to_tool_indices:
                    assistant_idx_to_tool_indices[assistant_idx] = []
                assistant_idx_to_tool_indices[assistant_idx].append(i)
    return assistant_idx_to_tool_indices


def _find_earliest_safe_cutoff_for_tool_group(
    candidate_idx: int,
    messages: List[Message],
    tool_call_id_to_assistant_idx: dict,
    assistant_idx_to_tool_indices: dict,
) -> int:
    """Find the earliest safe cutoff index that keeps a complete tool call/response group.
    
    Given a candidate cutoff index, this function checks if cutting at that point would
    split a tool call from its responses. If so, it returns an adjusted cutoff that
    keeps the entire tool group together.
    
    Args:
        candidate_idx: The proposed cutoff index
        messages: The list of messages
        tool_call_id_to_assistant_idx: Mapping from tool_call_id to assistant message index
        assistant_idx_to_tool_indices: Mapping from assistant index to tool response indices
        
    Returns:
        The earliest safe cutoff index that preserves tool pairing (may be <= candidate_idx)
    """
    adjusted_idx = candidate_idx
    
    # Check all messages that would be kept (from candidate_idx onwards)
    # and find any tool responses whose assistant would be evicted
    for kept_idx in range(candidate_idx, len(messages)):
        kept_msg = messages[kept_idx]
        if kept_msg.role == MessageRole.tool and kept_msg.tool_call_id:
            assistant_idx = tool_call_id_to_assistant_idx.get(kept_msg.tool_call_id)
            if assistant_idx is not None and assistant_idx < adjusted_idx:
                # This tool response's assistant would be evicted - we need to include the assistant
                adjusted_idx = assistant_idx
    
    # Now check if the adjusted cutoff point is an assistant with tool calls
    # and ensure ALL its tool responses are also kept
    while adjusted_idx > 0:
        msg_at_cutoff = messages[adjusted_idx]
        if msg_at_cutoff.role == MessageRole.assistant and msg_at_cutoff.tool_calls:
            tool_response_indices = assistant_idx_to_tool_indices.get(adjusted_idx, [])
            if tool_response_indices:
                min_tool_response_idx = min(tool_response_indices)
                if min_tool_response_idx < adjusted_idx:
                    # Some tool responses come BEFORE the assistant - this is unusual but handle it
                    # Move cutoff to include the earliest tool response
                    adjusted_idx = min_tool_response_idx
                    continue
        break
    
    return adjusted_idx


def _find_safe_cutoff_index(
    messages: List[Message],
    target_cutoff_index: int,
    tool_call_id_to_assistant_idx: dict,
    assistant_idx_to_tool_indices: dict,
    valid_cutoff_roles: Set[MessageRole],
    maximum_message_index: Optional[int] = None,
) -> Optional[int]:
    """Find a safe cutoff index with improved algorithm and multiple fallback strategies.

    The cutoff index is the first message to KEEP (i.e., messages[cutoff:] are kept).
    We need to ensure that:
    1. The cutoff preserves proper conversation flow (preferably at assistant or user message)
    2. If we're keeping a tool response message, we must also keep its assistant message
    3. If we're evicting an assistant message with tool calls, we must also evict all its tool responses

    The algorithm uses a multi-strategy approach:
    1. First, try to find a cutoff at a preferred role (assistant first, then user)
    2. Adjust the cutoff backwards if needed to keep complete tool call groups
    3. Fall back to ANY safe position if preferred roles don't work

    Args:
        messages: The list of messages
        target_cutoff_index: The initial target cutoff index (first message to keep)
        tool_call_id_to_assistant_idx: Mapping from tool_call_id to assistant message index
        assistant_idx_to_tool_indices: Mapping from assistant index to tool response indices
        valid_cutoff_roles: Set of valid roles for cutoff point
        maximum_message_index: Optional upper bound for cutoff (to preserve terminal messages like approvals)

    Returns:
        A safe cutoff index, or None if no valid cutoff can be found
    """
    # Apply maximum message index constraint if provided
    if maximum_message_index is not None:
        target_cutoff_index = min(target_cutoff_index, maximum_message_index)

    # Expand valid cutoff roles to include user messages as fallback
    # User messages are natural conversation boundaries and safe cutoff points
    expanded_valid_roles = valid_cutoff_roles | {MessageRole.user}

    # Strategy 1: Try preferred roles (assistant first, then user)
    # Search in priority order: assistant messages first, then user messages
    preferred_roles = [MessageRole.assistant, MessageRole.user]

    for role in preferred_roles:
        if role not in expanded_valid_roles:
            continue

        # Search backwards from target_cutoff_index to find a message of this role
        for candidate_idx in reversed(range(1, target_cutoff_index + 1)):
            if candidate_idx >= len(messages):
                continue

            candidate_msg = messages[candidate_idx]

            # Skip if not the role we're looking for
            if candidate_msg.role != role:
                continue

            # Adjust the cutoff to ensure complete tool groups are preserved
            adjusted_idx = _find_earliest_safe_cutoff_for_tool_group(
                candidate_idx,
                messages,
                tool_call_id_to_assistant_idx,
                assistant_idx_to_tool_indices,
            )

            # Validate that this cutoff doesn't orphan any tool calls or responses
            if _is_cutoff_safe(
                adjusted_idx, messages, tool_call_id_to_assistant_idx, assistant_idx_to_tool_indices
            ):
                # Don't evict everything including the last message (respect maximum_message_index)
                if maximum_message_index is None or adjusted_idx < maximum_message_index:
                    return adjusted_idx

    # Strategy 2: Final fallback - try ANY position that maintains tool safety
    # This handles edge cases where the conversation has unusual structure
    # or where preferred role cutoffs would all break tool pairing constraints
    for candidate_idx in reversed(range(1, target_cutoff_index + 1)):
        adjusted_idx = _find_earliest_safe_cutoff_for_tool_group(
            candidate_idx,
            messages,
            tool_call_id_to_assistant_idx,
            assistant_idx_to_tool_indices,
        )

        if _is_cutoff_safe(
            adjusted_idx, messages, tool_call_id_to_assistant_idx, assistant_idx_to_tool_indices
        ):
            if maximum_message_index is None or adjusted_idx < maximum_message_index:
                return adjusted_idx

    return None


def _is_cutoff_safe(
    cutoff_idx: int,
    messages: List[Message],
    tool_call_id_to_assistant_idx: dict,
    assistant_idx_to_tool_indices: dict,
) -> bool:
    """Check if a cutoff index is safe (doesn't orphan tool calls or responses).
    
    A cutoff is safe if:
    1. No tool response in the kept portion has its assistant evicted
    2. No assistant with tool calls in the kept portion has any tool responses evicted
    
    Args:
        cutoff_idx: The proposed cutoff index (first message to keep)
        messages: The list of messages
        tool_call_id_to_assistant_idx: Mapping from tool_call_id to assistant message index
        assistant_idx_to_tool_indices: Mapping from assistant index to tool response indices
        
    Returns:
        True if the cutoff is safe, False otherwise
    """
    if cutoff_idx < 1 or cutoff_idx >= len(messages):
        return False
    
    # Check all kept messages for orphaned tool responses
    for kept_idx in range(cutoff_idx, len(messages)):
        kept_msg = messages[kept_idx]
        
        # Check if this is a tool response whose assistant would be evicted
        if kept_msg.role == MessageRole.tool and kept_msg.tool_call_id:
            assistant_idx = tool_call_id_to_assistant_idx.get(kept_msg.tool_call_id)
            if assistant_idx is not None and assistant_idx < cutoff_idx:
                return False
        
        # Check if this is an assistant with tool calls where some responses would be evicted
        if kept_msg.role == MessageRole.assistant and kept_msg.tool_calls:
            tool_response_indices = assistant_idx_to_tool_indices.get(kept_idx, [])
            for tool_idx in tool_response_indices:
                if tool_idx < cutoff_idx:
                    return False
    
    return True


async def count_tokens(actor: User, llm_config: LLMConfig, messages: List[Message]) -> int:
    """Count tokens in messages using the appropriate token counter for the model configuration."""
    token_counter = create_token_counter(
        model_endpoint_type=llm_config.model_endpoint_type,
        model=llm_config.model,
        actor=actor,
    )
    converted_messages = token_counter.convert_messages(messages)
    tokens = await token_counter.count_message_tokens(converted_messages)

    # Apply safety margin for approximate counting to avoid underestimating
    from letta.services.context_window_calculator.token_counter import ApproxTokenCounter

    if isinstance(token_counter, ApproxTokenCounter):
        return int(tokens * APPROX_TOKEN_SAFETY_MARGIN)
    return tokens


@trace_method
async def summarize_via_sliding_window(
    # Required to tag LLM calls
    actor: User,
    # Actual summarization configuration
    llm_config: LLMConfig,
    summarizer_config: CompactionSettings,
    in_context_messages: List[Message],
    # new_messages: List[Message],
) -> Tuple[str, List[Message]]:
    """
    If the total tokens is greater than the context window limit (or force=True),
    then summarize and rearrange the in-context messages (with the summary in front).

    Finding the summarization cutoff point (target of final post-summarize count is N% of configured context window):
    1. Start at a message index cutoff (1-N%)
    2. Count tokens with system prompt, prior summary (if it exists), and messages past cutoff point (messages[0] + messages[cutoff:])
    3. Is count(post_sum_messages) <= N% of configured context window?
      3a. Yes -> create new summary with [prior summary, cutoff:], and safety truncate summary with char count
      3b. No -> increment cutoff by 10%, and repeat

    IMPORTANT: This function ensures tool_use → tool_result message pairs are treated atomically.
    A tool response message (role=tool) will never be orphaned from its corresponding assistant
    message (which contains the tool_use block).

    IMPORTANT: Memory context messages (identified by name tag or content patterns) are NOT
    included in summarization. They are filtered out before processing and should be re-added
    by the caller (compact() in LettaAgentV3) with fresh memory state.

    Returns:
    - The summary string
    - The list of messages to keep in-context (system prompt + retained conversation, NO memory messages)
    """
    system_prompt = in_context_messages[0]
    
    # Filter out memory messages - they should not be summarized
    # Memory will be re-injected fresh by the caller after compaction
    conversation_messages_raw = in_context_messages[1:]  # Everything after system prompt
    conversation_messages, memory_messages = filter_memory_messages(conversation_messages_raw)
    
    if memory_messages:
        logger.info(f"Filtered out {len(memory_messages)} memory message(s) from summarization")
    
    # If no conversation messages left after filtering, nothing to summarize
    if not conversation_messages:
        logger.warning("No conversation messages to summarize after filtering memory messages")
        return "", [system_prompt]
    
    total_message_count = len(conversation_messages)

    # cannot evict a pending approval request (will cause client-side errors)
    if conversation_messages[-1].role == MessageRole.approval:
        maximum_message_index = total_message_count - 2
    else:
        maximum_message_index = total_message_count - 1

    # Build tool call pairing maps for atomic eviction (on conversation messages only)
    tool_call_id_to_assistant_idx = _build_tool_call_id_to_assistant_index(conversation_messages)
    assistant_idx_to_tool_indices = _build_assistant_index_to_tool_response_indices(
        conversation_messages, tool_call_id_to_assistant_idx
    )

    # Starts at N% (eg 70%), and increments up until 100%
    message_count_cutoff_percent = max(
        1 - summarizer_config.sliding_window_percentage, 0.10
    )  # Some arbitrary minimum value (10%) to avoid negatives from badly configured summarizer percentage
    eviction_percentage = summarizer_config.sliding_window_percentage
    assert summarizer_config.sliding_window_percentage <= 1.0, "Sliding window percentage must be less than or equal to 1.0"
    safe_cutoff_index = None
    approx_token_count = llm_config.context_window
    # valid_cutoff_roles = {MessageRole.assistant, MessageRole.approval}
    valid_cutoff_roles = {MessageRole.assistant}

    # simple version: summarize(conversation[0:round(summarizer_config.sliding_window_percentage * len(conversation))])
    # this evicts 30% of the messages (via summarization) and keeps the remaining 70%
    # problem: we need the cutoff point to be an assistant message, so will grow the cutoff point until we find an assistant message
    # also need to grow the cutoff point until the token count is less than the target token count
    # ADDITIONALLY: we must respect tool_use → tool_result pairing to avoid breaking LLM API constraints

    while approx_token_count >= (1 - summarizer_config.sliding_window_percentage) * llm_config.context_window and eviction_percentage < 1.0:
        # more eviction percentage
        eviction_percentage += 0.10

        # calculate message_cutoff_index (relative to conversation_messages, not in_context_messages)
        message_cutoff_index = round(eviction_percentage * total_message_count)

        # Find a safe cutoff index that respects tool pairing constraints
        safe_cutoff_index = _find_safe_cutoff_index(
            messages=conversation_messages,
            target_cutoff_index=message_cutoff_index,
            tool_call_id_to_assistant_idx=tool_call_id_to_assistant_idx,
            assistant_idx_to_tool_indices=assistant_idx_to_tool_indices,
            valid_cutoff_roles=valid_cutoff_roles,
            maximum_message_index=maximum_message_index,
        )
        
        if safe_cutoff_index is None:
            logger.warning(f"No safe cutoff found for evicting up to index {message_cutoff_index}, incrementing eviction percentage")
            continue

        # update token count - note: memory will be added back by caller, so we only count system + retained conversation
        logger.info(f"Attempting to compact messages index 0:{safe_cutoff_index} (conversation only)")
        post_summarization_buffer = [system_prompt] + conversation_messages[safe_cutoff_index:]
        approx_token_count = await count_tokens(actor, llm_config, post_summarization_buffer)
        logger.info(
            f"Compacting messages index 0:{safe_cutoff_index} messages resulted in {approx_token_count} tokens, goal is {(1 - summarizer_config.sliding_window_percentage) * llm_config.context_window}"
        )

    if safe_cutoff_index is None or eviction_percentage >= 1.0:
        raise ValueError("No safe cutoff found for sliding window summarization (could not find valid assistant message that respects tool pairing)")  # fall back to complete summarization

    if safe_cutoff_index >= maximum_message_index:
        # need to keep the last message (might contain an approval request)
        raise ValueError(f"Safe cutoff index {safe_cutoff_index} is at the end of the message buffer, skipping summarization")

    messages_to_summarize = conversation_messages[:safe_cutoff_index]
    logger.info(
        f"Summarizing {len(messages_to_summarize)} messages, from index 0 to {safe_cutoff_index} (out of {total_message_count} conversation messages)"
    )

    summary_message_str = await simple_summary(
        messages=messages_to_summarize,
        llm_config=llm_config,
        actor=actor,
        include_ack=bool(summarizer_config.prompt_acknowledgement),
        prompt=summarizer_config.prompt,
    )

    if summarizer_config.clip_chars is not None and len(summary_message_str) > summarizer_config.clip_chars:
        logger.warning(f"Summary length {len(summary_message_str)} exceeds clip length {summarizer_config.clip_chars}. Truncating.")
        summary_message_str = summary_message_str[: summarizer_config.clip_chars] + "... [summary truncated to fit]"

    # Return system prompt + retained conversation messages (NO memory messages - caller will add fresh memory)
    updated_in_context_messages = conversation_messages[safe_cutoff_index:]
    return summary_message_str, [system_prompt] + updated_in_context_messages
