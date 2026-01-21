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


def _find_safe_cutoff_index(
    messages: List[Message],
    target_cutoff_index: int,
    tool_call_id_to_assistant_idx: dict,
    assistant_idx_to_tool_indices: dict,
    valid_cutoff_roles: Set[MessageRole],
) -> Optional[int]:
    """Find a safe cutoff index that respects tool_use → tool_result pairing.
    
    The cutoff index is the first message to KEEP (i.e., messages[cutoff:] are kept).
    We need to ensure that:
    1. The cutoff is at an assistant message (for proper conversation flow)
    2. If we're keeping a tool response message, we must also keep its assistant message
    3. If we're evicting an assistant message with tool calls, we must also evict all its tool responses
    
    Args:
        messages: The list of messages
        target_cutoff_index: The initial target cutoff index (first message to keep)
        tool_call_id_to_assistant_idx: Mapping from tool_call_id to assistant message index
        assistant_idx_to_tool_indices: Mapping from assistant index to tool response indices
        valid_cutoff_roles: Set of valid roles for cutoff point (typically just assistant)
    
    Returns:
        A safe cutoff index, or None if no valid cutoff can be found
    """
    # Search backwards from target_cutoff_index to find a valid assistant message
    for candidate_idx in reversed(range(1, target_cutoff_index + 1)):
        if candidate_idx >= len(messages):
            continue
        
        candidate_msg = messages[candidate_idx]
        
        # Must be a valid cutoff role (assistant)
        if candidate_msg.role not in valid_cutoff_roles:
            continue
        
        # Check if this assistant message has tool calls
        if candidate_msg.tool_calls:
            # Get all tool response indices for this assistant message
            tool_response_indices = assistant_idx_to_tool_indices.get(candidate_idx, [])
            
            if tool_response_indices:
                # The last tool response must come BEFORE this cutoff for us to safely evict
                # Or all tool responses must come AFTER (meaning we keep all of them)
                max_tool_response_idx = max(tool_response_indices)
                
                # If any tool response is >= candidate_idx, they would be in the "keep" portion
                # which is fine - we're keeping the assistant message too
                # But if the assistant message is being evicted (< candidate_idx), then
                # all its tool responses must also be evicted (< candidate_idx)
                # 
                # Wait - the candidate_idx IS the assistant message index, and we're considering
                # keeping messages[candidate_idx:]. So if candidate has tool calls, we're keeping
                # the assistant message. We need to check that ALL its tool responses are also kept.
                min_tool_response_idx = min(tool_response_indices)
                
                if min_tool_response_idx < candidate_idx:
                    # Some tool responses would be evicted while assistant message is kept - NOT SAFE
                    # We need to move the cutoff earlier to include all tool responses
                    # Actually, we need to either:
                    # a) Move cutoff to include all tool responses (cutoff = min_tool_response_idx or earlier)
                    # b) Skip this assistant message and find an earlier one
                    # 
                    # Since we want to evict as much as possible, let's try to adjust the cutoff
                    # to include the tool responses. But we need to find a valid assistant message
                    # before the first tool response.
                    continue  # Skip this candidate, try to find an earlier valid cutoff
        
        # This candidate is safe to use as cutoff.
        # But we also need to verify that we're not orphaning any tool responses in the KEPT portion.
        # Check all kept messages (candidate_idx onwards) for tool responses whose assistant is evicted.
        safe = True
        for kept_idx in range(candidate_idx, len(messages)):
            kept_msg = messages[kept_idx]
            if kept_msg.role == MessageRole.tool and kept_msg.tool_call_id:
                assistant_idx = tool_call_id_to_assistant_idx.get(kept_msg.tool_call_id)
                if assistant_idx is not None and assistant_idx < candidate_idx:
                    # This tool response's assistant message would be evicted - NOT SAFE
                    safe = False
                    break
        
        if safe:
            return candidate_idx
    
    return None


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

    Returns:
    - The summary string
    - The list of message IDs to keep in-context
    """
    system_prompt = in_context_messages[0]
    total_message_count = len(in_context_messages)

    # cannot evict a pending approval request (will cause client-side errors)
    if in_context_messages[-1].role == MessageRole.approval:
        maximum_message_index = total_message_count - 2
    else:
        maximum_message_index = total_message_count - 1

    # Build tool call pairing maps for atomic eviction
    tool_call_id_to_assistant_idx = _build_tool_call_id_to_assistant_index(in_context_messages)
    assistant_idx_to_tool_indices = _build_assistant_index_to_tool_response_indices(
        in_context_messages, tool_call_id_to_assistant_idx
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

    # simple version: summarize(in_context[1:round(summarizer_config.sliding_window_percentage * len(in_context_messages))])
    # this evicts 30% of the messages (via summarization) and keeps the remaining 70%
    # problem: we need the cutoff point to be an assistant message, so will grow the cutoff point until we find an assistant message
    # also need to grow the cutoff point until the token count is less than the target token count
    # ADDITIONALLY: we must respect tool_use → tool_result pairing to avoid breaking LLM API constraints

    while approx_token_count >= (1 - summarizer_config.sliding_window_percentage) * llm_config.context_window and eviction_percentage < 1.0:
        # more eviction percentage
        eviction_percentage += 0.10

        # calculate message_cutoff_index
        message_cutoff_index = round(eviction_percentage * total_message_count)

        # Find a safe cutoff index that respects tool pairing constraints
        safe_cutoff_index = _find_safe_cutoff_index(
            messages=in_context_messages,
            target_cutoff_index=message_cutoff_index,
            tool_call_id_to_assistant_idx=tool_call_id_to_assistant_idx,
            assistant_idx_to_tool_indices=assistant_idx_to_tool_indices,
            valid_cutoff_roles=valid_cutoff_roles,
        )
        
        if safe_cutoff_index is None:
            logger.warning(f"No safe cutoff found for evicting up to index {message_cutoff_index}, incrementing eviction percentage")
            continue

        # update token count
        logger.info(f"Attempting to compact messages index 1:{safe_cutoff_index} messages")
        post_summarization_buffer = [system_prompt] + in_context_messages[safe_cutoff_index:]
        approx_token_count = await count_tokens(actor, llm_config, post_summarization_buffer)
        logger.info(
            f"Compacting messages index 1:{safe_cutoff_index} messages resulted in {approx_token_count} tokens, goal is {(1 - summarizer_config.sliding_window_percentage) * llm_config.context_window}"
        )

    if safe_cutoff_index is None or eviction_percentage >= 1.0:
        raise ValueError("No safe cutoff found for sliding window summarization (could not find valid assistant message that respects tool pairing)")  # fall back to complete summarization

    if safe_cutoff_index >= maximum_message_index:
        # need to keep the last message (might contain an approval request)
        raise ValueError(f"Safe cutoff index {safe_cutoff_index} is at the end of the message buffer, skipping summarization")

    messages_to_summarize = in_context_messages[1:safe_cutoff_index]
    logger.info(
        f"Summarizing {len(messages_to_summarize)} messages, from index 1 to {safe_cutoff_index} (out of {total_message_count})"
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

    updated_in_context_messages = in_context_messages[safe_cutoff_index:]
    return summary_message_str, [system_prompt] + updated_in_context_messages
