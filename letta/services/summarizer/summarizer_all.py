from typing import List, Tuple

from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageRole
from letta.schemas.user import User
from letta.services.summarizer.memory_message_utils import filter_memory_messages
from letta.services.summarizer.summarizer import simple_summary
from letta.services.summarizer.summarizer_config import CompactionSettings

logger = get_logger(__name__)


@trace_method
async def summarize_all(
    # Required to tag LLM calls
    actor: User,
    # LLM config for the summarizer model
    llm_config: LLMConfig,
    # Actual summarization configuration
    summarizer_config: CompactionSettings,
    in_context_messages: List[Message],
    # new_messages: List[Message],
) -> Tuple[str, List[Message]]:
    """
    Summarize the entire conversation history into a single summary.

    IMPORTANT: Memory context messages (identified by name tag or content patterns) are NOT
    included in summarization. They are filtered out before processing and should be re-added
    by the caller (compact() in LettaAgentV3) with fresh memory state.

    Returns:
    - The summary string
    - The list of messages to keep in-context (system prompt + protected messages, NO memory messages)
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
    
    logger.info(
        f"Summarizing all messages (index 0 to {len(conversation_messages) - 2}), keeping last message: {conversation_messages[-1].role if conversation_messages else 'N/A'}"
    )
    
    if conversation_messages[-1].role == MessageRole.approval:
        # cannot evict a pending approval request (will cause client-side errors)
        # Also protect the assistant message before it if they share the same step_id
        # (both are part of the same LLM response - assistant has thinking/tool_calls, approval has approval-required subset)
        protected_messages = [conversation_messages[-1]]

        # Check if the message before approval is also from the same step (has reasoning/tool_calls)
        if len(conversation_messages) >= 2:
            potential_assistant = conversation_messages[-2]
            approval_request = conversation_messages[-1]
            if potential_assistant.role == MessageRole.assistant and potential_assistant.step_id == approval_request.step_id:
                # They're part of the same LLM response - protect both
                protected_messages = [potential_assistant, approval_request]
                messages_to_summarize = conversation_messages[:-2]
            else:
                messages_to_summarize = conversation_messages[:-1]
        else:
            messages_to_summarize = conversation_messages[:-1]
    else:
        messages_to_summarize = conversation_messages
        protected_messages = []

    # TODO: add fallback in case this has a context window error
    summary_message_str = await simple_summary(
        messages=messages_to_summarize,
        llm_config=llm_config,
        actor=actor,
        include_ack=bool(summarizer_config.prompt_acknowledgement),
        prompt=summarizer_config.prompt,
    )
    logger.info(f"Summarized {len(messages_to_summarize)} messages")

    if summarizer_config.clip_chars is not None and len(summary_message_str) > summarizer_config.clip_chars:
        logger.warning(f"Summary length {len(summary_message_str)} exceeds clip length {summarizer_config.clip_chars}. Truncating.")
        summary_message_str = summary_message_str[: summarizer_config.clip_chars] + "... [summary truncated to fit]"

    # Return system prompt + protected messages (NO memory messages - caller will add fresh memory)
    return summary_message_str, [system_prompt] + protected_messages
