import asyncio
import json
import traceback
from typing import List, Optional, Tuple, Union

from letta.agents.ephemeral_summary_agent import EphemeralSummaryAgent
from letta.constants import (
    DEFAULT_MESSAGE_TOOL,
    DEFAULT_MESSAGE_TOOL_KWARG,
    MESSAGE_SUMMARY_REQUEST_ACK,
    TOOL_RETURN_TRUNCATION_CHARS,
)
from letta.errors import ContextWindowExceededError
from letta.helpers.message_helper import convert_message_creates_to_messages
from letta.llm_api.llm_client import LLMClient
from letta.log import get_logger
from letta.otel.tracing import trace_method
from letta.prompts import gpt_summarize
from letta.schemas.enums import AgentType, MessageRole, ProviderType
from letta.schemas.letta_message_content import ImageContent, TextContent
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, MessageCreate
from letta.schemas.user import User
from letta.services.agent_manager import AgentManager
from letta.services.message_manager import MessageManager
from letta.services.summarizer.enums import SummarizationMode
from letta.system import package_summarize_message_no_counts
from letta.utils import safe_create_task

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Chunked Summarization Constants
# ---------------------------------------------------------------------------

# Default chunk size (in messages) for chunked summarization
# This is conservative - most models can handle more, but we want safety margin
DEFAULT_CHUNK_SIZE_MESSAGES = 30

# Default char budget per chunk (rough estimate: ~4 chars per token, 60% of context window)
# This provides headroom for system prompt, output, etc.
DEFAULT_CHUNK_CHAR_BUDGET = 40000

# Maximum recursion depth for hierarchical summarization
MAX_HIERARCHICAL_DEPTH = 3

# Prompt for combining multiple chunk summaries into one
CHUNK_COMBINE_PROMPT = """You are combining multiple summaries of different parts of a conversation into one coherent summary.

The summaries below are from consecutive chunks of the same conversation, in chronological order.
Combine them into a single, coherent summary that:
1. Preserves all important information from each chunk
2. Removes redundancy between chunks
3. Maintains chronological flow
4. Stays concise while being complete

Output only the combined summary, nothing else."""


# NOTE: legacy, new version is functional
class Summarizer:
    """
    Handles summarization or trimming of conversation messages based on
    the specified SummarizationMode. For now, we demonstrate a simple
    static buffer approach but leave room for more advanced strategies.
    """

    def __init__(
        self,
        mode: SummarizationMode,
        summarizer_agent: Optional[Union[EphemeralSummaryAgent, "VoiceSleeptimeAgent"]] = None,
        message_buffer_limit: int = 10,
        message_buffer_min: int = 3,
        partial_evict_summarizer_percentage: float = 0.30,
        agent_manager: Optional[AgentManager] = None,
        message_manager: Optional[MessageManager] = None,
        actor: Optional[User] = None,
        agent_id: Optional[str] = None,
    ):
        self.mode = mode

        # Need to do validation on this
        # TODO: Move this to config
        self.message_buffer_limit = message_buffer_limit
        self.message_buffer_min = message_buffer_min
        self.summarizer_agent = summarizer_agent
        self.partial_evict_summarizer_percentage = partial_evict_summarizer_percentage

        # for partial buffer only
        self.agent_manager = agent_manager
        self.message_manager = message_manager
        self.actor = actor
        self.agent_id = agent_id

    @trace_method
    async def summarize(
        self,
        in_context_messages: List[Message],
        new_letta_messages: List[Message],
        force: bool = False,
        clear: bool = False,
    ) -> Tuple[List[Message], bool]:
        """
        Summarizes or trims in_context_messages according to the chosen mode,
        and returns the updated messages plus any optional "summary message".

        Args:
            in_context_messages: The existing messages in the conversation's context.
            new_letta_messages: The newly added Letta messages (just appended).
            force: Force summarize even if the criteria is not met

        Returns:
            (updated_messages, summary_message)
            updated_messages: The new context after trimming/summary
            summary_message: Optional summarization message that was created
                             (could be appended to the conversation if desired)
        """
        if self.mode == SummarizationMode.STATIC_MESSAGE_BUFFER:
            return self._static_buffer_summarization(
                in_context_messages,
                new_letta_messages,
                force=force,
                clear=clear,
            )
        elif self.mode == SummarizationMode.PARTIAL_EVICT_MESSAGE_BUFFER:
            return await self._partial_evict_buffer_summarization(
                in_context_messages,
                new_letta_messages,
                force=force,
                clear=clear,
            )
        else:
            # Fallback or future logic
            return in_context_messages, False

    def fire_and_forget(self, coro):
        task = safe_create_task(coro, label="summarizer_background_task")

        def callback(t):
            try:
                t.result()  # This re-raises exceptions from the task
            except Exception:
                logger.exception("Background task failed")

        task.add_done_callback(callback)
        return task

    async def _partial_evict_buffer_summarization(
        self,
        in_context_messages: List[Message],
        new_letta_messages: List[Message],
        force: bool = False,
        clear: bool = False,
    ) -> Tuple[List[Message], bool]:
        """Summarization as implemented in the original MemGPT loop, but using message count instead of token count.
        Evict a partial amount of messages, and replace message[1] with a recursive summary.

        Note that this can't be made sync, because we're waiting on the summary to inject it into the context window,
        unlike the version that writes it to a block.

        Unless force is True, don't summarize.
        Ignore clear, we don't use it.
        """
        all_in_context_messages = in_context_messages + new_letta_messages

        if not force:
            logger.debug("Not forcing summarization, returning in-context messages as is.")
            return all_in_context_messages, False

        # First step: determine how many messages to retain
        total_message_count = len(all_in_context_messages)
        assert self.partial_evict_summarizer_percentage >= 0.0 and self.partial_evict_summarizer_percentage <= 1.0
        target_message_start = round((1.0 - self.partial_evict_summarizer_percentage) * total_message_count)
        logger.info(f"Target message count: {total_message_count}->{(total_message_count - target_message_start)}")

        # The summary message we'll insert is role 'user' (vs 'assistant', 'tool', or 'system')
        # We are going to put it at index 1 (index 0 is the system message)
        # That means that index 2 needs to be role 'assistant', so walk up the list starting at
        # the target_message_count and find the first assistant message
        for i in range(target_message_start, total_message_count):
            if all_in_context_messages[i].role == MessageRole.assistant:
                assistant_message_index = i
                break
        else:
            raise ValueError(f"No assistant message found from indices {target_message_start} to {total_message_count}")

        # The sequence to summarize is index 1 -> assistant_message_index
        messages_to_summarize = all_in_context_messages[1:assistant_message_index]
        logger.info(f"Eviction indices: {1}->{assistant_message_index}(/{total_message_count})")

        # Dynamically get the LLMConfig from the summarizer agent
        # Pretty cringe code here that we need the agent for this but we don't use it
        agent_state = await self.agent_manager.get_agent_by_id_async(agent_id=self.agent_id, actor=self.actor)

        # TODO if we do this via the "agent", then we can more easily allow toggling on the memory block version
        summary_message_str = await simple_summary(
            messages=messages_to_summarize,
            llm_config=agent_state.llm_config,
            actor=self.actor,
            include_ack=True,
            agent_id=self.agent_id,
            agent_tags=agent_state.tags,
        )

        # TODO add counts back
        # Recall message count
        # num_recall_messages_current = await self.message_manager.size_async(actor=self.actor, agent_id=agent_state.id)
        # num_messages_evicted = len(messages_to_summarize)
        # num_recall_messages_hidden = num_recall_messages_total - len()

        # Create the summary message
        summary_message_str_packed = package_summarize_message_no_counts(
            summary=summary_message_str,
            timezone=agent_state.timezone,
        )
        summary_message_obj = (
            await convert_message_creates_to_messages(
                message_creates=[
                    MessageCreate(
                        role=MessageRole.user,
                        content=[TextContent(text=summary_message_str_packed)],
                    )
                ],
                agent_id=agent_state.id,
                timezone=agent_state.timezone,
                # We already packed, don't pack again
                wrap_user_message=False,
                wrap_system_message=False,
                run_id=None,  # TODO: add this
            )
        )[0]

        # Create the message in the DB
        await self.message_manager.create_many_messages_async(
            pydantic_msgs=[summary_message_obj],
            actor=self.actor,
            project_id=agent_state.project_id,
            template_id=agent_state.template_id,
        )

        updated_in_context_messages = all_in_context_messages[assistant_message_index:]
        return [all_in_context_messages[0], summary_message_obj] + updated_in_context_messages, True

    def _static_buffer_summarization(
        self,
        in_context_messages: List[Message],
        new_letta_messages: List[Message],
        force: bool = False,
        clear: bool = False,
    ) -> Tuple[List[Message], bool]:
        """
        Implements static buffer summarization by maintaining a fixed-size message buffer (< N messages).

        Logic:
        1. Combine existing context messages with new messages
        2. If total messages <= buffer limit and not forced, return unchanged
        3. Calculate how many messages to retain (0 if clear=True, otherwise message_buffer_min)
        4. Find the trim index to keep the most recent messages while preserving user message boundaries
        5. Evict older messages (everything between system message and trim index)
        6. If summarizer agent is available, trigger background summarization of evicted messages
        7. Return updated context with system message + retained recent messages

        Args:
            in_context_messages: Existing conversation context messages
            new_letta_messages: Newly added messages to append
            force: Force summarization even if buffer limit not exceeded
            clear: Clear all messages except system message (retain_count = 0)

        Returns:
            Tuple of (updated_messages, was_summarized)
            - updated_messages: New context after trimming/summarization
            - was_summarized: True if messages were evicted and summarization triggered
        """

        all_in_context_messages = in_context_messages + new_letta_messages

        if len(all_in_context_messages) <= self.message_buffer_limit and not force:
            logger.info(
                f"Nothing to evict, returning in context messages as is. Current buffer length is {len(all_in_context_messages)}, limit is {self.message_buffer_limit}."
            )
            return all_in_context_messages, False

        retain_count = 0 if clear else self.message_buffer_min

        if not force:
            logger.info(f"Buffer length hit {self.message_buffer_limit}, evicting until we retain only {retain_count} messages.")
        else:
            logger.info(f"Requested force summarization, evicting until we retain only {retain_count} messages.")

        target_trim_index = max(1, len(all_in_context_messages) - retain_count)

        while target_trim_index < len(all_in_context_messages) and all_in_context_messages[target_trim_index].role != MessageRole.user:
            target_trim_index += 1

        # If the first retained message is an approval request, also keep the assistant message before it
        # (they're part of the same LLM response - assistant has reasoning/tool_calls, approval has approval-required subset)
        if target_trim_index < len(all_in_context_messages):
            first_retained = all_in_context_messages[target_trim_index]
            if first_retained.role == MessageRole.approval and target_trim_index > 1:
                # Check if the message before it is an assistant from the same step
                prev_message = all_in_context_messages[target_trim_index - 1]
                if prev_message.role == MessageRole.assistant and prev_message.step_id == first_retained.step_id:
                    # Back up to include the assistant message with reasoning
                    target_trim_index -= 1

        evicted_messages = all_in_context_messages[1:target_trim_index]  # everything except sys msg
        updated_in_context_messages = all_in_context_messages[target_trim_index:]  # may be empty

        # If *no* messages were evicted we really have nothing to do
        if not evicted_messages:
            logger.info("Nothing to evict, returning in-context messages as-is.")
            return all_in_context_messages, False

        if self.summarizer_agent:
            # Only invoke if summarizer agent is passed in
            # Format
            formatted_evicted_messages = format_transcript(evicted_messages)
            formatted_in_context_messages = format_transcript(updated_in_context_messages)

            # TODO: This is hyperspecific to voice, generalize!
            # Update the message transcript of the memory agent
            if not isinstance(self.summarizer_agent, EphemeralSummaryAgent):
                self.summarizer_agent.update_message_transcript(
                    message_transcripts=formatted_evicted_messages + formatted_in_context_messages
                )

            # Add line numbers to the formatted messages
            offset = len(formatted_evicted_messages)
            formatted_evicted_messages = [f"{i}. {msg}" for (i, msg) in enumerate(formatted_evicted_messages)]
            formatted_in_context_messages = [f"{i + offset}. {msg}" for (i, msg) in enumerate(formatted_in_context_messages)]

            summary_request_text = build_summary_request_text(
                retain_count=retain_count,
                evicted_messages=formatted_evicted_messages,
                in_context_messages=formatted_in_context_messages,
            )

            # Fire-and-forget the summarization task
            self.fire_and_forget(
                self.summarizer_agent.step([MessageCreate(role=MessageRole.user, content=[TextContent(text=summary_request_text)])])
            )

        return [all_in_context_messages[0]] + updated_in_context_messages, True


def simple_formatter(
    messages: List[Message],
    include_system: bool = False,
    tool_return_truncation_chars: int | None = None,
) -> str:
    """Go from an OpenAI-style list of messages to a concatenated string.

    Optionally clamps tool-return content to avoid ballooning the summarizer transcript.
    """

    parsed_messages = Message.to_openai_dicts_from_list(
        [message for message in messages if message.role != MessageRole.system or include_system],
        tool_return_truncation_chars=tool_return_truncation_chars,
    )
    return "<start_transcript>\n" + "\n".join(json.dumps(msg) for msg in parsed_messages) + "\n<end_transcript>\n. Generate the summary."


def middle_truncate_text(
    text: str,
    budget_chars: int,
    head_frac: float = 0.3,
    tail_frac: float = 0.3,
) -> tuple[str, int]:
    """Middle-truncate a string to fit within a character budget.

    Keeps the first `head_frac` and last `tail_frac` portions (by budget chars)
    and drops the middle. Returns (truncated_text, dropped_char_count).

    Fractions are relative to budget, not original text length.
    """
    if budget_chars <= 0 or len(text) <= budget_chars:
        return text, 0

    head_len = max(0, int(budget_chars * head_frac))
    tail_len = max(0, int(budget_chars * tail_frac))
    # Ensure head + tail <= budget; allocate remainder to tail preferentially
    if head_len + tail_len > budget_chars:
        tail_len = max(0, budget_chars - head_len)

    head = text[:head_len]
    tail = text[-tail_len:] if tail_len > 0 else ""
    dropped = max(0, len(text) - (len(head) + len(tail)))

    marker = f"\n[TRUNCATED: dropped {dropped} middle chars due to context budget]\n"
    # If marker would overflow budget, shrink tail to fit marker
    available_for_marker = budget_chars - (len(head) + len(tail))
    if available_for_marker < len(marker):
        # reduce tail to free up space
        over = len(marker) - available_for_marker
        tail = tail[:-over] if over < len(tail) else ""

    return head + marker + tail, dropped


def _summarizer_llm_config(base: LLMConfig) -> LLMConfig:
    """Create a safe LLMConfig for summarization.

    Summarization requests should:
    - not put inner thoughts in kwargs (provider formatting conflicts)
    - not enable extended reasoning
    """
    cfg = LLMConfig(**base.model_dump())
    cfg.put_inner_thoughts_in_kwargs = False
    cfg.enable_reasoner = False
    return cfg


def _summarizer_supports_provider_streaming(llm_config: LLMConfig) -> bool:
    return llm_config.model_endpoint_type in [ProviderType.anthropic, ProviderType.bedrock]


async def _run_summarizer_request(
    llm_client: LLMClient,
    summarizer_llm_config: LLMConfig,
    request_data: dict,
    input_messages_obj: list[Message],
) -> str:
    """Run a summarization request and return assistant text.

    DRY helper shared by both one-pass and chunked summarization.

    For Anthropic/Bedrock, use provider-side streaming to avoid long-request failures.
    Otherwise, use non-streaming request and normalize via chat-completions conversion.
    """
    if _summarizer_supports_provider_streaming(summarizer_llm_config):
        logger.info(
            "Summarizer: using provider streaming (%s/%s) to avoid long-request failures",
            summarizer_llm_config.model_endpoint_type,
            summarizer_llm_config.model,
        )
        from letta.interfaces.anthropic_parallel_tool_call_streaming_interface import (
            SimpleAnthropicStreamingInterface,
        )

        interface = SimpleAnthropicStreamingInterface(
            requires_approval_tools=[],
            run_id=None,
            step_id=None,
        )

        # Provider client sets request_data["stream"] = True internally.
        stream = await llm_client.stream_async_with_telemetry(request_data, summarizer_llm_config)
        async for _chunk in interface.process(stream):
            pass

        content_parts = interface.get_content()
        text = "".join(part.text for part in content_parts if isinstance(part, TextContent)).strip()
        if not text:
            raise Exception("Summary failed to generate")

        # Log telemetry after stream processing
        await llm_client.log_provider_trace_async(
            request_data=request_data,
            response_json={
                "content": text,
                "model": summarizer_llm_config.model,
                "usage": {
                    "input_tokens": getattr(interface, "input_tokens", None),
                    "output_tokens": getattr(interface, "output_tokens", None),
                },
            },
        )
        return text

    logger.debug(
        "Summarizer: using non-streaming request (%s/%s)",
        summarizer_llm_config.model_endpoint_type,
        summarizer_llm_config.model,
    )
    response_data = await llm_client.request_async_with_telemetry(request_data, summarizer_llm_config)
    response = await llm_client.convert_response_to_chat_completion(
        response_data,
        input_messages_obj,
        summarizer_llm_config,
    )
    content = response.choices[0].message.content
    if content is None:
        raise Exception("Summary failed to generate")
    return content.strip()


def _compute_safe_transcript_budget_chars(
    llm_config: LLMConfig,
    system_prompt: str,
    include_ack: bool,
    safety_frac: float = 0.6,
    min_budget_chars: int = 2000,
    default_budget_chars: int = 48000,
) -> int:
    """Compute a conservative transcript char budget for summarization.

    Uses a rough 4 chars/token heuristic with a safety fraction, and subtracts overhead
    for the system prompt + optional ACK + small JSON overhead.

    This is intentionally approximate; providers vary, and tool output can have high
    token density.
    """
    try:
        budget_chars = int(llm_config.context_window * safety_frac * 4)
    except Exception:
        budget_chars = default_budget_chars

    overhead = len(system_prompt) + (len(MESSAGE_SUMMARY_REQUEST_ACK) if include_ack else 0) + 1024
    return max(min_budget_chars, budget_chars - overhead)


def _truncate_to_budget(
    text: str,
    *,
    budget_chars: int,
    head_frac: float,
    tail_frac: float,
) -> str:
    truncated, _ = middle_truncate_text(
        text,
        budget_chars=budget_chars,
        head_frac=head_frac,
        tail_frac=tail_frac,
    )
    return truncated


def build_summary_request_text(retain_count: int, evicted_messages: List[str], in_context_messages: List[str]) -> str:
    parts: List[str] = []
    if retain_count == 0:
        parts.append(
            "You’re a memory-recall helper for an AI that is about to forget all prior messages. Scan the conversation history and write crisp notes that capture any important facts or insights about the conversation history."
        )
    else:
        parts.append(
            f"You’re a memory-recall helper for an AI that can only keep the last {retain_count} messages. Scan the conversation history, focusing on messages about to drop out of that window, and write crisp notes that capture any important facts or insights about the human so they aren’t lost."
        )

    if evicted_messages:
        parts.append("\n(Older) Evicted Messages:")
        for item in evicted_messages:
            parts.append(f"    {item}")

    if retain_count > 0 and in_context_messages:
        parts.append("\n(Newer) In-Context Messages:")
        for item in in_context_messages:
            parts.append(f"    {item}")

    return "\n".join(parts) + "\n"


def simple_message_wrapper(openai_msg: dict) -> Message:
    """Extremely simple way to map from role/content to Message object w/ throwaway dummy fields"""

    if "role" not in openai_msg:
        raise ValueError(f"Missing role in openai_msg: {openai_msg}")
    if "content" not in openai_msg:
        raise ValueError(f"Missing content in openai_msg: {openai_msg}")

    if openai_msg["role"] == "user":
        return Message(
            role=MessageRole.user,
            content=[TextContent(text=openai_msg["content"])],
        )
    elif openai_msg["role"] == "assistant":
        return Message(
            role=MessageRole.assistant,
            content=[TextContent(text=openai_msg["content"])],
        )
    elif openai_msg["role"] == "system":
        return Message(
            role=MessageRole.system,
            content=[TextContent(text=openai_msg["content"])],
        )
    else:
        raise ValueError(f"Unknown role: {openai_msg['role']}")


@trace_method
async def simple_summary(
    messages: List[Message],
    llm_config: LLMConfig,
    actor: User,
    include_ack: bool = True,
    prompt: str | None = None,
    telemetry_manager: "TelemetryManager | None" = None,
    agent_id: str | None = None,
    agent_tags: List[str] | None = None,
    run_id: str | None = None,
) -> str:
    """Generate a simple summary from a list of messages.

    Intentionally kept functional due to the simplicity of the prompt.
    """
    from letta.services.telemetry_manager import TelemetryManager

    # Create an LLMClient from the config
    llm_client = LLMClient.create(
        provider_type=llm_config.model_endpoint_type,
        put_inner_thoughts_first=True,
        actor=actor,
    )
    assert llm_client is not None

    # Always set telemetry context - create TelemetryManager if not provided
    tm = telemetry_manager or TelemetryManager()
    llm_client.set_telemetry_context(
        telemetry_manager=tm,
        agent_id=agent_id,
        agent_tags=agent_tags,
        run_id=run_id,
        call_type="summarization",
    )

    # Prepare the messages payload to send to the LLM
    system_prompt = prompt or gpt_summarize.SYSTEM
    # Build the initial transcript.
    # Do a conservative pre-clamp to reduce repeated provider 400s when the
    # conversation is extremely large (tool output can explode token counts).
    summary_transcript = simple_formatter(messages)
    try:
        preclamp_budget_chars = _compute_safe_transcript_budget_chars(
            llm_config=llm_config,
            system_prompt=system_prompt,
            include_ack=include_ack,
            safety_frac=0.6,
        )
        if len(summary_transcript) > preclamp_budget_chars:
            summary_transcript = _truncate_to_budget(
                summary_transcript,
                budget_chars=preclamp_budget_chars,
                head_frac=0.35,
                tail_frac=0.35,
            )
    except Exception:
        # Best-effort only
        pass
    logger.info(f"Summarizing {len(messages)} messages with prompt: {system_prompt}")

    if include_ack:
        logger.info(f"Summarizing with ACK for model {llm_config.model}")
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": MESSAGE_SUMMARY_REQUEST_ACK},
            {"role": "user", "content": summary_transcript},
        ]
    else:
        logger.info(f"Summarizing without ACK for model {llm_config.model}")
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_transcript},
        ]
    input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]
    # Build a local LLMConfig for v1-style summarization which uses native content and must not
    # include inner thoughts in kwargs to avoid conflicts in Anthropic formatting.
    # We also disable enable_reasoner to avoid extended thinking requirements (Anthropic requires
    # assistant messages to start with thinking blocks when extended thinking is enabled).
    summarizer_llm_config = LLMConfig(**llm_config.model_dump())
    summarizer_llm_config.put_inner_thoughts_in_kwargs = False
    summarizer_llm_config.enable_reasoner = False

    request_data = llm_client.build_request_data(AgentType.letta_v1_agent, input_messages_obj, summarizer_llm_config, tools=[])

    # Choose summarization strategy up-front.
    # For very large transcripts, prefer chunked/hierarchical summarization to avoid
    # repeated provider 400s and fallback churn.
    try:
        estimated_chars = len(summary_transcript)
    except Exception:
        estimated_chars = 0

    strategy_budget_chars = _compute_safe_transcript_budget_chars(
        llm_config=summarizer_llm_config,
        system_prompt=system_prompt,
        include_ack=include_ack,
        safety_frac=0.6,
        default_budget_chars=48000,
    )

    if estimated_chars and estimated_chars > strategy_budget_chars:
        logger.info(
            "Summarizer: transcript too large (%d chars > %d), using chunked summarization",
            estimated_chars,
            strategy_budget_chars,
        )
        return await chunked_summary(messages=messages, llm_config=llm_config, actor=actor, prompt=prompt)

    try:
        summary = await _run_summarizer_request(
            llm_client=llm_client,
            summarizer_llm_config=summarizer_llm_config,
            request_data=request_data,
            input_messages_obj=input_messages_obj,
        )
    except Exception as e:
        # handle LLM error (likely a context window exceeded error)
        try:
            raise llm_client.handle_llm_error(e)
        except ContextWindowExceededError as context_error:
            logger.warning(f"Context window exceeded during summarization. Applying clamping fallbacks. Original error: {context_error}")

            # Fallback A: rebuild transcript with clamped tool returns to shrink payload
            summary_transcript = simple_formatter(
                messages,
                tool_return_truncation_chars=TOOL_RETURN_TRUNCATION_CHARS,
            )
            # Avoid logging full payloads at INFO (can be extremely large and contain sensitive data)
            logger.debug("Summarization payload prepared (keys=%s, model=%s)", list(request_data.keys()), request_data.get("model"))

            if include_ack:
                logger.info(f"Fallback summarization with ACK for model {llm_config.model}")
                input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": MESSAGE_SUMMARY_REQUEST_ACK},
                    {"role": "user", "content": summary_transcript},
                ]
            else:
                logger.info(f"Fallback summarization without ACK for model {llm_config.model}")
                input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": summary_transcript},
                ]
            input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]

            request_data = llm_client.build_request_data(
                AgentType.letta_v1_agent,
                input_messages_obj,
                summarizer_llm_config,
                tools=[],
            )

            try:
                summary = await _run_summarizer_request(
                    llm_client=llm_client,
                    summarizer_llm_config=summarizer_llm_config,
                    request_data=request_data,
                    input_messages_obj=input_messages_obj,
                )
            except Exception as fallback_error_a:
                # Fallback B: hard-truncate the user transcript to fit a conservative char budget
                logger.warning(f"Clamped tool returns still overflowed ({fallback_error_a}). Falling back to transcript truncation.")
                logger.debug(
                    "Fallback summarization payload prepared (keys=%s, model=%s)",
                    list(request_data.keys()),
                    request_data.get("model"),
                )

                # Compute a conservative char budget for the transcript based on context window
                budget_chars = _compute_safe_transcript_budget_chars(
                    llm_config=summarizer_llm_config,
                    system_prompt=system_prompt,
                    include_ack=include_ack,
                    safety_frac=0.6,
                    default_budget_chars=48000,
                )

                truncated_transcript = _truncate_to_budget(
                    summary_transcript,
                    budget_chars=budget_chars,
                    head_frac=0.3,
                    tail_frac=0.3,
                )

                if include_ack:
                    input_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": MESSAGE_SUMMARY_REQUEST_ACK},
                        {"role": "user", "content": truncated_transcript},
                    ]
                else:
                    input_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": truncated_transcript},
                    ]
                input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]

                request_data = llm_client.build_request_data(
                    AgentType.letta_v1_agent,
                    input_messages_obj,
                    summarizer_llm_config,
                    tools=[],
                )
                try:
                    summary = await _run_summarizer_request(
                        llm_client=llm_client,
                        summarizer_llm_config=summarizer_llm_config,
                        request_data=request_data,
                        input_messages_obj=input_messages_obj,
                    )
                except Exception as fallback_error_b:
                    # Fallback C: Use chunked/hierarchical summarization
                    logger.warning(f"Transcript truncation fallback failed ({fallback_error_b}). Falling back to chunked summarization.")
                    
                    try:
                        summary = await chunked_summary(
                            messages=messages,
                            llm_config=llm_config,
                            actor=actor,
                            prompt=prompt,
                        )
                        logger.info(f"Chunked summarization succeeded for {len(messages)} messages")
                    except Exception as chunked_error:
                        logger.error(f"Chunked summarization also failed: {chunked_error}. Propagating original error.")
                        raise llm_client.handle_llm_error(fallback_error_b)

    logger.info(f"Summarized {len(messages)}: {summary}")

    return summary


# ---------------------------------------------------------------------------
# Chunked / Hierarchical Summarization
# ---------------------------------------------------------------------------


def _estimate_message_chars(messages: List[Message]) -> int:
    """Estimate the character count of messages when formatted as a transcript."""
    return len(simple_formatter(messages, tool_return_truncation_chars=TOOL_RETURN_TRUNCATION_CHARS))


def _compute_chunk_budget(llm_config: LLMConfig) -> int:
    """Compute a safe character budget for each chunk based on model context window."""
    try:
        # Use 50% of context window, assuming ~4 chars per token
        # This is conservative to leave room for system prompt, output, etc.
        budget = int(llm_config.context_window * 0.5 * 4)
        return max(10000, min(budget, 100000))  # Clamp between 10k and 100k chars
    except Exception:
        return DEFAULT_CHUNK_CHAR_BUDGET


def _split_messages_into_chunks(
    messages: List[Message],
    chunk_char_budget: int,
) -> List[List[Message]]:
    """
    Split messages into chunks that fit within the character budget.
    
    Tries to split on natural boundaries (user messages) to maintain context.
    """
    if not messages:
        return []
    
    chunks: List[List[Message]] = []
    current_chunk: List[Message] = []
    current_chars = 0
    
    for msg in messages:
        msg_chars = len(simple_formatter([msg], tool_return_truncation_chars=TOOL_RETURN_TRUNCATION_CHARS))
        
        # If adding this message would exceed budget and we have content, start new chunk
        if current_chars + msg_chars > chunk_char_budget and current_chunk:
            # Try to find a good split point (user message boundary)
            # If the last message is not a user message, include it in current chunk anyway
            chunks.append(current_chunk)
            current_chunk = [msg]
            current_chars = msg_chars
        else:
            current_chunk.append(msg)
            current_chars += msg_chars
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


@trace_method
async def _summarize_single_chunk(
    messages: List[Message],
    llm_config: LLMConfig,
    actor: User,
    prompt: str | None = None,
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """
    Summarize a single chunk of messages.
    
    This is a simplified version of simple_summary with aggressive truncation
    as the fallback since we know we're already working with smaller chunks.
    """
    llm_client = LLMClient.create(
        provider_type=llm_config.model_endpoint_type,
        put_inner_thoughts_first=True,
        actor=actor,
    )
    
    system_prompt = prompt or gpt_summarize.SYSTEM
    if total_chunks > 1:
        system_prompt = f"{system_prompt}\n\n(Note: This is chunk {chunk_index + 1} of {total_chunks} from a larger conversation.)"
    
    # Format with truncated tool returns from the start
    summary_transcript = simple_formatter(messages, tool_return_truncation_chars=TOOL_RETURN_TRUNCATION_CHARS)
    
    # Build a local LLMConfig for summarization
    summarizer_llm_config = _summarizer_llm_config(llm_config)
    
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summary_transcript},
    ]
    input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]
    
    async def _run_chunk_request(req_data: dict, req_messages_obj: list[Message]) -> str:
        return await _run_summarizer_request(
            llm_client=llm_client,
            summarizer_llm_config=summarizer_llm_config,
            request_data=req_data,
            input_messages_obj=req_messages_obj,
        )
    
    request_data = llm_client.build_request_data(
        AgentType.letta_v1_agent,
        input_messages_obj,
        summarizer_llm_config,
        tools=[],
    )
    
    try:
        return await _run_chunk_request(request_data, input_messages_obj)
    except Exception as e:
        # Fallback: aggressive truncation
        logger.warning(f"Chunk {chunk_index + 1}/{total_chunks} summarization failed ({e}), using aggressive truncation")
        budget_chars = _compute_safe_transcript_budget_chars(
            llm_config=summarizer_llm_config,
            system_prompt=system_prompt,
            include_ack=False,
            safety_frac=0.4,
            default_budget_chars=30000,
        )

        truncated_transcript = _truncate_to_budget(
            summary_transcript,
            budget_chars=budget_chars,
            head_frac=0.4,
            tail_frac=0.4,
        )
        
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": truncated_transcript},
        ]
        input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]
        request_data = llm_client.build_request_data(
            AgentType.letta_v1_agent,
            input_messages_obj,
            summarizer_llm_config,
            tools=[],
        )
        
        return await _run_chunk_request(request_data, input_messages_obj)


@trace_method
async def _combine_chunk_summaries(
    summaries: List[str],
    llm_config: LLMConfig,
    actor: User,
    depth: int = 0,
) -> str:
    """
    Combine multiple chunk summaries into one.
    
    If the combined summaries are still too large, recursively chunk and summarize.
    """
    if len(summaries) == 1:
        return summaries[0]
    
    if depth >= MAX_HIERARCHICAL_DEPTH:
        logger.warning(f"Max hierarchical summarization depth ({MAX_HIERARCHICAL_DEPTH}) reached, concatenating summaries")
        return "\n\n---\n\n".join(summaries)
    
    llm_client = LLMClient.create(
        provider_type=llm_config.model_endpoint_type,
        put_inner_thoughts_first=True,
        actor=actor,
    )
    
    combined_input = "\n\n---\n\n".join([f"[Summary {i+1}]\n{s}" for i, s in enumerate(summaries)])
    
    # Check if combined input fits in context
    chunk_budget = _compute_chunk_budget(llm_config)
    
    if len(combined_input) > chunk_budget:
        # Need to recursively summarize
        logger.info(f"Combined summaries ({len(combined_input)} chars) exceed budget ({chunk_budget}), recursively summarizing")
        
        # Split summaries into groups and summarize each group
        group_size = max(2, len(summaries) // 2)
        groups = [summaries[i:i + group_size] for i in range(0, len(summaries), group_size)]
        
        sub_summaries = []
        for group in groups:
            group_text = "\n\n---\n\n".join([f"[Summary]\n{s}" for s in group])
            sub_summary = await _summarize_text_directly(group_text, llm_config, actor, CHUNK_COMBINE_PROMPT)
            sub_summaries.append(sub_summary)
        
        return await _combine_chunk_summaries(sub_summaries, llm_config, actor, depth + 1)
    
    # Combine summaries with a single LLM call
    summarizer_llm_config = _summarizer_llm_config(llm_config)
    
    input_messages = [
        {"role": "system", "content": CHUNK_COMBINE_PROMPT},
        {"role": "user", "content": combined_input},
    ]
    input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]
    
    request_data = llm_client.build_request_data(
        AgentType.letta_v1_agent,
        input_messages_obj,
        summarizer_llm_config,
        tools=[],
    )
    
    try:
        return await _run_summarizer_request(
            llm_client=llm_client,
            summarizer_llm_config=summarizer_llm_config,
            request_data=request_data,
            input_messages_obj=input_messages_obj,
        )
    except Exception as e:
        logger.error(f"Failed to combine chunk summaries: {e}")
        # Fallback: just concatenate
        return "\n\n---\n\n".join(summaries)


async def _summarize_text_directly(
    text: str,
    llm_config: LLMConfig,
    actor: User,
    prompt: str,
) -> str:
    """Helper to summarize raw text directly."""
    llm_client = LLMClient.create(
        provider_type=llm_config.model_endpoint_type,
        put_inner_thoughts_first=True,
        actor=actor,
    )
    
    summarizer_llm_config = _summarizer_llm_config(llm_config)
    
    input_messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    input_messages_obj = [simple_message_wrapper(msg) for msg in input_messages]
    
    request_data = llm_client.build_request_data(
        AgentType.letta_v1_agent,
        input_messages_obj,
        summarizer_llm_config,
        tools=[],
    )
    
    return await _run_summarizer_request(
        llm_client=llm_client,
        summarizer_llm_config=summarizer_llm_config,
        request_data=request_data,
        input_messages_obj=input_messages_obj,
    )


@trace_method
async def chunked_summary(
    messages: List[Message],
    llm_config: LLMConfig,
    actor: User,
    prompt: str | None = None,
) -> str:
    """
    Summarize a large list of messages using chunked/hierarchical summarization.
    
    This is the main entry point for summarizing conversations that may exceed
    the model's context window. It:
    1. Splits messages into manageable chunks
    2. Summarizes each chunk in parallel
    3. Combines chunk summaries (recursively if needed)
    
    Args:
        messages: List of messages to summarize
        llm_config: LLM configuration for the summarizer model
        actor: User making the request
        prompt: Optional custom summarization prompt
    
    Returns:
        A single summary string covering all messages
    """
    if not messages:
        return ""
    
    # Compute chunk budget based on model
    chunk_budget = _compute_chunk_budget(llm_config)
    
    # Split messages into chunks
    chunks = _split_messages_into_chunks(messages, chunk_budget)
    
    logger.info(f"Chunked summarization: {len(messages)} messages -> {len(chunks)} chunks (budget={chunk_budget} chars)")
    
    if len(chunks) == 1:
        # Single chunk - just use regular summarization
        return await _summarize_single_chunk(
            chunks[0],
            llm_config,
            actor,
            prompt,
            chunk_index=0,
            total_chunks=1,
        )
    
    # Summarize chunks in parallel
    chunk_tasks = [
        _summarize_single_chunk(
            chunk,
            llm_config,
            actor,
            prompt,
            chunk_index=i,
            total_chunks=len(chunks),
        )
        for i, chunk in enumerate(chunks)
    ]
    
    chunk_summaries = await asyncio.gather(*chunk_tasks, return_exceptions=True)
    
    # Filter out failures and log them
    valid_summaries = []
    for i, result in enumerate(chunk_summaries):
        if isinstance(result, Exception):
            logger.error(f"Chunk {i + 1}/{len(chunks)} failed: {result}")
            # Add a placeholder for failed chunks
            valid_summaries.append(f"[Chunk {i + 1}: summarization failed]")
        else:
            valid_summaries.append(result)
    
    # Combine summaries
    return await _combine_chunk_summaries(valid_summaries, llm_config, actor)


def format_transcript(messages: List[Message], include_system: bool = False) -> List[str]:
    """
    Turn a list of Message objects into a human-readable transcript.

    Args:
        messages: List of Message instances, in chronological order.
        include_system: If True, include system-role messages. Defaults to False.

    Returns:
        A single string, e.g.:
          user: Hey, my name is Matt.
          assistant: Hi Matt! It's great to meet you...
          user: What's the weather like? ...
          assistant: The weather in Las Vegas is sunny...
    """
    lines = []
    for msg in messages:
        role = msg.role.value  # e.g. 'user', 'assistant', 'system', 'tool'
        # skip system messages by default
        if role == "system" and not include_system:
            continue

        # 1) Try plain content
        if msg.content:
            # Skip tool messages where the name is "send_message"
            if msg.role == MessageRole.tool and msg.name == DEFAULT_MESSAGE_TOOL:
                continue

            text = "".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()
            # Append a compact placeholder for any images
            image_count = len([c for c in msg.content if isinstance(c, ImageContent)])
            if image_count > 0:
                placeholder = "[Image omitted]" if image_count == 1 else f"[{image_count} images omitted]"
                text = (text + (" " if text else "")) + placeholder

        # 2) Otherwise, try extracting from function calls
        elif msg.tool_calls:
            parts = []
            for call in msg.tool_calls:
                args_str = call.function.arguments
                if call.function.name == DEFAULT_MESSAGE_TOOL:
                    try:
                        args = json.loads(args_str)
                        # pull out a "message" field if present
                        parts.append(args.get(DEFAULT_MESSAGE_TOOL_KWARG, args_str))
                    except json.JSONDecodeError:
                        parts.append(args_str)
                else:
                    parts.append(args_str)
            text = " ".join(parts).strip()

        else:
            # nothing to show for this message
            continue

        lines.append(f"{role}: {text}")

    return lines
