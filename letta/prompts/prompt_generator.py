from datetime import datetime
from typing import List, Literal, Optional

from letta.log import get_logger

logger = get_logger(__name__)

from letta.constants import IN_CONTEXT_MEMORY_KEYWORD
from letta.helpers import ToolRulesSolver
from letta.helpers.datetime_helpers import format_datetime
from letta.otel.tracing import trace_method
from letta.schemas.memory import Memory


class PromptGenerator:
    # TODO: This code is kind of wonky and deserves a rewrite
    @trace_method
    @staticmethod
    def compile_memory_metadata_block(
        memory_edit_timestamp: datetime,
        timezone: str,
        previous_message_count: int = 0,
        archival_memory_size: Optional[int] = 0,
        archive_tags: Optional[List[str]] = None,
        conversation_start_date: Optional[datetime] = None,
    ) -> str:
        """
        Generate a memory metadata block for the agent's system prompt.

        This creates a minimal, static metadata section to maximize prompt caching.
        Dynamic values (current time, message counts) are intentionally excluded
        to avoid cache invalidation on every request. The agent can use tools
        (e.g., Bash with `date`) to get current time when needed.

        Args:
            memory_edit_timestamp: When memory blocks were last modified (unused, kept for API compatibility)
            timezone: The timezone to use for formatting timestamps (e.g., 'America/Los_Angeles')
            previous_message_count: Number of messages in recall memory (unused, kept for API compatibility)
            archival_memory_size: Number of items in archival memory (unused, kept for API compatibility)
            archive_tags: List of unique tags available in archival memory (unused, kept for API compatibility)
            conversation_start_date: When the conversation/agent was created (fixed timestamp for caching)

        Returns:
            A formatted string containing the memory metadata block with XML-style tags

        Example Output:
            <memory_metadata>
            - Conversation started: January 15, 2024
            - Use Bash with `date` to check current date/time if needed
            - Use conversation_search to find past discussions
            </memory_metadata>
        """
        metadata_lines = ["<memory_metadata>"]

        # Only include conversation start date if provided (fixed, cacheable)
        if conversation_start_date:
            start_date_str = format_datetime(conversation_start_date, timezone)
            metadata_lines.append(f"- Conversation started: {start_date_str}")

        # Static instructions that don't change
        metadata_lines.append("- Use Bash with `date` to check current date/time if needed")
        metadata_lines.append("- Use conversation_search to find past discussions")

        metadata_lines.append("</memory_metadata>")
        memory_metadata_block = "\n".join(metadata_lines)
        return memory_metadata_block

    @staticmethod
    def safe_format(template: str, variables: dict) -> str:
        """
        Safely formats a template string, preserving empty {} and {unknown_vars}
        while substituting known variables.

        If we simply use {} in format_map, it'll be treated as a positional field
        """
        # First escape any empty {} by doubling them
        escaped = template.replace("{}", "{{}}")

        # Now use format_map with our custom mapping
        return escaped.format_map(PreserveMapping(variables))

    @trace_method
    @staticmethod
    def get_system_message_from_compiled_memory(
        system_prompt: str,
        memory_with_sources: str,
        in_context_memory_last_edit: datetime,  # TODO move this inside of BaseMemory?
        timezone: str,
        user_defined_variables: Optional[dict] = None,
        append_icm_if_missing: bool = True,
        template_format: Literal["f-string", "mustache"] = "f-string",
        previous_message_count: int = 0,
        archival_memory_size: int = 0,
        archive_tags: Optional[List[str]] = None,
        conversation_start_date: Optional[datetime] = None,
        exclude_memory: bool = False,
    ) -> str:
        """Prepare the final/full system message that will be fed into the LLM API

        The base system message may be templated, in which case we need to render the variables.

        The following are reserved variables:
        - CORE_MEMORY: the in-context memory of the LLM

        Args:
            exclude_memory: If True, memory is excluded (sent separately as a context message).
                           The {CORE_MEMORY} placeholder will be replaced with empty string or
                           a minimal note about memory being provided separately.
        """
        if user_defined_variables is not None:
            # TODO eventually support the user defining their own variables to inject
            raise NotImplementedError
        else:
            variables = {}

        # Add the protected memory variable
        if IN_CONTEXT_MEMORY_KEYWORD in variables:
            raise ValueError(f"Found protected variable '{IN_CONTEXT_MEMORY_KEYWORD}' in user-defined vars: {str(user_defined_variables)}")
        else:
            if exclude_memory:
                # When memory is excluded, don't add memory or metadata to system prompt
                # Memory will be sent as a separate developer/user message
                full_memory_string = ""
            else:
                # TODO should this all put into the memory.__repr__ function?
                memory_metadata_string = PromptGenerator.compile_memory_metadata_block(
                    memory_edit_timestamp=in_context_memory_last_edit,
                    previous_message_count=previous_message_count,
                    archival_memory_size=archival_memory_size,
                    timezone=timezone,
                    archive_tags=archive_tags,
                    conversation_start_date=conversation_start_date,
                )

                full_memory_string = memory_with_sources + "\n\n" + memory_metadata_string

            # Add to the variables list to inject
            variables[IN_CONTEXT_MEMORY_KEYWORD] = full_memory_string

        if template_format == "f-string":
            memory_variable_string = "{" + IN_CONTEXT_MEMORY_KEYWORD + "}"

            # Catch the special case where the system prompt is unformatted
            if append_icm_if_missing and not exclude_memory:
                if memory_variable_string not in system_prompt:
                    # In this case, append it to the end to make sure memory is still injected
                    # logger.warning(f"{IN_CONTEXT_MEMORY_KEYWORD} variable was missing from system prompt, appending instead")
                    system_prompt += "\n\n" + memory_variable_string

            # render the variables using the built-in templater
            try:
                if user_defined_variables:
                    formatted_prompt = PromptGenerator.safe_format(system_prompt, variables)
                else:
                    formatted_prompt = system_prompt.replace(memory_variable_string, full_memory_string)
            except Exception as e:
                raise ValueError(f"Failed to format system prompt - {str(e)}. System prompt value:\n{system_prompt}")

        else:
            # TODO support for mustache
            raise NotImplementedError(template_format)

        return formatted_prompt

    @trace_method
    @staticmethod
    async def compile_system_message_async(
        system_prompt: str,
        in_context_memory: Memory,
        in_context_memory_last_edit: datetime,  # TODO move this inside of BaseMemory?
        timezone: str,
        user_defined_variables: Optional[dict] = None,
        append_icm_if_missing: bool = True,
        template_format: Literal["f-string", "mustache"] = "f-string",
        previous_message_count: int = 0,
        archival_memory_size: int = 0,
        tool_rules_solver: Optional[ToolRulesSolver] = None,
        sources: Optional[List] = None,
        max_files_open: Optional[int] = None,
        llm_config: Optional[object] = None,
        conversation_start_date: Optional[datetime] = None,
        exclude_memory: bool = False,
    ) -> str:
        """Compile the system message for the agent.

        Args:
            exclude_memory: If True, memory blocks are NOT included in the system prompt.
                           Use this when memory_mode='context_message' and memory is
                           sent as a separate developer/user message instead.
        """
        tool_constraint_block = None
        if tool_rules_solver is not None:
            tool_constraint_block = tool_rules_solver.compile_tool_rule_prompts()

        if user_defined_variables is not None:
            # TODO eventually support the user defining their own variables to inject
            raise NotImplementedError
        else:
            pass

        # When exclude_memory is True, don't include memory in the system prompt
        # (memory will be sent as a separate context message for better caching)
        if exclude_memory:
            # Compile only tool rules and sources, without memory blocks
            memory_with_sources = ""
            if tool_constraint_block:
                memory_with_sources = f"\n\n<tool_usage_rules>\n{tool_constraint_block.description or ''}\n\n{tool_constraint_block.value or ''}\n</tool_usage_rules>"
            if sources:
                # Still need to render directories/sources
                from io import StringIO

                s = StringIO()
                in_context_memory._render_directories_common(s, sources, max_files_open)
                memory_with_sources += s.getvalue()
        else:
            memory_with_sources = in_context_memory.compile(
                tool_usage_rules=tool_constraint_block, sources=sources, max_files_open=max_files_open, llm_config=llm_config
            )

        return PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt=system_prompt,
            memory_with_sources=memory_with_sources,
            in_context_memory_last_edit=in_context_memory_last_edit,
            timezone=timezone,
            user_defined_variables=user_defined_variables,
            append_icm_if_missing=append_icm_if_missing,
            template_format=template_format,
            previous_message_count=previous_message_count,
            archival_memory_size=archival_memory_size,
            conversation_start_date=conversation_start_date,
            exclude_memory=exclude_memory,
        )
