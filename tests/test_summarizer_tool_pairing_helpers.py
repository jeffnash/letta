from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIFunction

from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message, ToolReturn
from letta.services.summarizer.summarizer_sliding_window import (
    _build_assistant_index_to_tool_response_indices,
    _find_earliest_safe_cutoff_for_tool_group,
    _is_cutoff_safe,
    _build_tool_call_id_to_assistant_index,
)


def _tool_call(call_id: str, name: str = "Bash") -> OpenAIToolCall:
    return OpenAIToolCall(
        id=call_id,
        type="function",
        function=OpenAIFunction(name=name, arguments="{}"),
    )


def test_tool_call_index_includes_approval_messages():
    messages = [
        Message(
            role=MessageRole.approval,
            content=[TextContent(text="approval required")],
            tool_calls=[_tool_call("call_approval")],
        ),
        Message(
            role=MessageRole.tool,
            content=[TextContent(text="approved")],
            tool_call_id="call_approval",
            tool_returns=[
                ToolReturn(
                    tool_call_id="call_approval",
                    status="success",
                    func_response="approved",
                )
            ],
        ),
    ]

    tool_call_to_idx = _build_tool_call_id_to_assistant_index(messages)
    assert tool_call_to_idx == {"call_approval": 0}

    response_idx = _build_assistant_index_to_tool_response_indices(messages, tool_call_to_idx)
    assert response_idx == {0: [1]}


def test_tool_response_index_uses_all_tool_returns_call_ids():
    messages = [
        Message(
            role=MessageRole.assistant,
            content=[TextContent(text="running tools")],
            tool_calls=[_tool_call("call_a", "Read"), _tool_call("call_b", "Write")],
        ),
        Message(
            role=MessageRole.tool,
            content=[TextContent(text="multi-return")],
            tool_call_id="call_a",
            tool_returns=[
                ToolReturn(tool_call_id="call_a", status="success", func_response="A"),
                ToolReturn(tool_call_id="call_b", status="success", func_response="B"),
            ],
        ),
    ]

    tool_call_to_idx = _build_tool_call_id_to_assistant_index(messages)
    assert tool_call_to_idx == {"call_a": 0, "call_b": 0}

    response_idx = _build_assistant_index_to_tool_response_indices(messages, tool_call_to_idx)
    # The same tool message index should be associated once, even with multiple tool_returns.
    assert response_idx == {0: [1]}


def test_find_earliest_safe_cutoff_uses_tool_returns_when_tool_call_id_missing():
    messages = [
        Message(
            role=MessageRole.assistant,
            content=[TextContent(text="calling tool")],
            tool_calls=[_tool_call("call_from_returns")],
        ),
        Message(
            role=MessageRole.tool,
            content=[TextContent(text="result")],
            tool_call_id=None,
            tool_returns=[
                ToolReturn(
                    tool_call_id="call_from_returns",
                    status="success",
                    func_response="ok",
                )
            ],
        ),
    ]

    tool_call_to_idx = _build_tool_call_id_to_assistant_index(messages)
    response_idx = _build_assistant_index_to_tool_response_indices(messages, tool_call_to_idx)

    # Candidate 1 would keep only the tool response and orphan its assistant.
    adjusted = _find_earliest_safe_cutoff_for_tool_group(
        candidate_idx=1,
        messages=messages,
        tool_call_id_to_assistant_idx=tool_call_to_idx,
        assistant_idx_to_tool_indices=response_idx,
    )
    assert adjusted == 0


def test_is_cutoff_safe_detects_tool_returns_only_orphaning():
    messages = [
        Message(
            role=MessageRole.assistant,
            content=[TextContent(text="calling tool")],
            tool_calls=[_tool_call("call_from_returns")],
        ),
        Message(
            role=MessageRole.tool,
            content=[TextContent(text="result")],
            tool_call_id=None,
            tool_returns=[
                ToolReturn(
                    tool_call_id="call_from_returns",
                    status="success",
                    func_response="ok",
                )
            ],
        ),
    ]

    tool_call_to_idx = _build_tool_call_id_to_assistant_index(messages)
    response_idx = _build_assistant_index_to_tool_response_indices(messages, tool_call_to_idx)

    # Keeping from index 1 would orphan the response from its assistant and must be unsafe.
    assert _is_cutoff_safe(
        cutoff_idx=1,
        messages=messages,
        tool_call_id_to_assistant_idx=tool_call_to_idx,
        assistant_idx_to_tool_indices=response_idx,
    ) is False
