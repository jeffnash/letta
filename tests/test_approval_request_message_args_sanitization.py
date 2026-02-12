import json

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
    Function as OpenAIFunction,
)

from letta.schemas.enums import MessageRole
from letta.server.rest_api.utils import create_approval_request_message_from_llm_response


MALFORMED_TOOL_ARGS_KEY = "__letta_malformed_tool_args"


def _tool_call(call_id: str, name: str, arguments: str) -> OpenAIToolCall:
    return OpenAIToolCall(
        id=call_id,
        function=OpenAIFunction(name=name, arguments=arguments),
        type="function",
    )


def test_sanitizes_invalid_requested_tool_call_arguments():
    requested = [
        _tool_call("call_bad", "memory", '{"command":"str_replace","new_string":"truncated'),
    ]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="gpt-4o-mini",
        requested_tool_calls=requested,
        step_id="step-1",
        run_id="run-1",
    )

    assert len(messages) == 1
    approval_message = messages[0]
    assert approval_message.role == MessageRole.approval
    assert approval_message.tool_calls is not None
    assert approval_message.tool_calls[0].id == "call_bad"
    assert json.loads(approval_message.tool_calls[0].function.arguments) == {MALFORMED_TOOL_ARGS_KEY: True}


def test_sanitizes_invalid_allowed_tool_call_arguments():
    requested = [_tool_call("call_req", "Read", "{}")]
    allowed = [
        _tool_call("call_allowed_bad", "Bash", '{"command":"npm test'),
    ]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="gpt-4o-mini",
        requested_tool_calls=requested,
        allowed_tool_calls=allowed,
        step_id="step-2",
        run_id="run-2",
    )

    assert len(messages) == 2
    assistant_message = messages[0]
    approval_message = messages[1]

    assert assistant_message.role == MessageRole.assistant
    assert assistant_message.tool_calls is not None
    assert assistant_message.tool_calls[0].id == "call_allowed_bad"
    assert json.loads(assistant_message.tool_calls[0].function.arguments) == {MALFORMED_TOOL_ARGS_KEY: True}

    assert approval_message.role == MessageRole.approval
    assert approval_message.tool_calls is not None
    assert approval_message.tool_calls[0].id == "call_req"


def test_preserves_valid_tool_call_arguments_semantically():
    valid_args = '{"command":"search","query":"tool call"}'
    requested = [_tool_call("call_valid", "memory", valid_args)]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="gpt-4o-mini",
        requested_tool_calls=requested,
        step_id="step-3",
        run_id="run-3",
    )

    approval_message = messages[0]
    persisted_args = approval_message.tool_calls[0].function.arguments

    assert json.loads(persisted_args) == json.loads(valid_args)
