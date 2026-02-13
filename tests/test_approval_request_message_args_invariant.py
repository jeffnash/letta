import json

from letta.schemas.enums import MessageRole
from letta.server.rest_api.utils import create_approval_request_message_from_llm_response
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIToolCall,
    Function as OpenAIFunction,
)


def _tool_call(tool_call_id: str, name: str, arguments: str) -> OpenAIToolCall:
    return OpenAIToolCall(
        id=tool_call_id,
        function=OpenAIFunction(name=name, arguments=arguments),
        type="function",
    )


def test_approval_request_sanitizes_invalid_requested_tool_args():
    requested = [_tool_call("call_req_1", "Edit", '{"file_path":"a.py"')]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="claude-sonnet-4-5",
        requested_tool_calls=requested,
        allowed_tool_calls=[],
        run_id="run-test",
        step_id="step-test",
    )

    assert len(messages) == 1
    approval = messages[0]
    assert approval.role == MessageRole.approval
    assert approval.tool_calls is not None
    assert approval.tool_calls[0].function.arguments == "{}"


def test_approval_request_sanitizes_invalid_allowed_tool_args():
    requested = [_tool_call("call_req_1", "Read", '{"file_path":"a.py"}')]
    allowed = [_tool_call("call_allow_1", "LS", '{"path":"src"')]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="claude-sonnet-4-5",
        requested_tool_calls=requested,
        allowed_tool_calls=allowed,
        run_id="run-test",
        step_id="step-test",
    )

    assert len(messages) == 2
    assistant = messages[0]
    approval = messages[1]
    assert assistant.role == MessageRole.assistant
    assert approval.role == MessageRole.approval
    assert assistant.tool_calls is not None
    assert assistant.tool_calls[0].function.arguments == "{}"


def test_approval_request_keeps_valid_json_object_args():
    requested = [_tool_call("call_req_1", "Edit", '{"file_path":"a.py","old":"x","new":"y"}')]

    messages = create_approval_request_message_from_llm_response(
        agent_id="agent-test",
        model="claude-sonnet-4-5",
        requested_tool_calls=requested,
        allowed_tool_calls=[],
        run_id="run-test",
        step_id="step-test",
    )

    approval = messages[0]
    assert approval.tool_calls is not None
    persisted_args = approval.tool_calls[0].function.arguments
    assert json.loads(persisted_args) == {"file_path": "a.py", "old": "x", "new": "y"}
