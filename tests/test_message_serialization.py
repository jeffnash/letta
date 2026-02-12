from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall as OpenAIToolCall
from openai.types.chat.chat_completion_message_tool_call import Function as OpenAIFunction

from letta.llm_api.openai_client import fill_image_content_in_responses_input
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn, sanitize_tool_identifier


def _user_message_with_image_first(text: str) -> Message:
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    return Message(role=MessageRole.user, content=[image, TextContent(text=text)])


def test_to_openai_responses_dicts_handles_image_first_content():
    message = _user_message_with_image_first("hello world")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert any(part["type"] == "input_text" and part["text"] == "hello world" for part in parts)
    assert any(part["type"] == "input_image" for part in parts)


def test_fill_image_content_in_responses_input_includes_image_parts():
    message = _user_message_with_image_first("describe image")
    serialized = Message.to_openai_responses_dicts_from_list([message])
    rewritten = fill_image_content_in_responses_input(serialized, [message])
    assert rewritten == serialized


def test_to_openai_responses_dicts_handles_image_only_content():
    image = ImageContent(source=Base64Image(media_type="image/png", data="dGVzdA=="))
    message = Message(role=MessageRole.user, content=[image])
    serialized = Message.to_openai_responses_dicts_from_list([message])
    parts = serialized[0]["content"]
    assert parts[0]["type"] == "input_image"


def test_to_anthropic_dict_sanitizes_invalid_tool_use_id():
    invalid_tool_id = "next_message_not_tool_response)"
    message = Message(
        role=MessageRole.assistant,
        model="claude-sonnet-4-5",
        tool_calls=[
            OpenAIToolCall(
                id=invalid_tool_id,
                type="function",
                function=OpenAIFunction(name="Read", arguments="{}"),
            )
        ],
    )

    anthropic = message.to_anthropic_dict(current_model="claude-sonnet-4-5")
    tool_use_block = next(block for block in anthropic["content"] if block.get("type") == "tool_use")
    assert tool_use_block["id"] == sanitize_tool_identifier(invalid_tool_id)
    assert tool_use_block["id"] != invalid_tool_id


def test_to_anthropic_dict_sanitizes_tool_result_id_consistently():
    invalid_tool_id = "next_message_not_tool_response)"
    message = Message(
        role=MessageRole.tool,
        model="claude-sonnet-4-5",
        tool_returns=[
            ToolReturn(
                tool_call_id=invalid_tool_id,
                status="success",
                func_response='{"message":"ok","status":"OK"}',
            )
        ],
    )

    anthropic = message.to_anthropic_dict(current_model="claude-sonnet-4-5")
    tool_result_block = anthropic["content"][0]
    assert tool_result_block["tool_use_id"] == sanitize_tool_identifier(invalid_tool_id)


def test_to_anthropic_dict_sanitized_id_collision_stays_unique():
    raw_id_1 = "parallel-call)"
    raw_id_2 = "parallel-call!"

    assistant_message = Message(
        role=MessageRole.assistant,
        model="claude-sonnet-4-5",
        tool_calls=[
            OpenAIToolCall(
                id=raw_id_1,
                type="function",
                function=OpenAIFunction(name="Read", arguments="{}"),
            ),
            OpenAIToolCall(
                id=raw_id_2,
                type="function",
                function=OpenAIFunction(name="Write", arguments="{}"),
            ),
        ],
    )

    assistant_anthropic = assistant_message.to_anthropic_dict(current_model="claude-sonnet-4-5")
    tool_use_blocks = [block for block in assistant_anthropic["content"] if block.get("type") == "tool_use"]
    tool_use_ids = [block["id"] for block in tool_use_blocks]

    assert len(tool_use_ids) == 2
    assert len(set(tool_use_ids)) == 2
    assert all(id_value.startswith("parallel-call_") for id_value in tool_use_ids)

    tool_message = Message(
        role=MessageRole.tool,
        model="claude-sonnet-4-5",
        tool_returns=[
            ToolReturn(tool_call_id=raw_id_1, status="success", func_response='{"message":"ok1","status":"OK"}'),
            ToolReturn(tool_call_id=raw_id_2, status="success", func_response='{"message":"ok2","status":"OK"}'),
        ],
    )
    tool_anthropic = tool_message.to_anthropic_dict(current_model="claude-sonnet-4-5")
    tool_result_ids = [block["tool_use_id"] for block in tool_anthropic["content"]]

    assert len(tool_result_ids) == 2
    assert len(set(tool_result_ids)) == 2
    assert set(tool_result_ids) == set(tool_use_ids)
