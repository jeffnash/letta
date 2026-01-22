from letta.helpers.datetime_helpers import get_utc_time
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message import ApprovalReturn
from letta.schemas.letta_message_content import Base64Image, ImageContent, TextContent
from letta.schemas.message import Message, ToolReturn


def test_tool_return_message_preserves_multimodal_tool_return_payload():
    raw_tool_return = [
        TextContent(text="hello"),
        ImageContent(source=Base64Image(media_type="image/png", data="aGVsbG8=")),
    ]

    internal_tool_return = ToolReturn(
        tool_call_id="call_1",
        status="success",
        func_response='{"status":"OK","message":"hello","time":"2026-01-22T00:00:00Z"}',
        tool_return=raw_tool_return,
    )

    msg = Message(
        role=MessageRole.tool,
        content=[TextContent(text=internal_tool_return.func_response)],
        tool_call_id="call_1",
        tool_returns=[internal_tool_return],
        created_at=get_utc_time(),
    )

    tool_return_message = msg._convert_tool_return_message()
    assert tool_return_message.tool_return == "hello\n[image x1]"
    assert tool_return_message.tool_returns is not None
    assert len(tool_return_message.tool_returns) == 1
    assert isinstance(tool_return_message.tool_returns[0].tool_return, list)
    assert tool_return_message.tool_returns[0].tool_return[0].type == "text"
    assert tool_return_message.tool_returns[0].tool_return[1].type == "image"


def test_approval_response_preserves_multimodal_tool_return_payload():
    """Test that approval responses with image payloads correctly preserve multimodal content."""
    raw_tool_return = [
        TextContent(text="Image generated successfully"),
        ImageContent(source=Base64Image(media_type="image/png", data="aW1hZ2VkYXRh")),
    ]

    # Create an internal ToolReturn with multimodal payload
    internal_tool_return = ToolReturn(
        tool_call_id="call_approval_1",
        status="success",
        func_response='{"status":"OK","message":"Image generated successfully","time":"2026-01-22T00:00:00Z"}',
        tool_return=raw_tool_return,
        stdout=["Log output"],
        stderr=None,
    )

    # Create an approval message with the tool return
    msg = Message(
        role=MessageRole.approval,
        content=None,
        tool_calls=None,
        approvals=[
            ApprovalReturn(tool_call_id="call_approval_1", approve=True, reason="Approved"),
            internal_tool_return,
        ],
        created_at=get_utc_time(),
    )

    letta_messages = msg.to_letta_messages()
    assert len(letta_messages) == 1

    approval_response = letta_messages[0]
    assert approval_response.approvals is not None
    assert len(approval_response.approvals) == 2

    # First approval is the ApprovalReturn (unchanged)
    assert approval_response.approvals[0].approve is True

    # Second approval is the converted LettaToolReturn - should preserve multimodal content
    tool_return_approval = approval_response.approvals[1]
    assert tool_return_approval.tool_call_id == "call_approval_1"
    assert tool_return_approval.status == "success"
    assert tool_return_approval.stdout == ["Log output"]
    assert tool_return_approval.stderr is None

    # The tool_return should be the raw multimodal payload, not parsed from func_response
    assert isinstance(tool_return_approval.tool_return, list)
    assert len(tool_return_approval.tool_return) == 2
    assert tool_return_approval.tool_return[0].type == "text"
    assert tool_return_approval.tool_return[0].text == "Image generated successfully"
    assert tool_return_approval.tool_return[1].type == "image"
    assert tool_return_approval.tool_return[1].source.data == "aW1hZ2VkYXRh"


def test_approval_response_falls_back_to_func_response_when_tool_return_missing():
    """Test that approval responses fall back to parsing func_response when tool_return is missing."""
    # Create an internal ToolReturn without the tool_return field (legacy behavior)
    internal_tool_return = ToolReturn(
        tool_call_id="call_legacy_1",
        status="success",
        func_response='{"status":"OK","message":"Legacy response","time":"2026-01-22T00:00:00Z"}',
        tool_return=None,
    )

    msg = Message(
        role=MessageRole.approval,
        content=None,
        tool_calls=None,
        approvals=[internal_tool_return],
        created_at=get_utc_time(),
    )

    letta_messages = msg.to_letta_messages()
    assert len(letta_messages) == 1

    approval_response = letta_messages[0]
    assert approval_response.approvals is not None
    assert len(approval_response.approvals) == 1

    tool_return_approval = approval_response.approvals[0]
    assert tool_return_approval.tool_call_id == "call_legacy_1"
    # Status should be parsed from func_response
    assert tool_return_approval.status == "success"
    # tool_return should be parsed message from func_response
    assert tool_return_approval.tool_return == "Legacy response"


def test_approval_response_handles_none_func_response():
    """Test that approval responses handle None func_response gracefully."""
    internal_tool_return = ToolReturn(
        tool_call_id="call_none_1",
        status="error",
        func_response=None,
        tool_return=None,
    )

    msg = Message(
        role=MessageRole.approval,
        content=None,
        tool_calls=None,
        approvals=[internal_tool_return],
        created_at=get_utc_time(),
    )

    letta_messages = msg.to_letta_messages()
    assert len(letta_messages) == 1

    approval_response = letta_messages[0]
    assert approval_response.approvals is not None
    assert len(approval_response.approvals) == 1

    tool_return_approval = approval_response.approvals[0]
    assert tool_return_approval.tool_call_id == "call_none_1"
    assert tool_return_approval.status == "error"
    # When both tool_return and func_response are None, defaults to empty string
    assert tool_return_approval.tool_return == ""

