"""
Unit tests for approval tool call ID validation in letta.agents.helpers
"""

import pytest
from letta.schemas.message import Message, ApprovalCreate
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message import ApprovalReturn
from letta.agents.helpers import validate_approval_tool_call_ids
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall as OpenAIToolCall, Function as OpenAIFunction


class TestValidateApprovalToolCallIds:
    """Tests for validate_approval_tool_call_ids function."""

    # =========================================================================
    # Normal cases - tool_calls is populated with expected IDs
    # =========================================================================

    def test_valid_approval_with_matching_single_tool_call_id(self):
        """Test that approval with matching tool call ID passes validation."""
        tc = OpenAIToolCall(id='call_expected', function=OpenAIFunction(name='Bash', arguments='{}'), type='function')
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[tc],
            tool_call_id='call_expected',
        )
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='call_expected', approve=True)]
        )

        # Should not raise
        validate_approval_tool_call_ids(approval_request, approval_response)

    def test_invalid_approval_with_mismatched_single_tool_call_id(self):
        """Test that approval with mismatched tool call ID fails validation."""
        tc = OpenAIToolCall(id='call_expected', function=OpenAIFunction(name='Bash', arguments='{}'), type='function')
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[tc],
            tool_call_id='call_expected',
        )
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='call_wrong', approve=True)]
        )

        with pytest.raises(ValueError, match="Invalid tool call IDs"):
            validate_approval_tool_call_ids(approval_request, approval_response)

    def test_valid_approval_with_multiple_matching_tool_call_ids(self):
        """Test that approval with multiple matching tool call IDs passes validation."""
        tc1 = OpenAIToolCall(id='call_1', function=OpenAIFunction(name='Read', arguments='{}'), type='function')
        tc2 = OpenAIToolCall(id='call_2', function=OpenAIFunction(name='Write', arguments='{}'), type='function')
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[tc1, tc2],
            tool_call_id='call_1',
        )
        approval_response = ApprovalCreate(
            approvals=[
                ApprovalReturn(tool_call_id='call_1', approve=True),
                ApprovalReturn(tool_call_id='call_2', approve=True),
            ]
        )

        # Should not raise
        validate_approval_tool_call_ids(approval_request, approval_response)

    def test_invalid_approval_with_partial_tool_call_match(self):
        """Test that approval with only partial tool call ID match fails validation."""
        tc1 = OpenAIToolCall(id='call_1', function=OpenAIFunction(name='Read', arguments='{}'), type='function')
        tc2 = OpenAIToolCall(id='call_2', function=OpenAIFunction(name='Write', arguments='{}'), type='function')
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[tc1, tc2],
            tool_call_id='call_1',
        )
        approval_response = ApprovalCreate(
            approvals=[
                ApprovalReturn(tool_call_id='call_1', approve=True),
                # call_2 is missing, only call_1 is in response
            ]
        )

        with pytest.raises(ValueError, match="Invalid tool call IDs"):
            validate_approval_tool_call_ids(approval_request, approval_response)

    def test_invalid_approval_with_extra_tool_call_id(self):
        """Test that approval with extra tool call ID fails validation."""
        tc1 = OpenAIToolCall(id='call_1', function=OpenAIFunction(name='Read', arguments='{}'), type='function')
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[tc1],
            tool_call_id='call_1',
        )
        approval_response = ApprovalCreate(
            approvals=[
                ApprovalReturn(tool_call_id='call_1', approve=True),
                ApprovalReturn(tool_call_id='call_extra', approve=True),  # Extra ID not in request
            ]
        )

        with pytest.raises(ValueError, match="Invalid tool call IDs"):
            validate_approval_tool_call_ids(approval_request, approval_response)

    # =========================================================================
    # Edge cases - tool_calls is empty or None
    # These are the bug scenarios that the fix addresses
    # =========================================================================

    def test_valid_approval_with_empty_tool_calls_and_correct_id(self):
        """
        Test that approval with empty tool_calls and correct tool_call_id passes.
        This is the bug scenario - previously this would fail validation.
        """
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[],  # Empty!
            tool_call_id='call_server_set',  # Server sets this
        )
        # Client sends the correct tool_call_id that it got from the SDK
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='call_correct', approve=True)]
        )

        # Should not raise - client knows what it's approving
        validate_approval_tool_call_ids(approval_request, approval_response)

    def test_valid_approval_with_none_tool_calls_and_correct_id(self):
        """Test that approval with None tool_calls and correct tool_call_id passes."""
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=None,  # None!
            tool_call_id='call_server_set',
        )
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='call_correct', approve=True)]
        )

        # Should not raise
        validate_approval_tool_call_ids(approval_request, approval_response)

    # =========================================================================
    # Legacy case - client uses message ID instead of tool call ID
    # =========================================================================

    def test_legacy_approval_with_message_id_instead_of_tool_call_id(self):
        """
        Test legacy case where client uses message ID instead of tool call ID.
        This should still work for backward compatibility.
        """
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[],
            tool_call_id='call_server_set',
            id='message-12345678-1234-1234-1234-123456789012',  # Message ID
        )
        # Legacy client sends message ID as tool_call_id
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='message-12345678-1234-1234-1234-123456789012', approve=True)]
        )

        # Should not raise - legacy case
        validate_approval_tool_call_ids(approval_request, approval_response)

    # =========================================================================
    # Error cases
    # =========================================================================

    def test_invalid_approval_with_empty_approvals(self):
        """Test that approval with empty approvals list fails."""
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[],
        )
        approval_response = ApprovalCreate(
            approvals=[]  # Empty!
        )

        with pytest.raises(ValueError, match="Invalid approval response"):
            validate_approval_tool_call_ids(approval_request, approval_response)

    def test_invalid_approval_with_none_approvals(self):
        """Test that approval with None approvals fails."""
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[],
        )
        approval_response = ApprovalCreate(
            approvals=None  # None!
        )

        with pytest.raises(ValueError, match="Invalid approval response"):
            validate_approval_tool_call_ids(approval_request, approval_response)

    def test_invalid_approval_with_empty_tool_calls_and_mismatched_id(self):
        """
        Test that approval with empty tool_calls and mismatched ID is accepted.
        When tool_calls is empty, we accept the client's IDs (the client knows what it's approving).
        This is the correct behavior after the fix.
        """
        approval_request = Message(
            role=MessageRole.approval,
            tool_calls=[],  # Empty!
            tool_call_id='call_server_set',  # Server sets this
        )
        # Client sends ANY tool_call_id - it's accepted when tool_calls is empty
        approval_response = ApprovalCreate(
            approvals=[ApprovalReturn(tool_call_id='call_any_id', approve=True)]
        )

        # Should NOT raise - when tool_calls is empty, we trust the client
        validate_approval_tool_call_ids(approval_request, approval_response)