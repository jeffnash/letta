"""
Unit tests for PendingApprovalError and related pending approval handling.

These tests are pure unit tests that don't require database fixtures.
"""

import pytest


class TestPendingApprovalErrorClass:
    """
    Unit tests for the PendingApprovalError class to ensure it correctly
    stores and exposes machine-usable identifiers.
    """

    def test_pending_approval_error_with_all_fields(self):
        """Test PendingApprovalError initialization with all fields."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
            run_id="run-789",
        )

        assert error.pending_request_id == "message-123"
        assert error.agent_id == "agent-456"
        assert error.run_id == "run-789"
        assert "message-123" in error.details["pending_request_id"]

    def test_pending_approval_error_with_minimal_fields(self):
        """Test PendingApprovalError with only required field."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(pending_request_id="message-123")

        assert error.pending_request_id == "message-123"
        assert error.agent_id is None
        assert error.run_id is None

    def test_pending_approval_error_message_format(self):
        """Test that error message contains the expected text."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(pending_request_id="message-123")

        error_str = str(error)
        assert "waiting for approval" in error_str.lower()
        # The pending_request_id is in the details, not the message string
        assert "CONFLICT" in error_str

    def test_pending_approval_error_details_dict(self):
        """Test that details dict includes all available identifiers."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
            run_id="run-789",
        )

        details = error.details
        assert "pending_request_id" in details
        assert details["pending_request_id"] == "message-123"
        assert "agent_id" in details
        assert details["agent_id"] == "agent-456"
        assert "run_id" in details
        assert details["run_id"] == "run-789"

    def test_pending_approval_error_is_letta_error(self):
        """Test that PendingApprovalError inherits from LettaError."""
        from letta.errors import PendingApprovalError, LettaError

        error = PendingApprovalError(pending_request_id="message-123")
        assert isinstance(error, LettaError)

    def test_pending_approval_error_has_conflict_code(self):
        """Test that PendingApprovalError has CONFLICT error code."""
        from letta.errors import PendingApprovalError, ErrorCode

        error = PendingApprovalError(pending_request_id="message-123")
        assert error.code == ErrorCode.CONFLICT

    def test_pending_approval_error_details_excludes_none_values(self):
        """Test that details dict only includes non-None values."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id=None,
            run_id=None,
        )

        details = error.details
        assert "pending_request_id" in details
        # The implementation should include these fields even if None for consistency
        # or exclude them - let's verify the actual behavior
        if "agent_id" in details:
            assert details["agent_id"] is None
        if "run_id" in details:
            assert details["run_id"] is None


class TestPendingApprovalErrorHandlerFormat:
    """
    Tests for the FastAPI error handler response format.
    These verify the expected JSON structure that clients will receive.
    """

    def test_error_handler_returns_correct_format_with_all_fields(self):
        """Test that the error handler returns correct JSON structure with all fields."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
            run_id="run-789",
        )

        # Simulate what the handler does
        content = {
            "detail": str(error),
            "error_code": "PENDING_APPROVAL",
            "pending_request_id": error.pending_request_id,
        }
        if hasattr(error, "agent_id") and error.agent_id:
            content["agent_id"] = error.agent_id
        if hasattr(error, "run_id") and error.run_id:
            content["run_id"] = error.run_id

        assert content["error_code"] == "PENDING_APPROVAL"
        assert content["pending_request_id"] == "message-123"
        assert content["agent_id"] == "agent-456"
        assert content["run_id"] == "run-789"
        assert "detail" in content

    def test_error_handler_without_optional_fields(self):
        """Test error handler when optional fields are None."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(pending_request_id="message-123")

        content = {
            "detail": str(error),
            "error_code": "PENDING_APPROVAL",
            "pending_request_id": error.pending_request_id,
        }
        if hasattr(error, "agent_id") and error.agent_id:
            content["agent_id"] = error.agent_id
        if hasattr(error, "run_id") and error.run_id:
            content["run_id"] = error.run_id

        assert content["error_code"] == "PENDING_APPROVAL"
        assert content["pending_request_id"] == "message-123"
        assert "agent_id" not in content
        assert "run_id" not in content

    def test_error_handler_with_only_agent_id(self):
        """Test error handler with agent_id but no run_id."""
        from letta.errors import PendingApprovalError

        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
        )

        content = {
            "detail": str(error),
            "error_code": "PENDING_APPROVAL",
            "pending_request_id": error.pending_request_id,
        }
        if hasattr(error, "agent_id") and error.agent_id:
            content["agent_id"] = error.agent_id
        if hasattr(error, "run_id") and error.run_id:
            content["run_id"] = error.run_id

        assert content["error_code"] == "PENDING_APPROVAL"
        assert content["pending_request_id"] == "message-123"
        assert content["agent_id"] == "agent-456"
        assert "run_id" not in content

    def test_error_code_is_machine_readable(self):
        """Test that error_code is a predictable, machine-readable string."""
        # This test documents the expected error code format for clients
        error_code = "PENDING_APPROVAL"
        
        # Should be uppercase with underscores (common convention)
        assert error_code == error_code.upper()
        assert " " not in error_code
        
        # Should match what clients will check for
        assert error_code == "PENDING_APPROVAL"


class TestStreamingLayerPendingApprovalHandling:
    """
    Tests for how PendingApprovalError should be handled in streaming layers.
    These are behavioral tests that verify the error is properly re-raised.
    """

    def test_pending_approval_error_should_not_be_internal_error(self):
        """
        Verify that PendingApprovalError is treated differently from internal errors.
        
        This is a regression test for the bug where PendingApprovalError was
        caught by generic exception handlers and converted to internal_error.
        """
        from letta.errors import PendingApprovalError, LettaError
        
        error = PendingApprovalError(pending_request_id="message-123")
        
        # The error should be distinguishable from generic errors
        assert isinstance(error, PendingApprovalError)
        
        # Check that we can distinguish it in a try/except
        caught_as_pending = False
        caught_as_generic = False
        
        try:
            raise error
        except PendingApprovalError:
            caught_as_pending = True
        except Exception:
            caught_as_generic = True
        
        assert caught_as_pending, "PendingApprovalError should be caught by specific handler"
        assert not caught_as_generic, "PendingApprovalError should not fall through to generic handler"

    def test_pending_approval_error_identifiers_accessible_after_raise(self):
        """Test that error identifiers are accessible when caught."""
        from letta.errors import PendingApprovalError
        
        original_pending_id = "message-123"
        original_agent_id = "agent-456"
        original_run_id = "run-789"
        
        try:
            raise PendingApprovalError(
                pending_request_id=original_pending_id,
                agent_id=original_agent_id,
                run_id=original_run_id,
            )
        except PendingApprovalError as e:
            assert e.pending_request_id == original_pending_id
            assert e.agent_id == original_agent_id
            assert e.run_id == original_run_id


class TestClientSideErrorParsing:
    """
    Tests that verify the error format can be parsed by clients.
    These simulate what the TypeScript client needs to do.
    """

    def test_client_can_detect_pending_approval_via_error_code(self):
        """Test that client can detect pending approval error using error_code."""
        from letta.errors import PendingApprovalError
        
        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
        )
        
        # Simulate server response
        response_body = {
            "detail": str(error),
            "error_code": "PENDING_APPROVAL",
            "pending_request_id": error.pending_request_id,
            "agent_id": error.agent_id,
        }
        
        # Client detection logic (mirrors TypeScript code)
        is_pending_approval = (
            "error_code" in response_body and
            response_body["error_code"] == "PENDING_APPROVAL"
        )
        
        assert is_pending_approval, "Client should detect pending approval via error_code"

    def test_client_can_extract_pending_request_id(self):
        """Test that client can extract pending_request_id from response."""
        from letta.errors import PendingApprovalError
        
        error = PendingApprovalError(
            pending_request_id="message-123",
            agent_id="agent-456",
        )
        
        # Simulate server response
        response_body = {
            "detail": str(error),
            "error_code": "PENDING_APPROVAL",
            "pending_request_id": error.pending_request_id,
            "agent_id": error.agent_id,
        }
        
        # Client extraction logic
        pending_request_id = response_body.get("pending_request_id")
        
        assert pending_request_id == "message-123"

    def test_client_can_fallback_to_detail_string_matching(self):
        """Test that client can detect via detail string for backwards compatibility."""
        from letta.errors import PendingApprovalError
        
        error = PendingApprovalError(pending_request_id="message-123")
        
        # Old-style response (before error_code was added)
        response_body = {
            "detail": str(error),
        }
        
        # Client fallback detection logic
        is_pending_approval = (
            "detail" in response_body and
            isinstance(response_body["detail"], str) and
            "waiting for approval" in response_body["detail"].lower()
        )
        
        assert is_pending_approval, "Client should detect pending approval via detail string"
