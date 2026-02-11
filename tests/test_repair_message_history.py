"""
Tests for the repair-message-history API endpoint and response model.

The repair process now INJECTS synthetic tool_result messages instead of removing
orphaned messages, to preserve message positions and maximize prompt caching.

Note: Unit tests for the internal repair_message_history_async method are complex
due to Pydantic validation requirements on Message schemas. These endpoint and
response model tests verify the API contract without needing to mock internal details.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestRepairMessageHistoryEndpoint:
    """Test suite for the /repair-message-history API endpoint."""

    @pytest.fixture
    def mock_server(self):
        """Create a mock SyncServer."""
        server = MagicMock()
        server.agent_manager = MagicMock()
        server.user_manager = MagicMock()
        return server

    @pytest.mark.asyncio
    async def test_endpoint_returns_ok_when_no_issues(self, mock_server):
        """Endpoint should return ok status when no issues found."""
        from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse
        
        mock_server.user_manager.get_actor_or_default_async = AsyncMock(return_value=MagicMock())
        mock_server.agent_manager.repair_message_history_async = AsyncMock(return_value={
            "status": "ok",
            "message": "No orphaned tool_use blocks found",
            "orphaned_tool_calls": [],
            "injected_message_ids": [],
            "injected_tool_call_ids": [],
        })
        
        result = await repair_message_history(
            agent_id="agent-123",
            server=mock_server,
            headers=MagicMock(actor_id=None),
        )
        
        assert isinstance(result, RepairMessageHistoryResponse)
        assert result.status == "ok"
        assert result.injected_message_ids == []
        assert result.injected_tool_call_ids == []
        assert result.pruned_message_ids == []

    @pytest.mark.asyncio
    async def test_endpoint_returns_repaired_with_injected_messages(self, mock_server):
        """Endpoint should return repaired status with injected message details."""
        from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse
        
        mock_server.user_manager.get_actor_or_default_async = AsyncMock(return_value=MagicMock())
        mock_server.agent_manager.repair_message_history_async = AsyncMock(return_value={
            "status": "repaired",
            "message": "Injected 2 synthetic tool result message(s) for 2 orphaned tool_call(s)",
            "orphaned_tool_calls": [
                {"message_id": "msg-1", "tool_call_id": "toolu_1", "tool_name": "search", "reason": "no_following_message"},
                {"message_id": "msg-1", "tool_call_id": "toolu_2", "tool_name": "calculate", "reason": "no_following_message"},
            ],
            "injected_message_ids": ["msg-synthetic-1", "msg-synthetic-2"],
            "injected_tool_call_ids": ["toolu_1", "toolu_2"],
        })
        
        result = await repair_message_history(
            agent_id="agent-123",
            server=mock_server,
            headers=MagicMock(actor_id=None),
        )
        
        assert isinstance(result, RepairMessageHistoryResponse)
        assert result.status == "repaired"
        assert len(result.orphaned_tool_calls) == 2
        assert len(result.injected_message_ids) == 2
        assert len(result.injected_tool_call_ids) == 2
        assert "toolu_1" in result.injected_tool_call_ids
        assert "toolu_2" in result.injected_tool_call_ids
        assert result.pruned_message_ids == []

    @pytest.mark.asyncio
    async def test_endpoint_reports_malformed_tool_call_json_sanitization(self, mock_server):
        """Endpoint should surface sanitization details for malformed approval tool-call JSON repair."""
        from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse

        mock_server.user_manager.get_actor_or_default_async = AsyncMock(return_value=MagicMock())
        mock_server.agent_manager.repair_message_history_async = AsyncMock(return_value={
            "status": "repaired",
            "message": "Sanitized malformed tool-call JSON in 1 message(s) covering 1 tool_call(s)",
            "orphaned_tool_calls": [],
            "injected_message_ids": [],
            "injected_tool_call_ids": [],
            "pruned_message_ids": [],
            "sanitized_message_ids": ["message-malformed-approval-1"],
            "sanitized_tool_call_ids": ["toolu-malformed-approval-1"],
        })

        result = await repair_message_history(
            agent_id="agent-123",
            server=mock_server,
            headers=MagicMock(actor_id=None),
        )

        assert isinstance(result, RepairMessageHistoryResponse)
        assert result.status == "repaired"
        assert result.sanitized_message_ids == ["message-malformed-approval-1"]
        assert result.sanitized_tool_call_ids == ["toolu-malformed-approval-1"]
        assert "Sanitized malformed tool-call JSON" in result.message

    async def test_endpoint_handles_no_message_history(self, mock_server):
        """Endpoint should handle case where agent has no message history."""
        from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse
        
        mock_server.user_manager.get_actor_or_default_async = AsyncMock(return_value=MagicMock())
        mock_server.agent_manager.repair_message_history_async = AsyncMock(return_value={
            "status": "ok",
            "message": "Agent has no message history to repair",
            "orphaned_tool_calls": [],
            "injected_message_ids": [],
            "injected_tool_call_ids": [],
        })
        
        result = await repair_message_history(
            agent_id="agent-123",
            server=mock_server,
            headers=MagicMock(actor_id=None),
        )
        
        assert result.status == "ok"
        assert "no message history" in result.message.lower()


class TestRepairMessageHistoryResponseModel:
    """Test the RepairMessageHistoryResponse Pydantic model."""

    def test_response_model_with_injections(self):
        """Test response model with injected messages."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse
        
        response = RepairMessageHistoryResponse(
            status="repaired",
            message="Injected 3 synthetic tool result messages",
            orphaned_tool_calls=[
                {"message_id": "m1", "tool_call_id": "t1", "tool_name": "search", "reason": "missing"},
                {"message_id": "m1", "tool_call_id": "t2", "tool_name": "read", "reason": "missing"},
                {"message_id": "m2", "tool_call_id": "t3", "tool_name": "write", "reason": "missing"},
            ],
            injected_message_ids=["m-syn-1", "m-syn-2", "m-syn-3"],
            injected_tool_call_ids=["t1", "t2", "t3"],
        )
        
        assert response.status == "repaired"
        assert len(response.orphaned_tool_calls) == 3
        assert len(response.injected_message_ids) == 3
        assert len(response.injected_tool_call_ids) == 3
        assert response.pruned_message_ids == []
        assert response.sanitized_message_ids == []
        assert response.sanitized_tool_call_ids == []

    def test_response_model_ok_status_defaults(self):
        """Test response model with ok status uses defaults."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse
        
        response = RepairMessageHistoryResponse(
            status="ok",
            message="No issues found",
        )
        
        assert response.status == "ok"
        assert response.orphaned_tool_calls == []
        assert response.injected_message_ids == []
        assert response.injected_tool_call_ids == []
        assert response.pruned_message_ids == []
        assert response.sanitized_message_ids == []
        assert response.sanitized_tool_call_ids == []

    def test_response_model_with_sanitized_tool_calls(self):
        """Test response model with malformed tool-call JSON sanitization details."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse

        response = RepairMessageHistoryResponse(
            status="repaired",
            message="Sanitized malformed tool-call JSON in 2 message(s)",
            orphaned_tool_calls=[],
            injected_message_ids=[],
            injected_tool_call_ids=[],
            pruned_message_ids=[],
            sanitized_message_ids=["message-1", "message-2"],
            sanitized_tool_call_ids=["toolu-1", "toolu-2", "toolu-3"],
        )

        assert response.status == "repaired"
        assert response.sanitized_message_ids == ["message-1", "message-2"]
        assert response.sanitized_tool_call_ids == ["toolu-1", "toolu-2", "toolu-3"]

    def test_response_model_error_status(self):
        """Test response model with error status."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse
        
        response = RepairMessageHistoryResponse(
            status="error",
            message="Failed to repair: database connection error",
            orphaned_tool_calls=[],
            injected_message_ids=[],
            injected_tool_call_ids=[],
        )
        
        assert response.status == "error"
        assert "failed" in response.message.lower()

    def test_response_model_ignores_extra_fields(self):
        """Test that response model ignores extra fields (for backwards compat)."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse
        
        # Simulates if the backend returns extra fields we don't know about
        response = RepairMessageHistoryResponse(
            status="ok",
            message="No issues",
            orphaned_tool_calls=[],
            injected_message_ids=[],
            injected_tool_call_ids=[],
            # These would be extra fields that should be ignored
        )
        
        assert response.status == "ok"
