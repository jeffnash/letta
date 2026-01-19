"""
Tests for repair_message_history_async in AgentManager and the repair-message-history API endpoint.

This module tests the ability to detect and repair corrupted agent message history
where tool_use blocks exist without corresponding tool_result blocks.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from letta.schemas.enums import MessageRole
from letta.schemas.message import Message as PydanticMessage, ToolReturn
from letta.schemas.openai.chat_completion_response import ToolCall, FunctionCall


class TestRepairMessageHistoryAsync:
    """Test suite for AgentManager.repair_message_history_async method."""

    @pytest.fixture
    def mock_agent_manager(self):
        """Create a mock AgentManager with necessary methods."""
        from letta.services.agent_manager import AgentManager
        manager = AgentManager()
        return manager

    @pytest.fixture
    def sample_tool_call(self):
        """Create a sample ToolCall object."""
        return ToolCall(
            id="toolu_test_123",
            function=FunctionCall(name="get_weather", arguments='{"location": "NYC"}'),
        )

    @pytest.fixture
    def message_with_tool_call(self, sample_tool_call):
        """Create an assistant message with a tool call."""
        return PydanticMessage(
            id="msg-assistant-1",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Let me check the weather."}],
            tool_calls=[sample_tool_call],
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def message_with_tool_result(self, sample_tool_call):
        """Create a tool message with the corresponding result."""
        return PydanticMessage(
            id="msg-tool-1",
            role=MessageRole.tool,
            content=[{"type": "text", "text": "Sunny, 72°F"}],
            tool_call_id=sample_tool_call.id,
            tool_returns=[
                ToolReturn(
                    tool_call_id=sample_tool_call.id,
                    function_return="Sunny, 72°F",
                    status="success",
                )
            ],
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def orphaned_assistant_message(self):
        """Create an assistant message with tool call but no following tool result."""
        return PydanticMessage(
            id="msg-orphaned-1",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Searching..."}],
            tool_calls=[
                ToolCall(
                    id="toolu_orphaned_456",
                    function=FunctionCall(name="web_search", arguments='{"query": "test"}'),
                )
            ],
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_no_messages_returns_ok(self, mock_agent_manager):
        """Agent with no messages should return ok status."""
        with patch.object(mock_agent_manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            
            with patch.object(mock_agent_manager, 'get_agent_by_id_async', new_callable=AsyncMock):
                result = await mock_agent_manager.repair_message_history_async(
                    agent_id="agent-test-123",
                    actor=MagicMock()
                )
        
        assert result["status"] == "ok"
        assert "no message history" in result["message"].lower()
        assert result["orphaned_tool_calls"] == []
        assert result["removed_message_ids"] == []

    @pytest.mark.asyncio
    async def test_valid_messages_returns_ok(
        self, mock_agent_manager, message_with_tool_call, message_with_tool_result
    ):
        """Valid tool_use/tool_result pair should return ok status."""
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, message_with_tool_call, message_with_tool_result]
        
        with patch.object(mock_agent_manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            result = await mock_agent_manager.repair_message_history_async(
                agent_id="agent-test-123",
                actor=MagicMock()
            )
        
        assert result["status"] == "ok"
        assert result["orphaned_tool_calls"] == []

    @pytest.mark.asyncio
    async def test_orphaned_tool_use_detected_and_repaired(
        self, mock_agent_manager, orphaned_assistant_message
    ):
        """Orphaned tool_use (no following message) should be detected and removed."""
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        user_msg = PydanticMessage(
            id="msg-user-1",
            role=MessageRole.user,
            content=[{"type": "text", "text": "Search for something."}],
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, user_msg, orphaned_assistant_message]
        
        mock_agent = MagicMock()
        mock_agent.message_ids = ["msg-system-1", "msg-user-1", "msg-orphaned-1"]
        
        with patch.object(mock_agent_manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            with patch.object(mock_agent_manager, 'get_agent_by_id_async', new_callable=AsyncMock) as mock_get_agent:
                mock_get_agent.return_value = mock_agent
                
                with patch.object(mock_agent_manager, 'set_in_context_messages_async', new_callable=AsyncMock) as mock_set:
                    result = await mock_agent_manager.repair_message_history_async(
                        agent_id="agent-test-123",
                        actor=MagicMock()
                    )
        
        assert result["status"] == "repaired"
        assert len(result["orphaned_tool_calls"]) == 1
        assert result["orphaned_tool_calls"][0]["tool_call_id"] == "toolu_orphaned_456"
        assert result["orphaned_tool_calls"][0]["tool_name"] == "web_search"
        assert "msg-orphaned-1" in result["removed_message_ids"]
        
        # Verify set_in_context_messages was called with orphaned message removed
        mock_set.assert_called_once()
        call_args = mock_set.call_args
        new_message_ids = call_args[1]["message_ids"]
        assert "msg-orphaned-1" not in new_message_ids
        assert "msg-system-1" in new_message_ids
        assert "msg-user-1" in new_message_ids

    @pytest.mark.asyncio
    async def test_tool_use_followed_by_wrong_role_detected(self, mock_agent_manager):
        """Tool_use followed by non-tool message should be detected."""
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        
        assistant_with_tool = PydanticMessage(
            id="msg-assistant-tool",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Searching..."}],
            tool_calls=[
                ToolCall(
                    id="toolu_wrong_role",
                    function=FunctionCall(name="search", arguments='{}'),
                )
            ],
            created_at=datetime.now(timezone.utc),
        )
        
        # Next message is user (wrong - should be tool)
        user_msg = PydanticMessage(
            id="msg-user-1",
            role=MessageRole.user,
            content=[{"type": "text", "text": "Thanks!"}],
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, assistant_with_tool, user_msg]
        
        mock_agent = MagicMock()
        mock_agent.message_ids = ["msg-system-1", "msg-assistant-tool", "msg-user-1"]
        
        with patch.object(mock_agent_manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            with patch.object(mock_agent_manager, 'get_agent_by_id_async', new_callable=AsyncMock) as mock_get_agent:
                mock_get_agent.return_value = mock_agent
                
                with patch.object(mock_agent_manager, 'set_in_context_messages_async', new_callable=AsyncMock):
                    result = await mock_agent_manager.repair_message_history_async(
                        agent_id="agent-test-123",
                        actor=MagicMock()
                    )
        
        assert result["status"] == "repaired"
        assert len(result["orphaned_tool_calls"]) == 1
        assert result["orphaned_tool_calls"][0]["reason"] == "next_message_not_tool_response"

    @pytest.mark.asyncio
    async def test_multiple_orphaned_tool_calls_in_one_message(self, mock_agent_manager):
        """Multiple orphaned tool_calls in a single message should all be detected."""
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        
        assistant_with_multiple_tools = PydanticMessage(
            id="msg-assistant-multi",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Let me do several things."}],
            tool_calls=[
                ToolCall(id="toolu_1", function=FunctionCall(name="tool_a", arguments='{}')),
                ToolCall(id="toolu_2", function=FunctionCall(name="tool_b", arguments='{}')),
                ToolCall(id="toolu_3", function=FunctionCall(name="tool_c", arguments='{}')),
            ],
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, assistant_with_multiple_tools]
        
        mock_agent = MagicMock()
        mock_agent.message_ids = ["msg-system-1", "msg-assistant-multi"]
        
        with patch.object(mock_agent_manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            with patch.object(mock_agent_manager, 'get_agent_by_id_async', new_callable=AsyncMock) as mock_get_agent:
                mock_get_agent.return_value = mock_agent
                
                with patch.object(mock_agent_manager, 'set_in_context_messages_async', new_callable=AsyncMock):
                    result = await mock_agent_manager.repair_message_history_async(
                        agent_id="agent-test-123",
                        actor=MagicMock()
                    )
        
        assert result["status"] == "repaired"
        assert len(result["orphaned_tool_calls"]) == 3
        tool_names = {tc["tool_name"] for tc in result["orphaned_tool_calls"]}
        assert tool_names == {"tool_a", "tool_b", "tool_c"}


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
            "removed_message_ids": [],
        })
        
        # Mock FastAPI dependencies
        with patch('letta.server.rest_api.routers.v1.agents.get_letta_server', return_value=mock_server):
            with patch('letta.server.rest_api.routers.v1.agents.get_headers', return_value=MagicMock(actor_id=None)):
                result = await repair_message_history(
                    agent_id="agent-123",
                    server=mock_server,
                    headers=MagicMock(actor_id=None),
                )
        
        assert isinstance(result, RepairMessageHistoryResponse)
        assert result.status == "ok"

    @pytest.mark.asyncio
    async def test_endpoint_returns_repaired_with_details(self, mock_server):
        """Endpoint should return repaired status with orphaned tool call details."""
        from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse
        
        mock_server.user_manager.get_actor_or_default_async = AsyncMock(return_value=MagicMock())
        mock_server.agent_manager.repair_message_history_async = AsyncMock(return_value={
            "status": "repaired",
            "message": "Removed 1 message(s) with 2 orphaned tool_use block(s)",
            "orphaned_tool_calls": [
                {"message_id": "msg-1", "tool_call_id": "toolu_1", "tool_name": "search", "reason": "no_following_message"},
                {"message_id": "msg-1", "tool_call_id": "toolu_2", "tool_name": "calculate", "reason": "no_following_message"},
            ],
            "removed_message_ids": ["msg-1"],
        })
        
        result = await repair_message_history(
            agent_id="agent-123",
            server=mock_server,
            headers=MagicMock(actor_id=None),
        )
        
        assert isinstance(result, RepairMessageHistoryResponse)
        assert result.status == "repaired"
        assert len(result.orphaned_tool_calls) == 2
        assert len(result.removed_message_ids) == 1

    def test_response_model_validation(self):
        """Test that RepairMessageHistoryResponse model validates correctly."""
        from letta.server.rest_api.routers.v1.agents import RepairMessageHistoryResponse
        
        # Valid response
        response = RepairMessageHistoryResponse(
            status="repaired",
            message="Fixed 1 issue",
            orphaned_tool_calls=[{"message_id": "m1", "tool_call_id": "t1", "tool_name": "test", "reason": "missing"}],
            removed_message_ids=["m1"],
        )
        assert response.status == "repaired"
        
        # Test with "ok" status
        response_ok = RepairMessageHistoryResponse(
            status="ok",
            message="No issues",
        )
        assert response_ok.orphaned_tool_calls == []
        assert response_ok.removed_message_ids == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_assistant_without_tool_calls_ignored(self):
        """Assistant messages without tool_calls should be ignored."""
        from letta.services.agent_manager import AgentManager
        
        manager = AgentManager()
        
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        
        assistant_no_tools = PydanticMessage(
            id="msg-assistant-1",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Hello! How can I help?"}],
            tool_calls=None,  # No tool calls
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, assistant_no_tools]
        
        with patch.object(manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            result = await manager.repair_message_history_async(
                agent_id="agent-test-123",
                actor=MagicMock()
            )
        
        assert result["status"] == "ok"
        assert result["orphaned_tool_calls"] == []

    @pytest.mark.asyncio  
    async def test_empty_tool_calls_list_ignored(self):
        """Assistant messages with empty tool_calls list should be ignored."""
        from letta.services.agent_manager import AgentManager
        
        manager = AgentManager()
        
        system_msg = PydanticMessage(
            id="msg-system-1",
            role=MessageRole.system,
            content=[{"type": "text", "text": "You are helpful."}],
            created_at=datetime.now(timezone.utc),
        )
        
        assistant_empty_tools = PydanticMessage(
            id="msg-assistant-1",
            role=MessageRole.assistant,
            content=[{"type": "text", "text": "Hello!"}],
            tool_calls=[],  # Empty list
            created_at=datetime.now(timezone.utc),
        )
        
        messages = [system_msg, assistant_empty_tools]
        
        with patch.object(manager, 'get_in_context_messages', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = messages
            
            result = await manager.repair_message_history_async(
                agent_id="agent-test-123",
                actor=MagicMock()
            )
        
        assert result["status"] == "ok"
