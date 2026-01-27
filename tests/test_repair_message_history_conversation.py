"""
Regression test for conversation-scoped repair-message-history (conversation_id).

Ensures that when repairing a non-default conversation, the response correctly
includes injected_message_ids and injected_tool_call_ids (instead of removed_message_ids).
This preserves message positions for prompt caching efficiency.

Note: Full integration testing requires a database. These tests verify the API contract
using mocked responses from the agent_manager.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from letta.schemas.user import User as PydanticUser
from letta.server.rest_api.routers.v1.agents import repair_message_history, RepairMessageHistoryResponse


@pytest.mark.asyncio
async def test_repair_message_history_conversation_returns_injected_messages():
    """Test that conversation repair returns injected_message_ids instead of removed_message_ids."""
    agent_id = "agent-123e4567-e89b-42d3-8456-426614174000"
    conversation_id = "conv-123e4567-e89b-42d3-8456-426614174000"

    server = MagicMock()
    actor = PydanticUser(name="test-user")
    server.user_manager.get_actor_or_default_async = AsyncMock(return_value=actor)
    
    # Mock the agent_manager to return the expected response format
    server.agent_manager.repair_message_history_async = AsyncMock(return_value={
        "status": "repaired",
        "message": "Injected 1 synthetic tool result message(s) for 1 orphaned tool_call(s)",
        "orphaned_tool_calls": [
            {
                "message_id": "message-00000002",
                "tool_call_id": "toolu_orphaned_456",
                "tool_name": "get_weather",
                "reason": "no_following_message",
            }
        ],
        "injected_message_ids": ["message-00000003"],
        "injected_tool_call_ids": ["toolu_orphaned_456"],
    })

    result = await repair_message_history(
        agent_id=agent_id,
        conversation_id=conversation_id,
        server=server,
        headers=MagicMock(actor_id=actor.id),
    )

    # Verify the response model uses injected_* fields
    assert isinstance(result, RepairMessageHistoryResponse)
    assert result.status == "repaired"
    
    # Verify injected message IDs (not removed)
    assert len(result.injected_message_ids) == 1
    assert "message-00000003" in result.injected_message_ids
    
    # Verify injected tool call IDs
    assert len(result.injected_tool_call_ids) == 1
    assert "toolu_orphaned_456" in result.injected_tool_call_ids
    
    # Verify orphaned tool calls were detected
    assert len(result.orphaned_tool_calls) == 1
    assert result.orphaned_tool_calls[0]["tool_name"] == "get_weather"
    
    # Verify the agent_manager was called with conversation_id
    server.agent_manager.repair_message_history_async.assert_called_once()
    call_kwargs = server.agent_manager.repair_message_history_async.call_args.kwargs
    assert call_kwargs["conversation_id"] == conversation_id
    assert call_kwargs["agent_id"] == agent_id


@pytest.mark.asyncio
async def test_repair_message_history_conversation_ok_status():
    """Test that conversation repair returns ok status when no issues found."""
    agent_id = "agent-123e4567-e89b-42d3-8456-426614174000"
    conversation_id = "conv-123e4567-e89b-42d3-8456-426614174000"

    server = MagicMock()
    actor = PydanticUser(name="test-user")
    server.user_manager.get_actor_or_default_async = AsyncMock(return_value=actor)
    
    server.agent_manager.repair_message_history_async = AsyncMock(return_value={
        "status": "ok",
        "message": "No orphaned tool_use blocks found",
        "orphaned_tool_calls": [],
        "injected_message_ids": [],
        "injected_tool_call_ids": [],
    })

    result = await repair_message_history(
        agent_id=agent_id,
        conversation_id=conversation_id,
        server=server,
        headers=MagicMock(actor_id=actor.id),
    )

    assert result.status == "ok"
    assert result.injected_message_ids == []
    assert result.injected_tool_call_ids == []


@pytest.mark.asyncio
async def test_repair_message_history_multiple_orphaned_tool_calls():
    """Test repair with multiple orphaned tool calls in a conversation."""
    agent_id = "agent-123e4567-e89b-42d3-8456-426614174000"
    conversation_id = "conv-123e4567-e89b-42d3-8456-426614174000"

    server = MagicMock()
    actor = PydanticUser(name="test-user")
    server.user_manager.get_actor_or_default_async = AsyncMock(return_value=actor)
    
    # Multiple orphaned tool calls from same message
    server.agent_manager.repair_message_history_async = AsyncMock(return_value={
        "status": "repaired",
        "message": "Injected 3 synthetic tool result message(s) for 3 orphaned tool_call(s)",
        "orphaned_tool_calls": [
            {"message_id": "msg-1", "tool_call_id": "toolu_1", "tool_name": "search", "reason": "no_following_message"},
            {"message_id": "msg-1", "tool_call_id": "toolu_2", "tool_name": "read", "reason": "no_following_message"},
            {"message_id": "msg-2", "tool_call_id": "toolu_3", "tool_name": "write", "reason": "next_message_not_tool_response"},
        ],
        "injected_message_ids": ["msg-syn-1", "msg-syn-2", "msg-syn-3"],
        "injected_tool_call_ids": ["toolu_1", "toolu_2", "toolu_3"],
    })

    result = await repair_message_history(
        agent_id=agent_id,
        conversation_id=conversation_id,
        server=server,
        headers=MagicMock(actor_id=actor.id),
    )

    assert result.status == "repaired"
    assert len(result.orphaned_tool_calls) == 3
    assert len(result.injected_message_ids) == 3
    assert len(result.injected_tool_call_ids) == 3
    
    # Verify all tool calls are accounted for
    tool_names = {tc["tool_name"] for tc in result.orphaned_tool_calls}
    assert tool_names == {"search", "read", "write"}
