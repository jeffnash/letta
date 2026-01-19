"""
Regression test for conversation-scoped repair-message-history (conversation_id).

Ensures that when repairing a non-default conversation, we update the conversation's
in-context message list (instead of only the agent-level message_ids).
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta.schemas.conversation import Conversation as PydanticConversation
from letta.schemas.enums import MessageRole
from letta.schemas.message import Message as PydanticMessage
from letta.schemas.user import User as PydanticUser
from letta.server.rest_api.routers.v1.agents import repair_message_history
from letta.services.agent_manager import AgentManager


@pytest.mark.asyncio
async def test_repair_message_history_repairs_conversation_in_context_list():
    agent_id = "agent-123e4567-e89b-42d3-8456-426614174000"
    conversation_id = "conv-123e4567-e89b-42d3-8456-426614174000"

    system_msg = PydanticMessage(
        id="message-00000001",
        agent_id=agent_id,
        conversation_id=conversation_id,
        role=MessageRole.system,
        content=[{"type": "text", "text": "You are helpful."}],
        created_at=datetime.now(timezone.utc),
    )
    orphaned_assistant_msg = PydanticMessage(
        id="message-00000002",
        agent_id=agent_id,
        conversation_id=conversation_id,
        role=MessageRole.assistant,
        content=[{"type": "text", "text": "Let me check."}],
        tool_calls=[
            {
                "id": "toolu_orphaned_456",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ],
        created_at=datetime.now(timezone.utc),
    )

    conversation = PydanticConversation(
        id=conversation_id,
        agent_id=agent_id,
        summary=None,
        in_context_message_ids=[system_msg.id, orphaned_assistant_msg.id],
    )

    mock_conversation_manager = MagicMock()
    mock_conversation_manager.get_conversation_by_id = AsyncMock(return_value=conversation)
    mock_conversation_manager.get_messages_for_conversation = AsyncMock(return_value=[system_msg, orphaned_assistant_msg])
    mock_conversation_manager.update_in_context_messages = AsyncMock()

    agent_manager = AgentManager()

    server = MagicMock()
    server.agent_manager = agent_manager
    actor = PydanticUser(name="test-user")
    server.user_manager.get_actor_or_default_async = AsyncMock(return_value=actor)

    with patch("letta.services.conversation_manager.ConversationManager", return_value=mock_conversation_manager):
        result = await repair_message_history(
            agent_id=agent_id,
            conversation_id=conversation_id,
            server=server,
            headers=MagicMock(actor_id=actor.id),
        )

    assert result.status == "repaired"
    assert orphaned_assistant_msg.id in result.removed_message_ids

    mock_conversation_manager.update_in_context_messages.assert_called_once()
    call_kwargs = mock_conversation_manager.update_in_context_messages.call_args.kwargs
    assert call_kwargs["conversation_id"] == conversation_id
    assert call_kwargs["in_context_message_ids"] == [system_msg.id]

