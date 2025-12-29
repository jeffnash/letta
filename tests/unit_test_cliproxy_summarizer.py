import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json

from letta.agents.letta_agent_v3 import LettaAgentV3
from letta.schemas.agent import AgentState
from letta.schemas.enums import MessageRole
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message, TextContent
from letta.errors import ContextWindowExceededError
from letta.schemas.letta_stop_reason import LettaStopReason

@pytest.mark.asyncio
async def test_letta_agent_v3_step_recovers_from_context_window_exceeded():
    """
    Unit test for LettaAgentV3._step to verify it catches ContextWindowExceededError,
    calls compact(), and retries the LLM request.
    """
    # 1. Mock dependencies
    mock_agent_state = MagicMock(spec=AgentState)
    mock_agent_state.id = "agent-123"
    mock_agent_state.llm_config = LLMConfig(
        model="gpt-4o-mini",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
        context_window=8000,
        provider_name="cliproxy"
    )
    mock_agent_state.agent_type = "letta_v1_agent"
    mock_agent_state.tools = []
    mock_agent_state.tool_rules = []
    mock_agent_state.message_ids = ["msg-1"]
    mock_agent_state.project_id = "proj-123"
    mock_agent_state.template_id = "temp-123"
    
    mock_actor = MagicMock()
    
    # 2. Instantiate LettaAgentV3
    # LettaAgentV2.__init__ only takes agent_state and actor
    agent = LettaAgentV3(
        agent_state=mock_agent_state,
        actor=mock_actor
    )
    
    # Mock manager attributes that are initialized in __init__
    agent.message_manager = AsyncMock()
    agent.agent_manager = AsyncMock()
    agent.step_manager = AsyncMock()
    agent.llm_client = MagicMock()
    
    # Mock internal methods that are not the focus of this test
    agent._get_valid_tools = AsyncMock(return_value=[])
    agent._step_checkpoint_start = AsyncMock(return_value=(None, None, MagicMock(), MagicMock()))
    agent._step_checkpoint_llm_request_start = MagicMock(return_value=(None, MagicMock()))
    agent._step_checkpoint_llm_request_finish = MagicMock(return_value=(None, MagicMock()))
    agent._step_checkpoint_finish = AsyncMock(return_value=(None, MagicMock()))
    agent._checkpoint_messages = AsyncMock()
    agent._refresh_messages = AsyncMock(side_effect=lambda msgs: msgs)
    agent._update_global_usage_stats = MagicMock()
    
    # Mock compact() to return a dummy summary message and updated message list
    summary_msg = Message(role=MessageRole.user, content=[TextContent(text="Summary")])
    agent.compact = AsyncMock(return_value=(summary_msg, [summary_msg]))
    
    # Mock _handle_ai_response
    agent._handle_ai_response = AsyncMock(return_value=([], False, LettaStopReason(stop_reason="end_turn")))

    # 3. Setup LLM adapter mock
    mock_llm_adapter = MagicMock()
    mock_llm_adapter.supports_token_streaming.return_value = False
    mock_llm_adapter.usage = MagicMock()
    mock_llm_adapter.usage.total_tokens = 1000
    mock_llm_adapter.usage.completion_tokens = 100
    mock_llm_adapter.usage.prompt_tokens = 900
    mock_llm_adapter.content = "Recovered"
    mock_llm_adapter.message_id = "msg-recovered"
    mock_llm_adapter.finish_reason = "stop"
    mock_llm_adapter.llm_request_finish_timestamp_ns = 12345
    
    # We want invoke_llm to raise error then succeed
    async def mock_invoke_llm(*args, **kwargs):
        if mock_invoke_llm.call_count == 0:
            mock_invoke_llm.call_count += 1
            raise ContextWindowExceededError("Context exceeded")
        # Second call returns an empty generator (since we're not testing streaming here)
        mock_invoke_llm.call_count += 1
        if False: yield # make it a generator
        return

    mock_invoke_llm.call_count = 0
    mock_llm_adapter.invoke_llm = mock_invoke_llm

    # 4. Execute _step
    messages = [Message(role=MessageRole.user, content=[TextContent(text="Hello")])]
    
    chunks = []
    async for chunk in agent._step(
        messages=messages,
        llm_adapter=mock_llm_adapter,
        run_id="run-123"
    ):
        chunks.append(chunk)
        
    # 5. Assertions
    assert agent.compact.called, "compact() should have been called after ContextWindowExceededError"
    assert mock_invoke_llm.call_count == 2, "Should have retried LLM invocation"
    assert agent.compact.call_args[1]["trigger_threshold"] == 8000
    
    # Check if checkpoint_messages was called with the summary message
    # It should be called twice: once during recovery (if fixed) and once at the end of the step
    checkpoint_calls = agent._checkpoint_messages.call_args_list
    assert len(checkpoint_calls) >= 1
    
    # Verify that at least one call included the summary message in new_messages
    found_summary_checkpoint = False
    for call in checkpoint_calls:
        new_msgs = call[1].get("new_messages", [])
        if any(m.content[0].text == "Summary" for m in new_msgs if hasattr(m, "content") and m.content):
            found_summary_checkpoint = True
            break
    assert found_summary_checkpoint, "Summary message should have been checkpointed"
    
    print("Unit test for LettaAgentV3 summarization recovery passed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_letta_agent_v3_step_recovers_from_context_window_exceeded())
