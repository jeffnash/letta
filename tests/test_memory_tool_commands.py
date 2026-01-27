"""Tests for memory tool read and search commands in LettaCoreToolExecutor."""

import pytest

from letta.schemas.block import Block
from letta.schemas.enums import AgentType
from letta.schemas.memory import Memory


class MockUser:
    """Mock user for testing."""

    id = "user-test-123"


class MockAgentState:
    """Mock agent state for testing."""

    def __init__(self, memory: Memory):
        self.id = "agent-test-123"
        self.memory = memory


class MockBlockManager:
    """Mock block manager for testing."""

    async def update_block_async(self, block_id, block_update, actor):
        pass


class MockAgentManager:
    """Mock agent manager for testing."""

    async def update_memory_if_changed_async(self, agent_id, new_memory, actor):
        pass


@pytest.fixture
def sample_memory():
    """Create a sample memory with blocks for testing."""
    return Memory(
        agent_type=AgentType.memgpt_agent,
        blocks=[
            Block(
                label="project",
                value="Line 1: Introduction\nLine 2: Setup\nLine 3: Database configuration\nLine 4: API endpoints\nLine 5: Testing\nLine 6: Database migrations\nLine 7: Deployment\nLine 8: Monitoring\nLine 9: Cleanup\nLine 10: Conclusion",
                description="Project documentation",
                limit=5000,
            ),
            Block(
                label="notes",
                value="TODO: Fix bug\nDONE: Review PR\nTODO: Write tests",
                description="Task notes",
                limit=1000,
            ),
        ],
    )


@pytest.fixture
def mock_actor():
    return MockUser()


@pytest.fixture
def tool_executor(sample_memory):
    """Create a tool executor instance for testing."""
    from letta.services.tool_executor.core_tool_executor import LettaCoreToolExecutor

    executor = LettaCoreToolExecutor.__new__(LettaCoreToolExecutor)
    executor.block_manager = MockBlockManager()
    executor.agent_manager = MockAgentManager()
    return executor


# ==================== memory_read tests ====================


@pytest.mark.asyncio
async def test_memory_read_full_block(tool_executor, sample_memory, mock_actor):
    """Test reading an entire memory block."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_read(agent_state, mock_actor, "/memories/project")

    assert "Memory block 'project':" in result
    assert "Description: Project documentation" in result
    assert "Total lines: 10" in result
    assert "   1\tLine 1: Introduction" in result
    assert "  10\tLine 10: Conclusion" in result


@pytest.mark.asyncio
async def test_memory_read_with_offset(tool_executor, sample_memory, mock_actor):
    """Test reading with offset."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_read(
        agent_state, mock_actor, "/memories/project", offset=5
    )

    assert "Showing lines 5-10 of 10" in result
    assert "   5\tLine 5: Testing" in result
    assert "  10\tLine 10: Conclusion" in result
    # Should not include lines before offset
    assert "Line 1: Introduction" not in result


@pytest.mark.asyncio
async def test_memory_read_with_limit(tool_executor, sample_memory, mock_actor):
    """Test reading with limit."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_read(
        agent_state, mock_actor, "/memories/project", limit=3
    )

    assert "Showing lines 1-3 of 10" in result
    assert "   1\tLine 1: Introduction" in result
    assert "   3\tLine 3: Database configuration" in result
    # Should not include lines after limit
    assert "Line 4:" not in result


@pytest.mark.asyncio
async def test_memory_read_with_offset_and_limit(tool_executor, sample_memory, mock_actor):
    """Test reading with both offset and limit."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_read(
        agent_state, mock_actor, "/memories/project", offset=3, limit=4
    )

    # offset=3 means start at line 3 (0-indexed: 2), limit=4 means 4 lines
    # So we get lines 3, 4, 5, 6
    assert "Showing lines 3-6 of 10" in result
    assert "   3\tLine 3: Database configuration" in result
    assert "   6\tLine 6: Database migrations" in result
    # Should not include lines outside range
    assert "Line 1:" not in result
    assert "Line 7:" not in result


@pytest.mark.asyncio
async def test_memory_read_nonexistent_block(tool_executor, sample_memory, mock_actor):
    """Test reading a block that doesn't exist."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_read(
        agent_state, mock_actor, "/memories/nonexistent"
    )

    assert "Error: Memory block 'nonexistent' does not exist" in result
    assert "Available blocks:" in result
    assert "project" in result
    assert "notes" in result


@pytest.mark.asyncio
async def test_memory_read_path_normalization(tool_executor, sample_memory, mock_actor):
    """Test that various path formats work."""
    agent_state = MockAgentState(sample_memory)

    # Test different path formats
    result1 = await tool_executor.memory_read(agent_state, mock_actor, "/memories/project")
    result2 = await tool_executor.memory_read(agent_state, mock_actor, "/project")
    result3 = await tool_executor.memory_read(agent_state, mock_actor, "project")

    # All should return the same content
    assert "Memory block 'project':" in result1
    assert "Memory block 'project':" in result2
    assert "Memory block 'project':" in result3


# ==================== memory_search tests ====================


@pytest.mark.asyncio
async def test_memory_search_single_match(tool_executor, sample_memory, mock_actor):
    """Test searching for a keyword with a single match."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/project", "Introduction"
    )

    assert "Found 1 match(es) for 'Introduction'" in result
    assert ">>>" in result  # Match marker
    assert "Line 1: Introduction" in result


@pytest.mark.asyncio
async def test_memory_search_multiple_matches(tool_executor, sample_memory, mock_actor):
    """Test searching for a keyword with multiple matches."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/project", "Database"
    )

    assert "Found 2 match(es) for 'Database'" in result
    assert "Line 3: Database configuration" in result
    assert "Line 6: Database migrations" in result


@pytest.mark.asyncio
async def test_memory_search_case_insensitive(tool_executor, sample_memory, mock_actor):
    """Test that search is case-insensitive."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/project", "database"
    )

    assert "Found 2 match(es)" in result
    assert "Database" in result


@pytest.mark.asyncio
async def test_memory_search_with_context(tool_executor, sample_memory, mock_actor):
    """Test that context lines are included."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/project", "Testing", context_lines=2
    )

    assert "Found 1 match(es)" in result
    # Should include 2 lines before and after
    assert "Line 3:" in result or "Line 4:" in result  # Before
    assert "Line 5: Testing" in result  # Match
    assert "Line 6:" in result or "Line 7:" in result  # After


@pytest.mark.asyncio
async def test_memory_search_no_matches(tool_executor, sample_memory, mock_actor):
    """Test searching for a keyword with no matches."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/project", "nonexistent_keyword_xyz"
    )

    assert "No matches found for 'nonexistent_keyword_xyz'" in result


@pytest.mark.asyncio
async def test_memory_search_nonexistent_block(tool_executor, sample_memory, mock_actor):
    """Test searching a block that doesn't exist."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/nonexistent", "test"
    )

    assert "Error: Memory block 'nonexistent' does not exist" in result
    assert "Available blocks:" in result


@pytest.mark.asyncio
async def test_memory_search_notes_block(tool_executor, sample_memory, mock_actor):
    """Test searching the notes block."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory_search(
        agent_state, mock_actor, "/memories/notes", "TODO"
    )

    assert "Found 2 match(es)" in result
    assert "TODO: Fix bug" in result
    assert "TODO: Write tests" in result


# ==================== memory dispatcher tests ====================


@pytest.mark.asyncio
async def test_memory_dispatcher_read(tool_executor, sample_memory, mock_actor):
    """Test the memory dispatcher routes 'read' command correctly."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory(
        agent_state, mock_actor, command="read", path="/memories/project"
    )

    assert "Memory block 'project':" in result


@pytest.mark.asyncio
async def test_memory_dispatcher_read_with_offset_limit(tool_executor, sample_memory, mock_actor):
    """Test the memory dispatcher passes offset and limit for 'read' command."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory(
        agent_state, mock_actor, command="read", path="/memories/project", offset=2, limit=3
    )

    # offset=2 means start at line 2 (0-indexed: 1), limit=3 means 3 lines
    # So we get lines 2, 3, 4
    assert "Showing lines 2-4 of 10" in result


@pytest.mark.asyncio
async def test_memory_dispatcher_search(tool_executor, sample_memory, mock_actor):
    """Test the memory dispatcher routes 'search' command correctly."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory(
        agent_state, mock_actor, command="search", path="/memories/project", query="Database"
    )

    assert "Found 2 match(es)" in result


@pytest.mark.asyncio
async def test_memory_dispatcher_search_with_context(tool_executor, sample_memory, mock_actor):
    """Test the memory dispatcher passes context_lines for 'search' command."""
    agent_state = MockAgentState(sample_memory)

    result = await tool_executor.memory(
        agent_state,
        mock_actor,
        command="search",
        path="/memories/project",
        query="Testing",
        context_lines=1,
    )

    assert "Found 1 match(es)" in result


@pytest.mark.asyncio
async def test_memory_dispatcher_read_missing_path(tool_executor, sample_memory, mock_actor):
    """Test that 'read' command requires path."""
    agent_state = MockAgentState(sample_memory)

    with pytest.raises(ValueError, match="path is required for read command"):
        await tool_executor.memory(agent_state, mock_actor, command="read")


@pytest.mark.asyncio
async def test_memory_dispatcher_search_missing_path(tool_executor, sample_memory, mock_actor):
    """Test that 'search' command requires path."""
    agent_state = MockAgentState(sample_memory)

    with pytest.raises(ValueError, match="path is required for search command"):
        await tool_executor.memory(agent_state, mock_actor, command="search", query="test")


@pytest.mark.asyncio
async def test_memory_dispatcher_search_missing_query(tool_executor, sample_memory, mock_actor):
    """Test that 'search' command requires query."""
    agent_state = MockAgentState(sample_memory)

    with pytest.raises(ValueError, match="query is required for search command"):
        await tool_executor.memory(
            agent_state, mock_actor, command="search", path="/memories/project"
        )


@pytest.mark.asyncio
async def test_memory_dispatcher_unknown_command(tool_executor, sample_memory, mock_actor):
    """Test that unknown commands raise an error."""
    agent_state = MockAgentState(sample_memory)

    with pytest.raises(ValueError, match="Unknown command"):
        await tool_executor.memory(
            agent_state, mock_actor, command="unknown", path="/memories/project"
        )
