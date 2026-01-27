"""
Tests for prompt caching optimizations in the system prompt generation.

These tests verify that the memory metadata block is static and cacheable,
without dynamic values that would invalidate the prompt cache on every request.
"""

import pytest
from datetime import datetime, timezone


class TestCompileMemoryMetadataBlock:
    """Test suite for PromptGenerator.compile_memory_metadata_block."""

    def test_metadata_block_is_static_without_conversation_start(self):
        """Metadata block should be static when no conversation_start_date is provided."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        # Call twice with different dynamic values
        result1 = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            timezone="UTC",
            previous_message_count=100,
            archival_memory_size=50,
        )
        
        result2 = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime(2026, 1, 20, 15, 30, 0, tzinfo=timezone.utc),  # Different timestamp
            timezone="UTC",
            previous_message_count=200,  # Different count
            archival_memory_size=100,    # Different count
        )
        
        # Both should be identical (no dynamic values)
        assert result1 == result2

    def test_metadata_block_includes_conversation_start_date(self):
        """Metadata block should include conversation_start_date when provided."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        conversation_start = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        
        result = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="UTC",
            conversation_start_date=conversation_start,
        )
        
        assert "<memory_metadata>" in result
        assert "</memory_metadata>" in result
        assert "Conversation started:" in result
        assert "2026-01-15" in result

    def test_metadata_block_is_stable_with_same_conversation_start(self):
        """Metadata block should be stable when conversation_start_date is the same."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        conversation_start = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        
        # Call twice with same conversation_start but different dynamic values
        result1 = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            timezone="UTC",
            previous_message_count=100,
            conversation_start_date=conversation_start,
        )
        
        result2 = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime(2026, 1, 20, 15, 30, 0, tzinfo=timezone.utc),
            timezone="UTC",
            previous_message_count=500,  # Much larger - but should be ignored
            conversation_start_date=conversation_start,  # Same start date
        )
        
        # Both should be identical
        assert result1 == result2

    def test_metadata_block_includes_tool_instructions(self):
        """Metadata block should include instructions for using tools."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        result = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="UTC",
        )
        
        # Should have instructions for getting current time
        assert "date" in result.lower() or "bash" in result.lower()
        # Should have instructions for searching conversations
        assert "conversation_search" in result

    def test_metadata_block_does_not_include_dynamic_date(self):
        """Metadata block should NOT include current system date (was cache-breaking)."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        result = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="UTC",
        )
        
        # Should NOT contain "current system date" (old dynamic field)
        assert "current system date" not in result.lower()
        # Should NOT contain "current time" (old dynamic field)
        assert "current time is:" not in result.lower()

    def test_metadata_block_does_not_include_message_count(self):
        """Metadata block should NOT include previous_message_count (was cache-breaking every turn)."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        result = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="UTC",
            previous_message_count=178,
        )
        
        # Should NOT contain the actual message count
        assert "178" not in result
        # Should NOT contain "previous messages" phrasing
        assert "previous messages" not in result.lower()

    def test_metadata_block_does_not_include_memory_edit_timestamp(self):
        """Metadata block should NOT include memory_edit_timestamp (was cache-breaking on memory edits)."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        result = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime(2026, 1, 27, 3, 43, 0, tzinfo=timezone.utc),
            timezone="UTC",
        )
        
        # Should NOT contain "last modified" or similar
        assert "last modified" not in result.lower()
        assert "were last modified" not in result.lower()

    def test_metadata_block_different_timezones(self):
        """Metadata block should format conversation_start_date in the given timezone."""
        from letta.prompts.prompt_generator import PromptGenerator
        
        conversation_start = datetime(2026, 1, 15, 18, 30, 0, tzinfo=timezone.utc)
        
        result_utc = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="UTC",
            conversation_start_date=conversation_start,
        )
        
        result_pst = PromptGenerator.compile_memory_metadata_block(
            memory_edit_timestamp=datetime.now(timezone.utc),
            timezone="America/Los_Angeles",
            conversation_start_date=conversation_start,
        )
        
        # Both should have conversation start, but formatted differently
        assert "Conversation started:" in result_utc
        assert "Conversation started:" in result_pst
        # Times should differ due to timezone conversion
        # (18:30 UTC = 10:30 PST)


class TestPromptCachingStability:
    """Integration tests for prompt caching stability."""

    def test_system_message_stable_across_turns(self):
        """System message should be identical across multiple turns (given same inputs)."""
        from letta.prompts.prompt_generator import PromptGenerator
        from letta.schemas.memory import Memory
        from letta.schemas.block import Block
        
        # Create a simple memory
        block = Block(
            label="test",
            value="Test value",
            limit=1000,
        )
        memory = Memory(blocks=[block])
        
        conversation_start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        # Simulate "turn 1" - 10 messages
        result1 = PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt="You are a helpful assistant.\n\n{CORE_MEMORY}",
            memory_with_sources=memory.compile(),
            in_context_memory_last_edit=datetime(2026, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
            timezone="UTC",
            previous_message_count=10,
            archival_memory_size=5,
            conversation_start_date=conversation_start,
        )
        
        # Simulate "turn 50" - 100 messages, memory edited
        result2 = PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt="You are a helpful assistant.\n\n{CORE_MEMORY}",
            memory_with_sources=memory.compile(),
            in_context_memory_last_edit=datetime(2026, 1, 20, 15, 30, 0, tzinfo=timezone.utc),  # Different!
            timezone="UTC",
            previous_message_count=100,  # Different!
            archival_memory_size=50,     # Different!
            conversation_start_date=conversation_start,  # Same!
        )
        
        # System messages should be IDENTICAL for cache efficiency
        assert result1 == result2

    def test_system_message_changes_only_with_memory_content(self):
        """System message should only change when actual memory content changes."""
        from letta.prompts.prompt_generator import PromptGenerator
        from letta.schemas.memory import Memory
        from letta.schemas.block import Block
        
        conversation_start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        # Memory version 1
        block1 = Block(label="test", value="Value version 1", limit=1000)
        memory1 = Memory(blocks=[block1])
        
        result1 = PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt="You are a helpful assistant.\n\n{CORE_MEMORY}",
            memory_with_sources=memory1.compile(),
            in_context_memory_last_edit=datetime.now(timezone.utc),
            timezone="UTC",
            conversation_start_date=conversation_start,
        )
        
        # Memory version 2 (content changed)
        block2 = Block(label="test", value="Value version 2 - CHANGED", limit=1000)
        memory2 = Memory(blocks=[block2])
        
        result2 = PromptGenerator.get_system_message_from_compiled_memory(
            system_prompt="You are a helpful assistant.\n\n{CORE_MEMORY}",
            memory_with_sources=memory2.compile(),
            in_context_memory_last_edit=datetime.now(timezone.utc),
            timezone="UTC",
            conversation_start_date=conversation_start,
        )
        
        # System messages should be DIFFERENT because memory content changed
        assert result1 != result2
        assert "Value version 1" in result1
        assert "Value version 2 - CHANGED" in result2
